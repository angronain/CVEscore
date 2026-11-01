#!/usr/bin/env python3
"""Build the training sets for the v2 label-harmonisation ablation, one folder per variant.

Usage:
    python src/make_variants.py corrector base tokens
    python src/make_variants.py corrected both --corrector-probs results/probs_corrector_relabel.npz

Variants (all use the stripped descriptions and the same time split):
    corrector  CNA vector + CNA identity + description -> NVD vector, on CVEs scored by both (pre-2025).
               relabel.parquet holds the CNA-only training CVEs it will relabel.
    base       v1 setup: one row per CVE, NVD label preferred, no tags.
    tokens     every label is a row tagged by who gave it: "[NVD] ..." or "[CNA: patchstack] ...".
               CVEs scored by both give two rows. test.parquet uses the tag of the test label's scorer,
               test_nvd.parquet asks for "[NVD]" everywhere.
    corrected  base, with CNA-only labels replaced by the corrector where it is confident.
    both       tokens, plus an "[NVD]" row with the corrected label for every CNA-only CVE.

Rows with is_aux=True are extra labels for a CVE that already has a main row; train.py keeps them out
of validation. Only CVEs published before --val-start are relabelled, so validation keeps raw labels.

Reads:  data/processed/{train,test}.parquet
Writes: data/variants/<name>/{train,test[,test_nvd,relabel]}.parquet
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cvss_text import METRICS
from model import LABELS

# ADPs and some CNAs are identified by UUID in the feeds
KNOWN_SCORERS = {"134c704f-9b21-4f2e-91b3-4a467353bcc0": "cisa-adp"}


def scorer_name(scorer: str) -> str:
    if scorer in KNOWN_SCORERS:
        return KNOWN_SCORERS[scorer]
    if "@" in scorer:
        return scorer.split("@")[1].rsplit(".", 1)[0]          # security@wordfence.com -> wordfence
    return scorer[:8]


def tag(scorer: str | None) -> str:
    return "[NVD]" if scorer is None else f"[CNA: {scorer_name(scorer)}]"


def labels_of(vectors: pd.Series) -> pd.DataFrame:
    parts = vectors.map(lambda v: dict(p.split(":") for p in v.split("/")[1:]))
    return pd.DataFrame({k: parts.map(lambda d: d[k]) for k in METRICS}, index=vectors.index)


def short_vector(vectors: pd.Series) -> pd.Series:
    return vectors.str.replace(r"^CVSS:3\.\d/", "", regex=True)


def rows(df: pd.DataFrame, text: pd.Series, vectors: pd.Series, is_aux: bool) -> pd.DataFrame:
    out = df[["id", "published", "source", "scorer"]].copy()
    out["description"] = text.values
    out["vector"] = vectors.values
    out[METRICS] = labels_of(vectors).values
    out["is_aux"] = is_aux
    return out


def corrector_input(df: pd.DataFrame) -> pd.Series:
    return df["cna_scorer"].map(tag) + " " + short_vector(df["cna_vector"]) + " | " + df["description"]


def matched_tags(df: pd.DataFrame) -> pd.Series:
    """Tag of whoever gave the row's label: NVD for Primary, else the CNA/ADP that scored it."""
    return np.where(df["source"] == "Primary", "[NVD]", df["scorer"].map(tag))


def tokens_train(train: pd.DataFrame) -> pd.DataFrame:
    nvd = train[train["nvd_vector"].notna()]
    cna = train[train["cna_vector"].notna()]
    return pd.concat([
        rows(nvd, "[NVD] " + nvd["description"], nvd["nvd_vector"], is_aux=False),
        # CNA label is the main row only when NVD never scored the CVE
        rows(cna, cna["cna_scorer"].map(tag) + " " + cna["description"], cna["cna_vector"], is_aux=False)
        .assign(is_aux=cna["nvd_vector"].notna().values),
    ]).sort_values("published", kind="stable").reset_index(drop=True)


def tagged_tests(test: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"test": test.assign(description=matched_tags(test) + " " + test["description"]),
            "test_nvd": test.assign(description="[NVD] " + test["description"])}


def corrected_labels(train: pd.DataFrame, probs_path: Path, threshold: float, val_start: str) -> pd.DataFrame:
    """Labels of the CNA-only CVEs before val_start, with each metric replaced where the corrector is sure."""
    z = np.load(probs_path, allow_pickle=True)
    pos = pd.Series(np.arange(len(z["ids"])), index=z["ids"])
    cand = train[train["nvd_vector"].isna() & (train["published"] < val_start)]
    new = labels_of(cand["vector"])
    idx = pos.loc[cand["id"]].values
    changed = 0
    for k in METRICS:
        p = z[k][idx]
        sure = p.max(1) >= threshold
        guess = np.array(LABELS[k])[p.argmax(1)]
        changed += int((sure & (guess != new[k].values)).sum())
        new[k] = np.where(sure, guess, new[k].values)
    new["vector"] = "CVSS:3.1/" + new[METRICS].apply(lambda r: "/".join(f"{k}:{r[k]}" for k in METRICS), axis=1)
    print(f"  corrector changed {changed:,} metric labels on {len(cand):,} CNA-only CVEs "
          f"({(new['vector'] != cand['vector']).mean():.1%} of their vectors)")
    return new


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("variants", nargs="+", choices=["corrector", "base", "tokens", "corrected", "both"])
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("data/variants"))
    p.add_argument("--val-start", default="2024-10-01")
    p.add_argument("--corrector-probs", type=Path, default=Path("results/probs_corrector_relabel.npz"))
    p.add_argument("--threshold", type=float, default=0.8)
    args = p.parse_args()

    train = pd.read_parquet(args.data / "train.parquet")
    test = pd.read_parquet(args.data / "test.parquet")

    for name in args.variants:
        files = {}
        if name == "corrector":
            pairs = lambda d: d[d["nvd_vector"].notna() & d["cna_vector"].notna()]
            tr, te = pairs(train), pairs(test)
            cand = train[train["nvd_vector"].isna()]
            files = {"train": rows(tr, corrector_input(tr), tr["nvd_vector"], False),
                     "test": rows(te, corrector_input(te), te["nvd_vector"], False),
                     # labels here are the CNA's own, so the "test" metrics on it measure how often it copies
                     "relabel": rows(cand, corrector_input(cand), cand["cna_vector"], False)}
        elif name == "base":
            files = {"train": rows(train, train["description"], train["vector"], False), "test": test}
        elif name == "tokens":
            files = {"train": tokens_train(train), **tagged_tests(test)}
        elif name == "corrected":
            new = corrected_labels(train, args.corrector_probs, args.threshold, args.val_start)
            t = train.copy()
            t.loc[new.index, ["vector", *METRICS]] = new[["vector", *METRICS]].values
            files = {"train": rows(t, t["description"], t["vector"], False), "test": test}
        elif name == "both":
            new = corrected_labels(train, args.corrector_probs, args.threshold, args.val_start)
            cand = train.loc[new.index]
            extra = rows(cand, "[NVD] " + cand["description"], new["vector"], is_aux=True)
            files = {"train": pd.concat([tokens_train(train), extra]).sort_values("published", kind="stable"),
                     **tagged_tests(test)}

        d = args.out / name
        d.mkdir(parents=True, exist_ok=True)
        for f, df in files.items():
            df.reset_index(drop=True).to_parquet(d / f"{f}.parquet", index=False)
        tr = files["train"]
        aux = int(tr["is_aux"].sum()) if "is_aux" in tr else 0
        print(f"{name}: train {len(tr):,} rows ({aux:,} aux) | "
              + " | ".join(f"{f} {len(df):,}" for f, df in files.items() if f != "train"))
        print(f"  e.g. {re.sub(r'\s+', ' ', tr['description'].iloc[-1])[:140]}")


if __name__ == "__main__":
    main()

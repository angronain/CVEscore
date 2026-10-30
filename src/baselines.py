#!/usr/bin/env python3
"""Majority-class and TF-IDF + logistic regression baselines for CVSS v3.1 prediction.

Usage:
    python src/baselines.py
    python src/baselines.py --max-features 100000 --C 2

Reads:   data/processed/train.parquet, data/processed/test.parquet
Writes:  results/preds_majority.parquet, results/preds_tfidf.parquet, results/baselines.json
"""
import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from cvss import CVSS3
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]
SEVERITIES = ["None", "Low", "Medium", "High", "Critical"]


# ---------- CVSS helpers ----------

def build_vectors(df: pd.DataFrame) -> pd.Series:
    """Rebuild 'CVSS:3.1/AV:N/AC:L/...' strings from the 8 metric columns."""
    vec = pd.Series("CVSS:3.1", index=df.index)
    for k in METRICS:
        vec = vec + f"/{k}:" + df[k].astype(str)
    return vec


@lru_cache(maxsize=None)   # only ~2.6k distinct vectors exist, so caching makes this instant
def score_vector(vector: str) -> tuple[float, str]:
    c = CVSS3(vector)
    return float(c.scores()[0]), c.severities()[0]


def add_scores(df: pd.DataFrame, vector_col: str = "vector") -> pd.DataFrame:
    scored = df[vector_col].map(score_vector)
    df["score"] = scored.str[0]
    df["severity"] = scored.str[1]
    return df


# ---------- Evaluation ----------

def evaluate(test: pd.DataFrame, pred: pd.DataFrame, seconds: float) -> dict:
    """Compare predictions to ground truth. Both frames must be row-aligned."""
    res = {"per_metric": {}}
    for k in METRICS:
        res["per_metric"][k] = {
            "acc": round(accuracy_score(test[k], pred[k]), 4),
            "macro_f1": round(f1_score(test[k], pred[k], average="macro"), 4),
        }
    res["exact_match"] = round(float((test[METRICS].values == pred[METRICS].values).all(axis=1).mean()), 4)
    res["score_mae"] = round(float(np.abs(test["score"].values - pred["score"].values).mean()), 3)
    res["severity_acc"] = round(float((test["severity"].values == pred["severity"].values).mean()), 4)
    res["severity_macro_f1"] = round(f1_score(test["severity"], pred["severity"], average="macro"), 4)
    res["ms_per_cve"] = round(1000 * seconds / len(test), 4)
    res["n_test"] = len(test)
    return res


def save_preds(name: str, test: pd.DataFrame, pred: pd.DataFrame, out: Path) -> None:
    cols = ["id", *METRICS, "vector", "score", "severity", "confidence"]
    pred.assign(id=test["id"].values)[cols].to_parquet(out / f"preds_{name}.parquet", index=False)


# ---------- Baselines ----------

def majority_baseline(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    pred = pd.DataFrame({k: [train[k].mode()[0]] * len(test) for k in METRICS})
    pred["confidence"] = np.nan   # no notion of confidence for a constant predictor
    return pred, time.perf_counter() - t0


def fit_one(X, y, C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=1000, class_weight=None).fit(X, y)


def tfidf_baseline(train, test, max_features: int, C: float, n_jobs: int):
    # Fit the vectorizer on train only: fitting on test would leak vocabulary from the future
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max_features,
                                 sublinear_tf=True, strip_accents="unicode")
    print("  vectorizing...")
    X_train = vectorizer.fit_transform(train["description"])

    print(f"  training {len(METRICS)} classifiers in parallel...")
    models = Parallel(n_jobs=n_jobs)(delayed(fit_one)(X_train, train[k], C) for k in METRICS)

    # Time only inference (vectorize + predict), which is what "speed per CVE" means
    t0 = time.perf_counter()
    X_test = vectorizer.transform(test["description"])
    pred, conf = {}, np.ones(len(test))
    for k, m in zip(METRICS, models):
        proba = m.predict_proba(X_test)
        pred[k] = m.classes_[proba.argmax(axis=1)]
        conf *= proba.max(axis=1)   # vector confidence = product of the 8 heads' top probabilities
    seconds = time.perf_counter() - t0

    pred = pd.DataFrame(pred)
    pred["confidence"] = conf
    return pred, seconds


# ---------- Main ----------

def print_table(results: dict) -> None:
    rows = {}
    for name, r in results.items():
        row = {k: r["per_metric"][k]["acc"] for k in METRICS}
        row.update(exact=r["exact_match"], sev_acc=r["severity_acc"],
                   mae=r["score_mae"], ms=r["ms_per_cve"])
        rows[name] = row
    print("\n" + pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--max-features", type=int, default=200_000)
    p.add_argument("--C", type=float, default=4.0)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--source", choices=["all", "Primary", "Secondary"], default="all",
                   help="restrict the test set to NVD-scored (Primary) or CNA-scored (Secondary) CVEs")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.data / "train.parquet")
    test = pd.read_parquet(args.data / "test.parquet")
    if args.source != "all":
        test = test[test["source"] == args.source]
    test = add_scores(test.reset_index(drop=True))   # recompute so truth and preds use the same formula
    print(f"Train: {len(train):,} CVEs | Test: {len(test):,} CVEs ({args.source})")

    results = {}
    for name, run in [
        ("majority", lambda: majority_baseline(train, test)),
        ("tfidf_logreg", lambda: tfidf_baseline(train, test, args.max_features, args.C, args.n_jobs)),
    ]:
        print(f"\n[{name}]")
        pred, seconds = run()
        pred["vector"] = build_vectors(pred)
        pred = add_scores(pred)
        results[name] = evaluate(test, pred, seconds)
        save_preds(name, test, pred, args.out)

    suffix = "" if args.source == "all" else f"_{args.source.lower()}"
    (args.out / f"baselines{suffix}.json").write_text(json.dumps(results, indent=2))
    print_table(results)
    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()
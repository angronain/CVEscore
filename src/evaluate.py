#!/usr/bin/env python3
"""Compare every system on the same test data, draw plots, write a markdown report.

Usage:
    python src/evaluate.py
    python src/evaluate.py --target 0.95     # coverage needed for 95% severity accuracy

Reads:  data/processed/test.parquet, results/preds_*.parquet, results/*.json (for timings)
Writes: results/results.md, results/summary.json, results/coverage.png, results/per_metric_f1.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines import METRICS, add_scores, evaluate

PRETTY = {"majority": "Majority class", "tfidf": "TF-IDF + LogReg", "llm": "Zero-shot LLM",
          "deberta": "DeBERTa-v3 (ours)", "modernbert": "ModernBERT (ours)"}


def load_timings(results_dir: Path) -> dict:
    """ms/CVE comes from the scripts that ran inference; pull it from their JSON outputs."""
    t = {}
    for f in results_dir.glob("*.json"):
        if f.name in ("summary.json",):
            continue
        data = json.loads(f.read_text())
        if "test" in data and "ms_per_cve" in data["test"]:            # train.py output
            t[f.stem] = data["test"]["ms_per_cve"]
        for name, r in data.items():                                    # baselines / llm outputs
            if isinstance(r, dict) and "ms_per_cve" in r and name != "test":
                t.setdefault(name, r["ms_per_cve"])
    return t


def aligned(test: pd.DataFrame, pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both frames to shared ids, same order."""
    ids = pred["id"][pred["id"].isin(test["id"])]
    t = test.set_index("id").loc[ids].reset_index()
    p = pred.set_index("id").loc[ids].reset_index()
    return t, p


def row(name: str, r: dict, ms: float | None) -> dict:
    return {"System": PRETTY.get(name, name), "N": r["n_test"],
            **{k: r["per_metric"][k]["macro_f1"] for k in METRICS},
            "Exact": r["exact_match"], "Sev. acc": r["severity_acc"], "Sev. F1": r["severity_macro_f1"],
            "Score MAE": r["score_mae"], "ms/CVE": ms}


def coverage_curve(t: pd.DataFrame, p: pd.DataFrame, points: int = 50):
    """Accuracy when only the top-x% most confident predictions are kept."""
    order = np.argsort(-p["confidence"].values)
    sev_ok = (t["severity"].values == p["severity"].values)[order]
    exact_ok = (t[METRICS].values == p[METRICS].values).all(axis=1)[order]
    n = len(order)
    cov = np.linspace(0.02, 1.0, points)
    k = np.maximum(1, (cov * n).astype(int))
    sev = np.array([sev_ok[:i].mean() for i in k])
    exact = np.array([exact_ok[:i].mean() for i in k])
    return cov, sev, exact, sev_ok


def coverage_at(sev_ok_sorted: np.ndarray, target: float) -> float:
    """Largest share of CVEs we can auto-score while keeping severity accuracy >= target."""
    running = np.cumsum(sev_ok_sorted) / np.arange(1, len(sev_ok_sorted) + 1)
    ok = np.where(running >= target)[0]
    return float((ok.max() + 1) / len(sev_ok_sorted)) if len(ok) else 0.0


def md_table(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.3f}"
        return "" if v is None else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--target", type=float, default=0.95)
    args = p.parse_args()

    test = add_scores(pd.read_parquet(args.data / "test.parquet").reset_index(drop=True))
    timings = load_timings(args.results)
    preds = {f.stem.removeprefix("preds_"): pd.read_parquet(f) for f in sorted(args.results.glob("preds_*.parquet"))}
    if not preds:
        raise SystemExit("No predictions found in results/. Run baselines.py and train.py first.")

    full = {n: pr for n, pr in preds.items() if len(pr) == len(test)}
    report, summary = ["# CVEscore results\n",
                       f"Test set: {len(test):,} CVEs published from "
                       f"{test['published'].min():%Y-%m-%d} to {test['published'].max():%Y-%m-%d}. "
                       "Per-metric columns are macro F1.\n"], {}

    # ---- Table 1: full test set ----
    rows = []
    for name, pr in full.items():
        t, pp = aligned(test, pr)
        r = evaluate(t, pp, seconds=0)
        rows.append(row(name, r, timings.get(name)))
        summary[name] = r
    report += ["## Full test set\n", md_table(pd.DataFrame(rows)), ""]

    # ---- Table 2: NVD-scored vs CNA-scored CVEs ----
    report.append("## By scoring source (exact vector match / severity accuracy)\n")
    src_rows = []
    for name, pr in full.items():
        t, pp = aligned(test, pr)
        rec = {"System": PRETTY.get(name, name)}
        for src in ["Primary", "Secondary"]:
            m = (t["source"] == src).values
            if m.sum():
                rec[f"{src} exact"] = float((t[METRICS].values[m] == pp[METRICS].values[m]).all(axis=1).mean())
                rec[f"{src} sev"] = float((t["severity"].values[m] == pp["severity"].values[m]).mean())
        src_rows.append(rec)
    counts = test["source"].value_counts().to_dict()
    report += [f"Primary = scored by NVD ({counts.get('Primary', 0):,}), "
               f"Secondary = scored by the CNA/vendor ({counts.get('Secondary', 0):,}).\n",
               md_table(pd.DataFrame(src_rows)), ""]

    # ---- Table 3: LLM comparison on its sample ----
    if "llm" in preds:
        llm_ids = preds["llm"]["id"]
        sub = test[test["id"].isin(llm_ids)]
        rows = []
        for name, pr in preds.items():
            t, pp = aligned(sub, pr[pr["id"].isin(llm_ids)])
            rows.append(row(name, evaluate(t, pp, seconds=0), timings.get(name)))
        report += [f"## Same {len(sub)}-CVE sample as the zero-shot LLM\n", md_table(pd.DataFrame(rows)), ""]

    # ---- Coverage plot ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    cov_rows = []
    for name, pr in full.items():
        if pr["confidence"].isna().all():
            continue
        t, pp = aligned(test, pr)
        cov, sev, exact, sev_sorted = coverage_curve(t, pp)
        label = PRETTY.get(name, name)
        ax[0].plot(cov * 100, sev * 100, label=label, linewidth=2)
        ax[1].plot(cov * 100, exact * 100, label=label, linewidth=2)
        c = coverage_at(sev_sorted, args.target)
        cov_rows.append({"System": label, f"Auto-scored at {args.target:.0%} severity acc.": f"{c:.1%}"})
        summary.setdefault(name, {})[f"coverage_at_{args.target}"] = c
    for a, title in zip(ax, ["Severity accuracy", "Exact vector match"]):
        a.set_xlabel("Coverage: % of CVEs auto-scored (most confident first)")
        a.set_ylabel(f"{title} (%)")
        a.set_title(title)
        a.grid(alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].axhline(args.target * 100, color="grey", linestyle="--", linewidth=1)
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.results / "coverage.png", dpi=150)
    if cov_rows:
        report += ["## Human-in-the-loop triage\n",
                   "Predictions are sorted by confidence (product of the 8 heads' top probabilities). "
                   "The rest go to an analyst.\n",
                   md_table(pd.DataFrame(cov_rows)), "", "![coverage](coverage.png)", ""]

    # ---- Per-metric F1 plot ----
    fig, ax = plt.subplots(figsize=(10, 4))
    names = list(full)
    width = 0.8 / len(names)
    x = np.arange(len(METRICS))
    for i, name in enumerate(names):
        vals = [summary[name]["per_metric"][k]["macro_f1"] for k in METRICS]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=PRETTY.get(name, name))
    ax.set_xticks(x, METRICS)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=len(names))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.results / "per_metric_f1.png", dpi=150)
    report += ["## Per-metric macro F1\n", "![per metric](per_metric_f1.png)", ""]

    (args.results / "results.md").write_text("\n".join(report))
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(report))
    print(f"\nWrote {args.results}/results.md, coverage.png, per_metric_f1.png")


if __name__ == "__main__":
    main()

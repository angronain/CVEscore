#!/usr/bin/env python3
"""Parse NVD feeds into time-split train/test parquet files.

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --test-start 2025-01-01

Reads:  data/raw/nvdcve-2.0-*.json.gz
Writes: data/processed/train.parquet, data/processed/test.parquet
"""
import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]


def parse_feed(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        feed = json.load(f)
    for item in feed.get("vulnerabilities", []):
        cve = item["cve"]
        if cve.get("vulnStatus") == "Rejected":
            continue
        m31 = cve.get("metrics", {}).get("cvssMetricV31", [])
        if not m31:
            continue
        # Prefer NVD's own score (Primary); fall back to the CNA's (Secondary)
        m = next((x for x in m31 if x.get("type") == "Primary"), m31[0])
        desc = next((d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"), None)
        if not desc or desc.startswith("** REJECT"):
            continue
        vector = m["cvssData"]["vectorString"]
        try:
            parts = dict(p.split(":") for p in vector.split("/")[1:])
            labels = {k: parts[k] for k in METRICS}
        except (KeyError, ValueError):
            continue   # malformed vector
        rows.append({
            "id": cve["id"],
            "published": cve["published"],
            "description": " ".join(desc.split()),   # normalize whitespace
            "vector": vector,
            "nvd_score": m["cvssData"]["baseScore"],
            "source": m.get("type", "Unknown"),
            "scorer": m.get("source", ""),
            **labels,
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--test-start", default="2025-01-01", help="CVEs published on/after this date go to test")
    args = p.parse_args()

    files = sorted(args.raw.glob("nvdcve-2.0-*.json.gz"))
    files = [f for f in files if f.name.split("-")[-1].split(".")[0].isdigit()]   # yearly feeds only
    if not files:
        raise SystemExit(f"No feeds in {args.raw}/. Run src/fetch_nvd.py first.")

    rows = []
    for f in files:
        r = parse_feed(f)
        print(f"  {f.name}: {len(r):,} CVEs with a CVSS v3.1 vector")
        rows.extend(r)

    df = pd.DataFrame(rows).drop_duplicates("id")
    df["published"] = pd.to_datetime(df["published"], format="ISO8601")
    df = df.sort_values("published").reset_index(drop=True)

    cutoff = pd.Timestamp(args.test_start)
    train, test = df[df.published < cutoff], df[df.published >= cutoff]

    # Exact-duplicate descriptions across the split would be free points: drop them from test
    leaked = test["description"].isin(set(train["description"]))
    test = test[~leaked]

    args.out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.out / "train.parquet", index=False)
    test.to_parquet(args.out / "test.parquet", index=False)

    print(f"\nTrain: {len(train):,} CVEs (published < {args.test_start})")
    print(f"Test:  {len(test):,} CVEs (published >= {args.test_start}, "
          f"{leaked.sum():,} exact-duplicate descriptions removed)")
    print("Test scored by:", test["source"].value_counts().to_dict())
    print("\nLabel distribution (train):")
    for k in METRICS:
        print(f"  {k:>2}: " + "  ".join(f"{v}={c:.1%}" for v, c in train[k].value_counts(normalize=True).items()))


if __name__ == "__main__":
    main()

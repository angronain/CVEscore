#!/usr/bin/env python3
"""Zero-shot LLM baseline through LM Studio's OpenAI-compatible server.

Usage:
    python src/llm_baseline.py                       # 300 CVEs, first loaded model
    python src/llm_baseline.py --n 500 --model qwen3-8b

Start LM Studio's server first (Developer tab > Start Server, default port 1234).
Writes: results/llm_raw.jsonl (cache, makes runs resumable), results/preds_llm.parquet,
        results/llm_baseline.json
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

from baselines import METRICS, add_scores, build_vectors, evaluate, save_preds

ALLOWED = {"AV": "NALP", "AC": "LH", "PR": "NLH", "UI": "NR", "S": "UC", "C": "NLH", "I": "NLH", "A": "NLH"}

SYSTEM = """You are an expert vulnerability analyst scoring CVEs with CVSS v3.1 base metrics.
Metric definitions:
- AV (Attack Vector): N=Network (remotely exploitable), A=Adjacent network, L=Local (local access or user opens a file), P=Physical
- AC (Attack Complexity): L=Low, H=High (requires race condition, specific config, MitM, etc.)
- PR (Privileges Required): N=None, L=Low (basic user account), H=High (admin)
- UI (User Interaction): N=None, R=Required (victim must click, open a file, visit a page)
- S (Scope): U=Unchanged, C=Changed (impact crosses a security boundary, e.g. XSS affecting the browser, VM escape)
- C, I, A (Confidentiality, Integrity, Availability impact): N=None, L=Low, H=High
Answer with ONLY the vector string, nothing else, in this exact format:
CVSS:3.1/AV:_/AC:_/PR:_/UI:_/S:_/C:_/I:_/A:_"""


def parse_vector(text: str) -> tuple[dict, bool]:
    """Extract each metric independently. Returns (values, fully_parsed)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)   # strip reasoning blocks
    values = {}
    for k, allowed in ALLOWED.items():
        m = re.search(rf"(?<![A-Z]){k}:([{allowed}])\b", text)
        values[k] = m.group(1) if m else None
    return values, all(v is not None for v in values.values())


def query(client: OpenAI, model: str, description: str, max_tokens: int) -> tuple[str, float]:
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model=model, temperature=0, max_tokens=max_tokens,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"CVE description:\n{description}"}],
    )
    return r.choices[0].message.content or "", time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default=os.environ.get("LLM_MODEL"), help="LM Studio model id (default: first loaded)")
    p.add_argument("--url", default=os.environ.get("LLM_URL", "http://localhost:1234/v1"))
    p.add_argument("--max-tokens", type=int, default=64, help="raise to ~2048 for reasoning models")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.url, api_key="lm-studio")
    model = args.model or client.models.list().data[0].id
    print(f"Model: {model} @ {args.url}")

    train = pd.read_parquet(args.data / "train.parquet")
    test = pd.read_parquet(args.data / "test.parquet")
    sample = test.sample(n=min(args.n, len(test)), random_state=args.seed).reset_index(drop=True)
    sample = add_scores(sample)
    fallback = {k: train[k].mode()[0] for k in METRICS}

    # Resumable cache: one JSON line per CVE, keyed by model + CVE id
    cache_path = args.out / "llm_raw.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            cache[(r["model"], r["id"])] = r

    rows = []
    with cache_path.open("a") as f:
        for i, row in sample.iterrows():
            key = (model, row["id"])
            if key not in cache:
                text, secs = query(client, model, row["description"], args.max_tokens)
                cache[key] = {"model": model, "id": row["id"], "response": text, "seconds": secs}
                f.write(json.dumps(cache[key]) + "\n")
                f.flush()
            rows.append(cache[key])
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(sample)}")

    # Parse; any metric the model failed to give falls back to the train majority value
    parsed, full = [], []
    for r in rows:
        values, ok = parse_vector(r["response"])
        parsed.append({k: v if v is not None else fallback[k] for k, v in values.items()})
        full.append(ok)

    pred = pd.DataFrame(parsed)
    pred["confidence"] = np.nan
    pred["vector"] = build_vectors(pred)
    pred = add_scores(pred)

    seconds = sum(r["seconds"] for r in rows)
    results = {"llm": evaluate(sample, pred, seconds)}
    results["llm"].update(model=model, parse_rate=round(float(np.mean(full)), 4))
    save_preds("llm", sample, pred, args.out)

    # Fair comparison: TF-IDF evaluated on the exact same CVEs
    tfidf_path = args.out / "preds_tfidf.parquet"
    if tfidf_path.exists():
        tf = pd.read_parquet(tfidf_path).set_index("id").loc[sample["id"]].reset_index()
        results["tfidf_same_sample"] = evaluate(sample, tf, seconds=0)
        results["tfidf_same_sample"].pop("ms_per_cve")   # timing came from the full run

    (args.out / "llm_baseline.json").write_text(json.dumps(results, indent=2))
    for name, r in results.items():
        print(f"\n[{name}] exact={r['exact_match']}  sev_acc={r['severity_acc']}  mae={r['score_mae']}")
    print(f"parse rate: {results['llm']['parse_rate']:.1%}  |  {1000 * seconds / len(rows):.0f} ms/CVE")


if __name__ == "__main__":
    main()

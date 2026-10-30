#!/usr/bin/env python3
"""Upload a trained checkpoint to the Hugging Face Hub with a generated model card.

Usage:
    HF_TOKEN=hf_xxx python scripts/push_to_hub.py --repo <your-username>/cvescore-deberta
"""
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

CARD = """---
license: mit
language: en
tags: [cvss, cve, vulnerability, security, text-classification]
datasets: [nvd]
base_model: {base_model}
---

# CVEscore: CVSS v3.1 vector prediction from CVE descriptions

Fine-tuned `{base_model}` with 8 classification heads (AV, AC, PR, UI, S, C, I, A).
The base score and severity are computed from the predicted vector with the official CVSS v3.1 formula.

## Training data and split
NVD CVE JSON 2.0 feeds, CVEs with a CVSS v3.1 vector (NVD score preferred, CNA score otherwise).
**Time-based split**: train on CVEs published before {val_start}, validate up to the end of 2024,
test on CVEs published from 2025 onwards. No random split, so no leakage from the future.

## Test results ({n_test:,} CVEs published in 2025+)
| Metric | Value |
|---|---|
| Exact vector match | {exact:.3f} |
| Severity accuracy | {sev:.3f} |
| Base score MAE | {mae:.3f} |
| Inference speed | {ms:.2f} ms/CVE (RTX 5080, batched) |

Per-metric macro F1: {per_metric}

## Intended use
Triage aid: pre-score new CVEs before official scoring is available, and route low-confidence
predictions (product of the 8 heads' top probabilities) to a human analyst.
Not a replacement for analyst scoring.

## Limitations
- English descriptions only; vague descriptions ("unspecified vulnerability") are scored poorly.
- Labels mix NVD and vendor (CNA) scores, which sometimes disagree.
- CVSS v3.1 only (no v4.0).

## Usage
Code, training pipeline and the `CVSSModel` class: https://github.com/atomic-crafter/CVEscore

```python
from model import CVSSModel, predict_proba   # src/model.py in the GitHub repo
model, tokenizer, config = CVSSModel.load("path/to/this/repo", device="cuda")
```
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="e.g. username/cvescore-deberta")
    p.add_argument("--run-name", default="deberta")
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints"))
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    run_dir = args.ckpt / args.run_name
    cfg = json.loads((run_dir / "cvss_config.json").read_text())
    res = json.loads((args.results / f"{args.run_name}.json").read_text())["test"]
    (run_dir / "README.md").write_text(CARD.format(
        base_model=cfg["base_model"], val_start=cfg["val_start"], n_test=res["n_test"],
        exact=res["exact_match"], sev=res["severity_acc"], mae=res["score_mae"], ms=res["ms_per_cve"],
        per_metric=", ".join(f"{k} {v['macro_f1']:.3f}" for k, v in res["per_metric"].items()),
    ))

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(run_dir), repo_id=args.repo, commit_message="Upload CVEscore model")
    print(f"Uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()

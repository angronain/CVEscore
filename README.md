# CVEscore

Predict the full **CVSS v3.1 base vector** (AV, AC, PR, UI, S, C, I, A) of a vulnerability from its raw CVE description, then compute its base score and severity with the official formula.

**Model:** [huggingface.co/angronain/cvescore-deberta](https://huggingface.co/angronain/cvescore-deberta)

## Results

Trained on 143,380 CVEs published before 2025, tested on **123,460 CVEs published from 2025-01-01 to 2026-10-30**: data the model has never seen, from the future relative to training.

| System | Exact vector | Severity acc. | Score MAE | Mean macro F1 (8 metrics) | ms/CVE |
|---|---|---|---|---|---|
| Majority class | 0.083 | 0.128 | 2.84 | 0.303 | <0.01 |
| TF-IDF + LogReg | 0.386 | 0.621 | 1.09 | 0.754 | 0.05 (CPU) |
| **DeBERTa-v3-base (ours)** | **0.401** | **0.629** | **1.04** | **0.756** | 0.99 (RTX 5080) |

| | Exact vector | Severity acc. |
|---|---|---|
| CVEs scored by NVD (31,575) | 0.546 | 0.710 |
| CVEs scored by the CNA/vendor (91,885) | 0.351 | 0.602 |

Per-metric macro F1, DeBERTa: AV 0.761 · AC 0.672 · PR 0.741 · UI 0.880 · S 0.843 · C 0.760 · I 0.764 · A 0.625.

![per-metric F1](results/per_metric_f1.png)

### Key findings

1. **DeBERTa beats TF-IDF, but only slightly.** On the same CVEs, a paired bootstrap gives +1.5 pts exact-vector match (95% CI +1.3 to +1.7) and +0.8 pts severity accuracy (+0.6 to +1.1). Both gains are statistically real but small. A well-tuned bag-of-words model already captures most of what the descriptions reveal.
2. **The model's edge grows over time.** In early 2025, TF-IDF actually matches it on exact vectors (0.471 vs 0.451 in Q1). From Q4 2025 on, DeBERTa leads by 2–3 pts every quarter (0.370 vs 0.338 in 2026 Q3): it generalises better as phrasing drifts away from the training data.
3. **Accuracy drops mostly because NVD stopped scoring, not because the model decays.** NVD's own scores were 38% of test labels in early 2025 and are 8% by late 2026. CNA labels are much noisier: per-metric accuracy on C/I/A falls from about 0.86 (NVD) to about 0.75 (CNA). Some CNAs score very differently from the descriptions they publish: VulDB at 25% severity accuracy, Patchstack at 52%, versus Microsoft at 83% and Adobe at 91%.
4. **Score is close even when the vector is wrong.** 62% of predictions are within ±1.0 of the true base score and 78% within ±2.0. Only 5.7% are off by two or more severity levels. The main error is High ↔ Medium/Critical confusion.
5. **Confidence is informative, so triage works.** The 10% most confident predictions are 90% severity-accurate (81% exact vector); the 10% least confident are 42%. At a severity-accuracy target of 85%, **28% of new CVEs can be auto-scored**. At 80% the share is 40%, and at 95% it's 4%. When DeBERTa and TF-IDF predict the same vector (59% of CVEs), severity accuracy is 71% versus 51% when they disagree, so agreement between the two is a cheap second check.
6. **Some descriptions quote the answer.** 3.6% of test descriptions, mostly Oracle's, include the CVSS vector (Oracle: 96% exact). Excluding them, DeBERTa scores 0.380 exact and 0.616 severity accuracy, and the ranking is unchanged.

Full tables: [`results/results.md`](results/results.md) (from `src/evaluate.py`) and [`results/analysis.md`](results/analysis.md) (from `scripts/analyze_results.py`: significance, confusion matrix, drift, per-CNA, calibration).

![coverage](results/coverage.png)

## Why this project

New CVEs often wait days or weeks before NVD scores them, and since 2025 most never get an NVD score at all. A fast local model that pre-scores them, and flags the ones it is unsure about, helps security teams triage.

- **Time-based split.** Train on CVEs published before 2025, test on 2025+ CVEs. Most published work uses random splits, which leak near-duplicate descriptions between train and test and overstate accuracy.
- **Full vector, not just a severity label.** 8 classification heads on one shared encoder.
- **Human in the loop.** Each prediction has a confidence score; low-confidence CVEs are routed to an analyst. The report shows how many CVEs can be auto-scored at a given accuracy target.
- **Optional zero-shot LLM baseline** (`make pipeline-llm`, LM Studio on the host). It was not included in this run.

## Quickstart (Docker, NVIDIA GPU)

Requirements: Docker with Compose v2.26+, NVIDIA driver, and the NVIDIA Container Toolkit
(Arch: `pacman -S nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`;
Windows: Docker Desktop with the WSL 2 backend).

```bash
cp .env.example .env          # optional: edit settings
make data                     # download NVD feeds + build train/test (~10 min)
make smoke                    # 1-minute GPU check
make pipeline                 # baselines + training + evaluation
make pipeline-llm             # same, plus the LM Studio zero-shot baseline
make demo                     # Gradio demo on http://localhost:7860
make push                     # upload model + model card to Hugging Face
```

Without Docker: `make local-install` then `make local-pipeline` (in a Python 3.12 venv).

The defaults are tuned for an RTX 5080 (16 GB): bf16 autocast, batch size 32, max 256 tokens, 3 epochs.
PyTorch comes from the CUDA 12.8 wheels, which are required for RTX 50-series GPUs.

## Pipeline

| Step | Script | Output |
|---|---|---|
| Download NVD JSON 2.0 feeds (skips unchanged ones, checks SHA-256) | `src/fetch_nvd.py` | `data/raw/` |
| Parse, keep CVSS v3.1 CVEs, split by publication date | `src/prepare_data.py` | `data/processed/{train,test}.parquet` |
| Majority + TF-IDF/LogReg baselines | `src/baselines.py` | `results/preds_{majority,tfidf}.parquet` |
| Zero-shot LLM baseline via LM Studio | `src/llm_baseline.py` | `results/preds_llm.parquet` |
| Fine-tune encoder with 8 heads | `src/train.py` | `checkpoints/deberta/`, `results/preds_deberta.parquet` |
| Tables, coverage plot, per-metric F1 | `src/evaluate.py` | `results/results.md`, `*.png` |
| Significance, drift, per-CNA, calibration | `scripts/analyze_results.py` | `results/analysis.{md,json}` |
| Demo | `app.py` | Gradio UI |
| Publish | `scripts/push_to_hub.py` | Hugging Face model + card |

**Splits:** train = published before 2024-10-01, validation (checkpoint selection) = Oct to Dec 2024, test = 2025-01-01 onwards. Test CVEs whose description exactly matches a training description are removed.

**Labels:** NVD's own score (`Primary`) when available, otherwise the CNA/vendor score (`Secondary`). Results are reported separately for both.

**Confidence:** product of the 8 heads' top softmax probabilities.

## Keeping the data fresh

`fetch_nvd.py` only downloads feeds whose checksum changed, so it can run daily (cron, systemd timer or CI) to grow the test set with new CVEs.

## Limitations

- English descriptions only. Vague descriptions ("unspecified vulnerability") are hard for any model.
- NVD and CNA scores sometimes disagree, which caps achievable accuracy.
- The zero-shot LLM may have seen 2025 CVEs during pretraining if it was released after them.
- CVSS v3.1 only; CVSS v4.0 is not covered yet.

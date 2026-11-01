# CVEscore results

Test set: 123,459 CVEs published from 2025-01-01 to 2026-10-30. Per-metric columns are macro F1.

## Full test set

| System | N | AV | AC | PR | UI | S | C | I | A | Exact | Sev. acc | Sev. F1 | Score MAE | ms/CVE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Majority class | 123459 | 0.217 | 0.472 | 0.227 | 0.428 | 0.442 | 0.221 | 0.196 | 0.217 | 0.083 | 0.128 | 0.045 | 2.840 | 0.001 |
| TF-IDF + LogReg | 123459 | 0.744 | 0.686 | 0.706 | 0.876 | 0.844 | 0.745 | 0.762 | 0.667 | 0.384 | 0.623 | 0.398 | 1.091 | 0.051 |
| DeBERTa v1 (raw text) | 123459 | 0.761 | 0.672 | 0.741 | 0.880 | 0.843 | 0.760 | 0.764 | 0.625 | 0.401 | 0.629 | 0.403 | 1.036 | 0.985 |
| DeBERTa v2 base | 123459 | 0.772 | 0.672 | 0.734 | 0.874 | 0.842 | 0.752 | 0.764 | 0.589 | 0.392 | 0.613 | 0.392 | 1.064 | 0.978 |
| DeBERTa v2 + scorer tokens (ours) | 123459 | 0.782 | 0.707 | 0.746 | 0.903 | 0.851 | 0.819 | 0.819 | 0.796 | 0.467 | 0.691 | 0.477 | 0.860 | 1.009 |

## By scoring source (exact vector match / severity accuracy)

Primary = scored by NVD (31,575), Secondary = scored by the CNA/vendor (91,884).

| System | Primary exact | Primary sev | Secondary exact | Secondary sev |
|---|---|---|---|---|
| Majority class | 0.144 | 0.176 | 0.061 | 0.112 |
| TF-IDF + LogReg | 0.485 | 0.678 | 0.349 | 0.604 |
| DeBERTa v1 (raw text) | 0.546 | 0.710 | 0.351 | 0.602 |
| DeBERTa v2 base | 0.558 | 0.720 | 0.335 | 0.576 |
| DeBERTa v2 + scorer tokens (ours) | 0.552 | 0.714 | 0.438 | 0.684 |

## Human-in-the-loop triage

Predictions are sorted by confidence (product of the 8 heads' top probabilities). The rest go to an analyst.

| System | Auto-scored at 95% severity acc. |
|---|---|
| TF-IDF + LogReg | 0.0% |
| DeBERTa v1 (raw text) | 3.8% |
| DeBERTa v2 base | 2.9% |
| DeBERTa v2 + scorer tokens (ours) | 12.7% |

![coverage](coverage.png)

## Per-metric macro F1

![per metric](per_metric_f1.png)

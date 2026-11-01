#!/usr/bin/env python3
"""Gradio demo: paste a CVE description, get its CVSS v3.1 vector, score and severity.

Usage:
    python app.py                                  # loads checkpoints/deberta
    CKPT=checkpoints/v2_tokens python app.py       # v2: asks who scores the CVE

CVSS vectors and scores quoted in the description are stripped before prediction. A complete v3.x vector
quoted there is shown next to the prediction, since it is usually the vendor's own score.
"""
import os
import sys
from pathlib import Path

import gradio as gr
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))
from baselines import score_vector  # noqa: E402
from cvss_text import find_vector, strip_cvss  # noqa: E402
from model import LABELS, METRICS, CVSSModel, predict_proba  # noqa: E402

CKPT = os.environ.get("CKPT", "checkpoints/deberta")
THRESHOLD = float(os.environ.get("REVIEW_THRESHOLD", "0.5"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NAMES = {
    "AV": ("Attack Vector", {"N": "Network", "A": "Adjacent", "L": "Local", "P": "Physical"}),
    "AC": ("Attack Complexity", {"L": "Low", "H": "High"}),
    "PR": ("Privileges Required", {"N": "None", "L": "Low", "H": "High"}),
    "UI": ("User Interaction", {"N": "None", "R": "Required"}),
    "S":  ("Scope", {"U": "Unchanged", "C": "Changed"}),
    "C":  ("Confidentiality", {"N": "None", "L": "Low", "H": "High"}),
    "I":  ("Integrity", {"N": "None", "L": "Low", "H": "High"}),
    "A":  ("Availability", {"N": "None", "L": "Low", "H": "High"}),
}

model, tokenizer, config = CVSSModel.load(CKPT, DEVICE)
SCORER_TOKENS = config.get("scorer_tokens", False)

# Most frequent CNA/ADP tags among 2025+ CVEs; any other "org" from the CNA's email domain also works
SCORERS = ["NVD", "cisa-adp", "patchstack", "vulncheck", "github", "vuldb", "wordfence", "microsoft", "oracle",
           "adobe", "redhat", "us.ibm", "mitre", "cisco", "nvidia", "hpe", "gitlab"]

EXAMPLES = [
    "SQL injection vulnerability in the login form of Acme Portal 3.2 allows remote unauthenticated "
    "attackers to execute arbitrary SQL commands via the username parameter.",
    "A stored cross-site scripting (XSS) vulnerability in the comment feature of ExampleCMS before 2.4.1 "
    "allows authenticated users to inject arbitrary web script via a crafted comment.",
    "A use-after-free in the USB driver of ExampleOS allows a local attacker with physical access to "
    "cause a denial of service by plugging in a crafted device.",
]


def predict(description: str, scorer: str = "NVD"):
    if not description or not description.strip():
        return "", "", "", "", pd.DataFrame()
    text = strip_cvss(description)
    if SCORER_TOKENS:
        scorer = (scorer or "NVD").strip()
        text = ("[NVD] " if scorer.upper() == "NVD" else f"[CNA: {scorer}] ") + text
    probs = predict_proba(model, tokenizer, [text], config.get("max_length", 256), device=DEVICE)
    letters, rows, conf = {}, [], 1.0
    for k in METRICS:
        p = probs[k][0]
        i = int(p.argmax())
        letters[k] = LABELS[k][i]
        conf *= float(p[i])
        name, values = NAMES[k]
        rows.append({"Metric": f"{name} ({k})", "Prediction": values[letters[k]], "Confidence": f"{p[i]:.0%}"})
    vector = "CVSS:3.1/" + "/".join(f"{k}:{letters[k]}" for k in METRICS)
    score, severity = score_vector(vector)
    status = (f"Confident ({conf:.0%}): OK to auto-score" if conf >= THRESHOLD
              else f"Low confidence ({conf:.0%}): send to an analyst for review")
    quoted = find_vector(description)
    quoted = f"{quoted}  ({'%.1f %s' % score_vector(quoted)})" if quoted else "none"
    return vector, f"{score:.1f} ({severity})", status, quoted, pd.DataFrame(rows)


with gr.Blocks(title="CVEscore") as demo:
    gr.Markdown("# CVEscore\nPredict the CVSS v3.1 base vector of a CVE from its description. "
                f"Model: `{config.get('base_model', '?')}` fine-tuned on NVD data. Triage aid only, "
                "not a replacement for analyst scoring.")
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="CVE description", lines=6)
            scorer = gr.Dropdown(SCORERS, value="NVD", allow_custom_value=True, visible=SCORER_TOKENS,
                                 label="Scored by", info="Predict the vector this scorer would give")
            btn = gr.Button("Predict", variant="primary")
            gr.Examples(EXAMPLES, inp)
        with gr.Column():
            out_vec = gr.Textbox(label="CVSS vector", show_copy_button=True)
            out_score = gr.Textbox(label="Base score")
            out_status = gr.Textbox(label="Review status")
            out_quoted = gr.Textbox(label="Vector quoted in the description (not used by the model)")
            out_table = gr.Dataframe(label="Per-metric predictions")
    outputs = [out_vec, out_score, out_status, out_quoted, out_table]
    btn.click(predict, [inp, scorer], outputs)
    inp.submit(predict, [inp, scorer], outputs)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))

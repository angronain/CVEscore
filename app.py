#!/usr/bin/env python3
"""Gradio demo: paste a CVE description, get its CVSS v3.1 vector, score and severity.

Usage:
    python app.py                                  # loads checkpoints/deberta
    CKPT=checkpoints/modernbert python app.py
"""
import os
import sys
from pathlib import Path

import gradio as gr
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))
from baselines import score_vector  # noqa: E402
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

EXAMPLES = [
    "SQL injection vulnerability in the login form of Acme Portal 3.2 allows remote unauthenticated "
    "attackers to execute arbitrary SQL commands via the username parameter.",
    "A stored cross-site scripting (XSS) vulnerability in the comment feature of ExampleCMS before 2.4.1 "
    "allows authenticated users to inject arbitrary web script via a crafted comment.",
    "A use-after-free in the USB driver of ExampleOS allows a local attacker with physical access to "
    "cause a denial of service by plugging in a crafted device.",
]


def predict(description: str):
    if not description or not description.strip():
        return "", "", "", pd.DataFrame()
    probs = predict_proba(model, tokenizer, [description.strip()], config.get("max_length", 256), device=DEVICE)
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
    return vector, f"{score:.1f} ({severity})", status, pd.DataFrame(rows)


with gr.Blocks(title="CVEscore") as demo:
    gr.Markdown("# CVEscore\nPredict the CVSS v3.1 base vector of a CVE from its description. "
                f"Model: `{config.get('base_model', '?')}` fine-tuned on NVD data. Triage aid only, "
                "not a replacement for analyst scoring.")
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="CVE description", lines=6)
            btn = gr.Button("Predict", variant="primary")
            gr.Examples(EXAMPLES, inp)
        with gr.Column():
            out_vec = gr.Textbox(label="CVSS vector", show_copy_button=True)
            out_score = gr.Textbox(label="Base score")
            out_status = gr.Textbox(label="Review status")
            out_table = gr.Dataframe(label="Per-metric predictions")
    btn.click(predict, inp, [out_vec, out_score, out_status, out_table])
    inp.submit(predict, inp, [out_vec, out_score, out_status, out_table])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))

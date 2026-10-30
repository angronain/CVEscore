"""CVSS multi-head classifier: one shared encoder, 8 linear heads."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Order must match METRICS in baselines.py
LABELS = {
    "AV": ["N", "A", "L", "P"],
    "AC": ["L", "H"],
    "PR": ["N", "L", "H"],
    "UI": ["N", "R"],
    "S":  ["U", "C"],
    "C":  ["N", "L", "H"],
    "I":  ["N", "L", "H"],
    "A":  ["N", "L", "H"],
}
METRICS = list(LABELS)


class CVSSModel(nn.Module):
    def __init__(self, encoder_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({k: nn.Linear(hidden, len(v)) for k, v in LABELS.items()})

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Mean pooling over real tokens (ignores padding)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = self.dropout(pooled)
        return {k: head(pooled) for k, head in self.heads.items()}

    def save(self, path, tokenizer, extra: dict) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / "encoder")
        tokenizer.save_pretrained(path / "encoder")
        torch.save(self.heads.state_dict(), path / "heads.pt")
        (path / "cvss_config.json").write_text(json.dumps({"labels": LABELS, **extra}, indent=2))

    @classmethod
    def load(cls, path, device: str = "cuda"):
        path = Path(path)
        model = cls(str(path / "encoder"))
        model.heads.load_state_dict(torch.load(path / "heads.pt", map_location="cpu"))
        tokenizer = AutoTokenizer.from_pretrained(path / "encoder")
        config = json.loads((path / "cvss_config.json").read_text())
        return model.to(device).eval(), tokenizer, config


@torch.no_grad()
def predict_proba(model, tokenizer, texts: list[str], max_length: int = 256,
                  batch_size: int = 128, device: str = "cuda") -> dict[str, np.ndarray]:
    """Returns {metric: array of shape (n_texts, n_classes)} with softmax probabilities."""
    model.eval()
    # Sort by length so each batch has similar lengths (much less padding), then restore order
    order = np.argsort([len(t) for t in texts])
    probs = {k: [] for k in METRICS}
    for i in range(0, len(texts), batch_size):
        batch = [texts[j] for j in order[i:i + batch_size]]
        enc = tokenizer(batch, truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            out = model(enc["input_ids"], enc["attention_mask"])
        for k in METRICS:
            probs[k].append(out[k].float().softmax(-1).cpu())
    inverse = np.argsort(order)
    return {k: torch.cat(v).numpy()[inverse] for k, v in probs.items()}


def probs_to_frame(probs: dict[str, np.ndarray]):
    """Turn probabilities into a DataFrame of predicted letters + vector confidence."""
    import pandas as pd
    pred = {k: np.array(LABELS[k])[p.argmax(1)] for k, p in probs.items()}
    conf = np.prod([p.max(1) for p in probs.values()], axis=0)
    df = pd.DataFrame(pred)
    df["confidence"] = conf
    return df
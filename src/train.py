#!/usr/bin/env python3
"""Fine-tune an encoder with 8 CVSS heads.

Usage:
    python src/train.py --limit 2000 --epochs 1          # smoke test (~1 min)
    python src/train.py                                   # DeBERTa-v3-base, full run
    python src/train.py --model answerdotai/ModernBERT-base --run-name modernbert

Writes: checkpoints/<run-name>/ (best model), results/preds_<run-name>.parquet,
        results/<run-name>.json (history + test metrics)
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from baselines import add_scores, build_vectors, evaluate, save_preds
from model import LABELS, METRICS, CVSSModel, predict_proba, probs_to_frame

LABEL_IDX = {k: {v: i for i, v in enumerate(vals)} for k, vals in LABELS.items()}


class CVEDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.enc = tokenizer(df["description"].tolist(), truncation=True, max_length=max_length)
        self.labels = np.stack([df[k].map(LABEL_IDX[k]).values for k in METRICS], axis=1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "labels": self.labels[i]}


def make_collate(tokenizer):
    def collate(batch):
        labels = torch.tensor(np.stack([b.pop("labels") for b in batch]), dtype=torch.long)
        padded = tokenizer.pad(batch, return_tensors="pt")
        return padded["input_ids"], padded["attention_mask"], labels
    return collate


def validate(model, tokenizer, val: pd.DataFrame, max_length: int) -> dict:
    probs = predict_proba(model, tokenizer, val["description"].tolist(), max_length)
    pred = probs_to_frame(probs)
    f1s = {k: f1_score(val[k], pred[k], average="macro") for k in METRICS}
    exact = (val[METRICS].values == pred[METRICS].values).all(axis=1).mean()
    return {"mean_macro_f1": float(np.mean(list(f1s.values()))), "exact": float(exact),
            **{f"f1_{k}": round(v, 4) for k, v in f1s.items()}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="microsoft/deberta-v3-base")
    p.add_argument("--run-name", default="deberta")
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.06)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--val-start", default="2024-10-01", help="train CVEs published after this go to validation")
    p.add_argument("--evals-per-epoch", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="subsample train/val/test for a quick smoke test")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda"
    assert torch.cuda.is_available(), "CUDA not available: check the cu128 PyTorch install"

    # ---- Data: time-based train / validation / test ----
    full = pd.read_parquet(args.data / "train.parquet")
    test = pd.read_parquet(args.data / "test.parquet")
    val = full[full["published"] >= args.val_start].reset_index(drop=True)
    train = full[full["published"] < args.val_start].reset_index(drop=True)
    if args.limit:
        train = train.sample(min(args.limit, len(train)), random_state=args.seed)
        val = val.sample(min(args.limit // 4, len(val)), random_state=args.seed)
        test = test.sample(min(args.limit // 4, len(test)), random_state=args.seed)
    test = add_scores(test.reset_index(drop=True))
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ds = CVEDataset(train, tokenizer, args.max_length)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=make_collate(tokenizer), num_workers=2, pin_memory=True)

    # ---- Model + optimizer: small LR for the pretrained encoder, larger for the new heads ----
    model = CVSSModel(args.model).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr},
        {"params": model.heads.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * total_steps), total_steps)
    eval_every = max(1, len(loader) // args.evals_per_epoch)

    # ---- Training loop ----
    best, history, step = -1.0, [], 0
    run_dir = args.ckpt / args.run_name
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for input_ids, attention_mask, labels in loader:
            input_ids, attention_mask, labels = (t.to(device, non_blocking=True)
                                                 for t in (input_ids, attention_mask, labels))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)
            # Sum of the 8 cross-entropies, computed in fp32 for stability
            loss = sum(F.cross_entropy(logits[k].float(), labels[:, j]) for j, k in enumerate(METRICS))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            running += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch + 1} step {step}/{total_steps} loss {running / 100:.4f} "
                      f"({(time.time() - t_start) / 60:.1f} min)")
                running = 0.0

            if step % eval_every == 0 or step == total_steps:
                metrics = validate(model, tokenizer, val, args.max_length)
                metrics.update(step=step, epoch=epoch + 1)
                history.append(metrics)
                print(f"  [val] mean macro F1 {metrics['mean_macro_f1']:.4f} | exact {metrics['exact']:.4f}")
                if metrics["mean_macro_f1"] > best:
                    best = metrics["mean_macro_f1"]
                    model.save(run_dir, tokenizer, {"base_model": args.model, "max_length": args.max_length,
                                                    "val_start": args.val_start, "best_val": metrics})
                    print(f"  saved best checkpoint to {run_dir}")
                model.train()

    # ---- Test the best checkpoint ----
    print("\nEvaluating best checkpoint on test...")
    del model
    torch.cuda.empty_cache()
    model, tokenizer, _ = CVSSModel.load(run_dir, device)
    predict_proba(model, tokenizer, test["description"].tolist()[:256], args.max_length)  # GPU warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    probs = predict_proba(model, tokenizer, test["description"].tolist(), args.max_length)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    pred = probs_to_frame(probs)
    pred["vector"] = build_vectors(pred)
    pred = add_scores(pred)
    results = evaluate(test, pred, seconds)
    args.out.mkdir(parents=True, exist_ok=True)
    save_preds(args.run_name, test, pred, args.out)
    np.savez(args.out / f"probs_{args.run_name}.npz", **probs)   # raw probabilities for calibration plots
    (args.out / f"{args.run_name}.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "history": history, "test": results}, indent=2))

    print(f"\nTEST  exact {results['exact_match']:.4f} | severity acc {results['severity_acc']:.4f} | "
          f"MAE {results['score_mae']:.3f} | {results['ms_per_cve']:.3f} ms/CVE")
    print(f"Training time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
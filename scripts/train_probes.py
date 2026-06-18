#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModel, AutoTokenizer
import joblib


LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}


def parse_args():
    p = argparse.ArgumentParser(description="Train linear probes on BERT hidden states for CommitmentBank.")
    p.add_argument("--model", required=True, help="Hugging Face model id or local path (BERT-style).")
    p.add_argument("--input-csv", default="data/CommitmentBank-items-labeled.csv")
    p.add_argument("--output-dir", default="results/probes")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    return p.parse_args()


def load_data(path: str, max_samples: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if max_samples is not None:
        df = df.iloc[:max_samples]
    # Expect columns 'Context' and 'Target' and 'gold_label' and 'Verb' for grouping
    if not {"Context", "Target", "gold_label"}.issubset(df.columns):
        raise ValueError("Input CSV must contain Context, Target, and gold_label columns.")
    df = df.dropna(subset=["Context", "Target", "gold_label"])
    return df


def texts_from_row(row: pd.Series) -> str:
    # Concatenate premise and hypothesis with separator token handled by tokenizer
    return row["Context"] + " [SEP] " + row["Target"]


def extract_layer_reps(model_name: str, texts: List[str], device: str = "cpu", batch_size: int = 16, max_length: int = 256):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    model.eval()

    all_layers: List[List[np.ndarray]] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(batch_texts, truncation=True, padding=True, return_tensors="pt", max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            # out.hidden_states: tuple(layer_count, batch, seq, hidden)
            hidden_states = out.hidden_states
            # Initialize storage on first batch
            if not all_layers:
                all_layers = [[] for _ in range(len(hidden_states))]

            # For each layer, take CLS token (index 0)
            for layer_idx, layer_tensor in enumerate(hidden_states):
                # layer_tensor shape: (batch, seq, hidden)
                cls_vecs = layer_tensor[:, 0, :].cpu().numpy()
                for vec in cls_vecs:
                    all_layers[layer_idx].append(vec)

    # Convert lists to arrays
    all_layers = [np.vstack(layer) for layer in all_layers]
    return all_layers, tokenizer


def train_probes_per_layer(X_layers: List[np.ndarray], y: np.ndarray, groups: np.ndarray, output_dir: Path, seed: int = 7):
    os.makedirs(output_dir, exist_ok=True)
    n_layers = len(X_layers)
    results = {}

    # Choose number of splits based on available groups to avoid errors on small samples
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    n_splits = min(5, n_groups) if n_groups >= 2 else 2
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for li in range(n_layers):
        X = X_layers[li]
        layer_name = f"layer_{li}"
        preds = np.zeros_like(y)
        probs = np.zeros((len(y), len(np.unique(y))))

        for train_idx, val_idx in cv.split(X, y, groups):
            # If training fold contains only one class, predict the majority class
            unique_train_labels = np.unique(y[train_idx])
            if len(unique_train_labels) < 2:
                majority = int(np.bincount(y[train_idx]).argmax())
                preds[val_idx] = majority
                probs[val_idx, :] = 0.0
                probs[val_idx, majority] = 1.0
                continue

            clf = LogisticRegression(max_iter=2000)
            clf.fit(X[train_idx], y[train_idx])
            preds[val_idx] = clf.predict(X[val_idx])
            try:
                probs[val_idx] = clf.predict_proba(X[val_idx])
            except Exception:
                pass

        acc = float(accuracy_score(y, preds))
        macro_f1 = float(f1_score(y, preds, average="macro"))
        results[layer_name] = {"accuracy": acc, "macro_f1": macro_f1}

        # Train final classifier on full data and save (or save majority if single-class)
        unique_full = np.unique(y)
        if len(unique_full) < 2:
            majority = int(np.bincount(y).argmax())
            joblib.dump({"majority": majority}, output_dir / f"{layer_name}_majority.joblib")
        else:
            final_clf = LogisticRegression(max_iter=2000)
            final_clf.fit(X, y)
            joblib.dump(final_clf, output_dir / f"{layer_name}_clf.joblib")

        # Save per-example predictions and probs for this layer
        np.save(output_dir / f"{layer_name}_preds.npy", preds)
        np.save(output_dir / f"{layer_name}_probs.npy", probs)

    # Save metrics
    with open(output_dir / "probe_report.json", "w") as fh:
        json.dump(results, fh, indent=2)

    return results


def main():
    args = parse_args()
    df = load_data(args.input_csv, args.max_samples)
    texts = [texts_from_row(r) for _, r in df.iterrows()]
    labels = [LABEL_MAP[l.strip().lower()] if isinstance(l, str) else LABEL_MAP[l] for l in df["gold_label"]]
    labels = np.array(labels)
    groups = df["Verb"].fillna("_NOGROUP_").to_numpy()

    print(f"Extracting representations for {len(texts)} examples from {args.model}...")
    layers, tokenizer = extract_layer_reps(args.model, texts, device=args.device, batch_size=args.batch_size, max_length=args.max_length)

    out_dir = Path(args.output_dir) / Path(args.model.replace('/', '_'))
    metrics = train_probes_per_layer(layers, labels, groups, out_dir, seed=args.seed)
    print("Done. Probe results written to:", out_dir)


if __name__ == "__main__":
    main()

Probe training for speaker-projection analysis

Usage (example):

1. Create or activate your Python environment and install dependencies:

```
python -m pip install -r requirements-probes.txt
```

2. Run the probe training on a BERT variant (example uses an MNLI-finetuned checkpoint):

```
python scripts/train_probes.py --model textattack/bert-base-uncased-MNLI --device cpu
```

Results are written to `results/probes/<model_name>/` and include per-layer classifiers and `probe_report.json`.

Notes:
- The script extracts the `[CLS]` token representation per layer and trains an L2-regularized logistic regression probe with grouped cross-validation by `Verb`.
- Adjust `--max-samples`, `--batch-size`, and `--device` for faster experimentation.

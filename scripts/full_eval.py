
import json

# Load predictions
predictions_file = 'results/Qwen_Qwen3-8B/initial_eval_low/test/run1/predictions.json'
with open(predictions_file) as f:
    predictions = json.load(f)

# Extract predictions and gold labels
pred_labels = [p['prediction'] for p in predictions]
gold_labels = [p['gold_label'] for p in predictions]

# Calculate basic metrics
correct = sum(1 for p, g in zip(pred_labels, gold_labels) if p == g)
accuracy = correct / len(predictions) if predictions else 0.0

# Per-class metrics
labels = ['E', 'N', 'C']
metrics = {}
all_tp = 0
all_fp = 0
all_fn = 0

for label in labels:
    tp = sum(1 for p, g in zip(pred_labels, gold_labels) if p == label and g == label)
    fp = sum(1 for p, g in zip(pred_labels, gold_labels) if p == label and g != label)
    fn = sum(1 for p, g in zip(pred_labels, gold_labels) if p != label and g == label)
    support = sum(1 for g in gold_labels if g == label)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics[label] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }
    
    all_tp += tp
    all_fp += fp
    all_fn += fn

# Calculate macro F1 (average of F1 scores)
macro_f1 = sum(metrics[label]['f1'] for label in labels) / len(labels)

# Calculate micro F1 (using aggregated TP, FP, FN)
micro_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
micro_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

# Weighted F1 (weighted by support)
total_samples = sum(metrics[label]['support'] for label in labels)
weighted_f1 = sum(metrics[label]['f1'] * metrics[label]['support'] for label in labels) / total_samples if total_samples > 0 else 0.0

# Print results
print("=" * 70)
print("EVALUATION RESULTS - Qwen3-8B")
print("=" * 70)
print(f"\nAccuracy: {accuracy:.4f} ({correct}/{len(predictions)})")
print(f"\nMacro F1:   {macro_f1:.4f}")
print(f"Micro F1:   {micro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

print("\nPer-class metrics:")
print(f"{'Label':<6} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<8}")
print("-" * 50)
for label in labels:
    m = metrics[label]
    print(f"{label:<6} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']:<8}")

print("\nConfusion Matrix:")
print(f"{'Predicted ->':>15} {'E':>8} {'N':>8} {'C':>8}")
print("-" * 42)
for label in labels:
    e_match = sum(1 for p, g in zip(pred_labels, gold_labels) if g == label and p == 'E')
    n_match = sum(1 for p, g in zip(pred_labels, gold_labels) if g == label and p == 'N')
    c_match = sum(1 for p, g in zip(pred_labels, gold_labels) if g == label and p == 'C')
    print(f"{'Gold_' + label:>15} {e_match:>8} {n_match:>8} {c_match:>8}")

# Count predictions by label
pred_counts = {label: sum(1 for p in pred_labels if p == label) for label in labels}
gold_counts = {label: sum(1 for g in gold_labels if g == label) for label in labels}

print("\nLabel Distribution:")
print(f"  Gold:        E={gold_counts['E']}, N={gold_counts['N']}, C={gold_counts['C']}")
print(f"  Predictions: E={pred_counts['E']}, N={pred_counts['N']}, C={pred_counts['C']}")

# Save summary
summary = {
    'model': 'Qwen3-8B',
    'num_examples': len(predictions),
    'num_with_gold': len(predictions),
    'accuracy': float(accuracy),
    'macro_f1': float(macro_f1),
    'micro_f1': float(micro_f1),
    'weighted_f1': float(weighted_f1),
    'correct_predictions': int(correct),
    'per_class_metrics': {
        label: {
            'precision': metrics[label]['precision'],
            'recall': metrics[label]['recall'],
            'f1': metrics[label]['f1'],
            'support': metrics[label]['support'],
        }
        for label in labels
    },
    'label_distribution': {
        'gold': gold_counts,
        'predictions': pred_counts,
    },
    'predictions_file': predictions_file,
}

summary_file = 'results/Qwen_Qwen3-8B/initial_eval_low/test/run1/evaluation_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
    f.write('\n')

print(f"\n✓ Evaluation summary saved to {summary_file}")
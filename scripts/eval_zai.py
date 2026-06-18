import json
import csv
import re
from collections import defaultdict

# ==========================================
# Configuration & File Paths
# ==========================================
PREDICTIONS_FILE = 'data/predictions_zai.jsonl'
DATASET_FILE = 'data/CommitmentBank-items.csv'
SUMMARY_FILE = 'data/evaluation_summary.json'

LABELS = ['entailment', 'neutral', 'contradiction']

# ==========================================
# Helper Functions
# ==========================================
def parse_responses(responses_str):
    """
    Parses the 'Reponses' string and returns the gold label and entropy category
    based on the >= 80% threshold rule.
    """
    try:
        votes = [int(v.strip()) for v in responses_str.strip('"').split(',') if v.strip()]
    except ValueError:
        return None, None
        
    if not votes:
        return None, None

    total_votes = len(votes)
    
    entailment_votes = sum(1 for v in votes if 1 <= v <= 3)
    neutral_votes = sum(1 for v in votes if v == 0)
    contradiction_votes = sum(1 for v in votes if -3 <= v <= -1)
    
    ent_pct = entailment_votes / total_votes
    neu_pct = neutral_votes / total_votes
    con_pct = contradiction_votes / total_votes
    
    max_pct = max(ent_pct, neu_pct, con_pct)
    if max_pct == ent_pct:
        gold_label = 'entailment'
    elif max_pct == con_pct:
        gold_label = 'contradiction'
    else:
        gold_label = 'neutral'
        
    entropy_class = 'low_entropy' if max_pct >= 0.8 else 'high_entropy'
    
    return gold_label, entropy_class

def extract_prediction(text):
    text = text.lower()
    if 'entailment' in text:
        return 'entailment'
    elif 'contradiction' in text:
        return 'contradiction'
    elif 'neutral' in text:
        return 'neutral'
    return 'unknown'

# ==========================================
# Main Execution
# ==========================================

# 1. Load and parse the Gold Dataset (CSV)
gold_data = {}
with open(DATASET_FILE, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    headers = next(reader)
    
    try:
        responses_idx = headers.index('Reponses')
    except ValueError:
        responses_idx = [i for i, h in enumerate(headers) if 'reponses' in h.lower() or 'responses' in h.lower()][0]
    
    for row in reader:
        item_id = row[0] 
        responses_str = row[responses_idx]
        
        gold_label, entropy_class = parse_responses(responses_str)
        if gold_label:
            gold_data[item_id] = {
                'gold_label': gold_label,
                'entropy_class': entropy_class
            }

# 2. Load Predictions (JSONL)
results = []
with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
            
        pred_item = json.loads(line)
        custom_id = pred_item.get('custom_id', '')
        
        match = re.search(r'id(\d+)', custom_id)
        item_id = match.group(1) if match else custom_id
            
        raw_content = pred_item['response']['body']['choices'][0]['message']['content']
        predicted_label = extract_prediction(raw_content)
        
        if item_id in gold_data:
            results.append({
                'id': item_id,
                'prediction': predicted_label,
                'gold_label': gold_data[item_id]['gold_label'],
                'entropy_class': gold_data[item_id]['entropy_class']
            })

# 3. Calculate Overall Metrics
pred_labels = [r['prediction'] for r in results]
gold_labels = [r['gold_label'] for r in results]

correct = sum(1 for p, g in zip(pred_labels, gold_labels) if p == g)
accuracy = correct / len(results) if results else 0.0

metrics = {}
all_tp = all_fp = all_fn = 0

for label in LABELS:
    tp = sum(1 for p, g in zip(pred_labels, gold_labels) if p == label and g == label)
    fp = sum(1 for p, g in zip(pred_labels, gold_labels) if p == label and g != label)
    fn = sum(1 for p, g in zip(pred_labels, gold_labels) if p != label and g == label)
    support = sum(1 for g in gold_labels if g == label)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics[label] = {
        'precision': precision, 'recall': recall, 'f1': f1,
        'support': support, 'tp': tp, 'fp': fp, 'fn': fn,
    }
    
    all_tp += tp
    all_fp += fp
    all_fn += fn

macro_f1 = sum(metrics[label]['f1'] for label in LABELS) / len(LABELS)
micro_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
micro_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

total_samples = sum(metrics[label]['support'] for label in LABELS)
weighted_f1 = sum(metrics[label]['f1'] * metrics[label]['support'] for label in LABELS) / total_samples if total_samples > 0 else 0.0

# 4. Entropy Breakdown with Sub-F1 Scores
entropy_stats = {
    'low_entropy': {'total': 0, 'correct': 0, 'errors': defaultdict(int), 'metrics': {l: {'tp':0, 'fp':0, 'fn':0, 'support':0} for l in LABELS}},
    'high_entropy': {'total': 0, 'correct': 0, 'errors': defaultdict(int), 'metrics': {l: {'tp':0, 'fp':0, 'fn':0, 'support':0} for l in LABELS}}
}

for r in results:
    e_class = r['entropy_class']
    pred = r['prediction']
    gold = r['gold_label']
    
    entropy_stats[e_class]['total'] += 1
    
    if pred == gold:
        entropy_stats[e_class]['correct'] += 1
        if gold in LABELS:
            entropy_stats[e_class]['metrics'][gold]['tp'] += 1
    else:
        error_type = f"Gold:{gold[:3].upper()} -> Pred:{pred[:3].upper()}"
        entropy_stats[e_class]['errors'][error_type] += 1
        
        if gold in LABELS:
            entropy_stats[e_class]['metrics'][gold]['fn'] += 1
        if pred in LABELS:
            entropy_stats[e_class]['metrics'][pred]['fp'] += 1
            
    if gold in LABELS:
        entropy_stats[e_class]['metrics'][gold]['support'] += 1

# Calculate subset F1s for each entropy class
for e_class in ['low_entropy', 'high_entropy']:
    stats = entropy_stats[e_class]
    sub_all_tp = sub_all_fp = sub_all_fn = 0
    valid_classes = 0
    sum_f1 = 0.0
    
    for label in LABELS:
        m = stats['metrics'][label]
        tp, fp, fn = m['tp'], m['fp'], m['fn']
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        m['precision'] = prec
        m['recall'] = rec
        m['f1'] = f1
        
        sub_all_tp += tp
        sub_all_fp += fp
        sub_all_fn += fn
        
        # Only include in Macro F1 if there was support or predictions for it
        if m['support'] > 0 or (tp + fp) > 0:
            valid_classes += 1
            sum_f1 += f1
            
    stats['macro_f1'] = sum_f1 / valid_classes if valid_classes > 0 else 0.0
    
    m_prec = sub_all_tp / (sub_all_tp + sub_all_fp) if (sub_all_tp + sub_all_fp) > 0 else 0.0
    m_rec = sub_all_tp / (sub_all_tp + sub_all_fn) if (sub_all_tp + sub_all_fn) > 0 else 0.0
    stats['micro_f1'] = 2 * (m_prec * m_rec) / (m_prec + m_rec) if (m_prec + m_rec) > 0 else 0.0


# ==========================================
# Print Results
# ==========================================
print("=" * 70)
print("EVALUATION RESULTS - GLM-5.1")
print("=" * 70)
print(f"\nOverall Accuracy: {accuracy:.4f} ({correct}/{len(results)})")
print(f"Macro F1:         {macro_f1:.4f}")
print(f"Micro F1:         {micro_f1:.4f}")

print("\n" + "=" * 70)
print("BREAKDOWN BY HUMAN AGREEMENT (ENTROPY)")
print("=" * 70)
for e_class, name in [('low_entropy', 'Low Entropy (>= 80% Agreement - Primary Dataset)'), 
                      ('high_entropy', 'High Entropy (< 80% Agreement - Disputed Items)')]:
    stats = entropy_stats[e_class]
    acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
    print(f"\n{name}")
    print(f"Total Items: {stats['total']}")
    print(f"Accuracy:    {acc:.4f} ({stats['correct']}/{stats['total']})")
    print(f"Macro F1:    {stats['macro_f1']:.4f}")
    print(f"Micro F1:    {stats['micro_f1']:.4f}")
    
    # Optional: Print subset F1s
    print("\n  Per-class (Entropy Subset):")
    print(f"  {'Label':<15} {'F1':<10} {'Support':<8}")
    print("  " + "-" * 35)
    for label in LABELS:
        m = stats['metrics'][label]
        print(f"  {label:<15} {m['f1']:<10.4f} {m['support']:<8}")

    if stats['total'] - stats['correct'] > 0:
        print("\n  Top Errors (Gold -> Predicted):")
        sorted_errors = sorted(stats['errors'].items(), key=lambda x: x[1], reverse=True)
        for err_type, count in sorted_errors[:3]:
            print(f"    - {err_type}: {count} occurrences")

# ==========================================
# Save JSON Summary
# ==========================================
summary = {
    'model': 'GLM-5.1',
    'num_examples': len(results),
    'overall_accuracy': float(accuracy),
    'macro_f1': float(macro_f1),
    'micro_f1': float(micro_f1),
    'entropy_breakdown': {
        e_class: {
            'total': entropy_stats[e_class]['total'],
            'accuracy': entropy_stats[e_class]['correct'] / max(1, entropy_stats[e_class]['total']),
            'macro_f1': entropy_stats[e_class]['macro_f1'],
            'micro_f1': entropy_stats[e_class]['micro_f1'],
            'per_class_metrics': {
                label: {
                    'f1': entropy_stats[e_class]['metrics'][label]['f1'],
                    'precision': entropy_stats[e_class]['metrics'][label]['precision'],
                    'recall': entropy_stats[e_class]['metrics'][label]['recall'],
                    'support': entropy_stats[e_class]['metrics'][label]['support']
                } for label in LABELS
            },
            'errors': dict(entropy_stats[e_class]['errors'])
        } for e_class in ['low_entropy', 'high_entropy']
    }
}

with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n✓ Evaluation summary saved to {SUMMARY_FILE}")
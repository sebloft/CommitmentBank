import json
import csv
import re

# ==========================================
# Configuration & File Paths
# ==========================================
PREDICTIONS_FILE = 'data/predictions_zai.jsonl'
DATASET_FILE = 'data/CommitmentBank-items.csv'
OUTPUT_JSONL_FILE = 'data/detailed_evaluation_results.jsonl'

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
    ent_pct = sum(1 for v in votes if 1 <= v <= 3) / total_votes
    neu_pct = sum(1 for v in votes if v == 0) / total_votes
    con_pct = sum(1 for v in votes if -3 <= v <= -1) / total_votes
    
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
    """Extracts the predicted label from the LLM's response text."""
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

# 1. Load and parse the Gold Dataset (CSV), storing all fields
gold_data = {}
with open(DATASET_FILE, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    headers = next(reader)
    
    # Handle the unnamed index column
    if not headers[0].strip():
        headers[0] = 'item_index'
        
    try:
        responses_idx = headers.index('Reponses')
    except ValueError:
        responses_idx = [i for i, h in enumerate(headers) if 'reponses' in h.lower() or 'responses' in h.lower()][0]
    
    for row in reader:
        item_id = row[0]
        responses_str = row[responses_idx]
        
        gold_label, entropy_class = parse_responses(responses_str)
        
        # Package the entire row into a dictionary mapped to the headers
        row_dict = {headers[i]: row[i] for i in range(len(headers))}
        
        if gold_label:
            gold_data[item_id] = {
                'csv_data': row_dict,
                'gold_label': gold_label,
                'entropy_class': entropy_class
            }

# 2. Load Predictions, check correctness, and combine all data
combined_records = []
with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
            
        pred_item = json.loads(line)
        custom_id = pred_item.get('custom_id', '')
        
        # Extract numeric ID
        match = re.search(r'id(\d+)', custom_id)
        item_id = match.group(1) if match else custom_id
            
        # Navigate to the message object to extract both content and reasoning
        message = pred_item['response']['body']['choices'][0]['message']
        raw_content = message.get('content', '')
        model_reasoning = message.get('reasoning', '') # Extract reasoning here
        
        predicted_label = extract_prediction(raw_content)
        
        if item_id in gold_data:
            gold_info = gold_data[item_id]
            is_correct = bool(predicted_label == gold_info['gold_label'])
            
            # Build the unified record
            record = {
                "custom_id": custom_id,
                "predicted_label": predicted_label,
                "gold_label": gold_info['gold_label'],
                "is_correct": is_correct,
                "entropy_class": gold_info['entropy_class'],
                "raw_model_response": raw_content,
                "model_reasoning": model_reasoning # Add reasoning to the final output
            }
            
            # Merge the original CSV columns into this dictionary
            record.update(gold_info['csv_data'])
            
            combined_records.append(record)

# 3. Write to the new JSONL file
with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as out_f:
    for record in combined_records:
        out_f.write(json.dumps(record, ensure_ascii=False, indent=1) + '\n')

print(f"Successfully processed {len(combined_records)} items.")
print(f"Saved detailed results to: {OUTPUT_JSONL_FILE}")
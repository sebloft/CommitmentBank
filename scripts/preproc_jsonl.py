#!/usr/bin/env python3
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Label definitions based on your rules
LABEL_RULES = {
    "entailment": lambda vote: 1 <= vote <= 3,
    "neutral": lambda vote: vote == 0,
    "contradiction": lambda vote: -3 <= vote <= -1,
}

def determine_label(votes_str: str):
    """
    Determines label based on 0.8 agreement or mean threshold.
    """
    # Clean up the string and handle empty/malformed inputs safely
    if pd.isna(votes_str) or not str(votes_str).strip():
        return "neutral", False
        
    try:
        votes = [int(v.strip()) for v in str(votes_str).replace('"', '').split(",")]
    except ValueError:
        # Fallback if a row has corrupted vote data
        return "neutral", False

    total = len(votes)
    if total == 0:
        return "neutral", False
    
    # 1. Calculate counts for majority rule
    counts = {k: sum(1 for v in votes if rule(v)) for k, rule in LABEL_RULES.items()}
    
    # 2. Check for 0.8 agreement
    agreement = False
    final_label = "neutral" 
    
    for k, count in counts.items():
        if count / total >= 0.8:
            agreement = True
            final_label = k
            break
            
    # 3. Fallback: If no 0.8 agreement, check absolute mean
    if not agreement:
        mean_val = np.mean(votes)
        if abs(mean_val) < 0.5:
            final_label = "neutral"
        else:
            final_label = "entailment" if mean_val > 0 else "contradiction"
                
    return final_label, agreement

def process_data(input_csv: Path, output_jsonl: Path, model: str, temp: float, reasoning: bool, max_tokens: int):
    # Read CSV without assuming headers, to keep index mapping consistent
    df = pd.read_csv(input_csv, header=None)
    
    jsonl_output = []
    
    for idx, row in df.iterrows():
        # Using .iloc on the row series forces positional indexing (0-indexed)
        # Based on your sample: 
        # Index 4 = Context/Premise
        # Index 5 = Hypothesis/Statement
        # Last index = Votes string
        premise = row.iloc[4]
        hypothesis = row.iloc[5]
        votes_str = row.iloc[-1] 
        
        label, agreement = determine_label(votes_str)
        
        # Build prompt
        prompt = (f"You are a helpful research assistant. Your task is to predict the label "
                  f"for an NLI item. Context: {premise}\nStatement: {hypothesis}\n\n"
                  f"Target label: {label}\n\n[PREDICTION]:")
        
        entry = {
            "custom_id": f"id{idx}_run1",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "reasoning": {"enabled": reasoning},
                "temperature": temp
            }
        }
        jsonl_output.append(entry)

    # Save to JSONL
    with output_jsonl.open("w", encoding="utf-8") as f:
        for entry in jsonl_output:
            f.write(json.dumps(entry) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Process NLI data to JSONL for API.")
    parser.add_argument("--model", required=True, help="Model name (e.g., anthropic/claude-haiku-4.5)")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature")
    parser.add_argument("--reasoning", action="store_true", help="Enable reasoning")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens")
    args = parser.parse_args()

    input_path = Path("data/CommitmentBank-items.csv")
    output_path = Path("data/nli_prompt.jsonl")
    
    process_data(input_path, output_path, args.model, args.temp, args.reasoning, args.max_tokens)
    print(f"Successfully processed {input_path} to {output_path}")

if __name__ == "__main__":
    main()
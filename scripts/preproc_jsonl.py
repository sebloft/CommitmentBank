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
        premise = str(row.iloc[4]) + ' ' + str(row.iloc[5])
        if premise.strip() == '':
            print(f"Warning: Empty premise at row {idx}. Skipping.")
            continue
        hypothesis = row.iloc[6]
        votes_str = row.iloc[-1] 
        print(f"Processing row {idx}: Premise='{premise}', Hypothesis='{hypothesis}', Votes='{votes_str}'")
        label, agreement = determine_label(votes_str)
        
        # Build prompt
        prompt = (f"You will be given a Context and a Statement. A group of annotators were asked to label the relationship between them using the following criteria:\n\n"
                  f"Assuming the Context is true, the Statement...\n"
                  f"* ... is most likely true -> entailment\n"
                  f"* ... could be either true or false -> neutral\n"
                  f"* ... is most likely false -> contradiction\n\n"
                  f"Your task is to predict the label that most annotators would assign to this item.\n\n"
                  f"Please output only a single label 'E', 'N' or 'C', depending on your prediction, after the flag '[PREDICTION]:'.\n\n"
                  f"[CONTEXT]: {premise}\n"
                  f"[HYPOTHESIS]: {hypothesis}\n\n"
                  f"[PREDICTION]:")
        
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

    input_path = Path("data/CB/CommitmentBank-items.csv")
    save_path = f"data/nli_prompt_{args.model.replace('/', '_')}_temp{args.temp}"


    if args.reasoning:
        save_path += "_reasoning.jsonl"
    else:
        save_path += "_no_reasoning.jsonl"

    process_data(input_path, Path(save_path), args.model, args.temp, args.reasoning, args.max_tokens)
    print(f"Successfully processed {input_path} to {save_path}")

if __name__ == "__main__":
    main()
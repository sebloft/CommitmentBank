import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr, chi2_contingency
import json
import os

# ==========================================
# Configuration
# ==========================================
INPUT_JSONL = 'data/detailed_evaluation_results.jsonl'
OUTPUT_REPORT = 'data/error_correlation_analysis.json'

def cramers_v(x, y):
    """Calculates Cramér's V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    if min((kcorr-1), (rcorr-1)) == 0:
        return 0.0
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def load_robust_json_stream(file_path):
    """
    Reads a file containing multiple JSON objects without requiring commas 
    between them, perfectly handling both strict JSONL and multi-line objects.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    data = []
    decoder = json.JSONDecoder()
    idx = 0
    content = content.lstrip()
    
    while idx < len(content):
        try:
            # Decode the next JSON object in the string
            obj, new_idx = decoder.raw_decode(content[idx:])
            data.append(obj)
            idx += new_idx
            
            # Skip any whitespace/newlines before the next object
            while idx < len(content) and content[idx].isspace():
                idx += 1
        except json.JSONDecodeError as e:
            print(f"Warning: Stopped parsing due to invalid JSON at character position {idx}: {e}")
            break
            
    return data

def analyze_correlations():
    # 1. Robust Data Loading
    if not os.path.exists(INPUT_JSONL):
        print(f"Error: Could not find {INPUT_JSONL}. Please check the path.")
        return

    print("Loading data...")
    data = load_robust_json_stream(INPUT_JSONL)

    if not data:
        print(f"Error: No valid JSON objects found in {INPUT_JSONL}.")
        return

    print(f"Successfully loaded {len(data)} items. Analyzing...")
    df = pd.DataFrame(data)
    
    # Create target variable: 1 if model was WRONG, 0 if CORRECT
    df['error_flag'] = (~df['is_correct']).astype(int)

    results = {
        "numerical_correlations": {},
        "categorical_associations": {},
        "error_rates_by_category": {}
    }

    # ==========================================
    # 2. Numerical Features Analysis
    # ==========================================
    numerical_cols = ['mean.noTarget', 'sd.noTarget', 'Mean', 'SD']
    print("\n" + "=" * 60)
    print("NUMERICAL FEATURES (Point-Biserial Correlation with Error)")
    print("=" * 60)
    print(f"{'Feature':<15} | {'Correlation (r)':<15} | {'p-value':<15}")
    print("-" * 60)

    for col in numerical_cols:
        if col in df.columns:
            # Clean data: force numeric, drop NaNs
            clean_df = df[[col, 'error_flag']].copy()
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
            clean_df = clean_df.dropna()
            
            if len(clean_df) > 0:
                corr, p_value = pointbiserialr(clean_df['error_flag'], clean_df[col])
                results["numerical_correlations"][col] = {"correlation": corr, "p_value": p_value}
                print(f"{col:<15} | {corr:>15.4f} | {p_value:>15.4e}")

    # ==========================================
    # 3. Categorical Features Analysis
    # ==========================================
    categorical_cols = ['Embedding', 'factive', 'ModalType', 'genre', 'entropy_class', 'Target']
    
    print("\n" + "=" * 60)
    print("CATEGORICAL FEATURES (Cramér's V & Chi-Square with Error)")
    print("=" * 60)
    print(f"{'Feature':<15} | {'Cramér\'s V':<15} | {'p-value':<15}")
    print("-" * 60)

    for col in categorical_cols:
        if col in df.columns:
            # Clean data: drop missing strings
            clean_df = df[[col, 'error_flag']].copy()
            clean_df[col] = clean_df[col].astype(str).str.strip()
            clean_df = clean_df[clean_df[col] != 'nan']
            clean_df = clean_df[clean_df[col] != '']
            
            if len(clean_df) > 0:
                # Calculate p-value via Chi-Square
                contingency = pd.crosstab(clean_df[col], clean_df['error_flag'])
                
                # Chi-Square requires contingency table to have shape > (1,1)
                if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                    chi2, p_val, dof, expected = chi2_contingency(contingency)
                    
                    # Calculate effect size via Cramér's V
                    v = cramers_v(clean_df[col], clean_df['error_flag'])
                    
                    results["categorical_associations"][col] = {"cramers_v": v, "p_value": p_val}
                    print(f"{col:<15} | {v:>15.4f} | {p_val:>15.4e}")
                else:
                    print(f"{col:<15} | Not enough variance to calculate correlation.")
                
                # Calculate specific error rates for this category
                category_stats = clean_df.groupby(col)['error_flag'].agg(['mean', 'count'])
                category_stats = category_stats.rename(columns={'mean': 'error_rate', 'count': 'support'})
                
                # Convert to dict for JSON saving
                cat_dict = category_stats.to_dict('index')
                results["error_rates_by_category"][col] = cat_dict

    # ==========================================
    # 4. Save and summarize deep dive
    # ==========================================
    with open(OUTPUT_REPORT, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "=" * 60)
    print("DEEP DIVE: NOTABLE ERROR RATES BY CATEGORY")
    print("=" * 60)
    
    # Print interesting categories (where support > 5 to avoid noise)
    for col, stats in results["error_rates_by_category"].items():
        if col == 'Target': # Skip printing raw text targets as they are too long
            continue
        print(f"\n--- {col.upper()} ---")
        # Sort by error rate descending
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['error_rate'], reverse=True)
        for cat_name, metrics in sorted_stats:
            if metrics['support'] >= 5: # Filter out rare edges
                print(f"  {cat_name:<15} : {metrics['error_rate']*100:>6.1f}% error rate (n={metrics['support']})")

    print(f"\n✓ Saved detailed correlation analysis to {OUTPUT_REPORT}")

if __name__ == "__main__":
    analyze_correlations()
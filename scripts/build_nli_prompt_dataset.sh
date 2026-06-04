#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${DATA_DIR}/nli_prompt_json"

ml --force purge
ml releases/2024a
module load CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0 Python/3.12.3-GCCcore-13.3.0
source "${PROJECT_ROOT}/ven/bin/activate"

mkdir -p "${OUTPUT_DIR}"

export PROJECT_ROOT
export DATA_DIR
export OUTPUT_DIR

python <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


DATA_DIR = Path(os.environ["DATA_DIR"])
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])

PROMPT_TEMPLATE = """You will be given a Context and a Statement. A group of annotators were asked to label the relationship between them using the following criteria:

Assuming the Context is true, the Statement...
* ... is most likely true -> entailment
* ... could be either true or false -> neutral
* ... is most likely false -> contradiction

Your task is to predict the label that most annotators would assign to this item.

Please output only a single label 'E', 'N' or 'C', depending on your prediction, after the flag '[PREDICTION]:'.

[CONTEXT]: {context}
[HYPOTHESIS]: {statement}

[PREDICTION]:
"""

def build_prompt(context: str, statement: str) -> str:
    return PROMPT_TEMPLATE.format(context=context.strip(), statement=statement.strip())


def process_file(input_path: Path, output_path: Path) -> int:
    frame = pd.read_parquet(input_path)
    if "premise" not in frame.columns or "hypothesis" not in frame.columns:
        missing = {"premise", "hypothesis"} - set(frame.columns)
        raise ValueError(f"{input_path.name} is missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["llm_prompt"] = [build_prompt(premise, hypothesis) for premise, hypothesis in zip(frame["premise"], frame["hypothesis"])]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(frame.to_dict(orient="records"), handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return len(frame)


def main() -> int:
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No parquet files found in {DATA_DIR}")

    total_rows = 0
    for parquet_path in parquet_files:
        output_path = OUTPUT_DIR / f"{parquet_path.stem}.json"
        row_count = process_file(parquet_path, output_path)
        total_rows += row_count
        print(f"Wrote {row_count} rows to {output_path}")

    print(f"Processed {len(parquet_files)} parquet files with {total_rows} total rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
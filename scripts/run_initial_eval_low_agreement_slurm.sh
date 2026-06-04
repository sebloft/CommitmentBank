#!/usr/bin/env bash
#SBATCH --job-name=cb_initial_eval_low_agree
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G

set -euo pipefail

# 1. Load cluster environment
ml --force purge
ml releases/2024a
module load CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0 Python/3.12.3-GCCcore-13.3.0

# 2. Define explicit paths
VENV_PATH="/home/ucl/cental/sloftus/CommitmentBank/ven"
PYTHON_BIN="$VENV_PATH/bin/python"

export PYTHONNOUSERSITE=1
unset PYTHONHOME

PROJECT_ROOT="/home/ucl/cental/sloftus/CommitmentBank"
cd "$PROJECT_ROOT"
mkdir -p logs
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

: "${MODEL:?Set MODEL, e.g. Qwen/Qwen3-8B}"
: "${SPLIT:?Set SPLIT to train_low_agreement, validation_low_agreement, or test_low_agreement}"
: "${RUN:?Set RUN number, e.g. 1}"

CACHE_ROOT="${CACHE_ROOT:-/globalscratch/ucl/cental/sloftus}"
MODEL_DIR="${MODEL//\//_}"
INPUT_JSON="${INPUT_JSON:-data/nli_prompt_json_low_agreement/${SPLIT#*_low_agreement}-00000-of-00001.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/${MODEL_DIR}/initial_eval_low/${SPLIT}/run${RUN}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-50}"
SEED="${SEED:-7}"

cmd=(
  "$PYTHON_BIN" "$PROJECT_ROOT/initial_eval.py"
  --model "$MODEL"
  --input-json "$INPUT_JSON"
  --output-dir "$OUTPUT_DIR"
  --cache-root "$CACHE_ROOT"
  --batch-size "$BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --seed "$SEED"
  --device cuda
)

if [[ -n "$MAX_SAMPLES" ]]; then
  cmd+=(--max-samples "$MAX_SAMPLES")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_split=($EXTRA_ARGS)
  cmd+=("${extra_split[@]}")
fi

printf 'Running: %q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

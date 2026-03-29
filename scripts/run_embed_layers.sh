#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-embed_layers.py}"
BASE_DIR="${BASE_DIR:-../outputs}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
DTYPE="${DTYPE:-bfloat16}"

# Set to 1 to add --print_sanity_check on every run, else 0.
PRINT_SANITY_CHECK="${PRINT_SANITY_CHECK:-1}"

# Datasets to run
DATASETS=(
  "gsm8k"
  "trivia_qa"
  "webq"
  "sciq"
  "polymath"
)

# Models:
# model_dir|hf_name|file_stem|layers
MODELS=(
  "Qwen3-0.6B|Qwen/Qwen3-0.6B|qwen3_0.6b|0,4,8,12,16,20,24,28"
  "Qwen3-1.7B|Qwen/Qwen3-1.7B|qwen3_1.7b|0,4,8,12,16,20,24,28"
  "Qwen3-4B-Thinking-2507|Qwen/Qwen3-4B-Thinking-2507|qwen3_4b_thinking_2507|0,4,8,12,16,20,24,28,32,36"
  "Qwen3-8B|Qwen/Qwen3-8B|qwen3_8b|0,4,8,12,16,20,24,28,32,36"
  "Qwen3-14B|Qwen/Qwen3-14B|qwen3_14b|0,4,8,12,16,20,24,28,32,36,40"
)

for dataset in "${DATASETS[@]}"; do
  for spec in "${MODELS[@]}"; do
    IFS='|' read -r model_dir hf_name file_stem layers <<< "$spec"

    in_csv="${BASE_DIR}/${model_dir}/${dataset}/n_1000_temp_0.7_k_100_out_df_temp_0.6_k_1.csv"
    out_dir="${BASE_DIR}/${model_dir}/${dataset}/layer_sweep"
    out_name="${file_stem}_${dataset}"

    if [[ ! -f "$in_csv" ]]; then
      echo "Skipping missing input: $in_csv"
      continue
    fi

    mkdir -p "$out_dir"

    cmd=(
      "$PYTHON_BIN" "$SCRIPT"
      --in_csv "$in_csv"
      --model_name "$hf_name"
      --out_dir "$out_dir"
      --out_name "$out_name"
      --layers "$layers"
      --max_length "$MAX_LENGTH"
      --dtype "$DTYPE"
    )

    if [[ "$PRINT_SANITY_CHECK" == "1" ]]; then
      cmd+=(--print_sanity_check)
    fi

    echo "============================================================"
    echo "Running model=${hf_name} dataset=${dataset}"
    echo "Input : $in_csv"
    echo "Output: $out_dir/${out_name}_probe_results.csv"
    echo "Layers: $layers"
    echo "============================================================"

    "${cmd[@]}"
  done
done
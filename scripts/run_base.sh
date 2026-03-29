#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-add_base_targets.py}"
BASE_DIR="${BASE_DIR:-../outputs}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-1}"

DATASETS=(
  "gsm8k"
  "polymath"
  "sciq"
  "trivia_qa"
  "webq"
)

# model_dir|hf_name|file_stem|base_hf_name
MODELS=(
  "Qwen3-0.6B|Qwen/Qwen3-0.6B|qwen3_0.6b|Qwen/Qwen3-0.6B-Base"
  "Qwen3-1.7B|Qwen/Qwen3-1.7B|qwen3_1.7b|Qwen/Qwen3-1.7B-Base"
  "Qwen3-4B-Thinking-2507|Qwen/Qwen3-4B-Thinking-2507|qwen3_4b_thinking_2507|Qwen/Qwen3-4B-Base"
  "Qwen3-8B|Qwen/Qwen3-8B|qwen3_8b|Qwen/Qwen3-8B-Base"
  "Qwen3-14B|Qwen/Qwen3-14B|qwen3_14b|Qwen/Qwen3-14B-Base"
)

for dataset in "${DATASETS[@]}"; do
  for spec in "${MODELS[@]}"; do
    IFS='|' read -r model_dir hf_name file_stem base_hf_name <<< "$spec"

    in_csv="${BASE_DIR}/${model_dir}/${dataset}/n_1000_temp_0.7_k_100_out_df_temp_0.6_k_1.csv"
    out_dir="${BASE_DIR}/${model_dir}/${dataset}"
    out_name="${file_stem}_${dataset}_with_base_targets"

    if [[ ! -f "$in_csv" ]]; then
      echo "Skipping missing input: $in_csv"
      continue
    fi

    mkdir -p "$out_dir"

    cmd=(
      "$PYTHON_BIN" "$SCRIPT"
      --in_csv "$in_csv"
      --model_name "$hf_name"
      --base_model_name "$base_hf_name"
      --out_dir "$out_dir"
      --out_name "$out_name"
      --batch_size "$BATCH_SIZE"
      --dtype "$DTYPE"
    )

    echo "============================================================"
    echo "Running model=${hf_name} dataset=${dataset}"
    echo "Base   : ${base_hf_name}"
    echo "Input  : $in_csv"
    echo "Output : $out_dir/${out_name}.csv"
    echo "============================================================"

    "${cmd[@]}"
  done
done
#!/usr/bin/env bash
set -euo pipefail

# Shared settings
N=1000

# Sweeps
MODELS=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B-Thinking-2507"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  "nvidia/Nemotron-Cascade-8B-Thinking"
  "nvidia/OpenReasoning-Nemotron-7B"
)

DATASETS=(
  "gsm8k"
  "trivia_qa"
  "polymath"
  "sciq"
  "webq"
)


# Pair the k/temperature settings by index
K_VALUES=(100 1)
TEMPS=("0.7" "0.6")   # empty means: don't pass --temperature

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for i in "${!K_VALUES[@]}"; do
      k="${K_VALUES[$i]}"
      temp="${TEMPS[$i]}"

      # Build the command safely as an array
      cmd=(python produce_data.py
           --model "$model"
           --dataset "$dataset"
           --n "$N"
           --k "$k")

      # Only include --temperature when specified
      if [[ -n "$temp" ]]; then
        cmd+=(--temperature "$temp")
      fi

      echo "Running: ${cmd[*]}"
      "${cmd[@]}"
    done
  done
done

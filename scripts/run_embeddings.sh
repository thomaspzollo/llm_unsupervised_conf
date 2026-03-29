#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# User config
# -----------------------------

BASE_DIR="../outputs"

# If your experiment naming differs per dataset, adjust these fixed parts:
N=1000
TEMP_TRAIN=0.7
K_TRAIN=100
TEMP_TEST=0.6
K_TEST=1

# Embed settings
MODE="question_response"               # "question" or "question_response"
EMBED_BATCH_SIZE=1024
N_GPU=4
SAVE_FORMAT="npz"             # npz recommended
# For vLLM embed models; set to 1 if you don't need it
ENFORCE_EAGER="--enforce_eager"

# Datasets to run
DATASETS=(
  "gsm8k"
  "polymath"
  "trivia_qa"
  "sciq"
  "webq"
)

# Models to run: pair of (run_id_model_dir, hf_model_name)
MODELS=(
  "Qwen3-0.6B|Qwen/Qwen3-0.6B"
  "Qwen3-1.7B|Qwen/Qwen3-1.7B"
  "Qwen3-4B-Thinking-2507|Qwen/Qwen3-4B-Thinking-2507"
  "Qwen3-8B|Qwen/Qwen3-8B"
  "Qwen3-14B|Qwen/Qwen3-14B"
  "Nemotron-Cascade-8B-Thinking|nvidia/Nemotron-Cascade-8B-Thinking"
  "OpenReasoning-Nemotron-7B|nvidia/OpenReasoning-Nemotron-7B"
  "DeepSeek-R1-Distill-Llama-8B|deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "DeepSeek-R1-Distill-Qwen-7B|deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

# -----------------------------
# Helper: choose embedding model for each generator model
# (edit this mapping if you change embedding backbones)
# -----------------------------
pick_embed_model () {
  local hf_model="$1"
  if [[ "$hf_model" == *"0.6B"* ]]; then
    echo "Qwen/Qwen3-0.6B"
  elif [[ "$hf_model" == *"1.7B"* ]]; then
    echo "Qwen/Qwen3-1.7B"
  elif [[ "$hf_model" == *"4B"* ]]; then
    echo "Qwen/Qwen3-4B-Thinking-2507"
  elif [[ "$hf_model" == *"8B"* ]]; then
    echo "Qwen/Qwen3-8B"
  else
    echo "Qwen/Qwen3-0.6B"
  fi
  echo $hf_model
  # echo "google/gemma-3-4b-it"
}

# -----------------------------
# Main loop
# -----------------------------
for ds in "${DATASETS[@]}"; do
  for entry in "${MODELS[@]}"; do
    run_dir="${entry%%|*}"
    hf_model="${entry#*|}"

    embed_model="$(pick_embed_model "$hf_model")"

    run_id="${run_dir}/${ds}/n_${N}_temp_${TEMP_TRAIN}_k_${K_TRAIN}_out_df_temp_${TEMP_TEST}_k_${K_TEST}"

    echo "============================================================"
    echo "Dataset:     $ds"
    echo "Gen model:   $hf_model"
    echo "Embed model: $embed_model"
    echo "Run ID:      $run_id"
    echo "Mode:        $MODE"
    echo "============================================================"

    python embed.py \
      --run_id "$run_id" \
      --model "$embed_model" \
      --base_dir "$BASE_DIR" \
      --mode "$MODE" \
      --n_device "$N_GPU" \
      --batch_size "$EMBED_BATCH_SIZE" \
      --save_format "$SAVE_FORMAT" \
      $ENFORCE_EAGER

    echo
  done
done

echo "All embeddings done."

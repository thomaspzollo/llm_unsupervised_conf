#!/usr/bin/env bash
set -euo pipefail

# Runs verbalized_confidence.py for a grid of (model, dataset, stem_idx).
# Everything else in the run_id is fixed:
#   n_1000_temp_0.7_k_100_out_df_temp_0.6_k_1

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-verbalized_confidence.py}"

if [[ -n "${N_DEVICE:-}" ]]; then
  N_DEVICE="${N_DEVICE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
  N_DEVICE="${#GPU_LIST[@]}"
else
  N_DEVICE=1
fi

# Datasets to run
DATASETS=(
  # "math"
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

# Stem indices to sweep
STEM_IDXS=(0 1 2 3 4 5 6 7 8 9)

FIXED_SUFFIX="n_1000_temp_0.7_k_100_out_df_temp_0.6_k_1"

for ds in "${DATASETS[@]}"; do
  for entry in "${MODELS[@]}"; do
    IFS="|" read -r model_dir hf_model <<< "${entry}"

    for stem_idx in "${STEM_IDXS[@]}"; do
      RUN_ID="${model_dir}/${ds}/${FIXED_SUFFIX}"

      echo "============================================================"
      echo "Dataset:  ${ds}"
      echo "Model:    ${hf_model}"
      echo "Stem idx: ${stem_idx}"
      echo "Run ID:   ${RUN_ID}"
      echo "Command:  ${PYTHON_BIN} ${SCRIPT} --run_id=\"${RUN_ID}\" --model=\"${hf_model}\" --n_device=${N_DEVICE} --stem_idx=${stem_idx}"
      echo "============================================================"

      ${PYTHON_BIN} "${SCRIPT}" \
        --run_id="${RUN_ID}" \
        --model="${hf_model}" \
        --n_device="${N_DEVICE}" \
        --stem_idx="${stem_idx}"
    done
  done
done


# #!/usr/bin/env bash
# set -euo pipefail

# # Runs verbalized_confidence.py for a grid of (model, dataset, stem_idx).
# # Fixed base suffix:
# #   n_1000_temp_0.7_k_100_out_df_temp_0.6_k_1
# # We append /stem_<idx> so each run writes to a unique location.

# PYTHON_BIN="${PYTHON_BIN:-python}"
# SCRIPT="${SCRIPT:-verbalized_confidence.py}"
# N_DEVICE="${N_DEVICE:-1}"

# # Datasets to run
# DATASETS=(
#   "gsm8k"
#   # "trivia_qa"
# )

# # Models to run: pair of (run_id_model_dir, hf_model_name)
# MODELS=(
#   "Qwen3-1.7B|Qwen/Qwen3-1.7B"
#   # "Qwen3-8B|Qwen/Qwen3-8B"
#   # "Nemotron-Cascade-8B-Thinking|nvidia/Nemotron-Cascade-8B-Thinking"
#   # "DeepSeek-R1-Distill-Llama-8B|deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# )

# # Stem indices to sweep
# STEM_IDXS=(0 1 2 3 4 5 6 7 8 9)

# FIXED_SUFFIX="n_1000_temp_0.7_k_100_out_df_temp_0.6_k_1"

# for ds in "${DATASETS[@]}"; do
#   for entry in "${MODELS[@]}"; do
#     IFS="|" read -r model_dir hf_model <<< "${entry}"

#     for stem_idx in "${STEM_IDXS[@]}"; do
#       RUN_ID="${model_dir}/${ds}/${FIXED_SUFFIX}/stem_${stem_idx}"

#       echo "============================================================"
#       echo "Dataset:  ${ds}"
#       echo "Model:    ${hf_model}"
#       echo "Stem idx: ${stem_idx}"
#       echo "Run ID:   ${RUN_ID}"
#       echo "Command:  ${PYTHON_BIN} ${SCRIPT} --run_id=\"${RUN_ID}\" --model=\"${hf_model}\" --n_device=${N_DEVICE} --stem_idx=${stem_idx}"
#       echo "============================================================"

#       ${PYTHON_BIN} "${SCRIPT}" \
#         --run_id="${RUN_ID}" \
#         --model="${hf_model}" \
#         --n_device="${N_DEVICE}" \
#         --stem_idx="${stem_idx}"
#     done
#   done
# done
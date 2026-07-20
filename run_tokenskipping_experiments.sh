#!/bin/bash
# Run TokenSkipping baseline on HotpotQA / 2Wiki / MuSiQue with 3 seeds each.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1
unset HF_ENDPOINT || true

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/autodl-tmp/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/hf_cache/hub}"

PYTHON=$(which python)
WIKI_SCRIPT=run_all_wiki_experiments_v2.py
WIKI2_SCRIPT=run_all_2wiki_experiments_v2.py
MUSIQUE_SCRIPT=run_all_musique_experiments_v2.py
METRICS_SCRIPT=record_experiment_metrics.py
LOGDIR=logs

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-7B-Instruct}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/root/autodl-tmp/hf_cache/models/Qwen2.5-7B-Instruct}"
MODEL_PATH="${MODEL_PATH:-$LOCAL_MODEL_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/tokenskipping_baseline}"
WIKI2_DATA_PATH="${WIKI2_DATA_PATH:-/root/autodl-tmp/kvmem/data/2wiki/dev.json}"
MUSIQUE_DATA_PATH="${MUSIQUE_DATA_PATH:-/root/autodl-tmp/kvmem/data/musique/dev.json}"
WIKI_INDEX_DIR="${WIKI_INDEX_DIR:-wiki_index}"

RUN_TAGS=("run1" "run2" "run3")
RUN_SEEDS=(233 42 3407)
CACHE_RATIOS=("0.5" "0.2")

mkdir -p "$LOGDIR"

if [ -f "$LOCAL_MODEL_DIR/config.json" ] && [ -f "$LOCAL_MODEL_DIR/tokenizer_config.json" ]; then
  echo "$(date): Found local model at ${LOCAL_MODEL_DIR}, skip download."
else
  echo "$(date): Local model not found, downloading ${MODEL_REPO} to ${LOCAL_MODEL_DIR}..."
  mkdir -p "$(dirname "$LOCAL_MODEL_DIR")"
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_REPO" --local-dir "$LOCAL_MODEL_DIR"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL_REPO" --local-dir "$LOCAL_MODEL_DIR"
  else
    echo "[ERROR] Neither 'hf' nor 'huggingface-cli' is installed."
    exit 1
  fi
fi

run_wiki() {
  local output_dir="$1"
  local seed="$2"
  local cache_ratio="$3"
  local tag="tokenskipping_r${cache_ratio/./}"
  local log_file="${LOGDIR}/logs_${tag}_hotpotqa_seed${seed}.log"
  local result_json="${output_dir}/react_kv_tokenskipping_wiki.json"

  echo "$(date): HotpotQA TokenSkipping seed=${seed} cache_ratio=${cache_ratio} -> ${output_dir}"
  $PYTHON -u "$WIKI_SCRIPT" \
    --experiment react_kv_tokenskipping \
    --model_path "$MODEL_PATH" \
    --output_dir "$output_dir" \
    --seed "$seed" \
    --cache_ratio "$cache_ratio" 2>&1 | tee "$log_file"

  if [ -f "$result_json" ]; then
    $PYTHON "$METRICS_SCRIPT" \
      --result_json "$result_json" \
      --dataset "hotpotqa" \
      --method "react_kv_tokenskipping" \
      --cache_ratio "$cache_ratio" \
      --output_file "${output_dir}/metrics_${tag}.md"
  fi
}

run_2wiki() {
  local output_dir="$1"
  local seed="$2"
  local cache_ratio="$3"
  local tag="tokenskipping_r${cache_ratio/./}"
  local log_file="${LOGDIR}/logs_${tag}_2wiki_seed${seed}.log"
  local result_json="${output_dir}/react_kv_tokenskipping_2wiki.json"
  local data_args=""
  if [ -f "$WIKI2_DATA_PATH" ]; then
    data_args="--data_path $WIKI2_DATA_PATH"
  fi

  echo "$(date): 2Wiki TokenSkipping seed=${seed} cache_ratio=${cache_ratio} -> ${output_dir}"
  $PYTHON -u "$WIKI2_SCRIPT" \
    --experiment react_kv_tokenskipping \
    --model_path "$MODEL_PATH" \
    --output_dir "$output_dir" \
    --wiki_index_dir "$WIKI_INDEX_DIR" \
    --seed "$seed" \
    --cache_ratio "$cache_ratio" \
    ${data_args} 2>&1 | tee "$log_file"

  if [ -f "$result_json" ]; then
    $PYTHON "$METRICS_SCRIPT" \
      --result_json "$result_json" \
      --dataset "2wiki" \
      --method "react_kv_tokenskipping" \
      --cache_ratio "$cache_ratio" \
      --output_file "${output_dir}/metrics_${tag}.md"
  fi
}

run_musique() {
  local output_dir="$1"
  local seed="$2"
  local cache_ratio="$3"
  local tag="tokenskipping_r${cache_ratio/./}"
  local log_file="${LOGDIR}/logs_${tag}_musique_seed${seed}.log"
  local pct
  pct=$($PYTHON -c "print(int(round(float('${cache_ratio}') * 100)))")
  local result_json="${output_dir}/react_kv_tokenskipping_musique_r${pct}.json"
  local data_args=""
  if [ -f "$MUSIQUE_DATA_PATH" ]; then
    data_args="--data_path $MUSIQUE_DATA_PATH"
  fi

  echo "$(date): MuSiQue TokenSkipping seed=${seed} cache_ratio=${cache_ratio} -> ${output_dir}"
  $PYTHON -u "$MUSIQUE_SCRIPT" \
    --experiment react_kv_tokenskipping \
    --model_path "$MODEL_PATH" \
    --output_dir "$output_dir" \
    --wiki_index_dir "$WIKI_INDEX_DIR" \
    --seed "$seed" \
    --cache_ratio "$cache_ratio" \
    ${data_args} 2>&1 | tee "$log_file"

  if [ -f "$result_json" ]; then
    $PYTHON "$METRICS_SCRIPT" \
      --result_json "$result_json" \
      --dataset "musique" \
      --method "react_kv_tokenskipping" \
      --cache_ratio "$cache_ratio" \
      --output_file "${output_dir}/metrics_${tag}.md"
  fi
}

for i in "${!RUN_TAGS[@]}"; do
  RUN="${RUN_TAGS[$i]}"
  SEED="${RUN_SEEDS[$i]}"
  echo "$(date): ===== ${RUN} (seed=${SEED}) ====="

  for cache_ratio in "${CACHE_RATIOS[@]}"; do
    ratio_tag="r${cache_ratio/./}"
    run_wiki "${OUTPUT_ROOT}/hotpotqa/${RUN}/${ratio_tag}" "$SEED" "$cache_ratio"
    run_2wiki "${OUTPUT_ROOT}/2wiki/${RUN}/${ratio_tag}" "$SEED" "$cache_ratio"
    run_musique "${OUTPUT_ROOT}/musique/${RUN}/${ratio_tag}" "$SEED" "$cache_ratio"
  done
done

echo "$(date): All TokenSkipping baseline runs complete."

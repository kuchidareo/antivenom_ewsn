#!/usr/bin/env zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN="${DRY_RUN:-0}"
DATA_ROOT="$ROOT_DIR/../data"

COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --epochs 10
  --batch-size 8
  --num-workers 0
  --img-size 224
  --disable-perf
  --train-frac 0.1
  --test-frac 0.1
)

FEASIBILITY_RUNNER="../feasibility/ml_running.py"

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_many() {
  local scenario="$1"
  local poison_type="$2"
  local log_dir="$3"
  local count="$4"

  mkdir -p "$log_dir"

  for ((i = 1; i <= count; i++)); do
    echo "[$scenario] run $i/$count for poison_type=$poison_type -> $log_dir"
    run_cmd python "senario_evaluation/${scenario}/ml_running.py" \
      "${COMMON_ARGS[@]}" \
      --poison-type "$poison_type" \
      --log-dir "$log_dir"
  done
}

run_many_feasibility() {
  local label="$1"
  local poison_type="$2"
  local log_dir="$3"
  local train_frac="$4"
  local test_frac="$5"
  local count="$6"
  shift 6
  local extra_args=("$@")

  mkdir -p "$log_dir"

  for ((i = 1; i <= count; i++)); do
    echo "[${label}] run $i/$count for poison_type=$poison_type -> $log_dir"
    run_cmd python "$FEASIBILITY_RUNNER" \
      --epochs 10 \
      --img-size 224 \
      --num-workers 0 \
      --data-root "$DATA_ROOT" \
      --log-dir "$ROOT_DIR/$log_dir" \
      --train-frac "$train_frac" \
      --test-frac "$test_frac" \
      --poison-type "$poison_type" \
      --dataset kuchidareo/small_trashnet \
      --batch-size 8 \
      "${extra_args[@]}"
  done
}

# Missing runs to reach 5 CSVs per scenario/poisoning type.
run_many_feasibility "baseclean" "clean" "senario_evaluation/baseclean" 0.1 0.1 2
run_many_feasibility "baseblurring" "blurring" "senario_evaluation/baseblurring" 0.05 0.05 4 --poison-frac 0.3

run_many "adamw_weight_decay" "clean" "senario_evaluation/adamw_weight_decay/clean" 1
run_many "adamw_weight_decay" "blurring" "senario_evaluation/adamw_weight_decay/blurring" 4

run_many "batch_norm" "clean" "senario_evaluation/batch_norm/clean" 2
run_many "batch_norm" "blurring" "senario_evaluation/batch_norm/blurring" 4

run_many "label_smooth" "clean" "senario_evaluation/label_smooth/clean" 5
run_many "label_smooth" "blurring" "senario_evaluation/label_smooth/blurring" 3

run_many "model_pruning" "clean" "senario_evaluation/model_pruning/clean" 3
run_many "model_pruning" "blurring" "senario_evaluation/model_pruning/blurring" 5

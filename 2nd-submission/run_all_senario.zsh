#!/bin/zsh
set -e

DATA_ROOT="../data"
PYTHON_BIN="${0:A:h}/../venv/bin/python"
EPOCHS=20
BATCH_SIZE=8
NUM_WORKERS=0
IMG_SIZE=224
TRAIN_FRAC=0.1
TEST_FRAC=0.1
POISON_FRAC=0.3
N_RUNS=5

COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --img-size "$IMG_SIZE"
  --disable-perf
  --train-frac "$TRAIN_FRAC"
  --test-frac "$TEST_FRAC"
)

run_one() {
  local scenario=$1
  local poison_type=$2
  shift 2

  local script="senario_evaluation/${scenario}/ml_running.py"
  local log_dir="senario_evaluation/${scenario}/${poison_type}_20rounds"

  echo ""
  echo "============================================================"
  echo "Scenario: ${scenario}"
  echo "Poison:   ${poison_type}"
  echo "Log dir:  ${log_dir}"
  echo "Extra:    $@"
  echo "============================================================"

  if [[ "$poison_type" == "blurring" ]]; then
    "$PYTHON_BIN" "$script" \
      "${COMMON_ARGS[@]}" \
      --poison-type blurring \
      --poison-frac "$POISON_FRAC" \
      --log-dir "$log_dir" \
      "$@"
  else
    "$PYTHON_BIN" "$script" \
      "${COMMON_ARGS[@]}" \
      --poison-type clean \
      --log-dir "$log_dir" \
      "$@"
  fi
}

run_repeated() {
  local scenario=$1
  shift

  for poison_type in clean blurring; do
    for ((run_id = 1; run_id <= N_RUNS; ++run_id)); do
      echo ""
      echo "############################################################"
      echo "# Run ${run_id}/${N_RUNS} | ${scenario} | ${poison_type}"
      echo "############################################################"
      run_one "$scenario" "$poison_type" "$@"
    done
  done
}

# 1. BatchNorm
run_repeated batch_norm

# 2. Label smoothing
run_repeated label_smooth

# 3. AdamW weight decay
run_repeated adamw_weight_decay \
  --weight-decay 1e-4

# 4. Weight normalization
run_repeated weight_normalization

# 5. Model pruning
run_repeated model_pruning \
  --final-sparsity 0.8 \
  --prune-start-ratio 0.1 \
  --prune-end-ratio 0.8 \
  --prune-freq 10

# 6. Backward stability
run_repeated backward_stabilization

echo ""
echo "All experiments finished successfully."

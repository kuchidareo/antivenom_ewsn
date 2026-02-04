

#!/usr/bin/env bash
set -euo pipefail

# Background workload: fixed ON duration + random OFF duration.
# Designed to create a bimodal background load (idle vs batch) for OT experiments.

ON_SEC=${ON_SEC:-10}
OFF_MIN_SEC=${OFF_MIN_SEC:-20}
OFF_MAX_SEC=${OFF_MAX_SEC:-60}
MODE=${MODE:-cpu}           # cpu | mem | io
THREADS=${THREADS:-0}
RUN_FOR_SEC=${RUN_FOR_SEC:-0}  # 0 = forever
WARMUP_SEC=${WARMUP_SEC:-2}    # seconds to skip at start of each ON window (transient)

usage() {
  cat <<EOF
Usage: $(basename "$0") [--on SEC] [--off-min SEC] [--off-max SEC] [--mode cpu|mem|io] [--threads N] [--run-for SEC] [--warmup SEC]

Env vars (override defaults):
  ON_SEC=$ON_SEC
  OFF_MIN_SEC=$OFF_MIN_SEC
  OFF_MAX_SEC=$OFF_MAX_SEC
  MODE=$MODE
  THREADS=$THREADS
  RUN_FOR_SEC=$RUN_FOR_SEC
  WARMUP_SEC=$WARMUP_SEC

Examples:
  MODE=cpu ON_SEC=10 OFF_MIN_SEC=20 OFF_MAX_SEC=60 ./run_bg_workloads.sh
  ./run_bg_workloads.sh --mode io --on 10 --off-min 10 --off-max 30 --run-for 600
EOF
}

# --- arg parsing ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --on) ON_SEC="$2"; shift 2;;
    --off-min) OFF_MIN_SEC="$2"; shift 2;;
    --off-max) OFF_MAX_SEC="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --run-for) RUN_FOR_SEC="$2"; shift 2;;
    --warmup) WARMUP_SEC="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ "$OFF_MIN_SEC" -gt "$OFF_MAX_SEC" ]]; then
  echo "OFF_MIN_SEC must be <= OFF_MAX_SEC" >&2
  exit 1
fi

# Determine thread count
if [[ "$THREADS" -le 0 ]]; then
  if command -v nproc >/dev/null 2>&1; then
    THREADS=$(nproc)
  elif command -v sysctl >/dev/null 2>&1; then
    THREADS=$(sysctl -n hw.ncpu 2>/dev/null || echo 1)
  else
    THREADS=1
  fi
fi

now_ts() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }

rand_int() {
  # Inclusive random integer in [min, max]
  local min="$1" max="$2"
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<PY
import random
print(random.randint(int("$min"), int("$max")))
PY
  else
    # $RANDOM is 0..32767; scale to range
    local r=$RANDOM
    echo $(( min + (r % (max - min + 1)) ))
  fi
}

PIDS=()
cleanup() {
  # Kill any running workers
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    for pid in "${PIDS[@]}"; do
      kill "$pid" >/dev/null 2>&1 || true
    done
    # Give them a moment, then force
    sleep 0.2 || true
    for pid in "${PIDS[@]}"; do
      kill -9 "$pid" >/dev/null 2>&1 || true
    done
  fi
  PIDS=()
}
trap cleanup EXIT INT TERM

# --- workload workers ---
start_cpu_workers() {
  # Prefer openssl if available (good CPU burn)
  if command -v openssl >/dev/null 2>&1; then
    # Each worker burns CPU via openssl speed; redirect output.
    for _ in $(seq 1 "$THREADS"); do
      (openssl speed -seconds 60 sha256 >/dev/null 2>&1) &
      PIDS+=("$!")
    done
    return
  fi

  # Fallback: stream zeros and checksum (usually available)
  if command -v cksum >/dev/null 2>&1; then
    for _ in $(seq 1 "$THREADS"); do
      (while :; do dd if=/dev/zero bs=1m count=32 2>/dev/null | cksum >/dev/null; done) &
      PIDS+=("$!")
    done
    return
  fi

  if command -v shasum >/dev/null 2>&1; then
    for _ in $(seq 1 "$THREADS"); do
      (while :; do dd if=/dev/zero bs=1m count=16 2>/dev/null | shasum >/dev/null; done) &
      PIDS+=("$!")
    done
    return
  fi

  # Last resort: pure bash arithmetic loop
  for _ in $(seq 1 "$THREADS"); do
    (x=0; while :; do x=$(( (x + 1103515245) ^ 12345 )); done) &
    PIDS+=("$!")
  done
}

start_mem_workers() {
  # Allocate memory for the ON window; do it in separate processes.
  # Prefer python (most controllable). Allocate ~256MB per worker by default.
  local per_mb=${MEM_MB_PER_WORKER:-256}
  if command -v python3 >/dev/null 2>&1; then
    for _ in $(seq 1 "$THREADS"); do
      (python3 - <<PY
import os, time
mb = int(os.environ.get('MEM_MB_PER_WORKER', '$per_mb'))
# Allocate and touch memory to force commit
b = bytearray(mb * 1024 * 1024)
for i in range(0, len(b), 4096):
    b[i] = 1
time.sleep(10**9)
PY
      ) &
      PIDS+=("$!")
    done
    return
  fi

  # Fallback: write big temp files to tmpfs or /tmp (page cache pressure)
  local dir=/tmp
  [[ -d /dev/shm ]] && dir=/dev/shm
  for i in $(seq 1 "$THREADS"); do
    local f="$dir/bg_mem_${$}_$i.bin"
    (dd if=/dev/zero of="$f" bs=1m count="$per_mb" conv=fsync 2>/dev/null; tail -f /dev/null >/dev/null) &
    PIDS+=("$!")
  done
}

start_io_workers() {
  local dir=/tmp
  [[ -d /dev/shm ]] && dir=/dev/shm
  local mb=${IO_MB:-512}
  for i in $(seq 1 "$THREADS"); do
    local f="$dir/bg_io_${$}_$i.bin"
    (while :; do
        dd if=/dev/zero of="$f" bs=1m count="$mb" conv=fsync 2>/dev/null
        dd if="$f" of=/dev/null bs=1m 2>/dev/null
        rm -f "$f"
      done) &
    PIDS+=("$!")
  done
}

start_workers() {
  cleanup
  case "$MODE" in
    cpu) start_cpu_workers;;
    mem) start_mem_workers;;
    io)  start_io_workers;;
    *) echo "Unknown MODE: $MODE (expected cpu|mem|io)" >&2; exit 1;;
  esac
}

# --- main loop ---
start_time=$(date +%s)
cycle=0
while :; do
  cycle=$((cycle + 1))
  off=$(rand_int "$OFF_MIN_SEC" "$OFF_MAX_SEC")

  echo "[$(now_ts)] cycle=$cycle OFF ${off}s (mode=$MODE threads=$THREADS)"
  sleep "$off"

  echo "[$(now_ts)] cycle=$cycle ON  ${ON_SEC}s (warmup=${WARMUP_SEC}s)"
  start_workers
  # Let the load stabilize
  if [[ "$WARMUP_SEC" -gt 0 ]]; then
    sleep "$WARMUP_SEC"
  fi
  # Keep workers running for the remainder of ON
  remain=$(( ON_SEC - WARMUP_SEC ))
  if [[ "$remain" -gt 0 ]]; then
    sleep "$remain"
  fi
  cleanup

  if [[ "$RUN_FOR_SEC" -gt 0 ]]; then
    now=$(date +%s)
    if [[ $((now - start_time)) -ge "$RUN_FOR_SEC" ]]; then
      echo "[$(now_ts)] finished after ${RUN_FOR_SEC}s"
      exit 0
    fi
  fi

done
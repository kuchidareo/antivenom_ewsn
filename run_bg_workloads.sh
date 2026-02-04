#!/usr/bin/env bash
# Background workloads (Perception / Comms / LogMap)
# Always run under systemd CPUQuota=20% (when systemd-run is available).

set -euo pipefail

# ---- tmpfs preferred (avoid SD/eMMC writes) ----
TMPDIR_DEFAULT="/dev/shm"
if [ -d "${TMPDIR_DEFAULT}" ] && [ -w "${TMPDIR_DEFAULT}" ]; then
  TMPDIR="${TMPDIR_DEFAULT}"
else
  TMPDIR="/tmp"
fi
LOG_DIR="${TMPDIR}"

# ---- Settings ----
# Perception: FPS is an upper bound; PERCEPTION_UTIL throttles CPU duty-cycle.
FEATURE="${FEATURE:-fast}"          # fast / orb
CAM_INDEX="${CAM_INDEX:-0}"
TARGET_FPS="${TARGET_FPS:-30}"
PERCEPTION_UTIL="${PERCEPTION_UTIL:-0.15}"  # approx CPU share for perception process

# Comms: throttle by duty-cycling iperf3 bursts.
IPERF_SERVER="${IPERF_SERVER:-192.0.2.10}"  # replace with real iperf3 server
UPLOAD_MBIT="${UPLOAD_MBIT:-2}"
BURST_SEC="${BURST_SEC:-0.5}"
IDLE_SEC="${IDLE_SEC:-1.0}"

IO_DIR="${IO_DIR:-${TMPDIR}/bg_io}"
TILE_URL="${TILE_URL:-https://tile.openstreetmap.org/10/550/340.png}"
TILE_PERIOD_SEC="${TILE_PERIOD_SEC:-7}"
CLEAN_PERIOD_SEC="${CLEAN_PERIOD_SEC:-3600}"

# LogMap defaults: run, but no disk I/O (SD/eMMC protection)
LOGMAP_IO=0
LOGMAP_NET=1

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

# ---- Generate perception workload ----
PERCEPTION_PY="$(mktemp -p "${TMPDIR}" bg_perception.XXXXXX.py)"
cat > "${PERCEPTION_PY}" <<'PY'
import os, time
import numpy as np
import cv2 as cv

FEATURE = os.getenv("FEATURE","fast").lower()
CAM_INDEX = int(os.getenv("CAM_INDEX","0"))
TARGET_FPS = float(os.getenv("TARGET_FPS","30"))
PERCEPTION_UTIL = float(os.getenv("PERCEPTION_UTIL","0.15"))
period = 1.0 / max(TARGET_FPS, 0.1)

cap = cv.VideoCapture(CAM_INDEX)
use_synthetic = not cap.isOpened()
if use_synthetic:
    w,h = 640,480
    frame = (np.random.rand(h,w,3)*255).astype(np.uint8)

if FEATURE == "orb":
    det = cv.ORB_create(nfeatures=500)
    def run_feat(img):
        det.detectAndCompute(img, None)
else:
    det = cv.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
    def run_feat(img):
        det.detect(img, None)

while True:
    t0 = time.time()

    if use_synthetic:
        img = frame
        _, buf = cv.imencode(".jpg", img, [int(cv.IMWRITE_JPEG_QUALITY), 85])
        img = cv.imdecode(buf, cv.IMREAD_COLOR)
    else:
        ok, img = cap.read()
        if not ok:
            use_synthetic = True
            continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    run_feat(gray)

    # Duty-cycle throttling: aim for PERCEPTION_UTIL CPU usage (approx).
    dt = time.time() - t0
    util = max(min(PERCEPTION_UTIL, 0.95), 0.01)
    desired_cycle = max(period, dt / util)
    to_sleep = desired_cycle - dt
    if to_sleep > 0:
        time.sleep(to_sleep)
PY

# ---- Generate logmap workload (I/O can be disabled) ----
LOGMAP_PY="$(mktemp -p "${TMPDIR}" bg_logmap.XXXXXX.py)"
cat > "${LOGMAP_PY}" <<'PY'
import os, time, random
import requests
from pathlib import Path

ENABLE_IO = os.getenv("LOGMAP_IO", "0") != "0"
ENABLE_NET = os.getenv("LOGMAP_NET", "1") != "0"

TILE_URL = os.getenv("TILE_URL", "https://tile.openstreetmap.org/10/550/340.png")
TILE_PERIOD = float(os.getenv("TILE_PERIOD_SEC", "7"))
CLEAN_PERIOD = float(os.getenv("CLEAN_PERIOD_SEC", "3600"))

IO_DIR = None
seq_path = None
seq_file = None

if ENABLE_IO:
    IO_DIR = Path(os.getenv("IO_DIR", "/tmp/bg_io"))
    IO_DIR.mkdir(parents=True, exist_ok=True)
    seq_path = IO_DIR / "seq.bin"
    seq_file = seq_path.open("ab", buffering=0)

SIZES = [4*1024, 8*1024, 16*1024, 64*1024, 128*1024]
SYNC_EVERY = 200

wcount = 0
last_tile = 0.0
last_cleanup = time.time()


def random_write():
    global wcount
    fname = IO_DIR / f"rnd_{random.randint(0,999999):06d}.bin"
    size = random.choice(SIZES)
    with open(fname, "ab", buffering=0) as f:
        f.write(os.urandom(size))
    wcount += 1
    if seq_file is not None and (wcount % SYNC_EVERY == 0):
        try:
            os.fsync(seq_file.fileno())
        except Exception:
            pass


def sequential_append():
    if seq_file is None:
        return
    seq_file.write(os.urandom(random.choice(SIZES)))


def fetch_tile():
    if not ENABLE_NET:
        return
    try:
        r = requests.get(TILE_URL, timeout=3)
        if r.ok and ENABLE_IO and IO_DIR is not None:
            (IO_DIR / "tile.cache").write_bytes(r.content)
    except Exception:
        pass


def periodic_cleanup():
    global seq_file
    if IO_DIR is None:
        return
    for f in IO_DIR.glob("rnd_*.bin"):
        try:
            f.unlink()
        except Exception:
            pass
    try:
        if seq_file is not None:
            seq_file.close()
    except Exception:
        pass
    try:
        if seq_path is not None and seq_path.exists():
            seq_path.unlink()
    except Exception:
        pass
    if seq_path is not None:
        try:
            seq_file = seq_path.open("ab", buffering=0)
        except Exception:
            seq_file = None


while True:
    now = time.time()

    if ENABLE_IO and IO_DIR is not None:
        random_write()
        sequential_append()
        time.sleep(0.02)
    else:
        time.sleep(0.05)

    if now - last_tile >= TILE_PERIOD:
        fetch_tile()
        last_tile = now

    if ENABLE_IO and (now - last_cleanup >= CLEAN_PERIOD):
        periodic_cleanup()
        last_cleanup = time.time()
PY

# ---- Launch ----
pids=()

cleanup() {
  echo "[bg] stopping..."
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  rm -f "${PERCEPTION_PY}" "${LOGMAP_PY}" 2>/dev/null || true
}
trap cleanup EXIT

start_perception() {
  echo "[bg] perception start"
  nice -n 10 "${PYTHON_BIN}" "${PERCEPTION_PY}" \
    1>"${LOG_DIR}/bg_perception.out" 2>&1 &
  pids+=($!)
}

start_comms() {
  echo "[bg] comms (iperf3 UDP bursts) start"
  nice -n 10 bash -lc "
    while true; do
      iperf3 -u -c ${IPERF_SERVER} -b ${UPLOAD_MBIT}M -t ${BURST_SEC} -l 1200 >\"${LOG_DIR}/bg_iperf3.out\" 2>&1 || true
      sleep ${IDLE_SEC}
    done
  " &
  pids+=($!)
}

start_logmap() {
  echo "[bg] logging/mapping start (LOGMAP_IO=${LOGMAP_IO}, LOGMAP_NET=${LOGMAP_NET})"
  LOGMAP_IO="${LOGMAP_IO}" LOGMAP_NET="${LOGMAP_NET}" \
  nice -n 10 "${PYTHON_BIN}" "${LOGMAP_PY}" \
    1>"${LOG_DIR}/bg_logmap.out" 2>&1 &
  pids+=($!)
}

start_perception
start_comms
start_logmap

echo "[bg] all workloads launched (throttled to ~20% CPU via pacing)."
echo "[bg] PIDs: ${pids[*]}"
echo "[bg] logs: ${LOG_DIR}/bg_perception.out , ${LOG_DIR}/bg_iperf3.out , ${LOG_DIR}/bg_logmap.out"

wait
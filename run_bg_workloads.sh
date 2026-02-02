#!/usr/bin/env bash
# run_bg_workloads.sh
# 背景負荷(Perception/Comms/Logging+定期クリーン)を起動
# 依存: python3, OpenCV(cv2), numpy, requests, iperf3

set -euo pipefail

### ====== 設定 ======
# Perception
FEATURE="${FEATURE:-orb}"            # orb / fast
CAM_INDEX="${CAM_INDEX:-0}"          # /dev/videoN -> N
TARGET_FPS="${TARGET_FPS:-30}"

# Communications
IPERF_SERVER="${IPERF_SERVER:-192.0.2.10}"   # 置き換え必須（iperf3 サーバ）
UPLOAD_MBIT="${UPLOAD_MBIT:-15}"             # 10–20 推奨
BURST_SEC="${BURST_SEC:-1}"                  # おおよそ 1 秒ごとにバースト

# Logging/Mapping
IO_DIR="${IO_DIR:-/tmp/bg_io}"
TILE_URL="${TILE_URL:-https://tile.openstreetmap.org/10/550/340.png}"  # 実験ではレート注意
TILE_PERIOD_SEC="${TILE_PERIOD_SEC:-7}"                                 # 5–10 秒レンジ
CLEAN_PERIOD_SEC="${CLEAN_PERIOD_SEC:-3600}"                             # ディスク掃除の周期(秒)

# CPU ピン留め（存在コアに合わせて調整）
CPU_PERCEPTION="${CPU_PERCEPTION:-0}"
CPU_COMMS="${CPU_COMMS:-1}"
CPU_LOGMAP="${CPU_LOGMAP:-2}"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

### ====== 依存チェック(甘め) ======
command -v iperf3 >/dev/null || { echo "iperf3 が必要"; exit 1; }
$PYTHON_BIN - <<'PY' >/dev/null 2>&1 || { echo "python3 必須"; exit 1; }
print("ok")
PY

# 最低限の Python パッケージ確認（なければ pip で入れろ）
$PYTHON_BIN - <<'PY' || true
try:
    import cv2, numpy, requests  # noqa
    print("python deps ok")
except Exception as e:
    print("WARNING: python deps missing:", e)
PY

# 作業ディレクトリを毎回クリーンに
mkdir -p "$IO_DIR"
rm -rf "$IO_DIR"/* || true

### ====== perception.py 生成 ======
PERCEPTION_PY="$(mktemp -p /tmp bg_perception.XXXXXX.py)"
cat > "$PERCEPTION_PY" <<'PY'
import os, time
import numpy as np
import cv2 as cv

FEATURE = os.getenv("FEATURE","orb").lower()
CAM_INDEX = int(os.getenv("CAM_INDEX","0"))
TARGET_FPS = float(os.getenv("TARGET_FPS","30"))
period = 1.0 / TARGET_FPS

cap = cv.VideoCapture(CAM_INDEX)
use_synthetic = False
if not cap.isOpened():
    use_synthetic = True
    w,h = 640,480
    frame = (np.random.rand(h,w,3)*255).astype(np.uint8)

if FEATURE == "fast":
    det = cv.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
    def run_feat(img): return det.detect(img, None)
else:
    det = cv.ORB_create(nfeatures=500)
    def run_feat(img): return det.detectAndCompute(img, None)

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
    _ = run_feat(gray)
    to_sleep = period - (time.time() - t0)
    if to_sleep > 0:
        time.sleep(to_sleep)
PY

### ====== logging_mapping.py 生成（定期クリーン入り） ======
LOGMAP_PY="$(mktemp -p /tmp bg_logmap.XXXXXX.py)"
cat > "$LOGMAP_PY" <<'PY'
import os, time, random
import requests
from pathlib import Path

IO_DIR = Path(os.getenv("IO_DIR","/tmp/bg_io"))
IO_DIR.mkdir(parents=True, exist_ok=True)
TILE_URL = os.getenv("TILE_URL","https://tile.openstreetmap.org/10/550/340.png")
TILE_PERIOD = float(os.getenv("TILE_PERIOD_SEC","7"))
CLEAN_PERIOD = float(os.getenv("CLEAN_PERIOD_SEC","3600"))

SIZES = [4*1024, 8*1024, 16*1024, 64*1024, 128*1024]
SYNC_EVERY = 20

wcount = 0
last_tile = 0.0
last_cleanup = time.time()
seq_path = IO_DIR/"seq.bin"
seq_file = seq_path.open("ab", buffering=0)

def random_write():
    global wcount
    fname = IO_DIR / f"rnd_{random.randint(0,999999):06d}.bin"
    size = random.choice(SIZES)
    data = os.urandom(size)
    with open(fname, "ab", buffering=0) as f:
        f.write(data)
        if random.random() < 0.3:
            f.flush()
            try:
                with open(fname, "rb") as fr:
                    fr.read(random.choice(SIZES))
            except Exception:
                pass
    wcount += 1
    if wcount % SYNC_EVERY == 0:
        try:
            os.fsync(seq_file.fileno())
        except Exception:
            pass

def sequential_append():
    size = random.choice(SIZES)
    seq_file.write(os.urandom(size))

def fetch_tile():
    try:
        r = requests.get(TILE_URL, timeout=3)
        if r.ok:
            (IO_DIR/"tile.cache").write_bytes(r.content)
    except Exception:
        pass

def periodic_cleanup():
    # rnd_*.bin を全削除し、seq.bin もリセット
    for f in IO_DIR.glob("rnd_*.bin"):
        try:
            f.unlink()
        except Exception:
            pass
    try:
        seq_file.close()
    except Exception:
        pass
    try:
        if seq_path.exists():
            seq_path.unlink()
    except Exception:
        pass
    # 再オープン
    globals()['seq_file'] = seq_path.open("ab", buffering=0)

t0 = time.time()
while True:
    now = time.time()
    random_write()
    sequential_append()

    if now - t0 > 0.2 and random.random() < 0.15:
        time.sleep(0.02)

    if now - last_tile >= TILE_PERIOD:
        fetch_tile()
        last_tile = now

    if now - last_cleanup >= CLEAN_PERIOD:
        periodic_cleanup()
        last_cleanup = time.time()
PY

### ====== 起動関数 ======
pids=()

start_perception() {
  echo "[bg] perception start on CPU $CPU_PERCEPTION"
  taskset -c "$CPU_PERCEPTION" nice -n 10 "$PYTHON_BIN" "$PERCEPTION_PY" \
    1>/tmp/bg_perception.out 2>&1 &
  pids+=($!)
}

start_comms() {
  echo "[bg] comms (iperf3 UDP bursts) start on CPU $CPU_COMMS"
  taskset -c "$CPU_COMMS" nice -n 10 bash -c "
    while true; do
      iperf3 -u -c ${IPERF_SERVER} -b ${UPLOAD_MBIT}M -t ${BURST_SEC} -l 1200 >/tmp/bg_iperf3.out 2>&1 || true
      sleep 0.2
    done
  " &
  pids+=($!)
}

start_logmap() {
  echo "[bg] logging/mapping start on CPU $CPU_LOGMAP (cleanup every ${CLEAN_PERIOD_SEC}s)"
  taskset -c "$CPU_LOGMAP" nice -n 10 "$PYTHON_BIN" "$LOGMAP_PY" \
    1>/tmp/bg_logmap.out 2>&1 &
  pids+=($!)
}

cleanup() {
  echo "[bg] stopping..."
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  # 片付け（IO_DIR 中身を消す）
  rm -rf "$IO_DIR"/* || true
}
trap cleanup EXIT

start_perception
start_comms
start_logmap

echo "[bg] all workloads launched."
echo "[bg] PIDs: ${pids[*]}"
echo "[bg] logs: /tmp/bg_perception.out , /tmp/bg_iperf3.out , /tmp/bg_logmap.out"

# フォアグラウンドで待つ（Ctrl-C で終了とクリーンアップ）
wait

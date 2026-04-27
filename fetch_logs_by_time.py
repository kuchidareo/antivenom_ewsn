from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> int:
    start = sys.argv[1] if len(sys.argv) > 1 else "20260131_1700"
    end = sys.argv[2] if len(sys.argv) > 2 else "20260201_0700"
    remote_root = sys.argv[3] if len(sys.argv) > 3 else "~/kuchida/antivenom_ewsn/logs_batch"
    dest_base = sys.argv[4] if len(sys.argv) > 4 else "logs_batch"

    devices = [114, 115, 116, 117, 118, 120]
    ts_pat = re.compile(r"[0-9]{8}_[0-9]{6}")

    for dev in devices:
        host = f"rasheed@192.168.0.{dev}"
        dest = f"{dest_base}_{dev}"
        tmp_list = Path(f"/tmp/rsync_files_{dev}.txt")
        user = host.split("@", 1)[0]
        if remote_root.startswith("~/"):
            remote_root_resolved = f"/home/{user}/{remote_root[2:]}"
        else:
            remote_root_resolved = remote_root

        list_cmd = [
            "rsync",
            "-av",
            "--list-only",
            "--recursive",
            f"{host}:{remote_root_resolved}/",
        ]
        result = subprocess.run(list_cmd, check=True, capture_output=True, text=True)

        with tmp_list.open("w") as f:
            for line in result.stdout.splitlines():
                parts = line.split()
                if not parts:
                    continue
                path = parts[-1]
                if not path.endswith(".csv"):
                    continue
                m = ts_pat.search(path)
                if not m:
                    continue
                ts = m.group(0)
                if start + "00" <= ts <= end + "00":
                    f.write(path + "\n")

        run_cmd([
            "rsync",
            "-av",
            f"--files-from={tmp_list}",
            f"{host}:{remote_root_resolved}/",
            f"{dest}/",
        ])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    for dev in devices:
        host = f"rasheed@192.168.0.{dev}"
        dest = f"{dest_base}_{dev}"
        tmp_list = Path(f"/tmp/rsync_files_{dev}.txt")

        py = r"""
import os, re
start = os.environ.get("START")
end = os.environ.get("END")
root = os.environ.get("ROOT")
pat = re.compile(r"[0-9]{8}_[0-9]{6}")

for dirpath, _, filenames in os.walk(root):
    for name in filenames:
        if not name.endswith(".csv"):
            continue
        m = pat.search(name)
        if not m:
            continue
        ts = m.group(0)
        if start + "00" <= ts <= end + "00":
            print(os.path.join(dirpath, name))
"""

        ssh_cmd = [
            "ssh",
            host,
            "python3",
            "-c",
            py,
        ]

        env = os.environ.copy()
        env.update({"START": start, "END": end, "ROOT": remote_root})

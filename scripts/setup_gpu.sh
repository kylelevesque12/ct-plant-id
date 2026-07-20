#!/bin/bash
# GPU-box setup. Checks headroom FIRST and refuses to start if there isn't
# room — the download filled a disk at 100% because nothing checked up front.
# Installs ONLY what training needs (no download/API deps).
#
# Usage:  bash scripts/setup_gpu.sh [/workspace/data]
set -e
DATA_DIR="${1:-/workspace/data}"
NEED_GB=80          # 67 GB dataset + checkpoints + headroom

echo "=== 1. disk check (before anything else) ==="
df -h "$(dirname "$DATA_DIR")" || df -h .
AVAIL=$(df -BG --output=avail "$(dirname "$DATA_DIR")" 2>/dev/null | tail -1 | tr -dc '0-9')
echo "available: ${AVAIL}G  (want >= ${NEED_GB}G)"
if [ -z "$AVAIL" ] || [ "$AVAIL" -lt "$NEED_GB" ]; then
    echo "STOP: not enough disk. Resize the volume before training."
    exit 1
fi

echo ""
echo "=== 2. installing only training deps ==="
pip install -q --upgrade pip
pip install -q torch torchvision timm pillow

echo ""
echo "=== 3. GPU check ==="
python3 - <<'PY'
import torch
ok = torch.cuda.is_available()
print("cuda available:", ok)
if ok:
    print("gpu:", torch.cuda.get_device_name(0))
    print("vram:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
else:
    raise SystemExit("STOP: no GPU visible — you are paying for a GPU pod without a GPU.")
PY

echo ""
echo "=== 4. dataset check ==="
python3 - "$DATA_DIR" <<'PY'
import csv, sys, os
d = sys.argv[1]
m = os.path.join(d, "manifest.csv")
if not os.path.exists(m):
    raise SystemExit(f"STOP: no manifest at {m} — is the volume attached?")
rows = list(csv.DictReader(open(m)))
from collections import Counter
s = Counter(r["split"] for r in rows)
print(f"images {len(rows):,} | species {len({r['species'] for r in rows}):,} | "
      f"train {s['train']:,} val {s['val']:,} test {s['test']:,}")
PY

echo ""
echo "setup OK — next: benchmark before committing to a full run:"
echo "  python3 scripts/train.py --data $DATA_DIR --benchmark 30 --img-size 224"

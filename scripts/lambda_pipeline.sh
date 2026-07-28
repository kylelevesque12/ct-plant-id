#!/bin/bash
# Unattended round-2 pipeline for the Lambda GPU box.
#
# Assumes: repo at ~/ct-plant-id, data already extracted to ~/round2,
# and runs/b_stage2/model.pt present for the warm start.
#
# Run:  cd ~/ct-plant-id && nohup bash scripts/lambda_pipeline.sh > ~/pipeline.log 2>&1 &
set -euo pipefail

REPO="$HOME/ct-plant-id"
DATA="$HOME/round2"
OUT="$REPO/runs/r2"
PY="${PY:-python3}"

cd "$REPO"
echo "=== $(date) round-2 pipeline starting ==="

# --- 0. sanity ---------------------------------------------------------------
[ -f "$DATA/manifest.csv" ] || { echo "STOP: no manifest at $DATA"; exit 1; }
[ -f "$REPO/runs/b_stage2/model.pt" ] || { echo "STOP: no warm-start checkpoint"; exit 1; }

echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "STOP: no GPU"; exit 1; }

# --- 1. fix manifest paths ---------------------------------------------------
# The manifest was written on the build droplet with absolute /root paths.
# Without this rewrite every image lookup fails.
if grep -q '/root/round2' "$DATA/manifest.csv"; then
    echo "--- rewriting manifest paths /root/round2 -> $DATA ---"
    cp "$DATA/manifest.csv" "$DATA/manifest.csv.droplet.bak"
    sed -i "s|/root/round2|$DATA|g" "$DATA/manifest.csv"
fi
BAD=$(grep -c '/root/round2' "$DATA/manifest.csv" || true)
[ "$BAD" = "0" ] || { echo "STOP: $BAD rows still point at /root"; exit 1; }

ROWS=$(( $(wc -l < "$DATA/manifest.csv") - 1 ))
echo "manifest: $ROWS images"

# Spot-check that files actually resolve before spending GPU hours on it.
MISSING=$(awk -F, 'NR>1 && NR<=200 {print $4}' "$DATA/manifest.csv" \
          | while read -r p; do [ -f "$p" ] || echo x; done | wc -l)
[ "$MISSING" = "0" ] || { echo "STOP: $MISSING of first 200 image paths missing"; exit 1; }
echo "path spot-check: OK"

# --- 2. deps -----------------------------------------------------------------
echo "--- installing deps ---"
$PY -m pip install -q timm requests numpy pillow 2>&1 | tail -2 || true

# --- 3. train ----------------------------------------------------------------
# Single 384px stage, warm-started from the deployed model. --init-from is
# shape-robust: the backbone transfers and the classifier head is reinitialised
# for the new class count (2,510 -> ~2,808).
echo "=== $(date) training ==="
$PY scripts/train.py \
    --data "$DATA" \
    --img-size 384 \
    --epochs 10 \
    --patience 3 \
    --batch 48 \
    --workers 8 \
    --init-from "$REPO/runs/b_stage2/model.pt" \
    --out "$OUT"

[ -f "$OUT/model.pt" ] || { echo "STOP: training produced no checkpoint"; exit 1; }
echo "=== $(date) training done ==="

# --- 4. calibration ----------------------------------------------------------
# Temperature is model-specific and invalid after any retrain.
echo "=== $(date) fitting temperature ==="
$PY scripts/fit_temperature.py --model "$OUT/model.pt" || echo "WARN: temperature fit failed (network?)"

# --- 5. OOD bank -------------------------------------------------------------
# Also model-specific. Needs the training images, which is why it runs here.
echo "=== $(date) building OOD bank ==="
$PY scripts/build_ood_bank.py --model "$OUT/model.pt" --data "$DATA" \
    || echo "WARN: OOD bank failed"

# --- 6. the "after" measurements ---------------------------------------------
# These MUST run here: they need the training images, which live only on this
# box. Once it is destroyed, re-running them means re-downloading 882k files.
# Baselines to beat (reports/interpretability.md, reports/stratified_eval.json):
#   trees 66.5% top-1 | position-2 accuracy 25.3% | tree attention entropy ~0.85
#   Quercus: no species structure in the embedding map
echo "=== $(date) post-train evaluation ==="
$PY -m pip install -q matplotlib umap-learn scikit-learn 2>&1 | tail -1 || true

$PY scripts/eval_stratified.py --model "$OUT/model.pt" --data "$DATA" \
    --per-class 6 --max-classes 500 \
    --out "$REPO/reports/stratified_eval_r2.json" || echo "WARN: stratified eval failed"

$PY scripts/tree_attention.py --model "$OUT/model.pt" --data "$DATA" \
    --max-species 25 --out "$REPO/reports/figures_r2" || echo "WARN: tree attention failed"

$PY scripts/embedding_map.py --model "$OUT/model.pt" --data "$DATA" \
    --genera Quercus,Acer,Pinus,Hydrangea,Toxicodendron --per-species 15 \
    --out "$REPO/reports/figures_r2" || echo "WARN: embedding map failed"

# --- 7. summary --------------------------------------------------------------
echo ""
echo "=== $(date) PIPELINE COMPLETE ==="
ls -lh "$OUT"
echo ""
echo "Pull back to the Mac:"
echo "  rsync -avz ubuntu@THIS_BOX:~/ct-plant-id/runs/r2/{model.pt,temperature.json,ood_bank.npz} ./runs/r2/"
echo "  rsync -avz ubuntu@THIS_BOX:~/ct-plant-id/reports/ ./reports/"
echo ""
echo "model.pt + temperature.json + ood_bank.npz are a MATCHED SET — the"
echo "temperature and OOD bank are meaningless with any other checkpoint."

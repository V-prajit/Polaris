#!/usr/bin/env bash
set -euo pipefail
VULTR_IP="${1:-144.202.53.143}"
echo "=== ParkSight H200 Pre-computation ==="
echo "Target Vultr: $VULTR_IP"

cd "$(dirname "$0")/.."
pip install -r requirements.txt

# Verify GPU
python -c "import torch; assert torch.cuda.is_available(), 'No GPU!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Step 1: Download tiles
echo "=== Downloading satellite tiles ==="
python scripts/download_map_tiles.py

# Step 2: Start background sync (every 5 min)
mkdir -p public/precomputed
while true; do
    rsync -avz --update public/precomputed/ root@${VULTR_IP}:/opt/parksight/public/precomputed/ 2>/dev/null || true
    sleep 300
done &
SYNC_PID=$!

# Step 3: Pre-compute (GPU)
echo "=== Pre-computing estimates ==="
python scripts/precompute_atlanta.py

# Final sync
kill $SYNC_PID 2>/dev/null || true
rsync -avz --progress public/precomputed/ root@${VULTR_IP}:/opt/parksight/public/precomputed/
echo "=== Done! ==="

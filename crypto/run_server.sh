#!/usr/bin/env bash
# Start the StratoCrypto bot fleet on a Linux server (Hostinger VPS).
# Usage:
#   chmod +x run_server.sh
#   ./run_server.sh
# Or run it under systemd (see stratocrypto-bots.service).
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
# Heavy zero-shot models are a liability on small VPS instances.
export FOUNDATION_MODELS="${FOUNDATION_MODELS:-0}"
# Keep BLAS from spawning dozens of threads on a 1–2 vCPU box.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

PYTHON="${PYTHON:-python3}"

if [ ! -f ".env" ]; then
  echo "ERROR: crypto/.env not found. Copy server.env.example to .env and fill it." >&2
  exit 1
fi

mkdir -p logs
echo "[run_server] python=$($PYTHON -V 2>&1)  foundation=$FOUNDATION_MODELS"
exec "$PYTHON" launch_bots.py

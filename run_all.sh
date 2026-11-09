#!/usr/bin/env bash
# Run the full invariant-taxonomy pipeline on a fresh GPU instance (RunPod/Vast.ai).
#
# Usage:
#   export HF_TOKEN=hf_...        # needed for gated models (Llama/Gemma)
#   bash run_all.sh              # runs default model list
#   bash run_all.sh "Qwen/Qwen2.5-1.5B mistralai/Mistral-7B-v0.1"   # custom list
#
# Env overrides: DEVICE=cuda|cpu   DTYPE=bf16|fp16|fp32
set -euo pipefail

MODELS="${1:-HuggingFaceTB/SmolLM2-1.7B mistralai/Mistral-7B-v0.1}"

echo "=== environment ==="
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

for M in $MODELS; do
  echo ""
  echo "=================================================================="
  echo "  RUNNING: $M"
  echo "=================================================================="
  uv run experiments/invariant_taxonomy_pilot.py "$M" || { echo "  [FAILED] $M — continuing"; continue; }
done

echo ""
echo "=== YaRN head-selection with controls (top/bottom/random 50% by rho) ==="
for M in $MODELS; do
  uv run experiments/yarn_hybrid.py "$M" || echo "  [YaRN FAILED] $M — continuing"
done

echo ""
echo "=== figures (rho vs induction) ==="
for J in results/invariant_taxonomy_pilot_*.json; do
  uv run experiments/plot_taxonomy.py "$J" || true
done

echo ""
echo "Done. Copy back: results/*.json and results/figures/*"

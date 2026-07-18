#!/usr/bin/env bash
# Serve the compressed medical model with vLLM's OpenAI-compatible server.
#
# Serves the AWQ W4A16 export (artifacts/awq-w4a16-stage3) — the vLLM-native
# format from run_awq_quantization_step.py. Requires a venv with vllm on PATH.
#
#   PORT=8801 ./serve_vllm.sh
set -euo pipefail
cd "$(dirname "$0")"
exec vllm serve artifacts/awq-w4a16-stage3 \
  --served-model-name medical-7b-compressed \
  --port "${PORT:-8801}" \
  --max-model-len 4096

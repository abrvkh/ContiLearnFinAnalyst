#!/usr/bin/env bash
# Usage: bash scripts/run_baseline_strategy.sh [additional baseline.py args]
# Example HF run: bash scripts/run_baseline_strategy.sh --provider hf --model Qwen/Qwen3-8B
# Example vLLM run: bash scripts/run_baseline_strategy.sh --provider vllm --model Qwen/Qwen3-8B
# Example resume run: bash scripts/run_baseline_strategy.sh --resume --save-every 100 --use-past-context
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

cd "${ROOT_DIR}"

uv run python src/strategies/baseline.py \
  --run-name baseline_strategy \
  "$@"

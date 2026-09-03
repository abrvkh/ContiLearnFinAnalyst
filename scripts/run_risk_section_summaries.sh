#!/usr/bin/env bash
# Usage: bash scripts/run_risk_section_summaries.sh [additional summarize_risk_sections.py args]
# Example resume run: bash scripts/run_risk_section_summaries.sh --resume --save-every 25
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

cd "${ROOT_DIR}"

uv run python src/data/summarize_risk_sections.py \
  --provider hf \
  --model Qwen/Qwen3-8B \
  "$@"

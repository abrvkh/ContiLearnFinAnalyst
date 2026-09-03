#!/usr/bin/env bash
# Usage: bash scripts/run_retrieval_agent.sh [additional run_agent.py args]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT_DIR}"

uv run python src/agent/run_agent.py "$@"

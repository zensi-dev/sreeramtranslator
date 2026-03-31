#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MOSES_BIN_DEFAULT="$REPO_ROOT/tools/mosesdecoder/bin/moses"
MODELS_ROOT_DEFAULT="$REPO_ROOT/models"

export MOSES_BIN="${MOSES_BIN:-$MOSES_BIN_DEFAULT}"
export PBSMT_MODELS_ROOT="${PBSMT_MODELS_ROOT:-$MODELS_ROOT_DEFAULT}"

if [ ! -x "$MOSES_BIN" ]; then
  printf 'Missing Moses binary: %s\n' "$MOSES_BIN" >&2
  printf 'Run scripts/setup_local_streamlit_tools.sh first.\n' >&2
  exit 1
fi

if [ ! -d "$PBSMT_MODELS_ROOT" ]; then
  printf 'Missing trained models directory: %s\n' "$PBSMT_MODELS_ROOT" >&2
  exit 1
fi

python3 -m streamlit run "$REPO_ROOT/app.py"

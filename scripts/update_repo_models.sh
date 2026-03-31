#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPORTS_DIR="${REPORTS_DIR:-$REPO_ROOT/reports}"
EXPORTS_DIR="$REPORTS_DIR"
TRAINED_MODELS_DIR="$REPORTS_DIR/models"
TARGET_MODELS_DIR="$REPO_ROOT/models"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

required_files=(
  "moses.ini"
  "phrase-table.gz"
  "reordering-table.wbe-msd-bidirectional-fe.gz"
  "lm.binary"
)

copy_model_files() {
  local source_dir="$1"
  local target_dir="$2"

  mkdir -p "$target_dir"
  rm -f "$target_dir"/*

  for file_name in "${required_files[@]}"; do
    if [ ! -f "$source_dir/$file_name" ]; then
      printf 'Missing required model file: %s\n' "$source_dir/$file_name" >&2
      exit 1
    fi
    cp "$source_dir/$file_name" "$target_dir/"
  done
}

load_from_archives() {
  local archive_en_de="$EXPORTS_DIR/model_en_de.tar.gz"
  local archive_de_en="$EXPORTS_DIR/model_de_en.tar.gz"

  if [ ! -f "$archive_en_de" ] || [ ! -f "$archive_de_en" ]; then
    return 1
  fi

  mkdir -p "$TMP_DIR/extracted"
  tar -xzf "$archive_en_de" -C "$TMP_DIR/extracted"
  tar -xzf "$archive_de_en" -C "$TMP_DIR/extracted"

  for direction in en-de de-en; do
    for file_name in "${required_files[@]}"; do
      if [ ! -f "$TMP_DIR/extracted/$direction/model/$file_name" ]; then
        printf 'Archive is missing required file, falling back to training directory: %s\n' "$TMP_DIR/extracted/$direction/model/$file_name" >&2
        return 1
      fi
    done
  done

  copy_model_files "$TMP_DIR/extracted/en-de/model" "$TARGET_MODELS_DIR/en-de/model"
  copy_model_files "$TMP_DIR/extracted/de-en/model" "$TARGET_MODELS_DIR/de-en/model"
}

load_from_training_dir() {
  if [ ! -d "$TRAINED_MODELS_DIR/en-de/model" ] || [ ! -d "$TRAINED_MODELS_DIR/de-en/model" ]; then
    printf 'Could not find trained models under %s\n' "$TRAINED_MODELS_DIR" >&2
    exit 1
  fi

  copy_model_files "$TRAINED_MODELS_DIR/en-de/model" "$TARGET_MODELS_DIR/en-de/model"
  copy_model_files "$TRAINED_MODELS_DIR/de-en/model" "$TARGET_MODELS_DIR/de-en/model"
}

if ! load_from_archives; then
  load_from_training_dir
fi

printf 'Updated repo inference models under %s\n' "$TARGET_MODELS_DIR"
printf 'English -> German model: %s\n' "$TARGET_MODELS_DIR/en-de/model"
printf 'German -> English model: %s\n' "$TARGET_MODELS_DIR/de-en/model"

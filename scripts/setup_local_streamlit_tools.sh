#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS_DIR="$REPO_ROOT/tools"

mkdir -p "$TOOLS_DIR"

if ! command -v git >/dev/null 2>&1; then
  printf 'Missing required command: git\n' >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  printf 'Missing required command: cmake\n' >&2
  exit 1
fi

if ! command -v make >/dev/null 2>&1; then
  printf 'Missing required command: make\n' >&2
  exit 1
fi

if ! command -v g++ >/dev/null 2>&1; then
  printf 'Missing required command: g++\n' >&2
  exit 1
fi

printf 'Building local SMT toolchain under %s\n' "$TOOLS_DIR"

if [ ! -d "$TOOLS_DIR/mosesdecoder" ]; then
  git clone --depth 1 https://github.com/moses-smt/mosesdecoder.git "$TOOLS_DIR/mosesdecoder"
fi

if [ ! -d "$TOOLS_DIR/kenlm" ]; then
  git clone --depth 1 https://github.com/kpu/kenlm.git "$TOOLS_DIR/kenlm"
fi

if [ ! -d "$TOOLS_DIR/fast_align" ]; then
  git clone --depth 1 https://github.com/clab/fast_align.git "$TOOLS_DIR/fast_align"
fi

if [ ! -d "$TOOLS_DIR/giza-pp" ]; then
  git clone --depth 1 https://github.com/moses-smt/giza-pp.git "$TOOLS_DIR/giza-pp"
fi

cd "$TOOLS_DIR/mosesdecoder"
git describe --dirty >/dev/null 2>&1 || git tag -f local-build >/dev/null 2>&1 || true
./bjam -j"$(nproc)" --without-tcmalloc --no-xmlrpc-c

mkdir -p "$TOOLS_DIR/kenlm/build"
cd "$TOOLS_DIR/kenlm/build"
cmake -Wno-dev .. >/dev/null
make -j"$(nproc)" >/dev/null

mkdir -p "$TOOLS_DIR/fast_align/build"
cd "$TOOLS_DIR/fast_align/build"
cmake -Wno-dev .. >/dev/null
make -j"$(nproc)" >/dev/null

make -C "$TOOLS_DIR/giza-pp/GIZA++-v2" -j"$(nproc)"
make -C "$TOOLS_DIR/giza-pp/mkcls-v2" -j"$(nproc)"

mkdir -p "$TOOLS_DIR/bin"
cp "$TOOLS_DIR/giza-pp/mkcls-v2/mkcls" "$TOOLS_DIR/bin/"
cp "$TOOLS_DIR/giza-pp/GIZA++-v2/GIZA++" "$TOOLS_DIR/bin/"
cp "$TOOLS_DIR/giza-pp/GIZA++-v2/snt2cooc.out" "$TOOLS_DIR/bin/"
cp "$TOOLS_DIR/fast_align/build/fast_align" "$TOOLS_DIR/bin/"
cp "$TOOLS_DIR/fast_align/build/atools" "$TOOLS_DIR/bin/"

printf '\nLocal SMT toolchain ready.\n'
printf 'Moses binary: %s\n' "$TOOLS_DIR/mosesdecoder/bin/moses"

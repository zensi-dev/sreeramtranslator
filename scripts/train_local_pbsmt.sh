#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="${BASE_DIR:-/workspace/local_runs/pbsmt_demo}"
TOOLS_DIR="${PBSMT_TOOLS:-/opt/pbsmt/tools}"

NEWS_PAIRS="${NEWS_PAIRS:-25000}"
EUROPARL_PAIRS="${EUROPARL_PAIRS:-15000}"
TRAIN_PAIRS="${TRAIN_PAIRS:-36000}"
VALID_PAIRS="${VALID_PAIRS:-2000}"
TEST_PAIRS="${TEST_PAIRS:-2000}"
MAX_SENTENCE_LENGTH="${MAX_SENTENCE_LENGTH:-50}"
MAX_PHRASE_LENGTH="${MAX_PHRASE_LENGTH:-5}"
THREADS="${THREADS:-4}"
LM_ORDER="${LM_ORDER:-5}"

TOTAL_PAIRS="$((NEWS_PAIRS + EUROPARL_PAIRS))"
SPLIT_TOTAL="$((TRAIN_PAIRS + VALID_PAIRS + TEST_PAIRS))"

if [ "$TOTAL_PAIRS" -ne "$SPLIT_TOTAL" ]; then
  printf 'Configured pair counts do not match splits: total=%s split_total=%s\n' "$TOTAL_PAIRS" "$SPLIT_TOTAL" >&2
  exit 1
fi

MOSES="$TOOLS_DIR/mosesdecoder"
KENLM_BIN="$TOOLS_DIR/kenlm/build/bin"
FAST_ALIGN="$TOOLS_DIR/bin/fast_align"

RAW="$BASE_DIR/data/raw"
PROC="$BASE_DIR/data/processed"
WORK="$BASE_DIR/data/work"
MODELS="$BASE_DIR/models"
EXPORTS="$BASE_DIR/exports"

for required in \
  "$MOSES/bin/moses" \
  "$MOSES/bin/symal" \
  "$MOSES/scripts/tokenizer/tokenizer.perl" \
  "$MOSES/scripts/recaser/train-truecaser.perl" \
  "$MOSES/scripts/recaser/truecase.perl" \
  "$MOSES/scripts/training/train-model.perl" \
  "$KENLM_BIN/lmplz" \
  "$KENLM_BIN/build_binary" \
  "$FAST_ALIGN"; do
  if [ ! -e "$required" ]; then
    printf 'Missing required tool: %s\n' "$required" >&2
    exit 1
  fi
done

mkdir -p "$RAW"
rm -rf "$PROC" "$WORK" "$MODELS" "$EXPORTS"
mkdir -p "$PROC" "$WORK" "$MODELS" "$EXPORTS"

if [ ! -f "$RAW/training-parallel-nc-v9.tgz" ]; then
  wget -q -O "$RAW/training-parallel-nc-v9.tgz" https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz
fi

if [ ! -f "$RAW/de-en.tgz" ]; then
  wget -q -O "$RAW/de-en.tgz" https://www.statmt.org/europarl/v7/de-en.tgz
fi

if [ ! -f "$RAW/nc/training/news-commentary-v9.de-en.en" ]; then
  mkdir -p "$RAW/nc"
  tar -xzf "$RAW/training-parallel-nc-v9.tgz" -C "$RAW/nc"
fi

if [ ! -f "$RAW/europarl/europarl-v7.de-en.en" ]; then
  mkdir -p "$RAW/europarl"
  tar -xzf "$RAW/de-en.tgz" -C "$RAW/europarl"
fi

BASE_DIR="$BASE_DIR" \
NEWS_PAIRS="$NEWS_PAIRS" \
EUROPARL_PAIRS="$EUROPARL_PAIRS" \
TRAIN_PAIRS="$TRAIN_PAIRS" \
VALID_PAIRS="$VALID_PAIRS" \
TEST_PAIRS="$TEST_PAIRS" \
MAX_SENTENCE_LENGTH="$MAX_SENTENCE_LENGTH" \
python3 - <<'PY'
import os
import re
from pathlib import Path

base_dir = Path(os.environ["BASE_DIR"])
raw = base_dir / "data" / "raw"
proc = base_dir / "data" / "processed"
proc.mkdir(parents=True, exist_ok=True)

news_pairs_limit = int(os.environ["NEWS_PAIRS"])
euro_pairs_limit = int(os.environ["EUROPARL_PAIRS"])
train_pairs_limit = int(os.environ["TRAIN_PAIRS"])
valid_pairs_limit = int(os.environ["VALID_PAIRS"])
test_pairs_limit = int(os.environ["TEST_PAIRS"])
max_sentence_length = int(os.environ["MAX_SENTENCE_LENGTH"])

news_en = raw / "nc" / "training" / "news-commentary-v9.de-en.en"
news_de = raw / "nc" / "training" / "news-commentary-v9.de-en.de"
euro_en = raw / "europarl" / "europarl-v7.de-en.en"
euro_de = raw / "europarl" / "europarl-v7.de-en.de"

def clean_line(text: str) -> str:
    text = text.strip()
    return re.sub(r"\s+", " ", text)

def good_pair(src: str, tgt: str) -> bool:
    if not src or not tgt:
        return False
    if src.startswith("<") or tgt.startswith("<"):
        return False
    src_len = len(src.split())
    tgt_len = len(tgt.split())
    if src_len == 0 or tgt_len == 0:
        return False
    if src_len > max_sentence_length or tgt_len > max_sentence_length:
        return False
    ratio = max(src_len / tgt_len, tgt_len / src_len)
    return ratio <= 2.5

def collect_pairs(src_path: Path, tgt_path: Path, limit: int):
    pairs = []
    seen = set()
    with src_path.open("r", encoding="utf-8", errors="ignore") as src_f, tgt_path.open("r", encoding="utf-8", errors="ignore") as tgt_f:
        for src, tgt in zip(src_f, tgt_f):
            src = clean_line(src)
            tgt = clean_line(tgt)
            if not good_pair(src, tgt):
                continue
            key = (src, tgt)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
            if len(pairs) >= limit:
                break
    if len(pairs) < limit:
        raise ValueError(f"Only found {len(pairs)} pairs in {src_path.name}, expected {limit}")
    return pairs

news_pairs = collect_pairs(news_en, news_de, news_pairs_limit)
euro_pairs = collect_pairs(euro_en, euro_de, euro_pairs_limit)
all_pairs = news_pairs + euro_pairs

expected_total = train_pairs_limit + valid_pairs_limit + test_pairs_limit
if len(all_pairs) != expected_total:
    raise ValueError(f"Subset total {len(all_pairs)} does not match expected total {expected_total}")

train_pairs = all_pairs[:train_pairs_limit]
valid_pairs = all_pairs[train_pairs_limit:train_pairs_limit + valid_pairs_limit]
test_pairs = all_pairs[train_pairs_limit + valid_pairs_limit:]

def write_split(name: str, pairs):
    with (proc / f"{name}.en").open("w", encoding="utf-8") as enf, (proc / f"{name}.de").open("w", encoding="utf-8") as def_:
        for en, de in pairs:
            enf.write(en + "\n")
            def_.write(de + "\n")

write_split("train", train_pairs)
write_split("valid", valid_pairs)
write_split("test", test_pairs)

print({"train": len(train_pairs), "valid": len(valid_pairs), "test": len(test_pairs)})
PY

mkdir -p "$WORK/tok" "$WORK/truecase"

for split in train valid test; do
  perl "$MOSES/scripts/tokenizer/normalize-punctuation.perl" -l en < "$PROC/$split.en" | perl "$MOSES/scripts/tokenizer/tokenizer.perl" -threads "$THREADS" -l en > "$WORK/tok/$split.tok.en"
  perl "$MOSES/scripts/tokenizer/normalize-punctuation.perl" -l de < "$PROC/$split.de" | perl "$MOSES/scripts/tokenizer/tokenizer.perl" -threads "$THREADS" -l de > "$WORK/tok/$split.tok.de"
done

perl "$MOSES/scripts/recaser/train-truecaser.perl" --model "$WORK/truecase/truecase-model.en" --corpus "$WORK/tok/train.tok.en"
perl "$MOSES/scripts/recaser/train-truecaser.perl" --model "$WORK/truecase/truecase-model.de" --corpus "$WORK/tok/train.tok.de"

for split in train valid test; do
  perl "$MOSES/scripts/recaser/truecase.perl" --model "$WORK/truecase/truecase-model.en" < "$WORK/tok/$split.tok.en" > "$WORK/$split.tc.en"
  perl "$MOSES/scripts/recaser/truecase.perl" --model "$WORK/truecase/truecase-model.de" < "$WORK/tok/$split.tok.de" > "$WORK/$split.tc.de"
done

perl "$MOSES/scripts/training/clean-corpus-n.perl" "$WORK/train.tc" en de "$WORK/train.clean" 1 "$MAX_SENTENCE_LENGTH"

mkdir -p "$MODELS/lm" "$MODELS/en-de/model" "$MODELS/de-en/model" "$EXPORTS/test_outputs"

"$KENLM_BIN/lmplz" -o "$LM_ORDER" -S 50% --discount_fallback < "$WORK/train.clean.de" > "$MODELS/lm/de.arpa"
"$KENLM_BIN/build_binary" trie "$MODELS/lm/de.arpa" "$MODELS/lm/de.binary"
"$KENLM_BIN/lmplz" -o "$LM_ORDER" -S 50% --discount_fallback < "$WORK/train.clean.en" > "$MODELS/lm/en.arpa"
"$KENLM_BIN/build_binary" trie "$MODELS/lm/en.arpa" "$MODELS/lm/en.binary"

ALIGN_HEURISTIC="grow-diag-final-and"
EXTBIN="$TOOLS_DIR/bin"

perl "$MOSES/scripts/ems/support/prepare-fast-align.perl" "$WORK/train.clean.en" "$WORK/train.clean.de" > "$WORK/train.en-de.fa"
"$FAST_ALIGN" -i "$WORK/train.en-de.fa" -d -o -v > "$WORK/train.en-de.forward"
"$FAST_ALIGN" -r -i "$WORK/train.en-de.fa" -d -o -v > "$WORK/train.en-de.reverse"
perl "$MOSES/scripts/ems/support/symmetrize-fast-align.perl" "$WORK/train.en-de.forward" "$WORK/train.en-de.reverse" "$WORK/train.clean.en" "$WORK/train.clean.de" "$MODELS/aligned_en_de" "$ALIGN_HEURISTIC" "$MOSES/bin/symal"

perl "$MOSES/scripts/ems/support/prepare-fast-align.perl" "$WORK/train.clean.de" "$WORK/train.clean.en" > "$WORK/train.de-en.fa"
"$FAST_ALIGN" -i "$WORK/train.de-en.fa" -d -o -v > "$WORK/train.de-en.forward"
"$FAST_ALIGN" -r -i "$WORK/train.de-en.fa" -d -o -v > "$WORK/train.de-en.reverse"
perl "$MOSES/scripts/ems/support/symmetrize-fast-align.perl" "$WORK/train.de-en.forward" "$WORK/train.de-en.reverse" "$WORK/train.clean.de" "$WORK/train.clean.en" "$MODELS/aligned_de_en" "$ALIGN_HEURISTIC" "$MOSES/bin/symal"

perl "$MOSES/scripts/training/train-model.perl" \
  -root-dir "$MODELS/en-de" \
  -corpus "$WORK/train.clean" \
  -f en -e de \
  -alignment "$ALIGN_HEURISTIC" \
  -alignment-file "$MODELS/aligned_en_de" \
  -reordering msd-bidirectional-fe \
  -max-phrase-length "$MAX_PHRASE_LENGTH" \
  -lm "0:${LM_ORDER}:$MODELS/lm/de.binary:8" \
  -external-bin-dir "$EXTBIN" \
  -cores "$THREADS" \
  -parallel \
  -first-step 4 -last-step 9

perl "$MOSES/scripts/training/train-model.perl" \
  -root-dir "$MODELS/de-en" \
  -corpus "$WORK/train.clean" \
  -f de -e en \
  -alignment "$ALIGN_HEURISTIC" \
  -alignment-file "$MODELS/aligned_de_en" \
  -reordering msd-bidirectional-fe \
  -max-phrase-length "$MAX_PHRASE_LENGTH" \
  -lm "0:${LM_ORDER}:$MODELS/lm/en.binary:8" \
  -external-bin-dir "$EXTBIN" \
  -cores "$THREADS" \
  -parallel \
  -first-step 4 -last-step 9

MODELS="$MODELS" python3 - <<'PY'
import os
import shutil
from pathlib import Path

models = Path(os.environ["MODELS"])
mapping = {
    "en-de": models / "lm" / "de.binary",
    "de-en": models / "lm" / "en.binary",
}

for direction, lm_path in mapping.items():
    model_dir = models / direction / "model"
    local_lm = model_dir / "lm.binary"
    shutil.copy2(lm_path, local_lm)
    ini_path = model_dir / "moses.ini"
    text = ini_path.read_text(encoding="utf-8")
    text = text.replace(str(model_dir) + "/", "")
    text = text.replace(str(lm_path), "lm.binary")
    ini_path.write_text(text, encoding="utf-8")
    print(f"Updated {ini_path}")
PY

(cd "$MODELS/en-de/model" && "$MOSES/bin/moses" -f moses.ini < "$WORK/test.tc.en" > "$EXPORTS/test_outputs/pred.de")
(cd "$MODELS/de-en/model" && "$MOSES/bin/moses" -f moses.ini < "$WORK/test.tc.de" > "$EXPORTS/test_outputs/pred.en")

python3 -m sacrebleu "$WORK/test.tc.de" -i "$EXPORTS/test_outputs/pred.de" -m bleu -b -w 4 | tee "$EXPORTS/test_outputs/bleu_en_de.txt"
python3 -m sacrebleu "$WORK/test.tc.en" -i "$EXPORTS/test_outputs/pred.en" -m bleu -b -w 4 | tee "$EXPORTS/test_outputs/bleu_de_en.txt"

tar -czf "$EXPORTS/model_en_de.tar.gz" -C "$MODELS" en-de/model
tar -czf "$EXPORTS/model_de_en.tar.gz" -C "$MODELS" de-en/model

printf '\nLocal training finished.\n'
printf 'Model archive: %s\n' "$EXPORTS/model_en_de.tar.gz"
printf 'Model archive: %s\n' "$EXPORTS/model_de_en.tar.gz"
printf 'BLEU summary: %s\n' "$EXPORTS/test_outputs"

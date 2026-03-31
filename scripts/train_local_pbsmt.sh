#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
REPORTS_DIR="${REPORTS_DIR:-$REPO_ROOT/reports}"
TOOLS_DIR="${PBSMT_TOOLS:-/opt/pbsmt/tools}"

NEWS_PAIRS="${NEWS_PAIRS:-60000}"
EUROPARL_PAIRS="${EUROPARL_PAIRS:-40000}"
TATOEBA_PAIRS="${TATOEBA_PAIRS:-20000}"
TRAIN_PAIRS="${TRAIN_PAIRS:-116000}"
VALID_PAIRS="${VALID_PAIRS:-2000}"
TEST_PAIRS="${TEST_PAIRS:-2000}"
MAX_SENTENCE_LENGTH="${MAX_SENTENCE_LENGTH:-50}"
MAX_PHRASE_LENGTH="${MAX_PHRASE_LENGTH:-5}"
THREADS="${THREADS:-}"
LM_ORDER="${LM_ORDER:-5}"
TUNING_MAX_ITERATIONS="${TUNING_MAX_ITERATIONS:-2}"
TUNING_JOBS="${TUNING_JOBS:-}"
TUNING_DECODER_THREADS="${TUNING_DECODER_THREADS:-}"
TUNING_MERT_THREADS="${TUNING_MERT_THREADS:-}"
TUNING_MOSES_PARALLEL_CMD="${TUNING_MOSES_PARALLEL_CMD:-}"
EVAL_DECODER_THREADS="${EVAL_DECODER_THREADS:-}"

resolve_path() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
}

detect_cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi

  python3 - <<'PY'
import os

print(os.cpu_count() or 1)
PY
}

DATA_DIR="$(resolve_path "$DATA_DIR")"
REPORTS_DIR="$(resolve_path "$REPORTS_DIR")"
TOOLS_DIR="$(resolve_path "$TOOLS_DIR")"
CPU_COUNT="$(detect_cpu_count)"

if [ -z "$THREADS" ]; then
  THREADS="$CPU_COUNT"
fi

if [ -z "$TUNING_DECODER_THREADS" ]; then
  TUNING_DECODER_THREADS="1"
fi

if [ -z "$TUNING_MERT_THREADS" ]; then
  TUNING_MERT_THREADS="$CPU_COUNT"
fi

if [ -z "$EVAL_DECODER_THREADS" ]; then
  EVAL_DECODER_THREADS="$(( (THREADS + 1) / 2 ))"
fi

printf 'Using %s training threads, tuning max %s iterations, %s tuning jobs, %s tuning decoder threads, %s tuning optimizer threads, %s evaluation decoder threads on %s CPUs\n' \
  "$THREADS" "$TUNING_MAX_ITERATIONS" "${TUNING_JOBS:-serial}" "$TUNING_DECODER_THREADS" "$TUNING_MERT_THREADS" "$EVAL_DECODER_THREADS" "$CPU_COUNT"

TOTAL_PAIRS="$((NEWS_PAIRS + EUROPARL_PAIRS + TATOEBA_PAIRS))"
SPLIT_TOTAL="$((TRAIN_PAIRS + VALID_PAIRS + TEST_PAIRS))"

if [ "$TOTAL_PAIRS" -ne "$SPLIT_TOTAL" ]; then
  printf 'Configured pair counts do not match splits: total=%s split_total=%s\n' "$TOTAL_PAIRS" "$SPLIT_TOTAL" >&2
  exit 1
fi

MOSES="$TOOLS_DIR/mosesdecoder"
KENLM_BIN="$TOOLS_DIR/kenlm/build/bin"
FAST_ALIGN="$TOOLS_DIR/bin/fast_align"
TUNING_SCRIPT="$MOSES/scripts/training/mert-moses.pl"

RAW="$DATA_DIR/raw"
PROC="$DATA_DIR/processed"
WORK="$DATA_DIR/work"
MODELS="$REPORTS_DIR/models"
EXPORTS="$REPORTS_DIR"

prepare_tuning_ini() {
  local source_ini="$1"
  local model_dir="$2"
  local lm_path="$3"
  local output_ini="$4"

  SOURCE_INI="$source_ini" MODEL_DIR="$model_dir" LM_PATH="$lm_path" OUTPUT_INI="$output_ini" python3 - <<'PY'
import os
from pathlib import Path

source_ini = Path(os.environ["SOURCE_INI"]).resolve()
model_dir = Path(os.environ["MODEL_DIR"]).resolve()
lm_path = Path(os.environ["LM_PATH"]).resolve()
output_ini = Path(os.environ["OUTPUT_INI"]).resolve()

text = source_ini.read_text(encoding="utf-8")
text = text.replace("path=phrase-table.gz", f"path={model_dir / 'phrase-table.gz'}")
text = text.replace(
    "path=reordering-table.wbe-msd-bidirectional-fe.gz",
    f"path={model_dir / 'reordering-table.wbe-msd-bidirectional-fe.gz'}",
)
text = text.replace("path=lm.binary", f"path={lm_path}")

output_ini.write_text(text, encoding="utf-8")
PY
}

run_tuning() {
  local label="$1"
  local source_file="$2"
  local reference_file="$3"
  local model_dir="$4"
  local lm_path="$5"
  local tuning_dir="$6"
  local tuning_ini="$tuning_dir/input.moses.ini"
  local -a tuning_cmd

  rm -rf "$tuning_dir"
  mkdir -p "$tuning_dir"
  prepare_tuning_ini "$model_dir/moses.ini" "$model_dir" "$lm_path" "$tuning_ini"

  tuning_cmd=(
    perl "$TUNING_SCRIPT"
    "$source_file"
    "$reference_file"
    "$MOSES/bin/moses"
    "$tuning_ini"
    --working-dir "$tuning_dir"
    --maximum-iterations "$TUNING_MAX_ITERATIONS"
    --mertdir "$MOSES/bin"
    --threads "$TUNING_MERT_THREADS"
    --decoder-flags "-threads $TUNING_DECODER_THREADS -v 0"
  )

  # Stock mert-moses.pl only uses --jobs through qsub-backed helpers.
  # Require an explicit replacement wrapper before enabling split-input parallelism.
  if [ -n "$TUNING_JOBS" ]; then
    if [ -z "$TUNING_MOSES_PARALLEL_CMD" ]; then
      printf 'TUNING_JOBS requires TUNING_MOSES_PARALLEL_CMD because standard Moses moses-parallel.pl submits through qsub\n' >&2
      exit 1
    fi
    if [ ! -x "$TUNING_MOSES_PARALLEL_CMD" ]; then
      printf 'TUNING_MOSES_PARALLEL_CMD is not executable: %s\n' "$TUNING_MOSES_PARALLEL_CMD" >&2
      exit 1
    fi

    tuning_cmd+=(
      --jobs "$TUNING_JOBS"
      --mosesparallelcmd "$TUNING_MOSES_PARALLEL_CMD"
    )
  fi

  "${tuning_cmd[@]}"

  if [ ! -f "$tuning_dir/moses.ini" ]; then
    printf 'Tuning did not produce tuned config for %s: %s\n' "$label" "$tuning_dir/moses.ini" >&2
    exit 1
  fi
}

for required in \
  "$MOSES/bin/moses" \
  "$MOSES/bin/mert" \
  "$MOSES/bin/symal" \
  "$MOSES/scripts/tokenizer/tokenizer.perl" \
  "$MOSES/scripts/recaser/train-truecaser.perl" \
  "$MOSES/scripts/recaser/truecase.perl" \
  "$TUNING_SCRIPT" \
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

if [ ! -f "$RAW/deu-eng.zip" ]; then
  wget -q -O "$RAW/deu-eng.zip" https://www.manythings.org/anki/deu-eng.zip
fi

if [ ! -f "$RAW/nc/training/news-commentary-v9.de-en.en" ]; then
  mkdir -p "$RAW/nc"
  tar -xzf "$RAW/training-parallel-nc-v9.tgz" -C "$RAW/nc"
fi

if [ ! -f "$RAW/europarl/europarl-v7.de-en.en" ]; then
  mkdir -p "$RAW/europarl"
  tar -xzf "$RAW/de-en.tgz" -C "$RAW/europarl"
fi

DATA_DIR="$DATA_DIR" \
NEWS_PAIRS="$NEWS_PAIRS" \
EUROPARL_PAIRS="$EUROPARL_PAIRS" \
TATOEBA_PAIRS="$TATOEBA_PAIRS" \
TRAIN_PAIRS="$TRAIN_PAIRS" \
VALID_PAIRS="$VALID_PAIRS" \
TEST_PAIRS="$TEST_PAIRS" \
MAX_SENTENCE_LENGTH="$MAX_SENTENCE_LENGTH" \
python3 - <<'PY'
import os
import random
import re
import zipfile
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR"])
raw = data_dir / "raw"
proc = data_dir / "processed"
proc.mkdir(parents=True, exist_ok=True)

news_pairs_limit = int(os.environ["NEWS_PAIRS"])
euro_pairs_limit = int(os.environ["EUROPARL_PAIRS"])
tatoeba_pairs_limit = int(os.environ["TATOEBA_PAIRS"])
train_pairs_limit = int(os.environ["TRAIN_PAIRS"])
valid_pairs_limit = int(os.environ["VALID_PAIRS"])
test_pairs_limit = int(os.environ["TEST_PAIRS"])
max_sentence_length = int(os.environ["MAX_SENTENCE_LENGTH"])

news_en = raw / "nc" / "training" / "news-commentary-v9.de-en.en"
news_de = raw / "nc" / "training" / "news-commentary-v9.de-en.de"
euro_en = raw / "europarl" / "europarl-v7.de-en.en"
euro_de = raw / "europarl" / "europarl-v7.de-en.de"
tatoeba_zip = raw / "deu-eng.zip"
name_blacklist = {"tom", "mary", "john", "bill", "jim"}
conversational_markers = {
    "i", "you", "we", "he", "she", "they", "it", "this", "that", "these", "those", "please", "thanks", "thank", "sorry",
    "yes", "no", "ok", "okay", "hello", "hi", "bye", "how", "what", "where", "when", "why", "who", "which", "can",
    "could", "would", "do", "does", "did", "is", "are", "am", "was", "were", "have", "has", "will", "shall", "need",
    "want", "know", "think", "like", "love", "go", "come", "help", "tell", "let", "good", "morning", "afternoon",
    "evening", "night", "here", "there"
}
allowed_tatoeba_re = re.compile(r"^[A-Za-z0-9 ,.'?!-]+$")

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


def collect_tatoeba_pairs(zip_path: Path, limit: int):
    conversational_pairs = []
    fallback_pairs = []
    seen = set()
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("deu.txt") as raw_lines:
            for raw_line in raw_lines:
                line = raw_line.decode("utf-8", errors="ignore").rstrip("\n")
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                src = clean_line(parts[0])
                tgt = clean_line(parts[1])
                if not good_pair(src, tgt):
                    continue
                key = (src, tgt)
                if key in seen:
                    continue
                seen.add(key)
                tokens = re.findall(r"[a-z']+", src.lower())
                is_conversational = (
                    2 <= len(tokens) <= 10
                    and not any(name in tokens for name in name_blacklist)
                    and not any(ch.isdigit() for ch in src)
                    and allowed_tatoeba_re.match(src) is not None
                    and any(token in conversational_markers for token in tokens)
                )
                if is_conversational:
                    conversational_pairs.append(key)
                else:
                    fallback_pairs.append(key)

    pairs = conversational_pairs[:limit]
    if len(pairs) < limit:
        pairs.extend(fallback_pairs[: limit - len(pairs)])
    if len(pairs) < limit:
        raise ValueError(f"Only found {len(pairs)} usable Tatoeba pairs, expected {limit}")
    return pairs

news_pairs = collect_pairs(news_en, news_de, news_pairs_limit)
euro_pairs = collect_pairs(euro_en, euro_de, euro_pairs_limit)
tatoeba_pairs = collect_tatoeba_pairs(tatoeba_zip, tatoeba_pairs_limit)
all_pairs = news_pairs + euro_pairs + tatoeba_pairs
random.Random(42).shuffle(all_pairs)

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

run_tuning "English -> German" "$WORK/valid.tc.en" "$WORK/valid.tc.de" "$MODELS/en-de/model" "$MODELS/lm/de.binary" "$MODELS/en-de/tuning"

run_tuning "German -> English" "$WORK/valid.tc.de" "$WORK/valid.tc.en" "$MODELS/de-en/model" "$MODELS/lm/en.binary" "$MODELS/de-en/tuning"

MODELS="$MODELS" python3 - <<'PY'
import os
import shutil
from pathlib import Path

models = Path(os.environ["MODELS"])
mapping = {
    "en-de": {
        "lm": models / "lm" / "de.binary",
        "source_ini": models / "en-de" / "tuning" / "moses.ini",
    },
    "de-en": {
        "lm": models / "lm" / "en.binary",
        "source_ini": models / "de-en" / "tuning" / "moses.ini",
    },
}

for direction, config in mapping.items():
    model_dir = models / direction / "model"
    lm_path = config["lm"].resolve()
    source_ini = config["source_ini"].resolve()
    local_lm = model_dir / "lm.binary"
    if not source_ini.exists():
        raise FileNotFoundError(f"Missing tuned config for {direction}: {source_ini}")
    shutil.copy2(lm_path, local_lm)
    ini_path = model_dir / "moses.ini"
    text = source_ini.read_text(encoding="utf-8")
    text = text.replace(str(model_dir.resolve()) + "/", "")
    text = text.replace(str(lm_path), "lm.binary")
    ini_path.write_text(text, encoding="utf-8")
    print(f"Updated tuned config {ini_path}")
PY

(cd "$MODELS/en-de/model" && "$MOSES/bin/moses" -f moses.ini -threads "$EVAL_DECODER_THREADS" -v 0 < "$WORK/test.tc.en" > "$EXPORTS/test_outputs/pred.de") &
pid_en_de=$!
(cd "$MODELS/de-en/model" && "$MOSES/bin/moses" -f moses.ini -threads "$EVAL_DECODER_THREADS" -v 0 < "$WORK/test.tc.de" > "$EXPORTS/test_outputs/pred.en") &
pid_de_en=$!

wait "$pid_en_de"
wait "$pid_de_en"

perl "$MOSES/scripts/recaser/detruecase.perl" < "$EXPORTS/test_outputs/pred.de" | perl "$MOSES/scripts/tokenizer/detokenizer.perl" -q -l de > "$EXPORTS/test_outputs/pred.detok.de"
perl "$MOSES/scripts/recaser/detruecase.perl" < "$EXPORTS/test_outputs/pred.en" | perl "$MOSES/scripts/tokenizer/detokenizer.perl" -q -l en > "$EXPORTS/test_outputs/pred.detok.en"
perl "$MOSES/scripts/recaser/detruecase.perl" < "$WORK/test.tc.de" | perl "$MOSES/scripts/tokenizer/detokenizer.perl" -q -l de > "$EXPORTS/test_outputs/ref.detok.de"
perl "$MOSES/scripts/recaser/detruecase.perl" < "$WORK/test.tc.en" | perl "$MOSES/scripts/tokenizer/detokenizer.perl" -q -l en > "$EXPORTS/test_outputs/ref.detok.en"

python3 -m sacrebleu "$EXPORTS/test_outputs/ref.detok.de" -i "$EXPORTS/test_outputs/pred.detok.de" -m bleu -b -w 4 | tee "$EXPORTS/test_outputs/bleu_en_de.txt"
python3 -m sacrebleu "$EXPORTS/test_outputs/ref.detok.en" -i "$EXPORTS/test_outputs/pred.detok.en" -m bleu -b -w 4 | tee "$EXPORTS/test_outputs/bleu_de_en.txt"

tar -czf "$EXPORTS/model_en_de.tar.gz" -C "$MODELS" en-de/model
tar -czf "$EXPORTS/model_de_en.tar.gz" -C "$MODELS" de-en/model

printf '\nLocal training finished.\n'
printf 'Model archive: %s\n' "$EXPORTS/model_en_de.tar.gz"
printf 'Model archive: %s\n' "$EXPORTS/model_de_en.tar.gz"
printf 'BLEU summary: %s\n' "$EXPORTS/test_outputs"
printf 'Training workspace: %s\n' "$DATA_DIR"
printf 'Training reports: %s\n' "$REPORTS_DIR"

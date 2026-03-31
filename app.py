from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import streamlit as st


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_ROOT = APP_ROOT / "models"
DEFAULT_MOSES_BIN = APP_ROOT / "runtime" / "bin" / "moses"
MOSES_SCRIPTS_DIR = APP_ROOT / "runtime" / "moses-scripts"
TRUECASE_DIR = APP_ROOT / "runtime" / "truecase"
MODELS_ROOT = Path(os.environ.get("PBSMT_MODELS_ROOT", str(DEFAULT_MODELS_ROOT))).resolve()
MOSES_BIN = Path(os.environ.get("MOSES_BIN", str(DEFAULT_MOSES_BIN))).resolve()
PERL_BIN = shutil.which("perl") or "perl"
MAX_INPUT_TOKENS = int(os.environ.get("PBSMT_MAX_INPUT_TOKENS", "50"))
MAX_PHRASE_LENGTH = int(os.environ.get("PBSMT_MAX_PHRASE_LENGTH", "5"))
DECODER_TIMEOUT_SECONDS = int(os.environ.get("PBSMT_DECODER_TIMEOUT", "60"))
DECODER_THREADS = int(os.environ.get("PBSMT_DECODER_THREADS", "2"))

REQUIRED_MOSES_SCRIPTS = [
    MOSES_SCRIPTS_DIR / "tokenizer" / "normalize-punctuation.perl",
    MOSES_SCRIPTS_DIR / "tokenizer" / "tokenizer.perl",
    MOSES_SCRIPTS_DIR / "tokenizer" / "detokenizer.perl",
    MOSES_SCRIPTS_DIR / "recaser" / "truecase.perl",
    MOSES_SCRIPTS_DIR / "recaser" / "detruecase.perl",
]

MODEL_DIRS = {
    "English -> German": MODELS_ROOT / "en-de" / "model",
    "German -> English": MODELS_ROOT / "de-en" / "model",
}

SAMPLE_TEXT = {
    "English -> German": "The region needs security and stability.",
    "German -> English": "Die Abstimmung findet nach der Aussprache statt.",
}

LANGUAGE_CODES = {
    "English -> German": ("en", "de"),
    "German -> English": ("de", "en"),
}


def validate_environment() -> list[str]:
    issues: list[str] = []
    if not MOSES_BIN.exists():
        issues.append(f"Missing Moses binary: {MOSES_BIN}")
    elif not os.access(MOSES_BIN, os.X_OK):
        issues.append(f"Moses binary is not executable: {MOSES_BIN}")
    if not MOSES_SCRIPTS_DIR.exists():
        issues.append(f"Missing Moses scripts directory: {MOSES_SCRIPTS_DIR}")
    for script_path in REQUIRED_MOSES_SCRIPTS:
        if not script_path.exists():
            issues.append(f"Missing Moses script: {script_path}")
    if shutil.which("perl") is None:
        issues.append("Missing Perl executable on PATH")
    for lang in ["en", "de"]:
        truecase_model = TRUECASE_DIR / f"truecase-model.{lang}"
        if not truecase_model.exists():
            issues.append(f"Missing truecase model: {truecase_model}")

    for label, model_dir in MODEL_DIRS.items():
        ini_path = model_dir / "moses.ini"
        if not model_dir.exists():
            issues.append(f"Missing model directory for {label}: {model_dir}")
            continue
        if not ini_path.exists():
            issues.append(f"Missing moses.ini for {label}: {ini_path}")
            continue
        for required_name in ["phrase-table.gz", "reordering-table.wbe-msd-bidirectional-fe.gz", "lm.binary"]:
            if not (model_dir / required_name).exists():
                issues.append(f"Missing {required_name} for {label}: {model_dir / required_name}")
    return issues


def run_perl_script(script_path: Path, args: list[str], text: str) -> str:
    try:
        completed = subprocess.run(
            [PERL_BIN, str(script_path), *args],
            input=text,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to start {script_path.name}: {exc}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or f"Failed script: {script_path.name}"
        raise RuntimeError(stderr)
    return completed.stdout


def build_source_phrases(tokens: list[str], max_phrase_length: int) -> set[str]:
    phrases: set[str] = set()
    for start in range(len(tokens)):
        max_length = min(max_phrase_length, len(tokens) - start)
        for length in range(1, max_length + 1):
            phrases.add(" ".join(tokens[start : start + length]))
    return phrases


def parse_phrase_pair(line: str) -> tuple[str, str] | None:
    parts = line.rstrip("\n").split(" ||| ", 2)
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def write_filtered_phrase_tables(model_dir: Path, source_phrases: set[str], output_dir: Path) -> tuple[Path, Path]:
    phrase_table_path = output_dir / "phrase-table.gz"
    reordering_path = output_dir / "reordering-table.wbe-msd-bidirectional-fe.gz"
    selected_pairs: set[tuple[str, str]] = set()

    with gzip.open(model_dir / "phrase-table.gz", "rt", encoding="utf-8", errors="ignore") as source_table:
        with gzip.open(phrase_table_path, "wt", encoding="utf-8", compresslevel=1) as filtered_table:
            for line in source_table:
                phrase_pair = parse_phrase_pair(line)
                if phrase_pair is None or phrase_pair[0] not in source_phrases:
                    continue
                selected_pairs.add(phrase_pair)
                filtered_table.write(line)

    if not selected_pairs:
        raise RuntimeError("The sentence does not match any phrase-table entries for this demo model.")

    with gzip.open(model_dir / "reordering-table.wbe-msd-bidirectional-fe.gz", "rt", encoding="utf-8", errors="ignore") as source_table:
        with gzip.open(reordering_path, "wt", encoding="utf-8", compresslevel=1) as filtered_table:
            for line in source_table:
                phrase_pair = parse_phrase_pair(line)
                if phrase_pair is not None and phrase_pair in selected_pairs:
                    filtered_table.write(line)

    return phrase_table_path, reordering_path


@contextmanager
def build_filtered_model(model_dir: Path, preprocessed_text: str):
    tokens = preprocessed_text.split()
    if not tokens:
        raise RuntimeError("The sentence became empty after preprocessing.")
    if len(tokens) > MAX_INPUT_TOKENS:
        raise RuntimeError(
            f"Input is too long for this PBSMT demo after preprocessing ({len(tokens)} tokens, limit {MAX_INPUT_TOKENS})."
        )

    source_phrases = build_source_phrases(tokens, MAX_PHRASE_LENGTH)

    with tempfile.TemporaryDirectory(prefix="pbsmt-filter-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        phrase_table_path, reordering_path = write_filtered_phrase_tables(model_dir, source_phrases, temp_dir)

        filtered_ini = (model_dir / "moses.ini").read_text(encoding="utf-8")
        filtered_ini = filtered_ini.replace("path=phrase-table.gz", f"path={phrase_table_path.resolve()}")
        filtered_ini = filtered_ini.replace(
            "path=reordering-table.wbe-msd-bidirectional-fe.gz",
            f"path={reordering_path.resolve()}",
        )
        filtered_ini = filtered_ini.replace("path=lm.binary", f"path={(model_dir / 'lm.binary').resolve()}")

        ini_path = temp_dir / "moses.ini"
        ini_path.write_text(filtered_ini, encoding="utf-8")
        yield ini_path, temp_dir


def preprocess(text: str, source_lang: str) -> str:
    normalized = run_perl_script(
        MOSES_SCRIPTS_DIR / "tokenizer" / "normalize-punctuation.perl",
        ["-l", source_lang],
        text + "\n",
    )
    tokenized = run_perl_script(
        MOSES_SCRIPTS_DIR / "tokenizer" / "tokenizer.perl",
        ["-l", source_lang],
        normalized,
    )
    truecased = run_perl_script(
        MOSES_SCRIPTS_DIR / "recaser" / "truecase.perl",
        ["--model", str(TRUECASE_DIR / f"truecase-model.{source_lang}")],
        tokenized,
    )
    return truecased.strip()


def postprocess(text: str, target_lang: str) -> str:
    detruecased = run_perl_script(
        MOSES_SCRIPTS_DIR / "recaser" / "detruecase.perl",
        [],
        text.strip() + "\n",
    )
    detokenized = run_perl_script(
        MOSES_SCRIPTS_DIR / "tokenizer" / "detokenizer.perl",
        ["-q", "-l", target_lang],
        detruecased,
    )
    return detokenized.strip()


def decode(text: str, model_dir: Path, source_lang: str, target_lang: str) -> str:
    preprocessed = preprocess(text, source_lang)
    try:
        with build_filtered_model(model_dir, preprocessed) as (ini_path, temp_dir):
            completed = subprocess.run(
                [str(MOSES_BIN), "-f", str(ini_path), "-threads", str(DECODER_THREADS), "-v", "0"],
                input=preprocessed + "\n",
                text=True,
                capture_output=True,
                cwd=str(temp_dir),
                check=False,
                timeout=DECODER_TIMEOUT_SECONDS,
            )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Moses timed out after {DECODER_TIMEOUT_SECONDS} seconds.") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to start Moses: {exc}") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "Unknown Moses error"
        raise RuntimeError(stderr)
    return postprocess(completed.stdout, target_lang)


st.set_page_config(page_title="English-German PBSMT", page_icon="translate", layout="centered")

st.title("English-German PBSMT Demo")
st.caption("Strict phrase-based statistical machine translation using trained local Moses models.")

issues = validate_environment()
if issues:
    st.error("Local model environment is incomplete.")
    for issue in issues:
        st.write(f"- {issue}")
    st.stop()

direction = st.selectbox("Translation direction", list(MODEL_DIRS.keys()))
source_lang, target_lang = LANGUAGE_CODES[direction]

default_text = SAMPLE_TEXT[direction]
text = st.text_area("Input text", value=default_text, height=140)

translate_clicked = st.button("Translate", type="primary")

if translate_clicked:
    cleaned = text.strip()
    if not cleaned:
        st.warning("Enter some text to translate.")
    else:
        with st.spinner("Running Moses decoder..."):
            try:
                translation = decode(cleaned, MODEL_DIRS[direction], source_lang, target_lang)
            except Exception as exc:
                st.error("Translation failed.")
                st.code(str(exc))
            else:
                st.subheader("Translation")
                st.write(translation)

with st.expander("Model details"):
    st.write(f"Models root: `{MODELS_ROOT}`")
    st.write(f"Moses binary: `{MOSES_BIN}`")
    for label, model_dir in MODEL_DIRS.items():
        st.write(f"{label}: `{model_dir}`")

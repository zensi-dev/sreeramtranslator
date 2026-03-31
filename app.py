from __future__ import annotations

import os
import subprocess
from pathlib import Path

import streamlit as st


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_ROOT = APP_ROOT / "models"
DEFAULT_MOSES_BIN = APP_ROOT / "runtime" / "bin" / "moses"
MOSES_SCRIPTS_DIR = APP_ROOT / "runtime" / "moses-scripts"
TRUECASE_DIR = APP_ROOT / "runtime" / "truecase"
MODELS_ROOT = Path(os.environ.get("PBSMT_MODELS_ROOT", str(DEFAULT_MODELS_ROOT))).resolve()
MOSES_BIN = Path(os.environ.get("MOSES_BIN", str(DEFAULT_MOSES_BIN))).resolve()

MODEL_DIRS = {
    "English -> German": MODELS_ROOT / "en-de" / "model",
    "German -> English": MODELS_ROOT / "de-en" / "model",
}

SAMPLE_TEXT = {
    "English -> German": "The government announced a new housing scheme.",
    "German -> English": "Die Regierung kuendigte ein neues Wohnungsbauprogramm an.",
}

LANGUAGE_CODES = {
    "English -> German": ("en", "de"),
    "German -> English": ("de", "en"),
}


def validate_environment() -> list[str]:
    issues: list[str] = []
    if not MOSES_BIN.exists():
        issues.append(f"Missing Moses binary: {MOSES_BIN}")
    if not MOSES_SCRIPTS_DIR.exists():
        issues.append(f"Missing Moses scripts directory: {MOSES_SCRIPTS_DIR}")
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
    completed = subprocess.run(
        ["perl", str(script_path), *args],
        input=text,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or f"Failed script: {script_path.name}"
        raise RuntimeError(stderr)
    return completed.stdout


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
    ini_path = model_dir / "moses.ini"
    preprocessed = preprocess(text, source_lang)
    completed = subprocess.run(
        [str(MOSES_BIN), "-f", str(ini_path), "-v", "0"],
        input=preprocessed + "\n",
        text=True,
        capture_output=True,
        cwd=str(model_dir),
        check=False,
    )
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
            except RuntimeError as exc:
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

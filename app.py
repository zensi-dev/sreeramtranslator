from __future__ import annotations

import os
import subprocess
from pathlib import Path

import streamlit as st


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_ROOT = APP_ROOT / "models"
DEFAULT_MOSES_BIN = APP_ROOT / "runtime" / "bin" / "moses"
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


def validate_environment() -> list[str]:
    issues: list[str] = []
    if not MOSES_BIN.exists():
        issues.append(f"Missing Moses binary: {MOSES_BIN}")

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


def decode(text: str, model_dir: Path) -> str:
    ini_path = model_dir / "moses.ini"
    completed = subprocess.run(
        [str(MOSES_BIN), "-f", str(ini_path), "-v", "0"],
        input=text.strip() + "\n",
        text=True,
        capture_output=True,
        cwd=str(model_dir),
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "Unknown Moses error"
        raise RuntimeError(stderr)
    return completed.stdout.strip()


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
                translation = decode(cleaned, MODEL_DIRS[direction])
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

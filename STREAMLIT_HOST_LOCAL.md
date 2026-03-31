# Streamlit On Host

## Purpose

Run the Streamlit app directly on your machine without Docker.

The app still uses only the trained local PBSMT Moses models.

## Prerequisites

1. trained models exist under `models/`
2. Python 3 is available
3. build tools are available on the host:
   - `git`
   - `cmake`
   - `make`
   - `g++`

## Install Streamlit

```bash
python3 -m pip install -r requirements-streamlit.txt
```

## Build local SMT tools

```bash
bash scripts/setup_local_streamlit_tools.sh
```

This builds the needed classical SMT binaries under `tools/`.

## Run the app

```bash
bash scripts/run_streamlit_local.sh
```

Then open:

`http://localhost:8501`

## Notes

1. Default Moses binary path: `tools/mosesdecoder/bin/moses`
2. Default model path: `models/`
3. You can override them with `MOSES_BIN` and `PBSMT_MODELS_ROOT`

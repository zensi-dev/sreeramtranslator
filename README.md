# English-German PBSMT Streamlit App

Strict bidirectional phrase-based statistical machine translation for:

1. English -> German
2. German -> English

This app uses:

1. Streamlit for the UI
2. Moses-style phrase-based SMT model artifacts for inference
3. no neural MT
4. no external translation API
5. no fallback translation path

## Repo Layout

Inference-only runtime files are committed under `models/`, and the repo-bundled classical inference runtime is under `runtime/`.

Included per direction:

1. `moses.ini`
2. `phrase-table.gz`
3. `reordering-table.wbe-msd-bidirectional-fe.gz`
4. `lm.binary`

Included for inference runtime:

1. `runtime/bin/moses`
2. Moses tokenizer/recaser scripts needed at inference time
3. English and German truecase models
4. Moses nonbreaking-prefix tokenizer data

Excluded from the repo:

1. raw corpora
2. preprocessing outputs
3. intermediate training work files
4. generated reports

## Local Run

Install Python dependency:

```bash
python3 -m pip install -r requirements.txt
```

Build the local SMT runtime:

```bash
bash scripts/setup_local_streamlit_tools.sh
```

Run the app:

```bash
bash scripts/run_streamlit_local.sh
```

Run the app in Docker against the repo-level inference models:

```bash
docker compose up streamlit-app
```

Train and refresh the repo-level deployable models in one Docker command:

```bash
docker compose run --rm pbsmt-train
```

This trains, evaluates, and then refreshes the minimal inference artifacts under `models/` automatically.

Training data and reports now live under:

1. `data/raw/`
2. `data/processed/`
3. `data/work/`
4. `reports/`

Current demo note:

1. The default improved demo variant mixes News Commentary, Europarl, and a Tatoeba-derived conversational slice biased toward short everyday sentences.
2. This is still a small classical PBSMT system, so short formal and semi-conversational sentences work better than rare compounds or broad open-domain text.
3. The training script keeps this variant reproducible through deterministic shuffling and explicit corpus-size controls.

## Streamlit Community Cloud Attempt

This repo is prepared for a best-effort Streamlit deployment attempt.

Important note:

1. Streamlit Community Cloud supports `requirements.txt`
2. Streamlit Community Cloud supports `packages.txt`
3. This repo includes a bundled `runtime/bin/moses` binary for a best-effort inference attempt
4. The deployment may still fail if Streamlit Cloud runtime compatibility or memory limits are hit

If Community Cloud fails, the recommended fallback hosting target is a Linux VM running Streamlit directly.

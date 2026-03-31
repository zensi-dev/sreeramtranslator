# Local Docker Training

## Purpose

This setup runs the strict English-German PBSMT demo locally with Docker Compose.

It will:

1. build the classical SMT toolchain inside the image
2. download News Commentary v9 and Europarl v7 automatically
3. download the Tatoeba-derived conversational slice automatically
4. train `en->de` and `de->en` phrase-based models
5. export model archives and BLEU sanity-check outputs
6. refresh the repo-level `models/` directory automatically after training

## Run

Build the image:

```bash
docker compose build pbsmt-train
```

Run training:

```bash
docker compose run --rm pbsmt-train
```

Then test the Streamlit app:

```bash
docker compose up streamlit-app
```

## Outputs

Generated files are written under:

1. raw and processed datasets under `data/`
2. training reports under `reports/`
3. deployable inference models under `models/`

Important outputs:

1. `reports/model_en_de.tar.gz`
2. `reports/model_de_en.tar.gz`
3. `reports/test_outputs/bleu_en_de.txt`
4. `reports/test_outputs/bleu_de_en.txt`
5. refreshed deployable models under `models/`

## Optional Overrides

You can override compose environment values such as:

1. `THREADS`
2. `DATA_DIR`
3. `NEWS_PAIRS`
4. `EUROPARL_PAIRS`

Example:

```bash
THREADS=8 docker compose run --rm pbsmt-train
```

## Notes

1. This is CPU-oriented training. GPU is not required.
2. The script preserves downloaded raw archives for faster reruns.
3. Generated processed data and working files are recreated on each run inside `data/`, while reports are recreated under `reports/`.
4. The training container runs as your host UID/GID so the workspace stays writable after training.

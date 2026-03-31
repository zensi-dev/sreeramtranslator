# Local Docker Training

## Purpose

This setup runs the strict English-German PBSMT demo locally with Docker Compose.

It will:

1. build the classical SMT toolchain inside the image
2. download News Commentary v9 and Europarl v7 automatically
3. create the 40k subset automatically
4. train `en->de` and `de->en` phrase-based models
5. export model archives and BLEU sanity-check outputs

## Run

Build the image:

```bash
docker compose build pbsmt-train
```

Run training:

```bash
docker compose run --rm pbsmt-train
```

## Outputs

Generated files are written under:

`local_runs/pbsmt_demo/`

Important outputs:

1. `local_runs/pbsmt_demo/exports/model_en_de.tar.gz`
2. `local_runs/pbsmt_demo/exports/model_de_en.tar.gz`
3. `local_runs/pbsmt_demo/exports/test_outputs/bleu_en_de.txt`
4. `local_runs/pbsmt_demo/exports/test_outputs/bleu_de_en.txt`

## Optional Overrides

You can override compose environment values such as:

1. `THREADS`
2. `BASE_DIR`
3. `NEWS_PAIRS`
4. `EUROPARL_PAIRS`

Example:

```bash
THREADS=8 docker compose run --rm pbsmt-train
```

## Notes

1. This is CPU-oriented training. GPU is not required.
2. The script preserves downloaded raw archives for faster reruns.
3. Generated processed data, working files, models, and exports are recreated on each run inside the selected `BASE_DIR`.

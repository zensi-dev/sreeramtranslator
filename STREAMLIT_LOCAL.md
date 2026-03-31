# Streamlit Local Test

## Purpose

This app loads the locally trained PBSMT Moses models and lets you test both directions locally:

1. English -> German
2. German -> English

## Prerequisite

Train the models first:

```bash
docker compose run --rm pbsmt-train
```

## Run the app

```bash
docker compose up streamlit-app
```

Then open:

`http://localhost:8501`

## Notes

1. The app uses only the trained local Moses models.
2. There is no fallback translator.
3. If the app says a model file is missing, rerun training first.
4. For the repo-ready layout, keep only the minimal inference artifacts under `models/`.

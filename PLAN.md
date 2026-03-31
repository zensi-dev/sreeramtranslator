# Bidirectional English-German PBSMT Plan

## Objective

Build, evaluate, and deploy a strict bidirectional Machine Translation demo system for:

1. English -> German
2. German -> English

The system must use Phrase-based Statistical Machine Translation only.

## Scope

This is a basic academic/demo system optimized for:

1. clean implementation
2. low training time
3. reproducible setup
4. basic general-text sentence coverage

This is not a production MT system.

## Dataset

### Selected Corpora

1. News Commentary v9
   - Source: `https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz`
2. Europarl v7 English-German
   - Source: `https://www.statmt.org/europarl/v7/de-en.tgz`

### Training Subset

Use a cleaned subset of about 100,000 parallel sentence pairs:

1. 60,000 pairs from News Commentary
2. 40,000 pairs from Europarl

### Data Split

1. Train: 96,000 pairs
2. Validation / tuning: 2,000 pairs
3. Test: 2,000 pairs

## Training Pipeline

### Core Tools

1. Moses
2. KenLM
3. fast_align

### Steps

1. Download corpora automatically
2. Extract English-German sentence pairs
3. Remove empty lines and obvious markup leftovers
4. Normalize punctuation
5. Tokenize English and German text
6. Clean sentence pairs by length and ratio
7. Build a deterministic 100k subset
8. Train truecaser models for English and German
9. Apply truecasing to train, validation, and test sets
10. Train German language model for `en->de`
11. Train English language model for `de->en`
12. Train phrase-based model for `en->de`
13. Train phrase-based model for `de->en`
14. Save model artifacts for deployment
15. Ensure inference uses the same normalization, tokenization, truecasing, detruecasing, and detokenization steps as training
16. Tune both directions on the validation split after the stronger 100k baseline is stable

## Quality Targets

Expected behavior:

1. reasonable translation for short factual or common sentences
2. acceptable output for simple general text
3. weaker handling of long sentences, idioms, and complex syntax

## Evaluation

Evaluate both directions independently.

### Metrics

1. BLEU
2. Optional: TER if time permits

### Error Analysis

Document examples of:

1. reordering issues
2. morphology problems
3. untranslated or rare words
4. domain mismatch errors

## Deployment

### Interface

Use Streamlit to provide:

1. language direction selector
2. text input box
3. translated output area

### Runtime Behavior

1. load the correct directional model based on selection
2. preprocess input with the same classical pipeline assumptions
3. decode using Moses
4. present translated output

### Repo Deployment Layout

For repository-hosted inference, keep only the minimal runtime artifacts under `models/`:

1. `moses.ini`
2. `phrase-table.gz`
3. `reordering-table.wbe-msd-bidirectional-fe.gz`
4. `lm.binary`

Do not commit raw corpora, intermediate training files, or local build outputs.

## Execution Order

1. finalize repository constraints in `AGENTS.md`
2. create training notebook for Google Colab
3. run notebook to produce both models
4. inspect outputs and sample translations
5. document evaluation and errors
6. build Streamlit UI for local/demo deployment

## Notes

1. GPU is not important for this project because PBSMT training is CPU-oriented.
2. Google Colab is acceptable for the demo baseline if the workflow stays small and self-contained.
3. A clean 100k subset from the approved corpora is preferred over a larger noisy corpus for this demo.

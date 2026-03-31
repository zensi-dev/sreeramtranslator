## Project Constraints

This repository is for a strict bidirectional Machine Translation project using only Phrase-based Statistical Machine Translation (PBSMT).

### Non-Negotiable Requirements

1. The core translation system must use Phrase-based Statistical Machine Translation only.
2. The supported translation directions are exactly:
   - English -> German
   - German -> English
3. Two separate directional models must be trained and saved.
4. The system must not use neural MT in the training or inference pipeline.
5. The system must not use external translation APIs or hosted translation services.
6. The system must not implement fallback translation paths using other MT paradigms.
7. Any comparison with neural MT, if ever added, must be clearly isolated from the core system and must not be used by the application.

### Approved Stack

Use tools from the classical SMT pipeline only.

1. Moses decoder and training scripts
2. KenLM for n-gram language models
3. fast_align or MGIZA++ for word alignment
4. Standard Moses preprocessing scripts for normalization, tokenization, truecasing, cleaning, and detokenization
5. BLEU and related classical MT evaluation tools for reporting
6. Streamlit only as the deployment UI layer

### Dataset Rules

1. Use aligned English-German parallel corpora only.
2. The primary approved corpora for the demo baseline are:
   - News Commentary v9
   - Europarl v7
3. For the fast demo baseline, use a cleaned subset of approximately 40,000 parallel sentence pairs:
   - 25,000 from News Commentary
   - 15,000 from Europarl
4. Do not switch to noisier corpora such as Common Crawl, ParaCrawl, OpenSubtitles, or WikiMatrix unless explicitly requested and clearly documented.
5. Keep the same cleaned parallel subset available for both translation directions.

### Training Rules

1. Prefer fast_align for the demo baseline because it is much faster and better suited to short training cycles.
2. Keep training CPU-friendly and reproducible.
3. Preprocessing must include:
   - normalization
   - tokenization
   - cleaning
   - truecasing
4. Maintain separate model directories for `en-de` and `de-en`.
5. Do not claim production-quality translation quality.
6. Optimize for a clean, reproducible academic demo.

### Evaluation Rules

1. Evaluate both directions separately.
2. Report metrics per direction, not only a combined summary.
3. Include qualitative error analysis covering at least:
   - word order
   - morphology
   - unknown words
   - long sentence failures
4. Clearly note the limitations of a small-data PBSMT demo system.

### Deployment Rules

1. Streamlit is the UI layer only.
2. The deployed app must call the trained PBSMT models only.
3. The app must allow the user to choose translation direction.
4. The app must not call any external translation service.

### Collaboration Guidance

1. Favor the smallest correct implementation.
2. Preserve strict PBSMT scope in all code and documentation.
3. If a requested change risks violating the project constraints, stop and ask for confirmation before implementing it.

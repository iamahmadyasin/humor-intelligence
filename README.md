# Humor Intelligence

Predicting humor in text. This is a fine-tuned transformer that scores how funny a joke is
(trained on 430k+ rated Reddit jokes), beating a classical baseline, with a full
analysis of where it succeeds and fails.

**Status:** complete (models trained, published, and analyzed).

## Headline results

| Model | Context | Test Spearman | Test Pearson |
|---|---|---|---|
| TF-IDF + Ridge baseline | — | 0.363 | — |
| DistilBERT-128 | 128 tokens | 0.4038 | 0.4397 |
| DistilBERT-256 | 256 tokens | 0.4059 | 0.4465 |

The fine-tuned transformer clearly beats the classical baseline. Between 128 and
256 token contexts the difference is within run-to-run noise (statistically
equivalent), so 128 is the preferred model because of it is same quality while being faster to train
(shorter sequences mean less compute per example). Both models were evaluated
and analyzed (`08_results_colab_128.ipynb`, `09_results_colab_256.ipynb`); they reproduce their training scores exactly
and behave near-identically. See `WRITEUP.md` for the full analysis and
`PROJECT_LOG.md` for the engineering narrative.

Models on the Hugging Face Hub:
- 128-token: iamahmadyasin/humor-intelligence-distilbert
- 256-token: iamahmadyasin/humor-intelligence-distilbert-256

## Project structure

```
humor-intelligence/
├── data/
│   ├── raw/            # original downloaded data (gitignored)
│   └── processed/      # cleaned/split data ready for training (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_clean.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_finetune_colab_128.ipynb
│   ├── 05_finetune_kaggle_128.ipynb
│   ├── 06_finetune_colab_256.ipynb
│   ├── 07_finetune_kaggle_256.ipynb
│   └── 08_results_colab_128.ipynb
│   └── 09_results_colab_256.ipynb
├── src/
│   ├── config.py       # paths and constants
│   ├── data.py         # download, clean, build processed splits
│   └── baseline.py     # TF-IDF & Ridge baseline
├── reports/
│   └── figures/        # saved charts (committed)
│   └── SUMMARY.md      # portfolio-facing write-up
├── models/             # local checkpoints (gitignored)
├── PROJECT_LOG.md      # engineering narrative
└── README.md
```

## Setup (Windows)

Open a PowerShell window in the project folder.

1. Create a virtual environment:
   ```powershell
   python -m venv venv
   ```
2. Activate it (prompt should show `(venv)`):
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks the script, run once then retry:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Register the venv as a Jupyter kernel:
   ```powershell
   python -m ipykernel install --user --name humor-intelligence
   ```
5. Download + clean the dataset (builds `data/processed/`):
   ```powershell
   python src\data.py
   ```
6. Run the baseline (CPU, a couple of minutes):
   ```powershell
   python src\baseline.py
   ```
7. Launch notebooks:
   ```powershell
   jupyter notebook
   ```

## Compute & workflow

- **Local (CPU):** data download, cleaning, EDA, TF-IDF baseline.
- **Colab / Kaggle (free GPU):** transformer fine-tuning. The training notebooks
  clone this repo, rebuild the cleaned data, train, and push the model to the
  Hugging Face Hub. Model weights are not committed to git.

Training notebooks are provided for both Colab and Kaggle, at **128 and 256
tokens**, to document the free-tier workflow and the sequence-length experiment.

> Multi-GPU note: effective batch size = per-device batch x number of GPUs. On a
> single-GPU platform use `BATCH_SIZE=32`; on Kaggle's 2-GPU T4 use `BATCH_SIZE=16`
> to keep the effective batch at 32. Mismatching this silently changes the number
> of gradient updates (see PROJECT_LOG.md).

## Dataset

**rJokes** (Weller & Seppi, LREC 2020) — ~432k Reddit r/Jokes posts.

- Paper: https://aclanthology.org/2020.lrec-1.753/
- Source: https://github.com/orionw/rJokesData

**The label is already log-scaled:** `score` is `round(ln(raw_score + 1))`,
giving integers 0-11. It is used directly as a regression target — do not
log-transform again.

### Data cleaning (see EDA)

- Removed 5,707 exact-duplicate jokes (Reddit reposts).
- Dropped ultra-short (<5 word) title-only fragments.
- Removed dev/test jokes that also appeared in train (~2.4%) to prevent leakage.
- Cleaned sizes: 339,499 train / 41,941 dev / 41,957 test.

## Task & evaluation

**Regression** on the 0-11 humor label, evaluated with **Spearman correlation**
against human-derived scores (matching the rJokes paper for benchmark
comparison). Pearson also reported.

## Possible future extensions

Not part of the current scope, but natural directions if the project is picked
up again: a multi-seed significance study of the 128-vs-256 comparison; trying a
larger encoder (RoBERTa); and an LLM-based layer. These are exploratory and may or may not be pursued.

## License / data use

Jokes are from Reddit under the Reddit User Agreement; see the dataset source.
This repo does not redistribute the data, it's downloaded on setup.

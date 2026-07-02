# Humor Intelligence

Predicting and explaining humor in text. A fine-tuned transformer learns to
score how funny a joke is (trained on 430k+ rated Reddit jokes), then an LLM
layer explains why a given joke scored the way it did.

**Status:** in progress — EDA complete, model baseline next.

## Project structure

```
humor-intelligence/
├── data/
│   ├── raw/            # original downloaded data (gitignored)
│   └── processed/      # cleaned/split data ready for training (gitignored)
├── notebooks/
│   └── 01_eda.ipynb    # exploration and visualization
├── src/
│   ├── config.py       # paths and constants
│   └── data.py         # load/clean helpers, reused by notebooks + scripts
├── reports/
│   └── figures/        # saved charts (committed, so they render on GitHub)
├── models/             # saved model checkpoints (gitignored)
├── app/                # demo app (built later)
├── requirements.txt
└── README.md
```

## Setup

These commands assume you're in the VS Code integrated terminal, in the
project folder, with PowerShell.

1. Create a virtual environment:
   ```powershell
   python -m venv venv
   ```
2. Activate it (your prompt should then show `(venv)`):
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks the script with a "running scripts is disabled" error,
   run this once, then retry activation:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Register the venv as a Jupyter kernel (so VS Code notebooks can use it):
   ```powershell
   python -m ipykernel install --user --name humor-intelligence
   ```
5. Download the dataset:
   ```powershell
   python src\data.py
   ```
   This pulls the rJokes dataset (~430k Reddit jokes with humor-score labels)
   into `data\raw\`.
6. Open `notebooks\01_eda.ipynb` in VS Code, select the `humor-intelligence`
   kernel (top-right of the notebook), and run the cells.

> Note: every new terminal session needs step 2 again to re-activate the venv.
> If you select the venv interpreter in VS Code (`Ctrl+Shift+P` → "Python:
> Select Interpreter"), the integrated terminal usually auto-activates it.

## Dataset

**rJokes** (Weller & Seppi, LREC 2020) ~432k Reddit r/Jokes posts, each
labeled with a humor score. Pre-split into train/dev/test. This is a
published benchmark, so results can be compared against numbers reported in
the original paper.

- Paper: https://aclanthology.org/2020.lrec-1.753/
- Source: https://github.com/orionw/rJokesData

**Important: the label is already log-scaled.** The `score` column is not the
raw Reddit score. The dataset authors applied `round(ln(raw_score + 1))`,
producing discrete integer labels 0–11 (raw scores run into the millions;
these don't). Do not log-transform it again during modeling. The label
can be used directly as a regression target, or bucketed into humor classes
(not funny / mild / funny / very funny).

## Components

- **EDA** — distribution of humor labels, joke-length analysis, class balance.
- **Baseline** — TF-IDF + linear model as a reference point *(next)*.
- **Fine-tuned model** — DistilBERT/RoBERTa predicting the humor score *(planned)*.
- **LLM explanation layer** — given a joke and the model's score, an LLM
  produces a natural-language explanation of *why* it's funny or not, and an
  LLM-as-judge experiment compares the LLM's own humor ranking against the
  trained model and the human labels *(planned)*.
- **Demo** — interactive app to score any input joke and explain it *(planned)*.

## License / data use

Jokes are from Reddit under the Reddit User Agreement; see the dataset source
above. This repo does not redistribute the data, it's downloaded on setup.

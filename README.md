# Humor Intelligence

Predicting and explaining humor in text. A fine-tuned transformer learns to
score how funny a joke is (trained on 430k+ rated Reddit jokes), then an LLM
layer explains *why* a given joke scored the way it did.

**Status:** in progress — EDA complete, data cleaning + baseline next.

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

## Setup (Windows)

Open a PowerShell window in the project folder. Easiest way: in File Explorer,
Shift + right-click the `humor-intelligence` folder and choose **"Open
PowerShell window here"**. (Or open PowerShell from the Start menu and
`cd "D:\path\to\humor-intelligence"`.)

1. Create a virtual environment:
   ```powershell
   python -m venv venv
   ```
2. Activate it (your prompt should then start with `(venv)`):
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks this with a "running scripts is disabled" error, run
   the following once, then retry activation:
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
5. Download the dataset:
   ```powershell
   python src\data.py
   ```
   This pulls the rJokes dataset (~430k Reddit jokes with humor-score labels)
   into `data\raw\`.
6. Launch Jupyter and open the notebook:
   ```powershell
   jupyter notebook
   ```
   This opens Jupyter in your browser. Open `notebooks\01_eda.ipynb` and, from
   the Kernel menu, select the **humor-intelligence** kernel, then run the
   cells. (If you use an editor with built-in notebook support instead, just
   pick the same kernel there.)

> Note: every new terminal session needs step 2 again to re-activate the venv
> before running anything.

## Compute & workflow

Model fine-tuning needs a GPU, which this project runs on **Google Colab or
Kaggle's free tier** — not locally. The workflow is split:

- **Local (this repo):** data download, cleaning, EDA, and the CPU-friendly
  baseline. Light work, done in the venv above.
- **Colab / Kaggle:** transformer fine-tuning. The training notebook clones
  this repo (`!git clone <your repo url>`), runs on the free GPU, and saves
  the trained model out (downloaded or pushed to the Hugging Face Hub — model
  weights are **not** committed to git; see `.gitignore`).

Because of this split, keep the repo pushed to GitHub so the Colab/Kaggle side
always has the latest code and data-loading logic.

## Dataset

**rJokes** (Weller & Seppi, LREC 2020) — ~432k Reddit r/Jokes posts, each
labeled with a humor score. Pre-split into train/dev/test. A published
benchmark, so results can be compared against numbers in the original paper.

- Paper: https://aclanthology.org/2020.lrec-1.753/
- Source: https://github.com/orionw/rJokesData

**Important — the label is already log-scaled.** The `score` column is not the
raw Reddit score. The authors applied `round(ln(raw_score + 1))`, giving
discrete integer labels 0–11 (raw scores run into the millions; these don't).
Do **not** log-transform it again during modeling. It's used directly as a
regression target.

### Known data-quality notes (from EDA)

- ~1.65% of jokes are exact duplicates (Reddit reposts) — removed during
  cleaning to avoid train/dev/test leakage.
- ~4.9% of entries are ≤8 words; many are title-only fragments or orphaned
  punchlines with no setup (e.g. jokes whose content was a linked image).
  Very short entries are filtered during cleaning.
- Joke length correlates only weakly with score (r ≈ 0.13), so length alone
  is not a shortcut the model can exploit.

## Task & evaluation

**Primary task: regression** — predict the 0–11 humor label directly.
Evaluated with **Spearman correlation** against the human labels, matching the
metric reported in the original rJokes paper (enables a direct benchmark
comparison). A bucketed 4-class view (not funny / mild / funny / very funny)
is kept as a secondary framing for confusion-matrix visualization.

## Roadmap

- [x] Project scaffold, data download, EDA
- [ ] **Data cleaning** — dedupe + drop ultra-short entries; save cleaned
      splits to `data/processed/` (`src/data.py`)
- [ ] **Baseline** — TF-IDF + linear regression (CPU, runs locally); record
      Spearman correlation as the reference number to beat
- [ ] **Fine-tuned model** — DistilBERT/RoBERTa regression head, trained on
      Colab/Kaggle free GPU; compare Spearman vs. baseline and vs. the paper
- [ ] **LLM explanation layer** — given a joke + predicted score, an LLM
      explains why it's funny/not; LLM-as-judge experiment compares the LLM's
      humor ranking against the trained model and human labels
- [ ] **Demo** — interactive app to score and explain any input joke
- [ ] **Writeup** — results, findings, and portfolio-facing summary

## License / data use

Jokes are from Reddit under the Reddit User Agreement; see the dataset source
above. This repo does not redistribute the data — it's downloaded on setup.

# Humor Intelligence

Predicting and explaining humor in text. A model learns to score how funny a
joke is (trained on real Reddit r/Jokes data), then an LLM layer explains the
prediction and suggests funnier rewrites.

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
├── models/              # saved model checkpoints (gitignored)
├── app/                 # demo app (built in week 4)
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Register the venv as a Jupyter kernel (so VS Code notebooks use it):
   ```bash
   python -m ipykernel install --user --name humor-intelligence
   ```
4. Download the dataset:
   ```bash
   python src/data.py
   ```
   This pulls the rJokes dataset (~430k Reddit jokes with score labels) from
   its public GitHub repo into `data/raw/`.
5. Open `notebooks/01_eda.ipynb` in VS Code, select the `humor-intelligence`
   kernel (top right of the notebook), and run the cells.

## Dataset

**rJokes** (Weller & Seppi, LREC 2020) — ~432k Reddit r/Jokes posts, each
labeled with its Reddit score (community upvotes). Pre-split into
train/dev/test. This is a published benchmark, which means later you can
compare your results against numbers reported in the original paper.

- Paper: https://aclanthology.org/2020.lrec-1.753/
- Source: https://github.com/orionw/rJokesData

Note: raw score is a noisy proxy for "funniness" — it's affected by post
time, subreddit traffic, and virality, not just joke quality. We'll address
this with a log-transform and by treating it as a *ranking* signal rather
than a precise score. This nuance is actually a good thing to discuss in
your portfolio writeup.

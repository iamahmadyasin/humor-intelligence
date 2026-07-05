# Reusable evaluation harness for the humor-intelligence project.

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm.auto import tqdm

from data import load_split, clean_split, remove_leakage


def compute_metrics(true, pred):
    """Standard metrics reported everywhere: Spearman, Pearson, RMSE, MAE."""
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float).squeeze()
    return {
        "n":        int(len(true)),
        "spearman": float(spearmanr(true, pred).correlation),
        "pearson":  float(pearsonr(true, pred)[0]),
        "rmse":     float(mean_squared_error(true, pred) ** 0.5),
        "mae":      float(mean_absolute_error(true, pred)),
    }


@torch.no_grad()
def predict(model, tokenizer, texts, max_length, batch_size=64, device=None):
    """Batched regression inference. Works for any HF seq-classification model."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    texts = list(texts)
    preds = []
    for i in tqdm(range(0, len(texts), batch_size)):
        enc = tokenizer(texts[i:i + batch_size], truncation=True,
                        max_length=max_length, padding=True,
                        return_tensors="pt").to(device)
        preds.extend(model(**enc).logits.squeeze(-1).cpu().numpy().tolist())
    return np.array(preds)


def build_test_variants(min_words: int = 5) -> dict:
    train = clean_split(load_split("train"), min_words=min_words)
    leaky = clean_split(load_split("test"),  min_words=min_words)
    clean = remove_leakage(train, leaky)
    leaked = leaky[leaky["joke"].isin(set(train["joke"]))].reset_index(drop=True)
    assert len(clean) + len(leaked) == len(leaky), "variant reconciliation failed"
    return {"clean": clean, "leaky": leaky, "leaked_only": leaked}

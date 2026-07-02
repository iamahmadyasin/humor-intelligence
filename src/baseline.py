# TF-IDF + Ridge regression baseline for humor-score prediction.

import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import PROCESSED_DIR


def load_processed(split: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. you need to run `python src\\data.py` first to build it."
        )
    return pd.read_csv(path)


def run_baseline(
    max_features: int | None = None,
    ngram_range: tuple = (1, 2),
    min_df: int = 5,
    alpha: float = 1.0,
    sample_size: int | None = None,
):
    print("Loading data...")
    train = load_processed("train")
    test = load_processed("test")
    if sample_size is not None:
        train = train.sample(sample_size, random_state=0).reset_index(drop=True)
        print(f"  (using a {sample_size:,}-row sample of train)")

    print(f"Vectorizing {len(train):,} jokes (TF-IDF, this is the slow step)...")
    t0 = time.time()
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
    )
    X_train = vec.fit_transform(train["joke"])
    X_test = vec.transform(test["joke"])
    print(f"  done in {time.time() - t0:.0f}s | matrix {X_train.shape}")

    print("Fitting Ridge regression...")
    t0 = time.time()
    model = Ridge(alpha=alpha)
    model.fit(X_train, train["score"])
    print(f"  done in {time.time() - t0:.0f}s")

    pred = model.predict(X_test)
    metrics = {
        "spearman": spearmanr(test["score"], pred).correlation,
        "pearson": pearsonr(test["score"], pred)[0],
        "mae": mean_absolute_error(test["score"], pred),
        "rmse": mean_squared_error(test["score"], pred) ** 0.5,
    }

    print("\nTest metrics")
    print(f"Spearman rho: {metrics['spearman']:.4f}   <- headline number")
    print(f"Pearson r:    {metrics['pearson']:.4f}")
    print(f"MAE:          {metrics['mae']:.4f}")
    print(f"RMSE:         {metrics['rmse']:.4f}")

    mean_rmse = mean_squared_error(
        test["score"], np.full(len(test), train["score"].mean())
    ) ** 0.5
    print(f"\n(predict-the-mean RMSE {mean_rmse:.4f} — model should be lower)")

    return model, vec, metrics


if __name__ == "__main__":
    run_baseline()
# Download and load the rJokes dataset.

import gzip
import shutil
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config import RAW_DIR, PROCESSED_DIR, RJOKES_BASE_URL, RJOKES_FILES


def download_rjokes(dest_dir: Path = RAW_DIR) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for filename in RJOKES_FILES:
        out_path = dest_dir / filename
        if out_path.exists():
            print(f"[skip] {filename} already downloaded")
            continue

        url = f"{RJOKES_BASE_URL}/{filename}"
        print(f"[download] {url}")
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with open(out_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=filename
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))


def load_split(split: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    assert split in {"train", "dev", "test"}, "split must be train/dev/test"
    path = raw_dir / f"{split}.tsv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found! run `python src/data.py` to download it first."
        )
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["score", "joke"],
        compression="gzip",
        on_bad_lines="skip",
    )
    return df

def clean_split(df: pd.DataFrame, min_words: int = 5) -> pd.DataFrame:
    # Drop empty/NaN jokes, exact duplicates, and ultra-short entries.
    # Returns a new DataFrame; does not modify the input.
    df = df.copy()
    df["joke"] = df["joke"].astype(str)
    # 1. drop empty/whitespace-only/literal 'nan'
    df = df[df["joke"].str.strip().ne("") & df["joke"].str.lower().ne("nan")]
    # 2. drop exact duplicate jokes within this split
    df = df.drop_duplicates(subset="joke")
    # 3. drop ultra-short entries (title fragments, junk, orphaned lines)
    word_counts = df["joke"].str.split().str.len()
    df = df[word_counts >= min_words]
    return df.reset_index(drop=True)


def remove_leakage(
    train: pd.DataFrame, other: pd.DataFrame
) -> pd.DataFrame:
    # Remove rows from `other` (dev/test) whose joke text also appears in `train`, preventing evaluation on memorized examples.
    train_jokes = set(train["joke"])
    mask = ~other["joke"].isin(train_jokes)
    return other[mask].reset_index(drop=True)


def build_processed(min_words: int = 5) -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train = clean_split(load_split("train"), min_words=min_words)
    dev = clean_split(load_split("dev"), min_words=min_words)
    test = clean_split(load_split("test"), min_words=min_words)
    # remove any dev/test joke that also appears in train
    dev = remove_leakage(train, dev)
    test = remove_leakage(train, test)
    splits = {"train": train, "dev": dev, "test": test}
    for name, frame in splits.items():
        out_path = PROCESSED_DIR / f"{name}.csv"
        frame.to_csv(out_path, index=False)
        print(f"[saved] {name}: {len(frame):,} rows -> {out_path}")
    return splits

if __name__ == "__main__":
    download_rjokes()
    print()
    splits = build_processed()
    print(f"\nSample cleaned training rows:")
    print(splits["train"].head())
# Download and load the rJokes dataset.

import gzip
import shutil
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config import RAW_DIR, RJOKES_BASE_URL, RJOKES_FILES


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


if __name__ == "__main__":
    download_rjokes()
    train = load_split("train")
    print(f"\nLoaded {len(train):,} training rows")
    print(train.head())

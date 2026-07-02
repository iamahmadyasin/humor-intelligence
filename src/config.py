# Shared paths and constants for the humor-intelligence project.

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RJOKES_BASE_URL = (
    "https://raw.githubusercontent.com/orionw/rJokesData/master/data"
)
RJOKES_FILES = ["train.tsv.gz", "dev.tsv.gz", "test.tsv.gz"]

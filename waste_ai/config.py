"""Configuration shared by the dataset, training, evaluation, and Telegram scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"
METRICS = ARTIFACTS / "metrics"
LABEL_MAP = ROOT / "config" / "label_map.json"
LABELS = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
CLASS_TO_INDEX = {name: index for index, name in enumerate(LABELS)}
BINARY_LABELS = ("non_recyclable", "recyclable")  # legacy helper compatibility only
IMAGE_SIZE = (224, 224)
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
SEED = 42


def ensure_dirs():
    for directory in (RAW_DATA, PROCESSED, ARTIFACTS, MODELS, METRICS):
        directory.mkdir(parents=True, exist_ok=True)

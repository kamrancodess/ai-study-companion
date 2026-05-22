from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "study_companion.db"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL_NAME = "deepset/minilm-uncased-squad2"
SUMMARIZER_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 35
TOP_K_RETRIEVAL = 5
OCR_DPI = 160
OCR_LANG = "eng"

DEFAULT_USER_ID = 1


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

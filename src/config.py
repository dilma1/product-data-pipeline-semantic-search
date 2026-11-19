import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw_batches"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

DB_PATH = GOLD_DIR / "products.db"        # SQLite: product details + pipeline state
FAISS_PATH = GOLD_DIR / "faiss.index"     # FAISS: embedding
REJECTED_PATH = SILVER_DIR / "rejected.csv"   # quarantine: rows dropped by Silver cleaning

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"   
EMBEDDING_DIM = 384    
EMBED_BATCH_SIZE = 32                   


def ensure_dirs() -> None:
    """Create the data folders if they don't exist yet."""
    for folder in (RAW_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    """Simple console logging used by scripts and the API."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

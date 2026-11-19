"""Step 4 - Gold embeddings (FAISS)"""

import logging
from functools import lru_cache

import faiss
import numpy as np
import pandas as pd

from src.config import EMBED_BATCH_SIZE, EMBEDDING_DIM, FAISS_PATH, MODEL_NAME, ensure_dirs, setup_logging
from src.db import get_connection, init_db

logger = logging.getLogger(__name__)
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    logger.info("Loading embedding model: %s", MODEL_NAME)
    try:
        # 1. Fast path: load straight from local cache with zero network calls
        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception:
        # 2. Download from Hugging Face and cache locally
        logger.info("Model not found in local cache, downloading from Hugging Face...")
        return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Turn texts into normalised float32 vectors (also used by search.py for the query)."""
    vectors = get_model().encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype("float32")


def _all_products() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT product_id, search_text FROM products "
            "WHERE search_text IS NOT NULL AND TRIM(search_text) != ''",
            conn,
        )


def _save_index(index: faiss.Index) -> None:
    """Write to a temp file, then rename, so a crash never leaves a broken index."""
    tmp_path = FAISS_PATH.with_suffix(".index.tmp")
    faiss.write_index(index, str(tmp_path))
    tmp_path.replace(FAISS_PATH)


def _embed(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ids = df["product_id"].astype("int64").to_numpy()
    vectors = embed_texts(df["search_text"].tolist())
    if vectors.shape[1] != EMBEDDING_DIM:
        logger.warning(
            "Model returns %d-dimensional vectors but EMBEDDING_DIM is %d - update config.py",
            vectors.shape[1], EMBEDDING_DIM,
        )
    return ids, vectors


def update_index(df: pd.DataFrame | None = None) -> int:
    """Create or update the FAISS index. Returns the number of vectors in the index."""
    ensure_dirs()
    init_db()

    incremental = df is not None and FAISS_PATH.exists()

    if incremental:
        df = df[["product_id", "search_text"]].drop_duplicates("product_id", keep="last")
        df = df[df["search_text"].notna() & (df["search_text"].str.strip() != "")]
        if df.empty:
            index = faiss.read_index(str(FAISS_PATH))
            logger.info("FAISS: nothing to update (%d vectors)", index.ntotal)
            return index.ntotal

        ids, vectors = _embed(df)
        index = faiss.read_index(str(FAISS_PATH))
        if index.d != vectors.shape[1]:
            raise ValueError(
                f"Existing index has {index.d} dimensions but the model gives {vectors.shape[1]}. "
                "The model was changed: delete data/gold/faiss.index and run again."
            )
        before_count = index.ntotal
        replaced = int(index.remove_ids(ids))
        added = len(ids) - replaced
        index.add_with_ids(vectors, ids)
        mode_label = "Incremental"
    else:
        products = _all_products()
        if products.empty:
            logger.warning("FAISS Summary | No searchable products in SQLite - nothing indexed")
            return 0
        ids, vectors = _embed(products)
        before_count = 0
        added = len(ids)
        replaced = 0
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
        index.add_with_ids(vectors, ids)
        mode_label = "Full Build"

    _save_index(index)

    with get_connection() as conn:
        expected = conn.execute(
            "SELECT COUNT(*) FROM products WHERE search_text IS NOT NULL AND TRIM(search_text) != ''"
        ).fetchone()[0]
    if index.ntotal != expected:
        logger.warning("Index has %d vectors but SQLite has %d searchable products", index.ntotal, expected)
    logger.info(
        "FAISS Summary:\n"
        "  Mode               : %s\n"
        "  Vectors embedded   : %d (dim=%d)\n"
        "  Newly Inserted     : %d\n"
        "  Updated (replaced) : %d\n"
        "  Total in index     : %d (was %d)",
        mode_label,
        len(ids),
        vectors.shape[1],
        added,
        replaced,
        index.ntotal,
        before_count,
    )
    return index.ntotal


if __name__ == "__main__":
    setup_logging()
    update_index()
"""Step 3 - Silver -> Gold (SQLite)
"""
import logging

import pandas as pd

from src.config import SILVER_DIR, ensure_dirs, setup_logging
from src.db import get_connection, init_db

logger = logging.getLogger(__name__)

MAX_SEARCH_TEXT_CHARS = 1000   # the embedding model only reads the first part of long text anyway
GOLD_COLUMNS = [
    "product_id", "product_title", "product_description", "brand",
    "category", "is_secondhand", "search_text", "updated_at",
]

UPSERT_SQL = """
    INSERT INTO products (
        product_id, product_title, product_description, brand,
        category, is_secondhand, search_text, updated_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(product_id) DO UPDATE SET
        product_title       = excluded.product_title,
        product_description = excluded.product_description,
        brand               = excluded.brand,
        category            = excluded.category,
        is_secondhand       = excluded.is_secondhand,
        search_text         = excluded.search_text,
        updated_at          = excluded.updated_at
"""


def build_search_text(df: pd.DataFrame) -> pd.Series:
    """Join title, category, brand and description, skipping anything missing."""
    parts = df[["product_title", "category", "brand", "product_description"]]
    text = parts.apply(
        lambda row: ". ".join(str(v) for v in row if pd.notna(v) and str(v).strip()),
        axis=1,
    )
    return text.str.slice(0, MAX_SEARCH_TEXT_CHARS)


def _read_silver() -> pd.DataFrame:
    """Read every Silver CSV."""
    files = sorted(SILVER_DIR.glob("silver_*.csv"))
    if not files:
        return pd.DataFrame(columns=GOLD_COLUMNS)

    df = pd.concat(
        [pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[""]) for f in files],
        ignore_index=True,
    )
    df["product_id"] = pd.to_numeric(df["product_id"]).astype(int)
    df["is_secondhand"] = pd.to_numeric(df["is_secondhand"], errors="coerce").astype("Int64")
    # The same product can appear in several Silver files: keep its newest version
    return df.sort_values("updated_at").drop_duplicates("product_id", keep="last")


def _to_records(df: pd.DataFrame) -> list[tuple]:
    """Plain Python values for SQLite, with every missing value as None (NULL)."""
    data = df[GOLD_COLUMNS].astype(object)
    data = data.where(data.notna(), None)
    return [
        tuple(int(v) if col in ("product_id", "is_secondhand") and v is not None else v
              for col, v in zip(GOLD_COLUMNS, row))
        for row in data.itertuples(index=False, name=None)
    ]


def load_to_gold(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Upsert Silver rows into SQLite. Returns the loaded rows (with search_text) for embedding."""
    ensure_dirs()
    init_db()

    if df is None:
        df = _read_silver()
    if df.empty:
        logger.info("Nothing to load into Gold")
        return pd.DataFrame(columns=GOLD_COLUMNS)

    df = df.copy()
    df["search_text"] = build_search_text(df)

    # Silver guarantees a title or description, so this should never remove rows
    empty = df["search_text"].str.strip() == ""
    if empty.any():
        logger.warning("Skipping %d rows with empty search_text", int(empty.sum()))
        df = df[~empty]

    with get_connection() as conn:
        existing = {row[0] for row in conn.execute("SELECT product_id FROM products")}
        n_new = int((~df["product_id"].isin(existing)).sum())
        conn.executemany(UPSERT_SQL, _to_records(df))
        total = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]

    n_updated = len(df) - n_new
    logger.info(
        "Gold Summary:\n"
        "  Records processed : %d\n"
        "  Inserted Records  : %d\n"
        "  Updated  Records  : %d\n"
        "  Total in Gold DB  : %d",
        len(df),
        n_new,
        n_updated,
        total,
    )
    return df[GOLD_COLUMNS].reset_index(drop=True)


if __name__ == "__main__":
    setup_logging()
    load_to_gold()

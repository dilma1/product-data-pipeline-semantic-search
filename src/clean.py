"""Bronze -> Silver cleaning, validation, and deduplication."""
import logging
import re
from datetime import datetime

import pandas as pd
from ftfy import fix_encoding

from src.config import (
    BRONZE_DIR,
    FAISS_PATH,
    REJECTED_PATH,
    SILVER_DIR,
    ensure_dirs,
    setup_logging,
)
from src.db import get_watermark, init_db

logger = logging.getLogger(__name__)

VECTOR_COLS = ["product_title", "product_description", "brand", "category"]
SILVER_COLUMNS = [
    "product_id", "product_title", "product_description",
    "brand", "category", "is_secondhand", "updated_at", "source_file",
]

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _repair(value):
    """Fix mojibake (UTF-8 text that was wrongly decoded as Windows-1252/Latin-1)"""
    return fix_encoding(value) if isinstance(value, str) else value


def _clean_text(s: pd.Series) -> pd.Series:
    """Repair mojibake, strip HTML tags, collapse whitespace, trim, and turn blanks into nulls."""
    s = s.astype(object).map(_repair).astype("string")
    s = s.str.replace(_TAG_RE, " ", regex=True).str.replace(_SPACE_RE, " ", regex=True).str.strip()
    return s.replace("", pd.NA)


def _read_bronze() -> pd.DataFrame | None:
    files = sorted(BRONZE_DIR.glob("bronze_*.csv"))
    if not files:
        return None
    return pd.concat(
        [pd.read_csv(f, dtype=str, keep_default_na=False) for f in files],
        ignore_index=True,
    )


def clean_to_silver() -> pd.DataFrame:
    ensure_dirs()
    init_db()

    watermark = get_watermark()
    incremental = FAISS_PATH.exists() and watermark is not None
    logger.info("Silver mode: %s", f"INCREMENTAL LOADING (watermark {watermark})" if incremental else "FULL LOAD")

    df = _read_bronze()
    if df is None:
        logger.warning("No Bronze files found in %s", BRONZE_DIR)
        return pd.DataFrame(columns=SILVER_COLUMNS)

    raw = df.copy()                      # untouched copy, used for the rejected file
    rows_read = len(df)
    rejected: list[pd.DataFrame] = []

    def reject(mask: pd.Series, reason: str) -> None:
        if mask.any():
            rejected.append(raw.loc[df.index[mask]].assign(
                reject_reason=reason,
                rejected_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ))

    # 1. Text cleaning
    for col in VECTOR_COLS:
        df[col] = _clean_text(df[col])

    # 2. Types; rows with an unusable id or timestamp cannot be processed
    df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce")
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
    bad = df["product_id"].isna() | df["updated_at"].isna()
    reject(bad, "invalid product_id or updated_at")
    df = df[~bad]

    # 3. Watermark filter (incremental runs only): skips old and stale records.
    skipped_old = 0
    if incremental:
        before = len(df)
        df = df[df["updated_at"] >= pd.Timestamp(watermark)]
        skipped_old = before - len(df)

    # 4. Reject (the embedding would be meaningless)
    no_text = df["product_title"].isna() & df["product_description"].isna()
    reject(no_text, "no title and no description")
    df = df[~no_text].copy()

    # 5. Flag: 1 / 0, or NULL when unknown 
    sh = df["is_secondhand"].astype("string").str.strip().str.lower()
    df["is_secondhand"] = sh.map({"true": 1, "1": 1, "false": 0, "0": 0}).astype("Int64")

    # 6. Deduplicate: newest updated_at wins; on a tie the earliest batch wins
    before = len(df)
    df = df.sort_values(["updated_at", "source_file"], ascending=[True, False], kind="stable")
    df = df.drop_duplicates("product_id", keep="last")
    dups_removed = before - len(df)

    # 7. Final shape
    df["product_id"] = df["product_id"].astype(int)
    df["updated_at"] = df["updated_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df[SILVER_COLUMNS].sort_values("product_id").reset_index(drop=True)

    # 8. Save Silver and the rejected rows
    if not incremental:
        for old in SILVER_DIR.glob("silver_*.csv"):
            old.unlink()
        if REJECTED_PATH.exists():
            REJECTED_PATH.unlink()
    if not df.empty:
        silver_path = SILVER_DIR / f"silver_{datetime.now():%Y%m%d_%H%M%S}.csv"
        df.to_csv(silver_path, index=False, encoding="utf-8-sig")
        logger.info("Saved %d rows to %s", len(df), silver_path.name)
    if rejected:
        rej = pd.concat(rejected, ignore_index=True)
        rej.to_csv(REJECTED_PATH, mode="a", header=not REJECTED_PATH.exists(), index=False, encoding="utf-8-sig")
        logger.info("Rejected %d rows -> %s", len(rej), REJECTED_PATH.name)

    rejected_count = sum(len(r) for r in rejected)
    logger.info(
        "Silver Summary:\n"
        "  Bronze Record Count     : %d\n"
        "  Rejected (invalid/empty): %d\n"
        "  Skipped (older than WM) : %d\n"
        "  Duplicates Removed      : %d\n"
        "  Output Records          : %d",
        rows_read,
        rejected_count,
        skipped_old,
        dups_removed,
        len(df),
    )
    return df

if __name__ == "__main__":
    setup_logging()
    clean_to_silver()

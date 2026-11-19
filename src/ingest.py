import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import BRONZE_DIR, RAW_DIR, ensure_dirs, setup_logging
from src.db import init_db

logger = logging.getLogger(__name__)


def ingest_to_bronze() -> list[Path]:
    """Load raw files into Bronz."""
    ensure_dirs()
    init_db()

    raw_files = sorted(RAW_DIR.glob("*.csv"))
    if not raw_files:
        logger.warning("No CSV files found in %s", RAW_DIR)
        return []

    created: list[Path] = []
    for raw_path in raw_files:
        bronze_path = BRONZE_DIR / f"bronze_{raw_path.name}"

        df = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
        df["source_file"] = raw_path.name
        df["ingested_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Write to .csv.tmp first, then replace bronze_<name>.csv
        # If a crash happens mid write, the previous valid Bronze file stays intact.
        tmp_path = bronze_path.with_suffix(".csv.tmp")
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(bronze_path)

        created.append(bronze_path)
        logger.info("Ingested %s -> %s (%d rows)", raw_path.name, bronze_path.name, len(df))

    return created


if __name__ == "__main__":
    setup_logging()
    ingest_to_bronze()
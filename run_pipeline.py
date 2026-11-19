"""Runs the whole pipeline: raw - Bronze - Silver - Gold (SQLite + FAISS).
"""
import logging

from src.clean import clean_to_silver
from src.config import setup_logging
from src.db import get_watermark, init_db, is_incremental_run, update_pipeline_state
from src.embedding import update_index
from src.ingest import ingest_to_bronze
from src.load_to_gold import load_to_gold

logger = logging.getLogger(__name__)


def run_pipeline() -> dict:

    init_db()
    mode = "Incremental" if is_incremental_run() else "Full"
    logger.info("=== Pipeline Starting (%s run) ===", mode)

    bronze_files = ingest_to_bronze()         
    silver = clean_to_silver() 
                   
    gold_rows = 0
    vectors = 0              

    if not silver.empty:
        gold = load_to_gold(silver)
        vectors = update_index(gold)
        gold_rows = len(gold)
        update_pipeline_state(silver["updated_at"].max(), silver["source_file"].tolist(),len(silver))
    else:
        logger.info("No new records, watermark unchanged")

    summary = {
        "mode": mode,
        "bronze_files": len(bronze_files),
        "silver_rows": len(silver),
        "gold_rows_loaded": gold_rows,
        "vectors_in_index": vectors,
        "watermark": get_watermark(),
    }
    logger.info(
        "=== Pipeline Completed  ===\n"
        "  Mode             : %s\n"
        "  Bronze Files     : %d\n"
        "  Silver Rows      : %d\n"
        "  Gold Rows Loaded : %d\n"
        "  Vectors in Index : %d\n"
        "  Watermark        : %s",
        
        mode,len(bronze_files),len(silver),gold_rows,vectors,summary["watermark"],
    )
    return summary


if __name__ == "__main__":
    setup_logging()
    run_pipeline()
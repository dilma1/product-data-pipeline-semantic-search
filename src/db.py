"""SQLite helpers: connection, table setup, and pipeline state."""
import sqlite3

from src.config import DB_PATH, FAISS_PATH, GOLD_DIR


def get_connection() -> sqlite3.Connection:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Create both tables if they don't exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                product_id          INTEGER PRIMARY KEY,
                product_title       TEXT,
                product_description TEXT,
                brand               TEXT,
                category            TEXT,
                is_secondhand       INTEGER,
                search_text         TEXT,
                updated_at          TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_state (
                run_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                max_updated_date       TEXT NOT NULL,
                pipeline_executed_date TEXT NOT NULL,
                run_count              INTEGER DEFAULT 0,
                source_files           TEXT
                );
            """
        )


def get_watermark() -> str | None:
    """Latest updated_at already processed, or None on the very first run."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT MAX(max_updated_date) FROM pipeline_state"
        ).fetchone()
    return row[0] if row else None


def is_incremental_run() -> bool:
    """True only when both faiss.index exists on disk AND a watermark is saved."""
    return FAISS_PATH.exists() and get_watermark() is not None


def update_pipeline_state(last_processed_date: str, source_files: list[str],batch_rows: int = 0) -> None:
    """Save watermark, run timestamp, incremented run_count, and source filenames."""
    files_str = ", ".join(sorted(set(source_files)))
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO pipeline_state (
                max_updated_date, pipeline_executed_date, run_count, source_files
            )
            VALUES (?, datetime('now', 'localtime'), ?, ?)
            """,
            (last_processed_date, batch_rows,files_str),
        )
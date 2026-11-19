"""FastAPI app.
  POST /ingest    raw files - Bronze only
  POST /process   the whole pipeline: Bronze - Silver - Gold (SQLite + FAISS)
  GET  /products  paged list with filters (SQLite)
  GET  /search    semantic search (FAISS finds ids, SQLite returns details)
  GET  /health    quick status
"""
import logging
import threading
from contextlib import asynccontextmanager, contextmanager
from datetime import date

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from run_pipeline import run_pipeline
from src.config import FAISS_PATH, setup_logging
from src.db import get_watermark, init_db
from src.ingest import ingest_to_bronze
from src.search import list_products, semantic_search

logger = logging.getLogger(__name__)

_pipeline_lock = threading.Lock()


@contextmanager
def _exclusive():
    if not _pipeline_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A pipeline run is already in progress")
    try:
        yield
    finally:
        _pipeline_lock.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    init_db()
    if FAISS_PATH.exists():
        try:  
            from src.embedding import get_model
            get_model()
        except Exception:
            logger.exception("Could not pre-load the model; the first search will load it")
    yield


app = FastAPI(
    title="Product Data Pipeline & Semantic Search",
    description="Medallion pipeline (Bronze/Silver/Gold) with SQLite and FAISS semantic search.",
    lifespan=lifespan,
)


# ---------- response models ----------
class Product(BaseModel):
    product_id: int
    product_title: str
    product_description: str | None = None
    brand: str
    category: str
    is_secondhand: bool | None = None
    updated_at: str


class SearchResult(Product):
    score: float


class ProductPage(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[Product]


class IngestResponse(BaseModel):
    files_written: list[str]


class PipelineSummary(BaseModel):
    mode: str
    bronze_files: int
    silver_rows: int
    gold_rows_loaded: int
    vectors_in_index: int
    watermark: str | None


# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "index_built": FAISS_PATH.exists(), "watermark": get_watermark()}


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """Copy the raw batch files into Bronze (no cleaning)."""
    with _exclusive():
        try:
            files = ingest_to_bronze()
        except Exception:
            logger.exception("Ingest failed")
            raise HTTPException(status_code=500, detail="Ingest failed - see server logs")
    return {"files_written": [f.name for f in files]}


@app.post("/process", response_model=PipelineSummary)
def process():
    """Run the full pipeline. Full on the first run, incremental afterwards."""
    with _exclusive():
        try:
            return run_pipeline()
        except Exception:
            logger.exception("Pipeline failed")
            raise HTTPException(status_code=500, detail="Pipeline failed - see server logs")


@app.get("/products", response_model=ProductPage)
def products(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: str | None = Query(None, description="Matches part of the category path"),
    secondhand: bool | None = None,
    date_from: date | None = Query(None, description="updated_at on or after (YYYY-MM-DD)"),
    date_to: date | None = Query(None, description="updated_at on or before (YYYY-MM-DD)"),
):
    """List products from SQLite with optional filters."""
    return list_products(
        page, page_size, category, secondhand,
        date_from.isoformat() if date_from else None,
        date_to.isoformat() if date_to else None,
    )


@app.get("/search", response_model=list[SearchResult])
def search(
    q: str = Query(..., min_length=1, max_length=300, description="Natural-language query"),
    k: int = Query(5, ge=1, le=50, description="Number of results"),
    category: str | None = None,
    secondhand: bool | None = None,
):
    """Semantic search: the products closest in meaning to the query, best first."""
    try:
        return semantic_search(q, k, category, secondhand)
    except FileNotFoundError:
        raise HTTPException(status_code=409, detail="Search index not built yet - call POST /process first")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

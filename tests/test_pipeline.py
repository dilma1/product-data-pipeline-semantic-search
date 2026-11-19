import hashlib
import importlib
import sqlite3
from types import SimpleNamespace

import faiss
import numpy as np
import pandas as pd
import pytest

from run_pipeline import run_pipeline
from src.db import get_watermark
from src.search import semantic_search

MODULES = [
    "src.config", "src.db", "src.ingest", "src.clean",
    "src.load_to_gold", "src.embedding", "src.search",
    "run_pipeline", "main",
]
COLUMNS = [
    "product_id", "product_title", "product_description",
    "brand", "category", "is_secondhand", "updated_at",
]


def _fake_embed_texts(texts):
    """Fake 384 dim vectors so tests run offline without loading the model."""
    vecs = []
    for text in texts:
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).normal(size=384)
        vecs.append(v / np.linalg.norm(v))
    return np.array(vecs, dtype="float32")


def _row(pid, title="__default__", desc="A description", brand="Brand",
         category="Cat > Sub", secondhand="False", updated="2026-09-10 10:00:00"):
    title = f"Product {pid}" if title == "__default__" else title
    return [pid, title, desc, brand, category, secondhand, updated]


def _write_raw(env, name, rows):
    pd.DataFrame(rows, columns=COLUMNS).to_csv(env.raw / name, index=False)


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    paths = SimpleNamespace(
        raw=data / "raw_batches",
        bronze=data / "bronze",
        silver=data / "silver",
        gold=data / "gold",
        db=data / "gold" / "products.db",
        faiss=data / "gold" / "products.index",
        rejected=data / "silver" / "rejected.csv",
    )
    for folder in (paths.raw, paths.bronze, paths.silver, paths.gold):
        folder.mkdir(parents=True)

    overrides = {
        "BASE_DIR": tmp_path, "DATA_DIR": data, "RAW_DIR": paths.raw,
        "BRONZE_DIR": paths.bronze, "SILVER_DIR": paths.silver,
        "GOLD_DIR": paths.gold, "DB_PATH": paths.db,
        "FAISS_PATH": paths.faiss, "REJECTED_PATH": paths.rejected,
    }
    for mod_name in MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            continue
        for attr, val in overrides.items():
            if hasattr(mod, attr):
                monkeypatch.setattr(mod, attr, val)

    embed_mod = importlib.import_module("src.embedding")
    if hasattr(embed_mod, "embed_texts"):
        monkeypatch.setattr(embed_mod, "embed_texts", _fake_embed_texts)

    return paths


def test_incremental_pipeline(env):
    # Batch 1: duplicate, empty text (rejected), missing fields ( NULL)
    _write_raw(env, "batch_1.csv", [
        _row(1, title="Blue running shoes", updated="2026-09-01 08:00:00"),
        _row(2, updated="2026-09-02 08:00:00"),
        _row(2, updated="2026-09-02 08:00:00"),
        _row(3, title="", desc="", updated="2026-09-03 08:00:00"),
        _row(4, desc="", brand="", updated="2026-09-04 08:00:00"),
    ])
    run_pipeline()

    # Batch 2: new product, updated product, and stale older copy
    _write_raw(env, "batch_2.csv", [
        _row(5, updated="2026-10-01 08:00:00"),
        _row(1, title="Blue running shoes (new edition)", updated="2026-10-02 08:00:00"),
        _row(2, desc="OLD STALE RECORD", updated="2026-09-02 08:00:00"),
    ])
    run_pipeline()

    with sqlite3.connect(env.db) as conn:
        products = {
            r[0]: r[1:]
            for r in conn.execute("SELECT product_id, product_title, product_description, brand FROM products")
        }
        text = conn.execute("SELECT search_text FROM products WHERE product_id = 5").fetchone()[0]
        audit_runs = conn.execute("SELECT COUNT(*) FROM pipeline_state").fetchone()[0]

    assert sorted(products) == [1, 2, 4, 5]
    assert products[1][0] == "Blue running shoes (new edition)"
    assert "OLD STALE" not in products[2][1]
    assert products[4][1] is None and products[4][2] is None
    assert audit_runs == 2
    assert get_watermark() == "2026-10-02 08:00:00"
    assert faiss.read_index(str(env.faiss)).ntotal == 4
    assert semantic_search(text, k=1)[0]["product_id"] == 5
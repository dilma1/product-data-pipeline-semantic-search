import logging

import faiss

import src.embedding as embed
from src.config import FAISS_PATH
from src.db import get_connection, init_db

logger = logging.getLogger(__name__)

COLUMNS = "product_id, product_title, product_description, brand, category, is_secondhand, updated_at"
FILTER_OVERFETCH = 10   # with filters, ask FAISS for k * 10 candidates so enough survive the filter


def format_product(row: tuple) -> dict:
    """One SQLite row -> API dict, with display placeholders for missing values."""
    product_id, title, description, brand, category, secondhand, updated_at = row
    return {
        "product_id": product_id,
        "product_title": title or "Untitled product",
        "product_description": description,
        "brand": brand or "Unknown",
        "category": category or "Uncategorized",
        "is_secondhand": None if secondhand is None else bool(secondhand),
        "updated_at": updated_at,
    }


def _build_where(category=None, secondhand=None, date_from=None, date_to=None) -> tuple[str, list]:
    """Optional filters as a parameterised WHERE clause (never string-formatted values)."""
    clauses, params = [], []
    if category:
        clauses.append("category LIKE ?")
        params.append(f"%{category}%")
    if secondhand is not None:
        clauses.append("is_secondhand = ?")
        params.append(int(secondhand))
    if date_from:
        clauses.append("date(updated_at) >= date(?)")
        params.append(date_from)
    if date_to:
        clauses.append("date(updated_at) <= date(?)")
        params.append(date_to)
    return " AND ".join(clauses), params


def list_products(page: int = 1, page_size: int = 20, category=None, secondhand=None,
                  date_from=None, date_to=None) -> dict:
    """Paged product list from SQLite. Dates are 'YYYY-MM-DD' and refer to updated_at."""
    init_db()
    page, page_size = max(page, 1), min(max(page_size, 1), 100)
    where, params = _build_where(category, secondhand, date_from, date_to)
    where_sql = f" WHERE {where}" if where else ""

    with get_connection() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM products{where_sql}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT {COLUMNS} FROM products{where_sql} ORDER BY product_id LIMIT ? OFFSET ?",
            params + [page_size, (page - 1) * page_size],
        ).fetchall()
    return {"total": total, "page": page, "page_size": page_size, "items": [format_product(r) for r in rows]}


def semantic_search(query: str, k: int = 5, category=None, secondhand=None) -> list[dict]:
    """Return the k products closest in meaning to the query, best first."""
    query = (query or "").strip()
    if not query:
        raise ValueError("Query must not be empty")
    if not FAISS_PATH.exists():
        raise FileNotFoundError("FAISS index not found - run the pipeline first")

    index = faiss.read_index(str(FAISS_PATH))
    filtered = category is not None or secondhand is not None
    n = min(index.ntotal, k * FILTER_OVERFETCH if filtered else k)

    # 1. FAISS: closest product_ids (IndexIDMap2 returns our ids directly; -1 means "no result")
    scores, ids = index.search(embed.embed_texts([query]), n)
    ranked = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
    if not ranked:
        return []

    # 2. SQLite: details for those ids, with the optional filters applied
    where, params = _build_where(category, secondhand)
    id_list = [pid for pid, _ in ranked]
    sql = f"SELECT {COLUMNS} FROM products WHERE product_id IN ({','.join('?' * len(id_list))})"
    if where:
        sql += f" AND {where}"
    with get_connection() as conn:
        rows = {row[0]: row for row in conn.execute(sql, id_list + params)}

    # 3. Keep FAISS's ranking order
    results = []
    for product_id, score in ranked:
        if product_id in rows:
            item = format_product(rows[product_id])
            item["score"] = round(score, 4)
            results.append(item)
            if len(results) == k:
                break
    return results
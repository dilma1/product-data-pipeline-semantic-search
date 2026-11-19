from pathlib import Path
import numpy as np
import pandas as pd
from datasets import load_dataset

OUT_DIR = Path("data/raw_batches")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def random_ts(n: int, start: str, end: str, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    s, e = pd.Timestamp(start).value // 10**9, pd.Timestamp(end).value // 10**9
    return pd.to_datetime(rng.integers(s, e, n), unit="s").strftime("%Y-%m-%d %H:%M:%S")


# 1. Stream 1,200 rows directly (fast, skips image column) and rename columns
print("Streaming 1,200 rows from Shopify/product-catalogue...")
stream = load_dataset("Shopify/product-catalogue", split="train", streaming=True)
stream = stream.select_columns([
    "product_title",
    "product_description",
    "ground_truth_brand",
    "ground_truth_category",
    "ground_truth_is_secondhand",
])
df = pd.DataFrame(list(stream.take(1200))).rename(columns={
    "ground_truth_brand": "brand",
    "ground_truth_category": "category",
    "ground_truth_is_secondhand": "is_secondhand",
})

# 2. Shuffle locally, keep 990 base products (450 + 250 + 290), assign sequential IDs
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
df = df.head(990).reset_index(drop=True)
df.insert(0, "product_id", range(1001, 1001 + len(df)))

# 3. Split into 3 chronological clean bases: 450 + 250 + 290
b1_clean = df.iloc[:450].copy()
b2_clean = df.iloc[450:700].copy()
b3_clean = df.iloc[700:990].copy()

b1_clean["updated_at"] = random_ts(len(b1_clean), "2026-09-01", "2026-09-30", seed=1)
b2_clean["updated_at"] = random_ts(len(b2_clean), "2026-10-01", "2026-10-31", seed=2)
b3_clean["updated_at"] = random_ts(len(b3_clean), "2026-11-01", "2026-11-19", seed=3)


def make_messy_batch(
    clean_df: pd.DataFrame,
    extra_dups: int,
    null_titles: int,
    null_counts: tuple[int, int, int],
    seed: int,
) -> pd.DataFrame:
    b = clean_df.copy()
    n_desc, n_brand, n_cat = null_counts

    # Varying nulls across title, description, brand, and category (all kept and imputed by Silver)
    title_null_idx = b.sample(null_titles, random_state=seed + 5).index
    b.loc[title_null_idx, "product_title"] = [
        None if i % 2 == 0 else "   " for i in range(null_titles)
    ]
    b.loc[b.sample(n_desc, random_state=seed + 1).index, "product_description"] = None
    b.loc[b.sample(n_brand, random_state=seed + 2).index, "brand"] = None
    b.loc[b.sample(n_cat, random_state=seed + 3).index, "category"] = None

    # Erratic leading/trailing whitespace on valid titles (trimmed by Silver)
    valid_title_pool = b.drop(index=title_null_idx)
    ws_idx = valid_title_pool.sample(35, random_state=seed + 4).index
    b.loc[ws_idx, "product_title"] = "  " + b.loc[ws_idx, "product_title"].astype(str) + " \n"


    rng = np.random.default_rng(seed)
    copies, rem = [], extra_dups
    for _, row in b.sample(frac=1.0, random_state=seed).iterrows():
        if rem <= 0:
            break
        n = min(int(rng.integers(1, 20)), rem)
        copies.append(pd.concat([row.to_frame().T] * n, ignore_index=True))
        rem -= n

    return pd.concat([b] + copies, ignore_index=True)


# 4. Build batches:
# Batch 1: 450 unique + 150 dups = 600 raw rows -> 450 clean
batch1 = make_messy_batch(
    b1_clean, extra_dups=150, null_titles=20, null_counts=(45, 35, 30), seed=10
).sample(frac=1.0, random_state=1)

# Batch 2: 250 unique + 150 dups = 400 raw rows -> 250 clean
batch2 = make_messy_batch(
    b2_clean, extra_dups=150, null_titles=12, null_counts=(30, 25, 20), seed=20
).sample(frac=1.0, random_state=2)

# Batch 3: 290 new + 10 updates (= 300 valid) + 85 dups + 15 stale = 400 raw rows -> 300 clean
updates = b1_clean.sample(10, random_state=30).copy()
updates["product_description"] = updates["product_description"].fillna("") + " Now available in new colors."
updates["updated_at"] = "2026-11-18 18:00:00"

stale = b1_clean.drop(index=updates.index).sample(15, random_state=31).copy()
stale["product_description"] = "OLD STALE RECORD - SHOULD BE SKIPPED"

b3_pool = pd.concat([b3_clean, updates], ignore_index=True)
batch3 = pd.concat(
    [make_messy_batch(b3_pool, extra_dups=85, null_titles=18, null_counts=(35, 20, 25), seed=30), stale],
    ignore_index=True,
).sample(frac=1.0, random_state=3)

# 5. Save CSVs
for i, b in enumerate([batch1, batch2, batch3], start=1):
    out_file = OUT_DIR / f"batch_{i}.csv"
    b.to_csv(out_file, index=False)
    print(f"{out_file} -> {len(b)} raw rows")
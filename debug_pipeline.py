"""
Diagnostic script — run this directly to see exactly where rows disappear.

Usage:
    python debug_pipeline.py
"""
from src.data_pipeline import fetch_ohlcv, fetch_market_context
from src.features import engineer_features

print("=" * 60)
print("STEP 1: Raw fetch")
raw = fetch_ohlcv("AAPL", force_refresh=True)
print(f"Raw AAPL shape: {raw.shape}")
print(raw.head(3))
print(f"Index dtype: {raw.index.dtype}, tz: {raw.index.tz}")

print("\n" + "=" * 60)
print("STEP 2: Market context fetch")
mkt = fetch_market_context(force_refresh=True)
print(f"Market context shape: {mkt.shape}")
print(mkt.head(3))
print(f"Index dtype: {mkt.index.dtype}, tz: {mkt.index.tz}")

print("\n" + "=" * 60)
print("STEP 3: Feature engineering WITHOUT market context")
processed_no_mkt = engineer_features(raw, market_context=None)
print(f"Processed (no market context) shape: {processed_no_mkt.shape}")

print("\n" + "=" * 60)
print("STEP 4: Feature engineering WITH market context")
processed_with_mkt = engineer_features(raw, market_context=mkt)
print(f"Processed (with market context) shape: {processed_with_mkt.shape}")

print("\n" + "=" * 60)
print("STEP 5: Join diagnostic (manual, to see overlap)")
import pandas as pd
overlap = raw.index.intersection(mkt.index)
print(f"Raw index range: {raw.index.min()} to {raw.index.max()}")
print(f"Mkt index range: {mkt.index.min()} to {mkt.index.max()}")
print(f"Overlapping dates: {len(overlap)} out of {len(raw)} raw rows")

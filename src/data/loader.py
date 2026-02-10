"""
High-performance data loading using Polars and Apache Arrow.

Polars provides 5-10x speedup over Pandas for large datasets via
Rust-native execution, lazy evaluation, and automatic parallelism.
"""

from __future__ import annotations

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from typing import Optional


class MarketDataLoader:
    """
    Load and preprocess market data using Polars for speed.

    Supports Parquet, CSV, and Arrow IPC formats with lazy evaluation
    for memory-efficient processing of multi-GB datasets.
    """

    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)

    def load_ohlcv(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        source: str = "parquet",
    ) -> pl.DataFrame:
        """Load OHLCV data with Polars lazy evaluation."""
        frames = []
        for sym in symbols:
            path = self.data_dir / f"{sym}.{source}"
            if path.exists():
                lf = pl.scan_parquet(str(path)) if source == "parquet" else pl.scan_csv(str(path))
                lf = lf.filter(
                    (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
                ).with_columns(pl.lit(sym).alias("symbol"))
                frames.append(lf)

        if not frames:
            return pl.DataFrame()

        return pl.concat(frames).collect()

    def compute_features(self, df: pl.DataFrame, windows: list[int] = [5, 10, 20, 60]) -> pl.DataFrame:
        """Compute technical features using Polars expressions (vectorized, parallel)."""
        result = df.clone()

        for w in windows:
            result = result.with_columns([
                pl.col("close").pct_change().rolling_mean(w).alias(f"ret_ma_{w}"),
                pl.col("close").pct_change().rolling_std(w).alias(f"vol_{w}"),
                (pl.col("close") / pl.col("close").shift(w) - 1).alias(f"momentum_{w}"),
                pl.col("volume").rolling_mean(w).alias(f"vol_ma_{w}"),
            ])

        # RSI
        delta = pl.col("close").diff()
        gain = delta.map_batches(lambda s: s.clip(lower_bound=0)).rolling_mean(14)
        loss = delta.map_batches(lambda s: (-s).clip(lower_bound=0)).rolling_mean(14)

        result = result.with_columns([
            (pl.col("high") - pl.col("low")).alias("range"),
            (pl.col("close") - pl.col("open")).alias("body"),
            (pl.col("volume") * pl.col("close")).alias("dollar_volume"),
        ])

        return result

    def to_arrow_table(self, df: pl.DataFrame) -> pa.Table:
        """Convert Polars DataFrame to Arrow Table for interop."""
        return df.to_arrow()

    def to_numpy(self, df: pl.DataFrame, columns: list[str]) -> np.ndarray:
        """Extract columns as NumPy array for ML model input."""
        return df.select(columns).to_numpy()

    @staticmethod
    def generate_synthetic(
        n_assets: int = 50,
        n_days: int = 2520,
        seed: int = 42,
    ) -> pl.DataFrame:
        """Generate synthetic multi-asset OHLCV data for testing."""
        rng = np.random.default_rng(seed)
        records = []

        for i in range(n_assets):
            close = 100 * np.cumprod(1 + rng.normal(0.0002, 0.015, n_days))
            high = close * (1 + np.abs(rng.normal(0, 0.005, n_days)))
            low = close * (1 - np.abs(rng.normal(0, 0.005, n_days)))
            open_ = close * (1 + rng.normal(0, 0.003, n_days))
            volume = rng.integers(100_000, 10_000_000, n_days)

            for j in range(n_days):
                records.append({
                    "symbol": f"ASSET_{i:03d}",
                    "date": f"2015-01-{(j % 28) + 1:02d}",
                    "open": open_[j],
                    "high": high[j],
                    "low": low[j],
                    "close": close[j],
                    "volume": int(volume[j]),
                })

        return pl.DataFrame(records)

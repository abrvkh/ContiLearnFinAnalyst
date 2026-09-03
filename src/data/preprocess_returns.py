#!/usr/bin/env python3
"""Preprocess return panels.

Currently supported:
- clean raw returns by dropping absurd names, then clipping causal 30-day
  standardized returns
- demean cleaned returns by subtracting the cross-sectional market mean on
  each date

Example
-------
::

    uv run python src/data/preprocess_returns.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def clean_returns(
    returns: pd.DataFrame,
    absurd_threshold: float = 10.0,
    max_absurd_fraction: float = 0.10,
    window: int = 30,
    z_clip: float = 2.0,
) -> pd.DataFrame:
    """Drop absurd names, then clip each ticker using a causal rolling z-score.

    A ticker is dropped entirely if more than ``max_absurd_fraction`` of its
    non-null observations have absolute value above ``absurd_threshold``.
    For the remaining tickers, returns are normalized using a rolling mean/std
    over ``window`` periods, both shifted by one period for causality. The
    resulting z-scores are clipped to ``[-z_clip, z_clip]`` and stored
    directly.
    """

    if absurd_threshold <= 0:
        raise ValueError("absurd_threshold must be positive")
    if not 0 <= max_absurd_fraction <= 1:
        raise ValueError("max_absurd_fraction must be between 0 and 1")
    if window < 2:
        raise ValueError("window must be at least 2")
    if z_clip <= 0:
        raise ValueError("z_clip must be positive")

    absolute = returns.abs()
    absurd_mask = absolute > absurd_threshold
    non_null_count = returns.notna().sum(axis=0)
    absurd_count = absurd_mask.sum(axis=0)
    absurd_fraction = absurd_count.divide(non_null_count.where(non_null_count > 0))
    keep_columns = absurd_fraction[(absurd_fraction <= max_absurd_fraction) | absurd_fraction.isna()].index

    cleaned = returns.loc[:, keep_columns].copy()
    rolling_mean = cleaned.rolling(window=window, min_periods=20).mean().shift(1)
    rolling_std = cleaned.rolling(window=window, min_periods=20).std(ddof=1).shift(1)
    z_score = cleaned.subtract(rolling_mean).divide(rolling_std)
    cleaned = z_score.clip(lower=-z_clip, upper=z_clip)
    cleaned.index.name = returns.index.name
    return cleaned


def demean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Subtract the equal-weight market return from each date's cross-section."""

    market_mean = returns.mean(axis=1, skipna=True)
    demeaned = returns.subtract(market_mean, axis=0)
    demeaned.index.name = returns.index.name
    return demeaned


def main() -> None:
    input_path = Path("data/financial-reports-sec/market_data/return.parquet")
    clean_path = Path("data/financial-reports-sec/market_data/return_clean.parquet")
    demeaned_path = Path("data/financial-reports-sec/market_data/returns_demeaned.parquet")

    returns = pd.read_parquet(input_path)
    cleaned = clean_returns(returns)
    demeaned = demean_returns(cleaned)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(clean_path)
    demeaned.to_parquet(demeaned_path)
    print(clean_path)
    print(demeaned_path)


if __name__ == "__main__":
    main()

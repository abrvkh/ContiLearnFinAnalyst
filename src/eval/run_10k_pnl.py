#!/usr/bin/env python3
"""Compute simple 10-K strategy PnL from prediction and return panels.

The script expects Parquet panels with a date index and ticker columns.
Predictions are loaded from ``predictions/<run_name>.parquet`` by default and
returns are loaded from ``data/financial-reports-sec/market_data/return.parquet``.

Workflow:
1. Load prediction and return DataFrames from Parquet.
2. Reindex predictions to the returns date index and align on common tickers.
3. Optionally forward-fill predictions after reindexing.
4. Compute raw positions as ``predictions.shift(2).mul(returns)``.
5. Sum across tickers to get daily gain.
6. Compute annualized Sharpe and t-stat.
7. Save a cumulative gain plot next to the prediction file.

The CLI prints only the annualized Sharpe ratio.

Examples
--------
Run the baseline PnL for ``predictions/my_run.parquet``::

    uv run python src/eval/run_10k_pnl.py --run-name my_run

Run with cross-sectional ranking::

    uv run python src/eval/run_10k_pnl.py \
        --run-name my_run \
        --postprocess cross-sectional-rank

Run with sector-aware cross-sectional ranking plus 252-day predictor
normalization::

    uv run python src/eval/run_10k_pnl.py \
        --run-name my_run \
        --postprocess cross-sectional-rank \
        --use-sectors \
        --postprocess normalize-predictor
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd


def resolve_predictions_path(predictions_dir: Path, run_name: str) -> Path:
    """Resolve the default prediction panel path from the run name."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("run name may contain only letters, numbers, '.', '_' and '-'")
    return predictions_dir / f"{run_name}.parquet"


def load_sector_mapping(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict):
        raise ValueError(f"invalid sector mapping: {path}")
    return {
        str(sector): [str(ticker) for ticker in tickers]
        for sector, tickers in mapping.items()
        if isinstance(tickers, list)
    }


def cross_sectional_rank(predictions: pd.DataFrame, sector_mapping: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Cross-sectional z-score, optionally within each sector."""

    result = predictions.copy()
    if sector_mapping is None:
        demeaned = predictions.subtract(predictions.mean(axis=1, skipna=True), axis=0)
        scale = demeaned.std(axis=1, skipna=True, ddof=1).replace(0.0, math.nan)
        return demeaned.divide(scale, axis=0).fillna(0.0)

    touched: set[str] = set()
    for tickers in sector_mapping.values():
        columns = [ticker for ticker in tickers if ticker in predictions.columns]
        if not columns:
            continue
        group = predictions.loc[:, columns]
        demeaned = group.subtract(group.mean(axis=1, skipna=True), axis=0)
        scale = demeaned.std(axis=1, skipna=True, ddof=1).replace(0.0, math.nan)
        result.loc[:, columns] = demeaned.divide(scale, axis=0).fillna(0.0)
        touched.update(columns)

    remaining = [ticker for ticker in predictions.columns if ticker not in touched]
    if remaining:
        group = predictions.loc[:, remaining]
        demeaned = group.subtract(group.mean(axis=1, skipna=True), axis=0)
        scale = demeaned.std(axis=1, skipna=True, ddof=1).replace(0.0, math.nan)
        result.loc[:, remaining] = demeaned.divide(scale, axis=0).fillna(0.0)
    return result


def normalize_predictor(
    predictions: pd.DataFrame,
    clip: float = 2.0,
) -> pd.DataFrame:
    """Normalize each ticker by a causal past std and clip."""

    if clip <= 0:
        raise ValueError("clip must be positive")
    scale = predictions.rolling(window=252, min_periods=2).std(ddof=1).shift(1)
    normalized = predictions.divide(scale).replace([math.inf, -math.inf], math.nan)
    return normalized.clip(lower=-clip, upper=clip)


def apply_postprocess(
    predictions: pd.DataFrame,
    postprocessings: list[str],
    sector_mapping: dict[str, list[str]] | None = None,
    normalize_clip: float = 2.0,
) -> pd.DataFrame:
    """Apply predictor post-processing steps in order."""

    result = predictions.copy()
    for step in postprocessings:
        if step == "cross-sectional-rank":
            result = cross_sectional_rank(result, sector_mapping=sector_mapping)
        elif step == "normalize-predictor":
            result = normalize_predictor(result, clip=normalize_clip)
        else:
            raise ValueError(f"unknown postprocess step: {step}")
    return result


def align_panels(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    ffill_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align on return dates and common tickers, with optional forward-fill."""

    if ffill_limit is not None and ffill_limit < 0:
        raise ValueError("ffill_limit must be non-negative")

    tickers = [ticker for ticker in predictions.columns if ticker in returns.columns]
    if not tickers:
        raise ValueError("predictions and returns have no ticker columns in common")

    aligned_predictions = predictions.loc[:, tickers].reindex(returns.index)
    if ffill_limit is not None:
        aligned_predictions = aligned_predictions.ffill(limit=ffill_limit)
    aligned_returns = returns.loc[:, tickers]
    return aligned_predictions, aligned_returns


def compute_pnl(predictions: pd.DataFrame, returns: pd.DataFrame, lag: int = 2) -> pd.Series:
    """Compute daily PnL from lagged predictions and realized returns."""
    if lag < 0:
        raise ValueError("lag must be non-negative")
    raw_position = predictions.shift(lag).mul(returns)
    pnl = raw_position.sum(axis=1, skipna=True)
    return pnl


def clip_returns(returns: pd.DataFrame, clip: float | None = None) -> pd.DataFrame:
    """Clip realized returns symmetrically to reduce outlier impact."""

    if clip is None:
        return returns
    if clip <= 0:
        raise ValueError("clip must be positive")
    return returns.clip(lower=-clip, upper=clip)


def annualized_sharpe(pnl: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized Sharpe as sqrt(252) * mean / std."""
    values = pnl.dropna()
    standard_deviation = values.std(ddof=1)
    if len(values) < 2 or standard_deviation == 0 or math.isnan(float(standard_deviation)):
        return math.nan
    return float(math.sqrt(periods_per_year) * values.mean() / standard_deviation)


def t_stat(pnl: pd.Series) -> float:
    """Compute the t-statistic of the daily mean pnl."""
    values = pnl.dropna()
    standard_deviation = values.std(ddof=1)
    if len(values) < 2 or standard_deviation == 0 or math.isnan(float(standard_deviation)):
        return math.nan
    return float(math.sqrt(len(values)) * values.mean() / standard_deviation)


def plot_gain(gain: pd.Series, run_name: str, output_path: Path) -> None:
    """Save a cumulative gain plot with Sharpe and t-stat in the legend."""

    cumulative = gain.fillna(0.0).cumsum()
    sharpe = annualized_sharpe(gain)
    statistic = t_stat(gain)
    label = f"{run_name} | sharpe={sharpe:.3f} | tstat={statistic:.3f}"

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(cumulative.index, cumulative, linewidth=2.0, color="#145DA0", label=label)
    axis.axhline(0.0, color="#4B5563", linewidth=0.8)
    axis.set_title(run_name, loc="left")
    axis.set_ylabel("Cumulative gain")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    figure.savefig(temporary, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    temporary.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--predictions-dir", type=Path, default=Path("predictions"))
    parser.add_argument(
        "--returns",
        type=Path,
        default=Path("data/financial-reports-sec/market_data/return.parquet"),
    )
    parser.add_argument(
        "--sector-mapping",
        type=Path,
        default=Path("data/financial-reports-sec/market_data/sector_companies.json"),
    )
    parser.add_argument(
        "--postprocess",
        action="append",
        default=[],
        choices=("cross-sectional-rank", "normalize-predictor"),
    )
    parser.add_argument("--use-sectors", action="store_true")
    parser.add_argument("--normalize-clip", type=float, default=2.0)
    parser.add_argument(
        "--ffill-limit",
        type=int,
        default=None,
        help="Forward-fill predictions after reindexing to returns dates. "
        "Use 0 to disable filling; omit for no filling.",
    )
    parser.add_argument(
        "--clip-returns",
        type=float,
        default=None,
        help="Optional symmetric clip for realized returns, applied after alignment and before PnL.",
    )
    parser.add_argument("--lag", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = resolve_predictions_path(args.predictions_dir, args.run_name)
    predictions = pd.read_parquet(predictions_path)
    returns = pd.read_parquet(args.returns)
    ffill_limit = None if args.ffill_limit is None or args.ffill_limit == 0 else args.ffill_limit
    predictions, returns = align_panels(predictions, returns, ffill_limit=ffill_limit)
    sector_mapping = load_sector_mapping(args.sector_mapping) if args.use_sectors else None
    predictions = apply_postprocess(
        predictions,
        args.postprocess,
        sector_mapping=sector_mapping,
        normalize_clip=args.normalize_clip,
    )
    returns = clip_returns(returns, clip=args.clip_returns)
    pnl = compute_pnl(predictions, returns, lag=args.lag)

    plot_path = predictions_path.with_name(f"{args.run_name}_pnl.png")
    plot_gain(pnl, args.run_name, plot_path)
    print(f"{annualized_sharpe(pnl):.10f}")

if __name__ == "__main__":
    main()

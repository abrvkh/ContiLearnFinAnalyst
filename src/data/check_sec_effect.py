#!/usr/bin/env python3
"""Measure average absolute returns around 10-K filing dates.

The script streams per-issuer filing dates from the SEC JSONL shards, maps each
issuer ticker to the Yahoo ticker convention used in ``return.parquet``, and then
computes the mean absolute return profile on trading-day offsets around each
filing date.

Offset ``0`` is the first trading day on or after the filing date.  This keeps
weekend and holiday filings aligned to the next available return observation.

Example
-------
::

    uv run python src/data/check_sec_effect.py --run-name sec_effect_v1
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
from matplotlib.ticker import PercentFormatter  # type: ignore
from tqdm import tqdm

from src.data.build_10k_market_data import sec_ticker_to_yahoo


_HEADER_RE = re.compile(
    rb'^\{"cik":"(?P<cik>[^"]*)","name":"(?P<name>(?:[^"\\]|\\.)*)",'
    rb'"tickers":(?P<tickers>\[[^\]]*\]),"exchanges":(?P<exchanges>\[[^\]]*\])'
)
_FILING_DATE_RE = re.compile(rb'"filingDate":"(\d{4}-\d{2}-\d{2})"')


@dataclass(frozen=True)
class FilingEvent:
    yahoo_ticker: str
    filing_date: date


def extract_filing_events(
    dataset_dir: Path,
    splits: Iterable[str] = ("train", "test"),
) -> list[FilingEvent]:
    """Stream all ticker/date filing events from the SEC shards."""

    events: set[tuple[str, date]] = set()
    found_any_shard = False
    for split in splits:
        shard_paths = sorted((dataset_dir / split).glob("*.jsonl"))
        if not shard_paths:
            continue
        found_any_shard = True

        for shard_path in tqdm(shard_paths, desc=f"Reading {split} shards", unit="shard"):
            with shard_path.open("rb") as handle:
                for line_number, line in enumerate(handle, start=1):
                    header = _HEADER_RE.search(line[:65536])
                    if header is None:
                        raise ValueError(f"Could not parse header at {shard_path}:{line_number}")
                    tickers = json.loads(header.group("tickers"))
                    filing_dates = [date.fromisoformat(x.decode("ascii")) for x in _FILING_DATE_RE.findall(line)]
                    if not filing_dates:
                        raise ValueError(f"No filingDate values at {shard_path}:{line_number}")
                    yahoo_tickers = [sec_ticker_to_yahoo(str(ticker)) for ticker in tickers]
                    for yahoo_ticker in yahoo_tickers:
                        for filing_date in filing_dates:
                            events.add((yahoo_ticker, filing_date))

    if not found_any_shard:
        raise FileNotFoundError(f"No JSONL shards found under {dataset_dir}")
    return [FilingEvent(yahoo_ticker, filing_date) for yahoo_ticker, filing_date in sorted(events)]


def load_returns(path: Path) -> Any:
    """Load the wide Parquet return panel used elsewhere in this repo."""

    if not path.exists():
        raise FileNotFoundError(f"Returns file does not exist: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        returns = pd.read_parquet(path)
    else:
        returns = pd.read_csv(path, index_col=0)
    returns.index = pd.to_datetime(returns.index, errors="raise", utc=True).tz_localize(None)
    returns.columns = returns.columns.astype(str)
    returns = returns.apply(pd.to_numeric, errors="raise").sort_index()
    if not returns.index.is_unique or not returns.columns.is_unique:
        raise ValueError("returns index and columns must be unique")
    return returns


def compute_abs_return_profile(
    returns: Any,
    events: Iterable[FilingEvent],
    window: int,
) -> Any:
    """Average absolute returns by trading-day offset around each filing."""

    if window < 0:
        raise ValueError("window must be non-negative")

    events = list(events)
    ticker_to_column = {ticker: index for index, ticker in enumerate(returns.columns)}
    values_by_offset: dict[int, list[float]] = {offset: [] for offset in range(-window, window + 1)}
    used_events = 0
    skipped_missing_ticker = 0
    skipped_after_panel = 0

    for event in tqdm(events, desc="Computing event profile", unit="event"):
        column_index = ticker_to_column.get(event.yahoo_ticker)
        if column_index is None:
            skipped_missing_ticker += 1
            continue

        anchor = returns.index.searchsorted(pd.Timestamp(event.filing_date))
        if anchor >= len(returns.index):
            skipped_after_panel += 1
            continue

        used_events += 1
        for offset in range(-window, window + 1):
            row_index = anchor + offset
            if row_index < 0 or row_index >= len(returns.index):
                continue
            value = returns.iat[row_index, column_index]
            if pd.notna(value):
                values_by_offset[offset].append(abs(float(value)))

    profile = pd.DataFrame(
        {
            "relative_day": list(range(-window, window + 1)),
            "mean_abs_return": [
                math.nan if not values_by_offset[offset] else sum(values_by_offset[offset]) / len(values_by_offset[offset])
                for offset in range(-window, window + 1)
            ],
            "num_observations": [len(values_by_offset[offset]) for offset in range(-window, window + 1)],
        }
    )
    profile["used_events"] = used_events
    profile["skipped_missing_ticker"] = skipped_missing_ticker
    profile["skipped_after_panel"] = skipped_after_panel
    return profile


def plot_profile(profile: Any, run_name: str, output_path: Path) -> None:
    """Save a simple event-study plot of mean absolute return by offset."""

    plt.style.use("default")
    figure, axis = plt.subplots(figsize=(10.5, 5.6))
    axis.plot(
        profile["relative_day"],
        profile["mean_abs_return"],
        color="#145DA0",
        linewidth=2.0,
        marker="o",
        markersize=4.5,
    )
    axis.axvline(0, color="#C0392B", linewidth=1.2, linestyle="--")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.7)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.5, alpha=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_xlabel("Trading days relative to filing date")
    axis.set_ylabel("Mean absolute return")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.set_title(run_name, loc="left", fontsize=16, fontweight="bold", pad=14)
    axis.text(
        0.0,
        1.01,
        f"Offset 0 = first trading day on or after filing date | events used: {int(profile['used_events'].iloc[0])}",
        transform=axis.transAxes,
        color="#4B5563",
        fontsize=9.5,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    figure.savefig(temporary, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    temporary.replace(output_path)


def _write_csv(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_csv(temporary, index=False)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/financial-reports-sec/large"),
    )
    parser.add_argument(
        "--returns",
        type=Path,
        default=Path("data/financial-reports-sec/market_data/return.parquet"),
    )
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--run-name", default="sec_effect")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        raise ValueError("run name may contain only letters, numbers, '.', '_' and '-'")

    returns = load_returns(args.returns)
    events = extract_filing_events(args.dataset_dir)
    profile = compute_abs_return_profile(returns, events, args.window)

    prefix = args.results_dir / args.run_name
    csv_path = prefix.with_name(f"{args.run_name}.csv")
    plot_path = prefix.with_name(f"{args.run_name}.png")
    _write_csv(profile, csv_path)
    plot_profile(profile, args.run_name, plot_path)

    peak_row = profile.loc[profile["mean_abs_return"].idxmax()]
    print(f"Events used: {int(profile['used_events'].iloc[0])}")
    print(f"Skipped missing ticker: {int(profile['skipped_missing_ticker'].iloc[0])}")
    print(f"Skipped after panel end: {int(profile['skipped_after_panel'].iloc[0])}")
    print(
        "Peak mean absolute return: "
        f"day {int(peak_row['relative_day'])} -> {float(peak_row['mean_abs_return']):.4%}"
    )
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()

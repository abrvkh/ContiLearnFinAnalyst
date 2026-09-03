#!/usr/bin/env python3
"""Download and save daily equity returns for the 10-K universe.

The script first extracts the train/test equity universe and the exact filing-date
span without deserializing the very large report bodies.  It then downloads one
resumable CSV per ticker, including Yahoo's corporate-action columns, and writes
an adjusted-close return panel plus current sector/industry metadata.  Prediction
normalisation and PnL analysis live in ``src/eval/run_10k_pnl.py``.

Outputs
-------
By default the script writes the following files under
``data/financial-reports-sec/market_data``:

``equity_universe.csv``
    Train/test ticker universe extracted from the SEC JSONL shards.

``prices/*.csv``
    One resumable Yahoo history file per ticker with daily prices and
    corporate-action columns.

``equity_metadata.csv``
    Per-ticker download status plus current Yahoo metadata such as sector and
    industry.

``sector_companies.json``
    JSON mapping of ``sector -> [Yahoo tickers]`` built from the metadata.

``return.parquet``
    Wide daily adjusted-close return panel stored as a DataFrame with ``date``
    index and Yahoo tickers as columns.

The price/return workflow and the sector-metadata workflow can be run together
or separately.  This is useful when Yahoo rate limits metadata requests more
aggressively than price-history requests.

Sector data is stored in two places: ``equity_metadata.csv`` contains one row
per ticker with current Yahoo ``sector`` and ``industry`` fields, while
``sector_companies.json`` stores the derived ``sector -> [Yahoo tickers]``
mapping used by downstream cross-sectional normalisation.  These are current
classifications, not point-in-time sector labels at the filing date.

Examples
--------
Download/resume price history and build ``return.parquet`` only::

    pip install yfinance
    uv run python src/data/build_10k_market_data.py --skip-metadata

Fetch sector metadata only and refresh ``equity_metadata.csv`` plus
``sector_companies.json``::

    uv run python src/data/build_10k_market_data.py --skip-prices

Run both workflows together in one pass::

    uv run python src/data/build_10k_market_data.py

Yahoo Finance is a current-symbol data source.  Delisted/reused tickers and
historical sector classifications require a point-in-time reference data source
and are recorded as failures rather than silently substituted.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


_HEADER_RE = re.compile(
    rb'^\{"cik":"(?P<cik>[^"]*)","name":"(?P<name>(?:[^"\\]|\\.)*)",'
    rb'"tickers":(?P<tickers>\[[^\]]*\]),"exchanges":(?P<exchanges>\[[^\]]*\])'
)
_FILING_DATE_RE = re.compile(rb'"filingDate":"(\d{4}-\d{2}-\d{2})"')


@dataclass(frozen=True)
class FilingSpan:
    start: date
    end: date


def _decode_json_string(raw: bytes) -> str:
    return json.loads(b'"' + raw + b'"')


def sec_ticker_to_yahoo(ticker: str) -> str:
    """Convert the common SEC share-class separator to Yahoo's convention."""

    return ticker.strip().upper().replace(".", "-")


def extract_universe(
    dataset_dir: Path,
    splits: Sequence[str] = ("train", "test"),
) -> tuple[list[dict[str, Any]], FilingSpan]:
    """Stream issuer/ticker metadata from the downloaded JSONL shards."""

    rows: list[dict[str, Any]] = []
    global_start: Optional[date] = None
    global_end: Optional[date] = None

    for split in splits:
        shard_paths = sorted((dataset_dir / split).glob("*.jsonl"))
        if not shard_paths:
            raise FileNotFoundError(f"No JSONL shards found for split {split!r} in {dataset_dir}")

        for shard_path in shard_paths:
            with shard_path.open("rb") as handle:
                for line_number, line in enumerate(handle, start=1):
                    header = _HEADER_RE.search(line[:65536])
                    if header is None:
                        raise ValueError(f"Could not parse header at {shard_path}:{line_number}")

                    tickers = json.loads(header.group("tickers"))
                    exchanges = json.loads(header.group("exchanges"))
                    filing_dates = [date.fromisoformat(x.decode("ascii")) for x in _FILING_DATE_RE.findall(line)]
                    if not filing_dates:
                        raise ValueError(f"No filingDate values at {shard_path}:{line_number}")

                    issuer_start, issuer_end = min(filing_dates), max(filing_dates)
                    global_start = issuer_start if global_start is None else min(global_start, issuer_start)
                    global_end = issuer_end if global_end is None else max(global_end, issuer_end)
                    name = _decode_json_string(header.group("name"))
                    cik = header.group("cik").decode("ascii")

                    # An SEC issuer can have multiple listed share classes.  Keep
                    # each as its own equity while retaining the common CIK.
                    for index, ticker in enumerate(tickers):
                        exchange = exchanges[index] if index < len(exchanges) else ""
                        rows.append(
                            {
                                "split": split,
                                "cik": cik,
                                "name": name,
                                "sec_ticker": ticker,
                                "yahoo_ticker": sec_ticker_to_yahoo(ticker),
                                "exchange": exchange,
                                "first_filing_date": issuer_start.isoformat(),
                                "last_filing_date": issuer_end.isoformat(),
                                "filing_count": len(filing_dates),
                            }
                        )

    if global_start is None or global_end is None:
        raise ValueError("The selected splits contained no filings")
    rows.sort(key=lambda row: (row["sec_ticker"], row["split"], row["cik"]))
    return rows, FilingSpan(global_start, global_end)


def write_csv(rows: Iterable[Mapping[str, Any]], path: Path, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_sector_company_mapping(
    metadata: Iterable[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Build a sorted ``sector -> [Yahoo tickers]`` mapping from metadata rows."""

    sectors: dict[str, set[str]] = {}
    for row in metadata:
        sector = str(row.get("sector", "")).strip()
        ticker = str(row.get("yahoo_ticker", "")).strip()
        if sector and ticker:
            sectors.setdefault(sector, set()).add(ticker)
    return {sector: sorted(tickers) for sector, tickers in sorted(sectors.items())}


def _unique_tickers(universe: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for row in universe:
        ticker = str(row["yahoo_ticker"])
        if ticker not in seen:
            seen.add(ticker)
            result.append(
                {
                    "sec_ticker": str(row["sec_ticker"]),
                    "yahoo_ticker": ticker,
                    "cik": str(row["cik"]),
                    "dataset_name": str(row["name"]),
                }
            )
    return result


def _ticker_price_path(prices_dir: Path, ticker: str) -> Path:
    safe_ticker = re.sub(r"[^A-Z0-9_.=-]", "_", ticker, flags=re.IGNORECASE)
    return prices_dir / f"{safe_ticker}.csv"


def _is_rate_limit_error(exc: Exception) -> bool:
    return type(exc).__name__ == "YFRateLimitError" or "Too Many Requests" in str(exc)


def _call_with_retries(
    operation: Any,
    *,
    retries: int,
    base_delay_seconds: float,
    ticker: str,
    label: str,
) -> Any:
    attempt = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            if attempt >= retries or not _is_rate_limit_error(exc):
                raise
            delay = base_delay_seconds * (2 ** attempt)
            print(
                f"{ticker}: {label} rate-limited on attempt {attempt + 1}/{retries + 1}; "
                f"sleeping {delay:.1f}s before retry",
                flush=True,
            )
            time.sleep(delay)
            attempt += 1


def _load_cached_history(price_path: Path) -> Any:
    import pandas as pd  # type: ignore

    history = pd.read_csv(price_path)
    if history.empty:
        raise ValueError("cached history is empty")
    return history


def _fetch_history(
    instrument: Any,
    span: FilingSpan,
    *,
    ticker: str,
    retries: int,
    base_delay_seconds: float,
) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "The default dtype for empty Series will be 'object' instead of "
                "'float64' in a future version. Specify a dtype explicitly to "
                "silence this warning."
            ),
            category=DeprecationWarning,
            module=r"yfinance\.scrapers\.history",
        )
        return _call_with_retries(
            lambda: instrument.history(
                start=span.start.isoformat(),
                end=(span.end + timedelta(days=1)).isoformat(),  # Yahoo end is exclusive.
                interval="1d",
                auto_adjust=False,
                actions=True,
                repair=True,
                keepna=False,
            ),
            retries=retries,
            base_delay_seconds=base_delay_seconds,
            ticker=ticker,
            label="history download",
        )


def _fetch_info(
    instrument: Any,
    *,
    ticker: str,
    retries: int,
    base_delay_seconds: float,
) -> tuple[dict[str, Any], str]:
    try:
        info = _call_with_retries(
            lambda: instrument.get_info() or {},
            retries=retries,
            base_delay_seconds=base_delay_seconds,
            ticker=ticker,
            label="metadata lookup",
        )
        return info, ""
    except Exception as exc:
        return {}, f"metadata lookup failed: {type(exc).__name__}: {exc}"


def _download_one(
    item: Mapping[str, str],
    span: FilingSpan,
    prices_dir: Path,
    overwrite: bool,
    retries: int,
    base_delay_seconds: float,
    fetch_prices: bool,
    fetch_metadata: bool,
) -> dict[str, Any]:
    import pandas as pd  # type: ignore
    import yfinance as yf  # type: ignore

    ticker = item["yahoo_ticker"]
    price_path = _ticker_price_path(prices_dir, ticker)
    yf.config.debug.hide_exceptions = False

    try:
        instrument = yf.Ticker(ticker) if (fetch_prices or fetch_metadata) else None
        history = None
        status = "skipped"
        if fetch_prices:
            if price_path.exists() and not overwrite:
                history = _load_cached_history(price_path)
                status = "cached"
            else:
                history = _fetch_history(
                    instrument,
                    span,
                    ticker=ticker,
                    retries=retries,
                    base_delay_seconds=base_delay_seconds,
                )
                if history is None or history.empty:
                    raise ValueError("Yahoo returned no price history")
                history = history.reset_index()
                date_column = "Date" if "Date" in history.columns else history.columns[0]
                history[date_column] = pd.to_datetime(history[date_column], utc=True).dt.date
                history.columns = [str(column).strip().lower().replace(" ", "_") for column in history.columns]
                prices_dir.mkdir(parents=True, exist_ok=True)
                temporary = price_path.with_suffix(".csv.tmp")
                history.to_csv(temporary, index=False)
                temporary.replace(price_path)
                status = "ok"
        elif price_path.exists():
            history = _load_cached_history(price_path)
            status = "cached"

        info: dict[str, Any] = {}
        info_error = ""
        if fetch_metadata:
            info, info_error = _fetch_info(
                instrument,
                ticker=ticker,
                retries=retries,
                base_delay_seconds=base_delay_seconds,
            )

        rows = len(history) if history is not None else 0
        first_price_date = history.iloc[0, 0] if history is not None and rows else ""
        last_price_date = history.iloc[-1, 0] if history is not None and rows else ""
        return {
            **item,
            "status": status,
            "rows": rows,
            "first_price_date": first_price_date,
            "last_price_date": last_price_date,
            "quote_type": info.get("quoteType", ""),
            "yahoo_name": info.get("longName") or info.get("shortName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "error": info_error,
        }
    except Exception as exc:  # Continue the large resumable job and expose failures.
        return {**item, "status": "failed", "rows": 0, "error": f"{type(exc).__name__}: {exc}"}


def download_market_data(
    universe: Sequence[Mapping[str, Any]],
    span: FilingSpan,
    output_dir: Path,
    workers: int = 1,
    overwrite: bool = False,
    limit: Optional[int] = None,
    retries: int = 4,
    base_delay_seconds: float = 5.0,
    fetch_prices: bool = True,
    fetch_metadata: bool = True,
) -> list[dict[str, Any]]:
    """Download prices/actions and current Yahoo metadata, one ticker at a time."""

    try:
        import yfinance  # noqa: F401  # type: ignore
    except ImportError as exc:
        raise RuntimeError("yfinance is required for downloads: pip install yfinance") from exc
    if workers < 1:
        raise ValueError("workers must be at least 1")

    items = _unique_tickers(universe)
    if limit is not None:
        items = items[:limit]
    prices_dir = output_dir / "prices"
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_one,
                item,
                span,
                prices_dir,
                overwrite,
                retries,
                base_delay_seconds,
                fetch_prices,
                fetch_metadata,
            ): item
            for item in items
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(f"[{completed}/{len(futures)}] {result['yahoo_ticker']}: {result['status']}", flush=True)
    results.sort(key=lambda row: row["yahoo_ticker"])
    return results


def build_return_panel(
    universe: Sequence[Mapping[str, Any]],
    prices_dir: Path,
) -> Any:
    """Build a date-by-ticker panel of total returns on a shared calendar.

    Yahoo's adjusted close accounts for both stock splits and cash dividends.
    Adjusted prices are first aligned to the union of observed trading dates and
    only then converted to returns.  Consequently, a missing price remains NaN
    and the following date does not become a disguised multi-day return.

    The individual price files retain close, dividends, and stock-split columns
    as an auditable resumable download cache.  A ticker with a missing or invalid
    cache file is reported and omitted.
    """

    import pandas as pd  # type: ignore

    adjusted_prices: list[Any] = []
    for item in _unique_tickers(universe):
        ticker = item["yahoo_ticker"]
        path = _ticker_price_path(prices_dir, ticker)
        if not path.exists():
            print(f"Returns skipped for {ticker}: no price file", flush=True)
            continue
        try:
            history = pd.read_csv(path)
            if "date" not in history.columns or "adj_close" not in history.columns:
                raise ValueError("expected date and adj_close columns")
            dates = pd.to_datetime(history["date"], errors="raise", utc=True).dt.tz_localize(None)
            adjusted_close = pd.to_numeric(history["adj_close"], errors="coerce")
            prices = pd.Series(adjusted_close.to_numpy(), index=dates, name=ticker)
            if prices.index.has_duplicates:
                raise ValueError("duplicate dates")
            prices = prices.sort_index()
            adjusted_prices.append(prices)
        except Exception as exc:
            print(f"Returns skipped for {ticker}: {type(exc).__name__}: {exc}", flush=True)

    if not adjusted_prices:
        raise RuntimeError(f"No usable adjusted-close histories found in {prices_dir}")

    # Concatenation creates the shared observed-trading-day grid.  Calculating
    # returns afterward prevents pct_change from jumping across a missing date.
    price_panel = pd.concat(adjusted_prices, axis=1).sort_index()
    returns = price_panel.pct_change(fill_method=None).dropna(how="all")
    returns.index.name = "date"
    return returns


def save_return_panel(returns: Any, path: Path) -> None:
    """Atomically save a return panel as Parquet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    returns.to_parquet(temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/financial-reports-sec/large")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/financial-reports-sec/market_data")
    )
    parser.add_argument("--universe-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="Download only the first N tickers (smoke tests)")
    parser.add_argument("--skip-prices", action="store_true", help="Do not fetch price history")
    parser.add_argument("--skip-metadata", action="store_true", help="Do not fetch sector/industry metadata")
    parser.add_argument("--retries", type=int, default=4, help="Retries after Yahoo rate limits")
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=5.0,
        help="Base sleep for exponential backoff after Yahoo rate limits",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_prices and args.skip_metadata:
        raise ValueError("At least one of prices or metadata must be requested")

    universe, span = extract_universe(args.dataset_dir)
    universe_fields = (
        "split", "cik", "name", "sec_ticker", "yahoo_ticker", "exchange",
        "first_filing_date", "last_filing_date", "filing_count",
    )
    write_csv(universe, args.output_dir / "equity_universe.csv", universe_fields)
    print(f"Equities: {len(_unique_tickers(universe))}")
    print(f"SEC filing span: {span.start} through {span.end} (inclusive)")
    print(f"Universe: {args.output_dir / 'equity_universe.csv'}")
    if args.universe_only:
        return

    metadata = download_market_data(
        universe,
        span,
        args.output_dir,
        args.workers,
        args.overwrite,
        args.limit,
        args.retries,
        args.retry_base_seconds,
        fetch_prices=not args.skip_prices,
        fetch_metadata=not args.skip_metadata,
    )
    metadata_fields = (
        "sec_ticker", "yahoo_ticker", "cik", "dataset_name", "status", "rows",
        "first_price_date", "last_price_date", "quote_type", "yahoo_name", "sector",
        "industry", "error",
    )
    # Normalize sparse failure/cached rows for DictWriter's fixed schema.
    normalized = [{field: row.get(field, "") for field in metadata_fields} for row in metadata]
    write_csv(normalized, args.output_dir / "equity_metadata.csv", metadata_fields)
    sector_companies = build_sector_company_mapping(normalized)
    sector_path = args.output_dir / "sector_companies.json"
    temporary = sector_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(sector_companies, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(sector_path)
    returns_path = args.output_dir / "return.parquet"
    if not args.skip_prices:
        returns = build_return_panel(universe, args.output_dir / "prices")
        save_return_panel(returns, returns_path)
        print(f"Returns: {returns_path} ({returns.shape[0]} dates x {returns.shape[1]} tickers)")
    print(f"Metadata/status: {args.output_dir / 'equity_metadata.csv'}")
    print(f"Sector mapping: {sector_path}")
    if args.skip_prices:
        print(f"Returns unchanged: {returns_path}")


if __name__ == "__main__":
    main()

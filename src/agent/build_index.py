#!/usr/bin/env python3
"""Build a local Chroma retrieval index over normalized company-date-information rows.

The expected input is a parquet table such as
``data/financial-reports-sec/summary/summaries.filtered.parquet`` with at least:

- ``ticker``
- ``date``
- ``summary``

The output index is a local directory containing:

- persistent Chroma collection data
- ``metadata.json``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import chromadb
from chromadb.utils import embedding_functions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/financial-reports-sec/summary/summaries.filtered.parquet"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/financial-reports-sec/summary/index"),
    )
    parser.add_argument("--collection-name", default="risk_summaries")
    parser.add_argument("--text-column", default="summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.input_path)
    if frame.empty:
        raise ValueError(f"no rows found in {args.input_path}")
    required_columns = {"date", "ticker", args.text_column}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["date"] = frame["date"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str)
    frame["company_name"] = frame.get("company_name", "").astype(str)
    frame[args.text_column] = frame[args.text_column].astype(str)
    frame = frame.rename(columns={args.text_column: "information"})

    args.index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(args.index_dir))
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    if args.collection_name in {collection.name for collection in client.list_collections()}:
        client.delete_collection(args.collection_name)
    collection = client.get_or_create_collection(
        name=args.collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []
    for row in frame.itertuples(index=False):
        ids.append(f"{row.ticker}|{row.date}")
        documents.append(str(row.information))
        metadatas.append(
            {
                "date": str(row.date),
                "ticker": str(row.ticker),
                "ticker_upper": str(row.ticker).upper(),
                "company_name": str(row.company_name),
                "company_name_upper": str(row.company_name).upper(),
            }
        )

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    metadata = {
        "input_path": str(args.input_path),
        "collection_name": args.collection_name,
        "text_column": args.text_column,
        "num_rows": int(len(frame)),
        "schema": ["date", "ticker", "company_name", "information"],
    }
    (args.index_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Built Chroma index with {len(frame)} rows at {args.index_dir}")


if __name__ == "__main__":
    main()

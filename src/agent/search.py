#!/usr/bin/env python3
"""Search a local Chroma retrieval index with optional exact company/date filters."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import chromadb
import numpy as np

from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

from agent.common import SearchResult, load_index_metadata


def build_where_clause(
    company: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, object] | None:
    """Build a Chroma metadata filter."""

    clauses: list[dict[str, object]] = []
    if company:
        company_upper = company.upper()
        clauses.append(
            {
                "$or": [
                    {"ticker_upper": {"$eq": company_upper}},
                    {"company_name_upper": {"$eq": company_upper}},
                ]
            }
        )
    if date_from:
        clauses.append({"date": {"$gte": date_from}})
    if date_to:
        clauses.append({"date": {"$lte": date_to}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def search_index(
    collection: Collection,
    embedding_function: object,
    query: str,
    top_k: int,
    company: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    hard_stop_date: str | None = None,
    max_context_chars: int = 12000,
) -> list[SearchResult]:
    """Run similarity search with optional exact ticker/company and date filters."""

    if hard_stop_date is not None:
        if date_from and date_from > hard_stop_date:
            raise ValueError(f"date_from={date_from} exceeds hard stop date {hard_stop_date}")
        if date_to and date_to > hard_stop_date:
            raise ValueError(f"date_to={date_to} exceeds hard stop date {hard_stop_date}")
        if date_to is None or date_to > hard_stop_date:
            date_to = hard_stop_date

    where = build_where_clause(company=company, date_from=date_from, date_to=date_to)
    result_limit = max(top_k * 5, top_k)

    if where is None:
        raw = collection.query(
            query_texts=[query],
            n_results=result_limit,
            include=["documents", "metadatas", "distances"],
        )
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        rows = list(zip(documents, metadatas, distances))
    else:
        filtered = collection.get(
            where=where,
            include=["documents", "metadatas", "embeddings"],
        )
        documents = filtered.get("documents", [])
        metadatas = filtered.get("metadatas", [])
        embeddings = filtered.get("embeddings", [])
        if not documents:
            return []
        query_vector = np.asarray(embedding_function([query])[0], dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vector))
        rows = []
        for document, metadata, embedding in zip(documents, metadatas, embeddings):
            emb = np.asarray(embedding, dtype=np.float32)
            denom = float(np.linalg.norm(emb) * query_norm)
            score = 0.0 if denom == 0.0 else float(np.dot(emb, query_vector) / denom)
            rows.append((document, metadata, 1.0 - score))
        rows.sort(key=lambda item: item[2])
        rows = rows[:result_limit]

    if not rows:
        return []

    results: list[SearchResult] = []
    used_chars = 0
    for document, metadata, distance in rows:
        summary = str(document)
        estimated_chars = len(summary) + 128
        if results and used_chars + estimated_chars > max_context_chars:
            break
        score = 1.0 - float(distance) if not math.isnan(float(distance)) else 0.0
        results.append(
            SearchResult(
                date=str(metadata.get("date", "")),
                ticker=str(metadata.get("ticker", "")),
                company_name=str(metadata.get("company_name", "")),
                summary=summary,
                score=score,
            )
        )
        used_chars += estimated_chars
        if len(results) >= top_k:
            break
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path("data/financial-reports-sec/summary/index"))
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--company")
    parser.add_argument("--date-from")
    parser.add_argument("--date-to")
    parser.add_argument("--max-context-chars", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_index_metadata(args.index_dir)
    client = chromadb.PersistentClient(path=str(args.index_dir))
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_collection(
        name=str(metadata["collection_name"]),
        embedding_function=embedding_function,
    )
    results = search_index(
        collection=collection,
        embedding_function=embedding_function,
        query=args.query,
        top_k=args.top_k,
        company=args.company,
        date_from=args.date_from,
        date_to=args.date_to,
        hard_stop_date=None,
        max_context_chars=args.max_context_chars,
    )
    for index, result in enumerate(results, start=1):
        print(f"[{index}] {result.ticker} {result.date} score={result.score:.4f}")
        print(result.summary)
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a simple retrieval agent over company-date-information rows.

The agent can issue iterative search actions of the form:

- ``search(query, top_k, company=None, date_from=None, date_to=None)``
- ``final_answer(text)``

All steps are logged to JSON for later analysis or RL trajectory collection.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from agent.common import (
    extract_json_object,
    build_ticker_to_sector,
    load_backend,
    load_index_metadata,
    load_sector_map,
    render_search_results,
    run_model_text,
)
from agent.search import search_index
from agent.starting_prompt import build_starting_prompt

CURRENT_DATE = date(2026, 8, 25)
FINAL_SYSTEM_PROMPT = """You are a financial analyst.

Based only on the gathered evidence, predict the stock movement outcome.

Return exactly one word:
UP
or
DOWN"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "hf", "vllm"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/financial-reports-sec/summary/index"))
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--filing-date", required=True, help="SEC filing release date in YYYY-MM-DD format")
    parser.add_argument(
        "--sector-map-path",
        type=Path,
        default=Path("data/financial-reports-sec/market_data/sector_companies.json"),
    )
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--default-top-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument(
        "--trajectory-path",
        type=Path,
        default=Path("data/financial-reports-sec/summary/agent_trajectory.json"),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_user_prompt(starting_prompt: str, history: list[dict[str, Any]], default_top_k: int) -> str:
    """Build the current turn prompt for the agent."""

    if not history:
        history_text = "No prior tool results."
    else:
        chunks: list[str] = []
        for step in history:
            chunks.append(
                "\n".join(
                    [
                        f"step={step['step']}",
                        f"action={json.dumps(step['action'], ensure_ascii=True)}",
                        f"results={step['results_text']}",
                    ]
                )
            )
        history_text = "\n\n".join(chunks)

    return "\n".join(
        [
            starting_prompt,
            "",
            f"Default top_k if omitted: {default_top_k}",
            "",
            "Previous search history:",
            history_text,
            "",
            "Return the next JSON action.",
        ]
    )


def build_final_answer_prompt(
    ticker: str,
    filing_date: str,
    hard_stop_date: str,
    history: list[dict[str, Any]],
) -> str:
    """Build the final prediction prompt from gathered evidence."""

    evidence_chunks: list[str] = []
    for step in history:
        if step["action"].get("action") != "search":
            continue
        evidence_chunks.append(
            "\n".join(
                [
                    f"step={step['step']}",
                    step["results_text"],
                ]
            )
        )
    evidence_text = "\n\n".join(evidence_chunks) if evidence_chunks else "No search evidence collected."
    return "\n".join(
        [
            f"Predict the expected movement of stock {ticker} after the filing release on {filing_date}.",
            f"Only use evidence dated on or before {hard_stop_date}.",
            "Return exactly one word: UP or DOWN.",
            "",
            "Evidence:",
            evidence_text,
        ]
    )


def parse_final_prediction(text: str) -> str:
    """Parse final model output into UP or DOWN."""

    upper = text.strip().upper()
    if "DOWN" in upper and "UP" not in upper:
        return "DOWN"
    if "UP" in upper:
        return "UP"
    raise ValueError(f"could not parse final prediction as UP/DOWN: {text!r}")


def resolve_hard_stop_date(filing_date: date) -> str:
    """Disallow access to information after the target filing date or after today."""

    return min(filing_date, CURRENT_DATE).isoformat()


def lookup_sector_peers(
    ticker: str,
    sector_map: dict[str, list[str]],
    ticker_to_sector: dict[str, str],
    limit: int = 25,
) -> dict[str, Any]:
    """Return sector and peer tickers for a given ticker."""

    ticker_upper = ticker.upper()
    sector = ticker_to_sector.get(ticker_upper)
    if sector is None:
        return {"ticker": ticker_upper, "sector": None, "peers": []}
    peers = [peer for peer in sector_map[sector] if peer != ticker_upper]
    return {"ticker": ticker_upper, "sector": sector, "peers": peers[:limit]}


def main() -> None:
    args = parse_args()
    filing_date = date.fromisoformat(args.filing_date)
    if filing_date > CURRENT_DATE:
        raise ValueError(
            f"filing date {filing_date.isoformat()} is in the future relative to {CURRENT_DATE.isoformat()}"
        )

    backend = load_backend(args.provider, args.model)
    metadata = load_index_metadata(args.index_dir)
    client = chromadb.PersistentClient(path=str(args.index_dir))
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_collection(
        name=str(metadata["collection_name"]),
        embedding_function=embedding_function,
    )
    sector_map = load_sector_map(args.sector_map_path)
    ticker_to_sector = build_ticker_to_sector(sector_map)
    hard_stop_date = resolve_hard_stop_date(filing_date)
    starting_prompt = build_starting_prompt(
        ticker=args.ticker.upper(),
        filing_date=filing_date.isoformat(),
        current_date=CURRENT_DATE.isoformat(),
        hard_stop_date=hard_stop_date,
    )

    history: list[dict[str, Any]] = []
    final_answer = ""
    for step in range(1, args.max_steps + 1):
        user_prompt = build_user_prompt(starting_prompt, history, args.default_top_k)
        raw = run_model_text(
            provider=args.provider,
            backend=backend,
            model=args.model,
            system_prompt="You are a retrieval agent. Output valid JSON only.",
            user_prompt=user_prompt,
            max_new_tokens=220,
        )
        action = extract_json_object(raw)
        action_type = str(action.get("action", ""))
        if action_type == "final_answer":
            final_prompt = build_final_answer_prompt(
                ticker=args.ticker.upper(),
                filing_date=filing_date.isoformat(),
                hard_stop_date=hard_stop_date,
                history=history,
            )
            final_raw = run_model_text(
                provider=args.provider,
                backend=backend,
                model=args.model,
                system_prompt=FINAL_SYSTEM_PROMPT,
                user_prompt=final_prompt,
                max_new_tokens=8,
            )
            final_answer = parse_final_prediction(final_raw)
            history.append(
                {
                    "step": step,
                    "action": action,
                    "results": [],
                    "results_text": f"Final prediction emitted: {final_answer}",
                    "final_prompt": final_prompt,
                    "final_raw": final_raw,
                }
            )
            break
        if action_type == "sector_peers":
            company = str(action.get("company", args.ticker)).strip() or args.ticker
            payload = lookup_sector_peers(company, sector_map, ticker_to_sector)
            results_text = json.dumps(payload, ensure_ascii=True)
            history.append(
                {
                    "step": step,
                    "action": action,
                    "results": payload,
                    "results_text": results_text,
                }
            )
            if args.verbose:
                print(f"Step {step} action: {json.dumps(action, ensure_ascii=True)}")
                print(results_text)
                print()
            continue
        if action_type != "search":
            raise ValueError(f"unsupported action: {action!r}")

        results = search_index(
            collection=collection,
            embedding_function=embedding_function,
            query=str(action.get("query", "")).strip(),
            top_k=int(action.get("top_k", args.default_top_k)),
            company=str(action["company"]).strip() if action.get("company") else None,
            date_from=str(action["date_from"]).strip() if action.get("date_from") else None,
            date_to=str(action["date_to"]).strip() if action.get("date_to") else None,
            hard_stop_date=hard_stop_date,
            max_context_chars=args.max_context_chars,
        )
        results_text = render_search_results(results)
        history.append(
            {
                "step": step,
                "action": action,
                "results": [result.__dict__ for result in results],
                "results_text": results_text,
            }
        )
        if args.verbose:
            print(f"Step {step} action: {json.dumps(action, ensure_ascii=True)}")
            print(results_text)
            print()

    if not final_answer:
        final_answer = "No final answer produced within the step limit."

    trajectory = {
        "ticker": args.ticker.upper(),
        "filing_date": filing_date.isoformat(),
        "hard_stop_date": hard_stop_date,
        "starting_prompt": starting_prompt,
        "provider": args.provider,
        "model": args.model,
        "index_dir": str(args.index_dir),
        "max_steps": args.max_steps,
        "history": history,
        "final_answer": final_answer,
    }
    args.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    args.trajectory_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    print(final_answer)
    print(f"Saved trajectory to {args.trajectory_path}")


if __name__ == "__main__":
    main()

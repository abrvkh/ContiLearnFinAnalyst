#!/usr/bin/env python3
"""Propose and backtest one simple retrieval/analysis strategy.

Example: run the baseline-style strategy agent
------------------------------------------------

From the repository root, using OpenAI::

    uv run python src/strategy_agent/run_strategy_agent.py \
        --provider openai \
        --model gpt-4.1-mini \
        --predictions-dir predictions/baseline_strategy \
        --strategy-path predictions/baseline_strategy/baseline_strategy.json \
        --run-name baseline_strategy_agent

Or using a local vLLM model::

    uv run python src/strategy_agent/run_strategy_agent.py \
        --provider vllm \
        --model Qwen/Qwen3-8B \
        --predictions-dir predictions/baseline_strategy \
        --strategy-path predictions/baseline_strategy/baseline_strategy.json \
        --run-name baseline_strategy_agent
        --batch-size 16

In the baseline example above, the strategy specification is::

    predictions/baseline_strategy/baseline_strategy.json

The runner then:

1. Loads the strategy specification.
2. Loads the normalized 10-K Item 1A risk-summary parquet.
3. Evaluates the strategy independently for every stock and filing date.
4. Resolves ``TARGET_STOCK``, ``SECTOR_PEERS``, and the requested years for
   each target case, while discarding retrieval years after the target year.
5. Sends each retrieval context and analysis question to the LLM.
6. Sends the analysis answers to the final prediction prompt.
7. Parses the strategy-defined prediction label and maps it to ``-1``, ``0``,
   or ``1``.
8. Writes the prediction parquet files.
9. Runs ``src/eval/run_10k_pnl.py`` with ``--ffill-limit 30``.
10. Writes the PnL plot.

With the default run name and output directory, the files are::

    predictions/baseline_strategy/baseline_strategy.json
    predictions/baseline_strategy/baseline_strategy_agent.parquet
    predictions/baseline_strategy/baseline_strategy_agent.rows.parquet
    predictions/baseline_strategy/baseline_strategy_agent_pnl.png

The ``.parquet`` panel is created before the PnL evaluator runs. The PnL plot
is created only after the prediction panel has been written and evaluated.
Use ``--no-pnl`` to skip the final PnL step.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from utils import extract_json_object
from strategies.baseline import load_backend, run_hf, run_hf_batch, run_openai, run_openai_parallel, run_vllm, run_vllm_batch
from strategy_agent.prompts import analysis_prompt, final_prompt
from strategy_agent.strategy import StrategySpec, _signal


# Keep substantial headroom for the system prompt, question, chat template,
# and generated answer under Qwen3's 16,384-token vLLM context limit.
MAX_RETRIEVED_CONTEXT_CHARS = 24000
SUMMARY_CHUNK_CHARS = 16000
MAX_SECTOR_PEERS = 5


def truncate_retrieved_context(context: str, max_chars: int = MAX_RETRIEVED_CONTEXT_CHARS) -> str:
    """Bound one retrieved context so no individual prompt exceeds the model window."""
    if len(context) <= max_chars:
        return context
    head_chars = int(max_chars * 0.8)
    tail_chars = max_chars - head_chars
    return (
        context[:head_chars]
        + "\n\n[Retrieved context truncated to fit the model context window.]\n\n"
        + context[-tail_chars:]
    )


def summarize_context(provider: str, backend: object, model: str, context: str) -> str:
    """Summarize one context; retained as a small compatibility wrapper."""
    return summarize_context_batch(provider, backend, model, [context])[0]


def summarize_context_batch(
    provider: str, backend: object, model: str, contexts: list[str]
) -> list[str]:
    """Summarize oversized evidence in batches, preserving case order."""
    results = list(contexts)
    pending_prompts: list[str] = []
    pending_locations: list[tuple[int, int]] = []
    prompt = (
        "Summarize the following 10-K Item 1A risk evidence faithfully. "
        "Preserve dates, companies, concrete risk changes, and comparisons. "
        "Do not invent facts. Return concise prose only.\n\n"
    )
    for context_index, context in enumerate(contexts):
        if len(context) <= MAX_RETRIEVED_CONTEXT_CHARS:
            continue
        chunks = [
            context[i : i + SUMMARY_CHUNK_CHARS]
            for i in range(0, len(context), SUMMARY_CHUNK_CHARS)
        ]
        for chunk_index, chunk in enumerate(chunks):
            pending_locations.append((context_index, chunk_index))
            pending_prompts.append(prompt + chunk)

    if not pending_prompts:
        return results

    summaries = run_model_batch(
        provider,
        backend,
        model,
        pending_prompts,
        system_prompt="You are a careful financial research summarizer.",
    )
    grouped: dict[int, list[tuple[int, str]]] = {}
    for (context_index, chunk_index), summary in zip(pending_locations, summaries):
        grouped.setdefault(context_index, []).append((chunk_index, summary))
    for context_index, chunks in grouped.items():
        chunks.sort()
        combined = "\n\n--- summarized evidence ---\n\n".join(
            summary for _, summary in chunks
        )
        results[context_index] = truncate_retrieved_context(combined)
    return results


def run_model_text(
    provider: str,
    backend: object,
    model: str,
    user_prompt: str,
    system_prompt: str = "",
    enable_thinking: bool = False,
) -> str:
    """Dispatch through the tested baseline.py model implementations."""
    if provider == "openai":
        return run_openai(backend, model, user_prompt, system_prompt=system_prompt)
    if provider == "hf":
        return run_hf(backend, user_prompt, system_prompt=system_prompt, enable_thinking=enable_thinking)
    if provider == "vllm":
        return run_vllm(backend, user_prompt, system_prompt=system_prompt, enable_thinking=enable_thinking)
    raise ValueError(f"unsupported provider: {provider}")


def run_model_batch(provider: str, backend: object, model: str, prompts: list[str], system_prompt: str = "", enable_thinking: bool = False) -> list[str]:
    """Run a prompt batch using the baseline's batching implementation."""
    if provider == "openai":
        return run_openai_parallel(backend, model, prompts, system_prompt=system_prompt)
    if provider == "hf":
        return run_hf_batch(backend, prompts, system_prompt=system_prompt, enable_thinking=enable_thinking)
    if provider == "vllm":
        return run_vllm_batch(backend, prompts, system_prompt=system_prompt, enable_thinking=enable_thinking)
    raise ValueError(f"unsupported provider: {provider}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "hf", "vllm"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--summaries-path", type=Path, default=Path("data/financial-reports-sec/summary/summaries.filtered.parquet"))
    parser.add_argument("--sector-map-path", type=Path, default=Path("data/financial-reports-sec/market_data/sector_companies.json"))
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--strategy-path", type=Path, required=True)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--verbose", action="store_true", help="Print model prompts and raw outputs.")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode when supported by the chat template.")
    parser.add_argument("--pnl-ffill-limit", type=int, default=30)
    parser.add_argument("--no-pnl", action="store_true")
    return parser.parse_args()


def load_strategy(args: argparse.Namespace, backend: object) -> StrategySpec:
    raw = json.loads(args.strategy_path.read_text(encoding="utf-8"))
    return StrategySpec.from_dict(raw)


def resolve_year(expression: str, target_year: int) -> int:
    expression = expression.replace(" ", "")
    if expression == "TARGET_YEAR":
        return target_year
    if expression.startswith("TARGET_YEAR-"):
        return target_year - int(expression.removeprefix("TARGET_YEAR-"))
    return int(expression)


def extract_label(text: str, labels: tuple[str, ...], name: str) -> str:
    """Parse one unambiguous label following an explicit answer marker."""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    normalized = {label.upper(): label for label in labels}
    label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    pattern = re.compile(
        rf"(?:FINAL\s+ANSWER|ANSWER)\s*(?::|\s+IS)\s*"
        rf"[\"']?({label_pattern})\b",
        flags=re.IGNORECASE,
    )
    matches = [normalized[match.group(1).upper()] for match in pattern.finditer(text)]
    if not matches or len(set(matches)) > 1:
        raise ValueError(f"ambiguous {name} output: multiple answer labels")
    return matches[0]


def retrieve(frame: pd.DataFrame, sector_map: dict[str, list[str]], ticker: str, target_year: int, step) -> str:
    '''
    Input:
        frame: DataFrame with columns ["date", "ticker", "summary"]
        sector_map: dict mapping tickers to lists of sector peer tickers
        ticker: the target stock ticker
        target_year: the target filing year
        step: a StrategyStep object with attributes stocks, years, question, output_labels
    Returns a string with the retrieved summaries, or "No matching evidence." if none found.
    '''
    requested = [(expression, resolve_year(expression, target_year)) for expression in step.years]
    # Enforce the information cutoff. Year expressions are exact requests:
    # TARGET_YEAR-1 means the previous calendar year. Missing filings are
    # represented as N/A in the retrieval context.
    requested = [item for item in requested if item[1] <= target_year]
    if not requested:
        raise ValueError(
            f"no requested retrieval years are valid for target year {target_year}: "
            f"requested={sorted(resolve_year(expression, target_year) for expression in step.years)}"
        )

    # A peer comparison needs the target in the same retrieved context. Treat
    # SECTOR_PEERS as "target plus peers" so a peer-only step can still answer
    # questions comparing the target company with its sector. Only count peers
    # that have evidence for at least one requested year; otherwise the top-five
    # slots can be consumed by tickers that produce only N/A blocks.
    tickers = {ticker} if "TARGET_STOCK" in step.stocks or "SECTOR_PEERS" in step.stocks else set()
    if "SECTOR_PEERS" in step.stocks:
        for peers in sector_map.values():
            if ticker in peers:
                eligible_peers = []
                for peer in sorted(set(peers) - {ticker}):
                    peer_rows = frame[frame["ticker"] == peer]
                    if not peer_rows.empty and bool((peer_rows["date"].dt.year <= max(year for _, year in requested)).any()):
                        eligible_peers.append(peer)
                # Keep retrieval prompts bounded: include the target plus the
                # first five peers that actually have requested-year evidence.
                tickers.update(eligible_peers[:MAX_SECTOR_PEERS])
                break

    documents: list[str] = []
    for company in sorted(tickers):
        company_rows = frame[frame["ticker"] == company].sort_values("date")
        emitted_years: set[int] = set()
        for expression, cutoff_year in requested:
            year = cutoff_year
            matching = company_rows[company_rows["date"].dt.year == year]

            # Do not print the same requested calendar year twice.
            if year in emitted_years:
                continue
            emitted_years.add(year)
            source_year = year
            if matching.empty:
                # If an exact comparison year is unavailable, use the latest
                # filing before it. This keeps year-over-year questions useful
                # for companies with irregular filing histories without ever
                # looking past the requested cutoff.
                prior_rows = company_rows[company_rows["date"].dt.year < year]
                if prior_rows.empty:
                    summary_text = "No prior year data exists."
                else:
                    source_year = int(prior_rows["date"].dt.year.max())
                    matching = company_rows[company_rows["date"].dt.year == source_year]
                    summary_text = "\n\n".join(
                        str(row.summary) for row in matching.itertuples()
                    )
            else:
                summary_text = "\n\n".join(
                    str(row.summary) for row in matching.itertuples()
                )
            documents.append(
                "\n".join(
                    [
                        "Item 1A Risk Factors",
                        f"Requested year: {year}",
                        f"Evidence year: {source_year}",
                        f"Company: {company}",
                        "",
                        summary_text,
                    ]
                )
            )
    return "\n\n".join(documents)


def main() -> None:
    args = parse_args()
    # open the parquet file with the summaries
    frame = pd.read_parquet(args.summaries_path).copy()
    required_columns = {"date", "ticker", "summary"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"summary parquet missing columns: {sorted(missing_columns)}")
    # Parquet may return date as strings, Python date objects, or timestamps.
    # Normalize it once so retrieval can safely use .dt.year.
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["summary"] = frame["summary"].fillna("").astype(str).str.strip()
    frame = frame[frame["summary"].ne("")].copy()
    # load the sector map and normalize ticker casing
    sector_map = json.loads(args.sector_map_path.read_text(encoding="utf-8"))
    sector_map = {
        str(sector): [str(ticker).upper() for ticker in tickers]
        for sector, tickers in sector_map.items()
    }
    # load the model backend and the strategy (either from file or generated by the model)
    backend = load_backend(args.provider, args.model)
    strategy = load_strategy(args, backend)
    # run the strategy on each unique (ticker, date) pair in the summaries
    rows: list[dict[str, object]] = []
    log_rows: list[dict[str, object]] = []
    cases = frame[["ticker", "date"]].drop_duplicates().sort_values(["date", "ticker"])
    if args.max_rows:
        cases = cases.head(args.max_rows)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    case_records = list(cases.itertuples(index=False))
    # Batch API jobs should contain the complete independent stage. Splitting
    # OpenAI requests into hundreds of sequential jobs defeats the point of
    # Batch and can leave the run waiting on each completion window in turn.
    effective_batch_size = len(case_records) if args.provider == "openai" else args.batch_size
    for batch_start in range(0, len(case_records), effective_batch_size):
        batch_cases = case_records[batch_start : batch_start + effective_batch_size]
        outputs_by_case: list[list[str]] = [[] for _ in batch_cases]
        for step in strategy.steps:
            prompts: list[str] = []
            raw_contexts: list[str] = []
            for case in batch_cases:
                ticker = str(case.ticker)
                target_year = pd.Timestamp(case.date).year
                raw_contexts.append(retrieve(frame, sector_map, ticker, target_year, step))
            contexts = summarize_context_batch(args.provider, backend, args.model, raw_contexts)
            for case, context in zip(batch_cases, contexts):
                ticker = str(case.ticker)
                target_year = pd.Timestamp(case.date).year
                prompts.append(analysis_prompt(
                    strategy.hypothesis,
                    ticker,
                    target_year,
                    step.question,
                    context,
                    list(step.output_labels),
                    step.label_descriptions,
                ))
            raw_outputs = run_model_batch(args.provider, backend, args.model, prompts, system_prompt=strategy.system_prompt, enable_thinking=args.thinking)
            if args.verbose:
                for prompt, raw in zip(prompts, raw_outputs):
                    print("\n--- analysis model input ---\n" + prompt, flush=True)
                    print("\n--- analysis model output ---\n" + raw, flush=True)
            for index, raw in enumerate(raw_outputs):
                try:
                    label = extract_label(raw, step.output_labels, "analysis")
                except ValueError:
                    label = "UNPARSED"
                case = batch_cases[index]
                log_rows.append({
                    "stage": "analysis",
                    "target_ticker": str(case.ticker),
                    "target_date": pd.Timestamp(case.date).strftime("%Y-%m-%d"),
                    "target_year": pd.Timestamp(case.date).year,
                    "step_index": strategy.steps.index(step),
                    "stocks": json.dumps(list(step.stocks)),
                    "years": json.dumps(list(step.years)),
                    "question": step.question,
                    "retrieved_context": contexts[index],
                    "raw_output": raw,
                    "parsed_answer": label,
                    "status": "ok" if label != "UNPARSED" else "unparsed",
                })
                outputs_by_case[index].append(f"question: {step.question}\nanswer: {label}")

        final_prompts = []
        final_case_indices: list[int] = []
        predictions = [float("nan")] * len(batch_cases)
        for case_index, (case, outputs) in enumerate(zip(batch_cases, outputs_by_case)):
            if any("answer: UNPARSED" in output for output in outputs):
                continue
            ticker = str(case.ticker)
            target_year = pd.Timestamp(case.date).year
            final_prompts.append(final_prompt(
                strategy.hypothesis,
                ticker,
                target_year,
                strategy.prediction_instructions,
                outputs,
                list(strategy.prediction_labels),
            ))
            final_case_indices.append(case_index)
        final_outputs = run_model_batch(args.provider, backend, args.model, final_prompts, system_prompt=strategy.system_prompt, enable_thinking=args.thinking) if final_prompts else []
        if args.verbose:
            for prompt, raw in zip(final_prompts, final_outputs):
                print("\n--- final model input ---\n" + prompt, flush=True)
                print("\n--- final model output ---\n" + raw, flush=True)
        for case_index, raw in zip(final_case_indices, final_outputs):
            case = batch_cases[case_index]
            target_date = pd.Timestamp(case.date)
            ticker = str(case.ticker)
            try:
                label = extract_label(raw, strategy.prediction_labels, "prediction")
                mapped_prediction = strategy.parse_map[label]
                predictions[case_index] = (
                    float("nan") if mapped_prediction is None else float(mapped_prediction)
                )
            except (KeyError, ValueError):
                predictions[case_index] = float("nan")
                label = "UNPARSED"
            log_rows.append({
                "stage": "final",
                "target_ticker": ticker,
                "target_date": target_date.strftime("%Y-%m-%d"),
                "target_year": target_date.year,
                "step_index": None,
                "stocks": None,
                "years": None,
                "question": None,
                "retrieved_context": None,
                "raw_output": raw,
                "parsed_answer": label,
                "status": "ok" if label != "UNPARSED" else "unparsed",
            })
        for case_index, case in enumerate(batch_cases):
            target_date = pd.Timestamp(case.date)
            ticker = str(case.ticker)
            prediction = predictions[case_index]
            rows.append({"date": target_date.strftime("%Y-%m-%d"), "ticker": ticker, "prediction": prediction})
            print(f"{ticker} {target_date.date()}: {prediction}", flush=True)

    result = pd.DataFrame(rows)
    panel = result.pivot_table(index="date", columns="ticker", values="prediction", aggfunc="last")
    panel.index = pd.to_datetime(panel.index)
    output = args.predictions_dir / f"{args.run_name}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output)
    result.to_parquet(args.predictions_dir / f"{args.run_name}.rows.parquet", index=False)
    pd.DataFrame(log_rows).to_parquet(
        args.predictions_dir / f"{args.run_name}.logs.parquet", index=False
    )
    print(f"Saved predictions to {output}")
    if not args.no_pnl:
        evaluations = (
            (30, (), False, f"{args.run_name}_ffill30"),
            (252, (), False, f"{args.run_name}_ffill252"),
            (30, ("cross-sectional-rank",), True, f"{args.run_name}_ffill30_xsec"),
            (252, ("cross-sectional-rank",), True, f"{args.run_name}_ffill252_xsec"),
        )
        pnl_results = []
        for ffill_limit, postprocess, use_sectors, artifact_name in evaluations:
            command = [
                sys.executable, "src/eval/run_10k_pnl.py",
                "--run-name", args.run_name,
                "--predictions-dir", str(args.predictions_dir),
                "--ffill-limit", str(ffill_limit),
                "--artifact-name", artifact_name,
            ]
            if use_sectors:
                command.extend(["--use-sectors", "--sector-mapping", str(args.sector_map_path)])
            for step in postprocess:
                command.extend(["--postprocess", step])
            print("Running PnL:", " ".join(command), flush=True)
            subprocess.run(command, check=True)
            metrics_path = args.predictions_dir / f"{artifact_name}_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            pnl_results.append((artifact_name, metrics))
        print("\nPnL summary:", flush=True)
        for artifact_name, metrics in pnl_results:
            print(
                f"  {artifact_name}: sharpe={metrics.get('annualized_sharpe')}, "
                f"t_stat={metrics.get('t_stat')}, "
                f"always_long_corr={metrics.get('always_long_pnl_correlation')}",
                flush=True,
            )


if __name__ == "__main__":
    main()

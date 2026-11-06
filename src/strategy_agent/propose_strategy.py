#!/usr/bin/env python3
"""Iteratively propose, backtest, and inspect strategy-agent strategies."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from eval.run_10k_pnl import annualized_sharpe
from utils import extract_json_object
from strategies.baseline import load_backend, run_hf, run_openai, run_vllm
from strategy_agent.prompts import PROPOSAL_SYSTEM_PROMPT, STRATEGY_INSTRUCTIONS, MAIN_TASK
from strategy_agent.strategy import StrategySpec


# Keep the available model actions together and easy to audit.
ACTION_SPECS = {
    "inspect_general_performance": "Inputs: none. Returns aggregate directional counts and errors for the whole backtest.",
    "inspect_per_year": "Inputs: target_year (required integer). Returns directional counts and errors for that filing year.",
    "inspect_per_sector": "Inputs: sector (required string, such as Technology). Returns directional counts and errors for that sector.",
    "propose_strategy": (
        "Uses the accumulated strategy, metrics, and inspection results to create a replacement strategy."
    ),
    "stop": (
        "Ends the search and keeps the most recent valid strategy."
    ),
}
# A small, recognizable cross-industry sample. These are Yahoo tickers and
# were checked against the current filtered summary parquet; some have only a
# few surviving rows after summary filtering.
TOP_TICKERS = ("AAPL", "AMZN", "MSFT", "NVDA", "XOM", "JNJ", "META")
# Compact, human-auditable inspection basket. These are deliberately major,
# recognizable companies rather than the full sector mapping, which contains
# many illiquid, obsolete, or obscure symbols.
UNIVERSE_INSPECTION_TEXT = """Representative stocks by sector:
- Basic Materials: APD, ECL, FCX, NEM, SHW
- Communication Services: CHTR, META, NFLX, TMUS, VZ
- Consumer Cyclical: AMZN, HD, MCD, NKE, TSLA
- Consumer Defensive: COST, KO, PEP, PG, WMT
- Energy: COP, CVX, EOG, MPC, SLB, XOM
- Financial Services: AXP, BLK, CME, MA, V, PGR
- Healthcare: ABBV, ABT, JNJ, MRK, PFE, UNH
- Industrials: BA, CAT, DE, GE, HON, UPS
- Real Estate: AMT, CCI, EQIX, EQR, SPG, WELL
- Technology: AAPL, AMD, CSCO, IBM, INTC, MSFT, NVDA, ORCL, QCOM, TXN
- Utilities: D, DUK, EXC, PEG, SO, WEC, XEL"""
MAX_SEARCH_ACTIONS = 10
MAX_TOOL_CONTEXT_CHARS = 45000
PRIMARY_VARIANT_MARKERS = ("ffill252", "xsec")


def initial_prompt(examples: str) -> str:
    return f"""
{MAIN_TASK}

This is the first strategy proposal and you will iteratively improve it. You have access to the following
past examples and their performance:
{examples}

Critically assess what worked and what did not in these examples. Use that
assessment to make deliberate design choices; do not merely copy an example.

Strategy requirements:
{STRATEGY_INSTRUCTIONS}

You may reason through the design before answering. When finished, end your
response exactly in this form:
FINAL ANSWER:
{{complete strategy JSON}}"""


def initial_prompt_v2(examples: str) -> str:
    return f"""
{MAIN_TASK}

This is the first strategy proposal. Generate a genuinely new hypothesis from
the ideas below. The ideas are independent starting points, not templates
to combine mechanically. Prefer a simple, testable strategy with one clear
comparison or question. Do not anchor on, copy, or make a cosmetic variation
of the prior example strategies included at the end; they are only background
about the task format.

Several hypothesis ideas:

1. Cross-year change in the target company's risk profile
   Compare the target company's risk summary for TARGET_YEAR with its own
   summary from an earlier year (for example TARGET_YEAR - 1). Ask whether
   severe, material, or company-specific risks have increased, decreased, or
   remained stable. Base the prediction on the direction of change, not merely
   on how much risk language appears.

2. Current company versus its own historical baseline
   Compare the target company's current risk summary with several prior
   summaries for that same company (for example TARGET_YEAR - 1, -2, and -3).
   Ask whether the current filing is unusually adverse relative to the
   company's own recent history. This is a within-company baseline, not a
   peer comparison.

3. Current company versus peers in the same year
   Compare the target company's summary with several SECTOR_PEERS summaries
   from TARGET_YEAR only. Ask whether the target has unusually severe or
   distinctive risks relative to its peers, without introducing any cross-year
   comparison. This tests whether relative risk is more useful than absolute
   risk language.

4. One narrow, high-signal risk question
   Ask a specific binary or three-way question such as whether the filing
   mentions a severe company-specific risk, a meaningful deterioration in a
   major risk, or an unusually consequential threat. Avoid vague overall
   sentiment. Define severity clearly and distinguish company-specific risks
   from generic systemic events.

The above serves just as examples. You are free to come up with a creative, different hypothesis. 

Strategy requirements:
{STRATEGY_INSTRUCTIONS}

You may reason through the design before answering. When finished, end your
response exactly in this form:
FINAL ANSWER:
{{complete strategy JSON}}"""


def revision_prompt(context: str, search_count: int) -> str:
    actions = "\n".join(f"- {name}: {description}" for name, description in ACTION_SPECS.items())
    return f"""You are iteratively improving a financial prediction strategy. The goal is to increase the Sharpe, achieve a stable t-stat and low always-long correlation of a strategy that consists of a 252-day forward-filled predictions with cross-sectional ranking.

Past strategy and measured performance: 
{context}

When improving the strategy, consider conditional instructions rather than
assuming one interpretation works equally well everywhere. Sector-specific
calibration may be useful because risk-factor language can have different
base rates or meanings across sectors. Macro-event calibration may also be
useful: distinguish company-specific risk from systemic conditions such as
COVID-era disruption, interest-rate shocks, or supply-chain events. If a risk
factor describes a systemic event rather than an increase in company-specific
risk, do not automatically interpret its mention as worsening risk for that
company. Use the error breakdown to decide whether such a conditional rule is
justified, and state the observed failure pattern it addresses. Be creative in your proposals. 

You have used {search_count}/{MAX_SEARCH_ACTIONS} inspection actions. 

Available actions:
{actions}

1. Inspect aggregate directional performance:
{{"action": "inspect_general_performance"}}

2. Inspect a particular filing year, such as a COVID-era year:
{{"action": "inspect_per_year", "target_year": 2020}}

3. Inspect a particular sector:
{{"action": "inspect_per_sector", "sector": "Technology"}}

4. After diagnosing the error patterns, request a replacement strategy:
{{"action": "propose_strategy"}}

5. End the search when further changes are not justified:
{{"action": "stop"}}

Choose exactly one action and return JSON only:
{{
  "action": "inspect_general_performance" | "inspect_per_year" | "inspect_per_sector" | "propose_strategy" | "stop",
  "target_year": "required for inspect_per_year",
  "sector": "required for inspect_per_sector"
}}"""


def proposal_prompt(context: str) -> str:
    return f"""
{MAIN_TASK}

Research context from the previous strategy and all inspections:
{context}

Improve the strategy using evidence from the diagnostics. Consider whether
the prompt should behave differently for particular sectors or systemic
macro-economic events. For example, risk-factor language in Technology may
have a different baseline than in other sectors, and a filing that mentions a
systemic event such as COVID-era disruption should not automatically be
treated as evidence that the company's own risk has increased. Add such
conditional instructions only when they address an observed error pattern;
do not assume these examples are universally true.

Use this evidence to make a critical, concrete improvement. The strategy must
satisfy these requirements:
{STRATEGY_INSTRUCTIONS}

You may reason through the design before answering. When finished, end your
response exactly in this form:
FINAL ANSWER:
{{complete replacement strategy JSON}}"""


def extract_strategy_response(raw: str) -> dict[str, Any]:
    """Extract the last valid JSON object from a strategy response."""
    decoder = json.JSONDecoder()
    last: dict[str, Any] | None = None
    offset = 0
    while True:
        start = raw.find("{", offset)
        if start < 0:
            break
        try:
            candidate, end = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            offset = start + 1
            continue
        if isinstance(candidate, dict):
            last = candidate
        offset = start + end
    if last is None:
        raise ValueError("no valid JSON object found in strategy response")
    return last


def summarize_tool_result(backend: object, provider: str, model: str, result: str) -> str:
    if len(result) <= MAX_TOOL_CONTEXT_CHARS:
        return result
    return generate_text(
        provider,
        backend,
        model,
        "Summarize this backtest inspection result faithfully. Preserve important "
        "patterns, labels, dates, and metrics. Return concise prose only.\n\n"
        + result,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "hf", "vllm"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--summaries-path", type=Path, default=Path("data/financial-reports-sec/summary/summaries.filtered.parquet"))
    parser.add_argument("--sector-map-path", type=Path, default=Path("data/financial-reports-sec/market_data/sector_companies.json"))
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-iterations", type=int, default=10)
    return parser.parse_args()


def generate_text(provider: str, backend: object, model: str, prompt: str) -> str:
    if provider == "openai":
        return run_openai(backend, model, prompt, system_prompt=PROPOSAL_SYSTEM_PROMPT)
    if provider == "hf":
        return run_hf(backend, prompt, system_prompt=PROPOSAL_SYSTEM_PROMPT, max_new_tokens=32768)
    if provider == "vllm":
        return run_vllm(backend, prompt, system_prompt=PROPOSAL_SYSTEM_PROMPT, max_tokens=32768)
    raise ValueError(f"unsupported provider: {provider}")


def inspect_metrics(run_dir: Path) -> str:
    paths = sorted(run_dir.glob("*_metrics.json"))
    if not paths:
        return "No metrics JSON found."
    metrics = {}
    for path in paths:
        name = path.stem.removesuffix("_metrics")
        if all(marker in name.lower() for marker in PRIMARY_VARIANT_MARKERS):
            metrics[name] = json.loads(path.read_text(encoding="utf-8"))
    if not metrics:
        return "Primary variant ffill252_xsec metrics not found."
    result = format_metrics_table(metrics)

    # A small or weak model can produce a signal that varies over time but is
    # identical across stocks on each date. Cross-sectional ranking then
    # removes the entire signal, producing a constant-zero PnL and NaN
    # Sharpe/t-stat/correlation. Give the proposal model the unranked result
    # as useful feedback instead of showing it only NaNs.
    xsec = next(
        (values for name, values in metrics.items() if "_xsec" in name.lower()),
        None,
    )
    xsec_invalid = xsec is not None and any(
        pd.isna(xsec.get(field))
        for field in (
            "annualized_sharpe",
            "t_stat",
            "always_long_pnl_correlation",
        )
    )
    unranked = {
        name: values
        for name, values in metrics.items()
        if name.lower().endswith("_ffill252")
    }
    if xsec_invalid and unranked:
        result += (
            "\n\nCross-sectional ranking produced NaN, implying that all predictions "
            "are the same. Results from a pure forward fill:\n"
            + format_metrics_table(unranked)
        )
    return result


def format_metrics_table(metrics: dict[str, dict[str, Any]]) -> str:
    """Format variant metrics compactly for model-facing context."""
    lines = [
        "| Variant | Annualized Sharpe | T-stat | Always-long PnL correlation |",
        "|---|---:|---:|---:|",
    ]
    for variant, values in metrics.items():
        lines.append(
            f"| {variant} | {values.get('annualized_sharpe', 'N/A')} | "
            f"{values.get('t_stat', 'N/A')} | "
            f"{values.get('always_long_pnl_correlation', 'N/A')} |"
        )
    return "\n".join(lines)


def format_strategy_section(strategy: StrategySpec, strategy_type: str) -> str:
    values = json.loads(strategy.to_json())
    fields = []
    for name in ("system_prompt", "hypothesis", "steps", "prediction_rule"):
        fields.append(f"{name}: {json.dumps(values[name], indent=2)}")
    return f"Strategy type: {strategy_type}\n" + "\n".join(fields)


def inspect_universe(summaries_path: Path, sector_map_path: Path) -> str:
    """Return a compact orientation list for targeted log inspection."""
    return UNIVERSE_INSPECTION_TEXT

def see_examples() -> str:
    """Return seeded strategies and metrics in compact model-facing format."""
    examples = []
    for strategy_path in (
        Path("predictions/agent_single_risk_assessment/agent_single_risk_assessment.json"),
        Path("predictions/agent_compare_two_years/agent_compare_two_years.json"),
    ):
        if not strategy_path.exists():
            continue
        metrics = {}
        for metrics_path in sorted(strategy_path.parent.glob("*_metrics.json")):
            name = metrics_path.stem.removesuffix("_metrics")
            if all(marker in name.lower() for marker in PRIMARY_VARIANT_MARKERS):
                metrics[name] = json.loads(metrics_path.read_text(encoding="utf-8"))
        strategy_type = strategy_path.stem
        examples.append(
            f"Strategy type: {strategy_type}\n"
            f"Strategy:\n{json.dumps(json.loads(strategy_path.read_text(encoding='utf-8')), indent=2)}\n"
            f"Primary performance (ffill252 + cross-sectional ranking):\n{format_metrics_table(metrics)}"
        )
    return "\n\n".join(examples) if examples else "No prior strategy examples found."


def inspect_performance(
    run_dir: Path,
    sector_map_path: Path,
    scope: str,
    target_year: int | None = None,
    sector: str | None = None,
) -> str:
    """Diagnose final directional predictions for one requested scope."""
    log_paths = sorted(run_dir.glob("*.logs.parquet"))
    if not log_paths:
        return "No logs parquet found."
    logs = pd.read_parquet(log_paths[-1])
    logs = logs[logs["stage"].eq("final")].copy()
    logs["predicted_direction"] = logs["parsed_answer"].astype(str).str.upper().map(
        {"UP": "UP", "DOWN": "DOWN", "NO_POSITION": "FLAT"}
    )
    logs = logs[logs["predicted_direction"].notna()].copy()
    returns_path = Path("data/financial-reports-sec/market_data/return.parquet")
    returns = pd.read_parquet(returns_path)
    returns.index = pd.to_datetime(returns.index)
    return_dates = returns.index.sort_values()
    logs["target_date"] = pd.to_datetime(logs["target_date"])
    next_signal_by_ticker = {}
    for ticker, group in logs.groupby("target_ticker"):
        dates = sorted(pd.Timestamp(date) for date in group["target_date"].unique())
        next_signal_by_ticker[ticker] = {
            date: dates[index + 1] if index + 1 < len(dates) else None
            for index, date in enumerate(dates)
        }
    realized = []
    holding_days = []
    daily_pnls = []
    for row in logs.itertuples(index=False):
        ticker = str(row.target_ticker).upper()
        if ticker not in returns.columns:
            realized.append(None)
            holding_days.append(None)
            daily_pnls.append(None)
            continue
        start = pd.Timestamp(row.target_date)
        next_signal = next_signal_by_ticker[ticker].get(start)
        dates = return_dates[return_dates > start]
        if next_signal is not None:
            dates = dates[dates < pd.Timestamp(next_signal)]
        dates = dates[:252]
        if not len(dates):
            realized.append(None)
            holding_days.append(None)
            daily_pnls.append(None)
            continue
        interval_returns = pd.to_numeric(returns.loc[dates, ticker], errors="coerce").dropna()
        compounded = float((1.0 + interval_returns).prod() - 1.0)
        signal = {"UP": 1.0, "DOWN": -1.0, "FLAT": 0.0}
        realized_direction = "UP" if compounded > 0 else "DOWN" if compounded < 0 else "FLAT"
        realized.append(realized_direction)
        holding_days.append(len(interval_returns))
        daily_pnls.append(interval_returns.mul(signal.get(str(row.predicted_direction), 0.0)))
    logs["realized_direction"] = realized
    logs["holding_days"] = holding_days
    logs["daily_pnl"] = daily_pnls
    logs = logs[logs["realized_direction"].notna()].copy()
    sector_map = json.loads(sector_map_path.read_text(encoding="utf-8"))
    sector_by_ticker = {
        str(ticker).upper(): str(sector)
        for sector, tickers in sector_map.items()
        for ticker in tickers
    }
    logs["sector"] = logs["target_ticker"].map(sector_by_ticker).fillna("Unknown")
    logs["correct"] = logs["predicted_direction"].eq(logs["realized_direction"])
    if scope == "year":
        logs = logs[logs["target_year"].eq(int(target_year))]
        title = f"Performance for filing year {target_year}"
    elif scope == "sector":
        logs = logs[logs["sector"].str.casefold().eq(str(sector).casefold())]
        title = f"Performance for sector {sector}"
    else:
        title = "General performance"

    def table(grouped: Any) -> str:
        def sharpe(group: pd.DataFrame) -> float:
            series = [value for value in group["daily_pnl"] if value is not None]
            if not series:
                return float("nan")
            daily_pnl = pd.concat(series).groupby(level=0).sum().sort_index()
            return annualized_sharpe(daily_pnl)

        rows = []
        for name, group in grouped:
            rows.append({
                "group": name,
                "up_rate_bias": float(
                    (group["predicted_direction"] == "UP").mean()
                    - (group["realized_direction"] == "UP").mean()
                ),
                "error_rate": round(float((~group["correct"]).mean()), 4),
                "sharpe": sharpe(group),
            })
        rows.append({
            "group": "OVERALL",
            "up_rate_bias": float(
                (logs["predicted_direction"] == "UP").mean()
                - (logs["realized_direction"] == "UP").mean()
            ),
            "error_rate": round(float((~logs["correct"]).mean()), 4),
            "sharpe": sharpe(logs),
        })
        header = "| Group | UP-rate bias | Error rate | Annualized Sharpe |\n|---|---:|---:|---:|"
        return header + "\n" + "\n".join(
            f"| {r['group']} | {r['up_rate_bias']:+.2%} | {r['error_rate']:.2%} | {r['sharpe']:.4f} |"
            for r in rows
        )

    return (
        title + "\n"
        "Directional diagnosis uses compounded daily returns from after each filing date until the next filing signal for that ticker, capped at 252 trading days.\n\n"
        + table([("ALL", logs)]) +
        "\n\nDiagnosis instruction: identify where errors concentrate, whether the model is systematically too positive or too negative, and plausible reasons grounded in the strategy and risk-factor task. Then propose concrete strategy-question or prediction-rule changes."
    )


def inspect_logs(run_dir: Path, request: dict[str, Any]) -> str:
    paths = sorted(run_dir.glob("*.logs.parquet"))
    if not paths:
        return "No logs parquet found."
    frame = pd.read_parquet(paths[-1])
    filters = {"target_ticker": request.get("ticker"), "target_year": request.get("target_year"), "step_index": request.get("step_index"), "stage": request.get("stage"), "parsed_answer": request.get("parsed_answer")}
    for column, value in filters.items():
        if value not in (None, "") and column in frame:
            frame = frame[frame[column].astype(str) == str(value)]
    if not any(request.get(key) for key in ("ticker", "target_year", "step_index", "stage", "parsed_answer")):
        frame = frame[frame["target_ticker"].isin(TOP_TICKERS)].head(12)
    else:
        frame = frame.head(20)
    if frame.empty:
        return "No matching log rows found."
    # Keep inspection prompts compact. The full retrieved evidence remains in
    # the parquet log for targeted debugging, but including it here can
    # overwhelm the revision prompt and leave no room for a JSON action.
    keep_columns = [
        "target_ticker",
        "target_date",
        "target_year",
        "stage",
        "step_index",
        "question",
        "parsed_answer",
        "status",
    ]
    compact = frame[[column for column in keep_columns if column in frame.columns]].copy()
    if "raw_output" in frame.columns:
        compact["raw_output_excerpt"] = (
            frame["raw_output"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str[:1500]
        )
    return compact.to_json(orient="records", date_format="iso")


def run_strategy(args: argparse.Namespace, strategy: StrategySpec, iteration_dir: Path) -> None:
    iteration_dir.mkdir(parents=True, exist_ok=True)
    strategy_path = iteration_dir / "strategy.json"
    strategy_path.write_text(strategy.to_json(), encoding="utf-8")
    command = [sys.executable, "src/strategy_agent/run_strategy_agent.py", "--provider", args.provider, "--model", args.model, "--predictions-dir", str(iteration_dir), "--run-name", args.run_name, "--strategy-path", str(strategy_path), "--summaries-path", str(args.summaries_path), "--sector-map-path", str(args.sector_map_path)]
    if args.max_rows is not None:
        command.extend(["--max-rows", str(args.max_rows)])
    command.extend(["--batch-size", str(args.batch_size)])
    subprocess.run(command, check=True)


def release_backend(backend: object) -> None:
    """Release model memory before starting the strategy subprocess."""
    del backend
    gc.collect()
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def record_trajectory(path: Path, event: str, **data: Any) -> None:
    """Append one proposal-agent event to the durable search trajectory."""
    record = {"event": event, **data}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    args = parse_args()
    if args.max_iterations < 1:
        raise ValueError("--max-iterations must be positive")
    backend = load_backend(args.provider, args.model)
    root = args.predictions_dir
    root.mkdir(parents=True, exist_ok=True)
    trajectory_path = root / "proposal_trajectory.jsonl"
    examples = see_examples()
    record_trajectory(trajectory_path, "initial_examples", result=examples)
    print(f"Recorded initial examples to {trajectory_path}", flush=True)
    initial_raw = generate_text(args.provider, backend, args.model, initial_prompt_v2(examples))
    strategy = StrategySpec.from_dict(extract_strategy_response(initial_raw))
    record_trajectory(
        trajectory_path,
        "initial_proposal",
        raw_response=initial_raw,
        strategy=json.loads(strategy.to_json()),
    )
    context = "Initial strategy proposal:\n" + format_strategy_section(strategy, args.run_name)
    for iteration in range(0, args.max_iterations):
        iteration_dir = root / f"iteration_{iteration:03d}"
        print(f"Running strategy iteration {iteration}: {iteration_dir}", flush=True)
        release_backend(backend)
        backend = None
        try:
            run_strategy(args, strategy, iteration_dir)
        finally:
            backend = load_backend(args.provider, args.model)
        context += f"\n\nIteration {iteration} metrics:\n{inspect_metrics(iteration_dir)}"
        record_trajectory(
            trajectory_path,
            "iteration_completed",
            iteration=iteration,
            strategy=json.loads(strategy.to_json()),
            metrics=inspect_metrics(iteration_dir),
        )
        if iteration == args.max_iterations - 1:
            break
        search_count = 0
        while True:
            decision_raw = generate_text(args.provider, backend, args.model, revision_prompt(context, search_count))
            record_trajectory(
                trajectory_path,
                "action_response",
                iteration=iteration,
                search_count=search_count,
                raw_response=decision_raw,
            )
            try:
                decision = extract_strategy_response(decision_raw)
                action = str(decision.get("action", "")).strip()
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                record_trajectory(trajectory_path, "invalid_action", iteration=iteration, error=str(exc))
                context += f"\nInvalid action response: {exc}. Return action JSON only."
                continue
            if action == "stop":
                record_trajectory(trajectory_path, "stop", iteration=iteration, decision=decision)
                print("Proposal agent stopped the search.", flush=True)
                return
            if action in {"inspect_general_performance", "inspect_per_year", "inspect_per_sector"}:
                if search_count >= MAX_SEARCH_ACTIONS:
                    context += "\nInspection limit reached; choose propose_strategy or stop."
                    continue
                search_count += 1
                if action == "inspect_general_performance":
                    result = inspect_performance(iteration_dir, args.sector_map_path, "general")
                elif action == "inspect_per_year":
                    if decision.get("target_year") in (None, ""):
                        context += "\ninspect_per_year requires target_year."
                        continue
                    result = inspect_performance(
                        iteration_dir, args.sector_map_path, "year", target_year=int(decision["target_year"])
                    )
                else:
                    if not decision.get("sector"):
                        context += "\ninspect_per_sector requires sector."
                        continue
                    result = inspect_performance(
                        iteration_dir, args.sector_map_path, "sector", sector=str(decision["sector"])
                    )
                # result = summarize_tool_result(backend, args.provider, args.model, result)
                context += f"\n\nTool result ({action}, search {search_count}):\n{result}"
                record_trajectory(
                    trajectory_path,
                    "inspection",
                    iteration=iteration,
                    search_count=search_count,
                    decision=decision,
                    result=result,
                )
                continue
            if action == "propose_strategy":
                proposal_prompt_text = proposal_prompt(context)
                try:
                    proposal_raw = generate_text(args.provider, backend, args.model, proposal_prompt_text)
                    strategy = StrategySpec.from_dict(extract_strategy_response(proposal_raw))
                    record_trajectory(
                        trajectory_path,
                        "proposal_accepted",
                        iteration=iteration,
                        decision=decision,
                        raw_response=proposal_raw,
                        strategy=json.loads(strategy.to_json()),
                    )
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    record_trajectory(trajectory_path, "invalid_proposal", iteration=iteration, error=str(exc))
                    context += f"\nInvalid strategy proposal: {exc}. Propose a valid strategy JSON."
                    continue
                context += "\n\nReplacement strategy:\n" + format_strategy_section(strategy, args.run_name)
                break
            record_trajectory(trajectory_path, "unknown_action", iteration=iteration, decision=decision)
            context += f"\nUnknown action {action!r}; choose one of: {', '.join(ACTION_SPECS)}."


if __name__ == "__main__":
    main()

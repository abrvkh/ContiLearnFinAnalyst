#!/usr/bin/env python3
"""Estimate baseline strategy token usage and API cost.

The estimate is based on the local 10-K risk sections (``section_1A``) and the
baseline prompting pattern that uses the current year's risk section plus the
previous year's risk section as context.

You can either:
- pass your own prices with ``--input-cost-per-1m`` and ``--output-cost-per-1m``
- or select an OpenAI model from the built-in pricing snapshot

The built-in OpenAI prices are a snapshot from the official OpenAI pricing page
on August 24, 2026, for standard processing and short-context pricing.

Examples
--------
Use built-in OpenAI pricing::

    uv run python src/strategies/util.py --model gpt-5.6-terra

Use custom pricing::

    uv run python src/strategies/util.py \
        --input-cost-per-1m 2.0 \
        --output-cost-per-1m 12.0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from data.build_10k_market_data import sec_ticker_to_yahoo


OPENAI_PRICING_PER_1M: dict[str, dict[str, float]] = {
    "gpt-5.6-sol": {"input": 5.00, "output": 30.00},
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00},
    "gpt-5.6-luna": {"input": 0.20, "output": 1.20},
    "gpt-5.5": {"input": 5.00, "output": 30.00},
    "gpt-5.4": {"input": 1.25, "output": 7.50},
    "gpt-5.4-mini": {"input": 0.375, "output": 2.25},
    "gpt-5.4-nano": {"input": 0.10, "output": 0.625},
    "gpt-5.4-pro": {"input": 15.00, "output": 90.00},
    "chat-latest": {"input": 5.00, "output": 30.00},
}


SYSTEM_PROMPT = """You are a financial analyst.
Read the 10-K risk-factor text and output exactly one word:
UP if the stock sentiment implication is positive,
DOWN if the stock sentiment implication is negative.
Do not explain your answer."""

USER_PROMPT_TEMPLATE = """Current year risk section:
{current_risk}

Previous year risk section:
{previous_risk}

Based on these risk disclosures, what is your stock sentiment prediction?
Answer with exactly one word: UP or DOWN."""


@dataclass(frozen=True)
class TokenStats:
    num_examples: int
    avg_current_tokens: float
    avg_previous_tokens: float
    avg_prompt_tokens: float
    total_input_tokens: int
    total_output_tokens: int


def estimate_tokens(text: str) -> int:
    """Simple token estimate using the common chars/4 heuristic."""

    return max(1, round(len(text) / 4))


def load_risk_examples(dataset_dir: Path, splits: tuple[str, ...]) -> list[tuple[str, str]]:
    """Load pairs of current and previous risk sections from SEC shards."""

    examples: list[tuple[str, str]] = []
    for split in splits:
        for shard_path in sorted((dataset_dir / split).glob("*.jsonl")):
            with shard_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    _ = [sec_ticker_to_yahoo(str(ticker)) for ticker in record.get("tickers", [])]
                    filings = sorted(record.get("filings", []), key=lambda filing: filing["filingDate"])
                    previous_risk = ""
                    for filing in filings:
                        report = filing.get("report", {})
                        section = report.get("section_1A")
                        if not section:
                            continue
                        current_risk = "\n".join(str(part) for part in section).strip()
                        if not current_risk:
                            continue
                        examples.append((current_risk, previous_risk))
                        previous_risk = current_risk
    return examples


def compute_token_stats(dataset_dir: Path, splits: tuple[str, ...], output_tokens_per_call: int) -> TokenStats:
    """Estimate average and total token usage for the baseline strategy."""

    examples = load_risk_examples(dataset_dir, splits)
    if not examples:
        raise ValueError(f"no risk-section examples found under {dataset_dir}")

    current_lengths: list[int] = []
    previous_lengths: list[int] = []
    prompt_lengths: list[int] = []

    for current_risk, previous_risk in examples:
        previous_text = previous_risk or "N/A"
        prompt = USER_PROMPT_TEMPLATE.format(
            current_risk=current_risk,
            previous_risk=previous_text,
        )
        current_lengths.append(estimate_tokens(current_risk))
        previous_lengths.append(estimate_tokens(previous_text))
        prompt_lengths.append(estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(prompt))

    num_examples = len(examples)
    total_input_tokens = sum(prompt_lengths)
    total_output_tokens = num_examples * output_tokens_per_call
    return TokenStats(
        num_examples=num_examples,
        avg_current_tokens=sum(current_lengths) / num_examples,
        avg_previous_tokens=sum(previous_lengths) / num_examples,
        avg_prompt_tokens=sum(prompt_lengths) / num_examples,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
    )


def resolve_prices(args: argparse.Namespace) -> tuple[float, float, str]:
    """Resolve input/output token prices from either CLI input or built-in table."""

    if args.input_cost_per_1m is not None or args.output_cost_per_1m is not None:
        if args.input_cost_per_1m is None or args.output_cost_per_1m is None:
            raise ValueError("provide both --input-cost-per-1m and --output-cost-per-1m")
        return args.input_cost_per_1m, args.output_cost_per_1m, "custom"

    if args.model is None:
        raise ValueError("provide either --model or both custom cost arguments")
    if args.model not in OPENAI_PRICING_PER_1M:
        raise ValueError(f"model not found in built-in OpenAI pricing table: {args.model}")
    pricing = OPENAI_PRICING_PER_1M[args.model]
    return pricing["input"], pricing["output"], args.model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--input-cost-per-1m", type=float)
    parser.add_argument("--output-cost-per-1m", type=float)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/financial-reports-sec/large"))
    parser.add_argument("--splits", nargs="+", default=("train", "test"))
    parser.add_argument("--output-tokens-per-call", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_cost_per_1m, output_cost_per_1m, pricing_source = resolve_prices(args)
    stats = compute_token_stats(args.dataset_dir, tuple(args.splits), args.output_tokens_per_call)

    input_cost = stats.total_input_tokens * input_cost_per_1m / 1_000_000
    output_cost = stats.total_output_tokens * output_cost_per_1m / 1_000_000
    total_cost = input_cost + output_cost

    print(f"pricing_source={pricing_source}")
    print(f"num_examples={stats.num_examples}")
    print(f"avg_current_section_tokens={stats.avg_current_tokens:.2f}")
    print(f"avg_previous_section_tokens={stats.avg_previous_tokens:.2f}")
    print(f"avg_prompt_tokens={stats.avg_prompt_tokens:.2f}")
    print(f"total_input_tokens={stats.total_input_tokens}")
    print(f"total_output_tokens={stats.total_output_tokens}")
    print(f"total_tokens_passed={stats.total_input_tokens + stats.total_output_tokens}")
    print(f"estimated_input_cost_usd={input_cost:.6f}")
    print(f"estimated_output_cost_usd={output_cost:.6f}")
    print(f"estimated_total_cost_usd={total_cost:.6f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Baseline LLM strategy on 10-K risk sections.

The script reads 10-K filings from the local SEC dataset shards, extracts the
``section_1A`` risk-factor text, optionally includes the prior year's
``section_1A`` for the same issuer via ``--use-past-context``, and asks an LLM
for a binary market view.

Predictions are saved to ``predictions/<run_name>.parquet`` as a wide panel with
filing dates as index, Yahoo tickers as columns, and values in ``{-1, 1}`` or
``NaN`` when a model response cannot be parsed. A row-level checkpoint is also
saved to ``predictions/<run_name>.rows.parquet`` so runs can resume reliably.

The strategy can optionally read precomputed risk summaries from
``data/financial-reports-sec/summary/summaries.parquet`` instead of raw
``section_1A`` text.

Examples
--------
OpenAI model::

    OPENAI_API_KEY=... uv run python src/strategies/baseline.py \
        --provider openai \
        --model gpt-4.1-nano

Hugging Face model::

    uv run python src/strategies/baseline.py \
        --provider hf \
        --model Qwen/Qwen3-8B \
        --batch-size 16

vLLM model on precomputed summaries instead of raw risk sections::

    uv run python src/strategies/baseline.py \
        --provider vllm \
        --model Qwen/Qwen3-8B \
        --batch-size 16
        --summaries-path data/financial-reports-sec/summary/summaries.filtered.parquet
        --verbose

Run only a few tickers::

    uv run python src/strategies/baseline.py \
        --provider hf \
        --model Qwen/Qwen3-8B \
        --tickers AAPL NVDA MSFT AMZN GOOGL META

Run with prior-year risk-section context::

    uv run python src/strategies/baseline.py \
        --provider vllm \
        --model Qwen/Qwen3-8B \
        --batch-size 16 \
        --tickers AAPL NVDA MSFT AMZN META \
        --use-past-context

Hugging Face models are loaded once per run and cached locally by the
``transformers`` / Hugging Face stack after the first download.

The repository dependency manifest is intended to support a vLLM-first runtime.
If you maintain a separate Hugging Face-only environment, keep its PyTorch/CUDA
stack aligned with your local installation.

Use ``--resume`` to continue from existing row-level checkpoints in
``predictions/<run_name>.rows.parquet`` and skip examples that were already
processed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from openai import OpenAI

from data.build_10k_market_data import sec_ticker_to_yahoo

SECTION_END_MARKERS = (
    "\nITEM 1B",
    "\nITEM 2",
    "\nUNRESOLVED STAFF COMMENTS",
    "\nPROPERTIES",
    "\nITEM 3",
)

SYSTEM_PROMPT = """You are a financial analyst.
You are reading only the Item 1A Risk Factors section of a 10-K.
Judge the signal based on whether the filing introduces materially new risks compared to the previous year.
Output exactly one label from this set:
NEW
NO_NEW
"""

USER_PROMPT = """Current year Item 1A Risk Factors:
{current_risk}

Previous year Item 1A Risk Factors:
{previous_risk}

Compare this year's risk disclosure with last year's and decide whether the current filing introduces materially new risks.
Choose exactly one label:
NEW = the current filing introduces meaningful new risks compared to last year
NO_NEW = the current filing is mostly the same as last year and any changes are minor or not meaningful

Answer with exactly one label only."""

USER_PROMPT_CURRENT_ONLY = """Current year Item 1A Risk Factors:
{current_risk}

This is only one section of the full 10-K, and risk language is expected to be negative.
Choose exactly one label:
NEW = the filing highlights specific or emerging risks that appear meaningfully new or newly emphasized rather than standard boilerplate
NO_NEW = the filing appears mostly routine, generic, or standard for a 10-K risk section, without meaningful new risks

Answer with exactly one label."""

LABEL_TO_PREDICTION = {
    "NEW": -1,
    "NO_NEW": 1,
}


@dataclass(frozen=True)
class FilingExample:
    filing_date: str
    yahoo_ticker: str
    current_risk: str
    previous_risk: str


def clean_risk_section_text(section_parts: list[object] | tuple[object, ...]) -> str:
    """Normalize and truncate noisy section_1A text to the actual risk section."""

    text = "\n".join(str(part) for part in section_parts).strip()
    if not text:
        return ""

    upper = text.upper()
    start_idx = upper.find("ITEM 1A")
    if start_idx >= 0:
        text = text[start_idx:]
        upper = upper[start_idx:]

    cut_idx = len(text)
    for marker in SECTION_END_MARKERS:
        marker_idx = upper.find(marker)
        if marker_idx >= 0:
            cut_idx = min(cut_idx, marker_idx)
    return text[:cut_idx].strip()


def load_examples(
    dataset_dir: Path,
    splits: tuple[str, ...],
    tickers_filter: set[str] | None = None,
) -> list[FilingExample]:
    """Load current and prior-year risk-section examples from SEC shards."""

    examples: list[FilingExample] = []
    for split in splits:
        for shard_path in sorted((dataset_dir / split).glob("*.jsonl")):
            with shard_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    tickers = [sec_ticker_to_yahoo(str(ticker)) for ticker in record.get("tickers", [])]
                    if tickers_filter is not None:
                        tickers = [ticker for ticker in tickers if ticker in tickers_filter]
                    if not tickers:
                        continue
                    filings = sorted(record.get("filings", []), key=lambda filing: filing["filingDate"])
                    previous_risk = ""
                    for filing in filings:
                        report = filing.get("report", {})
                        section = report.get("section_1A")
                        if not section:
                            continue
                        current_risk = clean_risk_section_text(section)
                        if not current_risk:
                            continue
                        for ticker in tickers:
                            examples.append(
                                FilingExample(
                                    filing_date=str(filing["filingDate"]),
                                    yahoo_ticker=ticker,
                                    current_risk=current_risk,
                                    previous_risk=previous_risk,
                                )
                            )
                        previous_risk = current_risk
    return examples


def load_summary_examples(
    summaries_path: Path,
    splits: tuple[str, ...],
    tickers_filter: set[str] | None = None,
) -> list[FilingExample]:
    """Load current and prior-year examples from normalized summary parquet."""

    frame = pd.read_parquet(summaries_path)
    if frame.empty:
        return []
    required_columns = {"date", "ticker", "summary", "split"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"summary parquet missing required columns: {sorted(missing)}")
    frame = frame[frame["split"].astype(str).isin(splits)].copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["date"] = frame["date"].astype(str)
    frame["summary"] = frame["summary"].astype(str)
    if tickers_filter is not None:
        frame = frame[frame["ticker"].isin(tickers_filter)].copy()
    # Summary generation can append duplicate rows for the same filing/ticker.
    # Keep the last occurrence so later reruns win deterministically.
    frame = frame.drop_duplicates(subset=["ticker", "date"], keep="last")
    frame = frame.sort_values(["ticker", "date"])

    examples: list[FilingExample] = []
    previous_by_ticker: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        ticker = str(row.ticker)
        current_summary = str(row.summary).strip()
        if not current_summary:
            continue
        examples.append(
            FilingExample(
                filing_date=str(row.date),
                yahoo_ticker=ticker,
                current_risk=current_summary,
                previous_risk=previous_by_ticker.get(ticker, ""),
            )
        )
        previous_by_ticker[ticker] = current_summary
    return examples


def load_backend(provider: str, model: str) -> object:
    """Load the inference backend once."""

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
        return OpenAI(api_key=api_key)

    if provider == "hf":
        try:
            import torch  # type: ignore
            from transformers import pipeline  # type: ignore
        except ImportError as exc:
            raise RuntimeError("transformers is required for provider=hf") from exc
        pipeline_kwargs: dict[str, object] = {"model": model}
        if torch.cuda.is_available():
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            pipeline_kwargs["torch_dtype"] = torch.float16
            pipeline_kwargs["device"] = 0
            pipeline_kwargs["model_kwargs"] = {"attn_implementation": "flash_attention_2"}
            try:
                generator = pipeline("text-generation", **pipeline_kwargs)
                setattr(generator, "_attn_implementation", "flash_attention_2")
                return generator
            except Exception as exc:
                print(
                    "FlashAttention-2 unavailable; falling back to default attention "
                    f"implementation ({exc})",
                    flush=True,
                )
                pipeline_kwargs.pop("model_kwargs", None)
        generator = pipeline("text-generation", **pipeline_kwargs)
        setattr(
            generator,
            "_attn_implementation",
            "default" if torch.cuda.is_available() else "default_cpu",
        )
        return generator

    if provider == "vllm":
        try:
            from vllm import LLM  # type: ignore
        except ImportError as exc:
            raise RuntimeError("vllm is required for provider=vllm") from exc
        return LLM(model=model, max_model_len=16384, gpu_memory_utilization=0.90)

    raise ValueError(f"unsupported provider: {provider}")


def run_openai(client: OpenAI, model: str, prompt: str) -> str:
    """Call the OpenAI Responses API via the official Python SDK."""

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return str(response.output_text).strip()


def build_hf_prompt(generator: object, system_prompt: str, user_prompt: str) -> str:
    """Format prompts for Hugging Face models, preferring chat templates when available."""

    tokenizer = getattr(generator, "tokenizer", None)
    chat_template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
    if tokenizer is not None and chat_template:
        return str(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return f"{system_prompt}\n\n{user_prompt}"


def run_hf(generator: object, prompt: str) -> str:
    """Run a local Hugging Face causal LM via transformers."""

    full_prompt = build_hf_prompt(generator, SYSTEM_PROMPT, prompt)
    generation_config = copy.deepcopy(generator.model.generation_config)
    generation_config.max_new_tokens = 512
    generation_config.max_length = None
    result = generator(
        full_prompt,
        do_sample=False,
        generation_config=generation_config,
        return_full_text=False,
        clean_up_tokenization_spaces=False,
    )
    return str(result[0]["generated_text"]).strip()


def run_hf_batch(generator: object, prompts: list[str]) -> list[str]:
    """Run a local Hugging Face causal LM on a batch of prompts."""

    full_prompts = [build_hf_prompt(generator, SYSTEM_PROMPT, prompt) for prompt in prompts]
    generation_config = copy.deepcopy(generator.model.generation_config)
    generation_config.max_new_tokens = 512
    generation_config.max_length = None
    result = generator(
        full_prompts,
        do_sample=False,
        generation_config=generation_config,
        return_full_text=False,
        clean_up_tokenization_spaces=False,
    )
    outputs: list[str] = []
    if isinstance(result, list) and result and isinstance(result[0], list):
        for item in result:
            outputs.append(str(item[0]["generated_text"]).strip())
        return outputs
    for item in result:
        outputs.append(str(item["generated_text"]).strip())
    return outputs


def run_vllm(generator: object, prompt: str) -> str:
    """Run a local vLLM model for one prompt."""

    outputs = run_vllm_batch(generator, [prompt])
    return outputs[0] if outputs else ""


def run_vllm_batch(generator: object, prompts: list[str]) -> list[str]:
    """Run a local vLLM model on a batch of prompts."""

    try:
        from vllm import SamplingParams  # type: ignore
    except ImportError as exc:
        raise RuntimeError("vllm is required for provider=vllm") from exc

    full_prompts = [build_hf_prompt(generator, SYSTEM_PROMPT, prompt) for prompt in prompts]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
    )
    results = generator.generate(full_prompts, sampling_params)
    outputs: list[str] = []
    for result in results:
        if not result.outputs:
            outputs.append("")
            continue
        outputs.append(str(result.outputs[0].text).strip())
    return outputs


def normalize_label_token(token: str) -> str:
    """Normalize common label variants to NEW / NO_NEW."""

    upper = token.upper().replace("\\_", "_")
    normalized = re.sub(r"[\s\-_]+", "_", upper).strip("_")
    if normalized == "NEW":
        return "NEW"
    if normalized == "NO_NEW":
        return "NO_NEW"
    raise ValueError(f"unsupported label token: {token!r}")


def extract_explicit_label(text: str) -> str | None:
    """Recover the model's intended label from noisy completions."""

    patterns = (
        re.compile(r"\\BOXED\{\s*(NO(?:\\_|[\s\-_])?NEW|NEW)\s*\}", flags=re.IGNORECASE),
        re.compile(
            r"(?:^|\n)\s*(?:ANSWER|FINAL ANSWER|LABEL|PREDICTION)\s*:\s*(NO(?:\\_|[\s\-_])?NEW|NEW)\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?:^|\n)\s*(NO(?:\\_|[\s\-_])?NEW|NEW)\s*(?:$|\n)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:THE ANSWER IS|THE LABEL IS|THE CORRECT LABEL IS|THE APPROPRIATE LABEL IS|"
            r"THE LABEL SHOULD BE|THE CORRECT LABEL SHOULD BE|THE APPROPRIATE LABEL SHOULD BE)\s*"
            r"(?:[:\-]\s*)?[\"']?(NO(?:\\_|[\s\-_])?NEW|NEW)\b",
            flags=re.IGNORECASE,
        ),
    )
    matches: list[tuple[int, str]] = []
    for pattern_index, pattern in enumerate(patterns):
        for match in pattern.finditer(text):
            token = normalize_label_token(str(match.group(1)))
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_prefix = text[line_start:match.start()].upper()
            if pattern_index != 0 and ("FORMAT" in line_prefix or "EXACTLY ONE LABEL" in line_prefix):
                continue
            matches.append((match.start(), token))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def to_prediction(text: str) -> int:
    """Convert model text output to {-1, 1} via NEW vs NO_NEW labels."""

    without_think = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    upper = without_think.strip().upper().replace("\\_", "_")
    boxed_match = re.fullmatch(r"(?:\\BOXED\{)?(NO(?:[\s\-_])?NEW|NEW)(?:\})?\W*", upper)
    if boxed_match:
        return LABEL_TO_PREDICTION[normalize_label_token(str(boxed_match.group(1)))]

    explicit_label = extract_explicit_label(without_think)
    if explicit_label is not None:
        return LABEL_TO_PREDICTION[explicit_label]

    labels = [normalize_label_token(match) for match in re.findall(r"\b(NO(?:[\s\-_])?NEW|NEW)\b", upper)]
    unique_labels = set(labels)
    if len(unique_labels) == 1 and labels:
        if "FORMAT" in upper or "EXACTLY ONE LABEL" in upper:
            raise ValueError(f"model echoed formatting instructions instead of answering: {text!r}")
        return LABEL_TO_PREDICTION[labels[0]]
    raise ValueError(f"could not parse model output as risk-change label: {text!r}")


def infer_one(
    provider: str,
    backend: object,
    model: str,
    example: FilingExample,
    use_past_context: bool,
    verbose: bool = False,
) -> tuple[int, str]:
    """Run one example through the chosen LLM backend."""

    if use_past_context:
        prompt = USER_PROMPT.format(
            current_risk=example.current_risk,
            previous_risk=example.previous_risk or "N/A",
        )
    else:
        prompt = USER_PROMPT_CURRENT_ONLY.format(current_risk=example.current_risk)
    if provider == "openai":
        raw = run_openai(backend, model, prompt)
    elif provider == "hf":
        raw = run_hf(backend, prompt)
    elif provider == "vllm":
        raw = run_vllm(backend, prompt)
    else:
        raise ValueError(f"unsupported provider: {provider}")
    if verbose:
        print(
            f"Raw model output for {example.yahoo_ticker} {example.filing_date}: {raw!r}",
            flush=True,
        )
    return to_prediction(raw), raw


def infer_batch(
    provider: str,
    backend: object,
    model: str,
    examples: list[FilingExample],
    use_past_context: bool,
    verbose: bool = False,
) -> list[tuple[float, str]]:
    """Run a batch of examples through the chosen backend."""

    if provider == "openai":
        predictions: list[tuple[float, str]] = []
        for example in examples:
            parsed_prediction, raw = infer_one(
                provider,
                backend,
                model,
                example,
                use_past_context,
                verbose=verbose,
            )
            predictions.append((float(parsed_prediction), raw))
        return predictions

    prompts = [
        USER_PROMPT.format(
            current_risk=example.current_risk,
            previous_risk=example.previous_risk or "N/A",
        )
        if use_past_context
        else USER_PROMPT_CURRENT_ONLY.format(current_risk=example.current_risk)
        for example in examples
    ]
    if provider == "hf":
        raw_outputs = run_hf_batch(backend, prompts)
    elif provider == "vllm":
        raw_outputs = run_vllm_batch(backend, prompts)
    else:
        raise ValueError(f"unsupported provider: {provider}")
    predictions: list[tuple[float, str]] = []
    for example, raw in zip(examples, raw_outputs):
        if verbose:
            print(
                f"Raw model output for {example.yahoo_ticker} {example.filing_date}: {raw!r}",
                flush=True,
            )
        try:
            prediction = float(to_prediction(raw))
        except ValueError:
            prediction = float("nan")
        predictions.append((prediction, raw))
    return predictions


def rows_to_panel(rows: pd.DataFrame) -> pd.DataFrame:
    """Convert long-form prediction rows into a wide date-by-ticker panel."""

    if rows.empty:
        return pd.DataFrame()
    panel = rows.pivot_table(index="date", columns="ticker", values="prediction", aggfunc="last")
    panel.index = pd.to_datetime(panel.index, errors="raise")
    panel = panel.sort_index()
    panel.columns = panel.columns.astype(str)
    return panel


def save_prediction_checkpoints(
    rows: pd.DataFrame,
    rows_output_path: Path,
    panel_output_path: Path,
) -> pd.DataFrame:
    """Persist row-level checkpoints and the wide panel output."""

    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(rows_output_path, index=False)
    panel = rows_to_panel(rows)
    panel.to_parquet(panel_output_path)
    return panel


def load_existing_predictions(
    rows_output_path: Path,
    panel_output_path: Path,
) -> pd.DataFrame:
    """Load prior predictions for resume, preferring row-level checkpoints."""

    if rows_output_path.exists():
        rows = pd.read_parquet(rows_output_path)
        if rows.empty:
            return pd.DataFrame(columns=["date", "ticker", "prediction"])
        rows = rows.loc[:, ["date", "ticker", "prediction"]].copy()
        rows["date"] = rows["date"].astype(str)
        rows["ticker"] = rows["ticker"].astype(str)
        return rows

    if panel_output_path.exists():
        panel = pd.read_parquet(panel_output_path)
        rows = (
            panel.stack(dropna=True)
            .rename("prediction")
            .reset_index()
            .rename(columns={"level_0": "date", "level_1": "ticker"})
        )
        rows["date"] = pd.to_datetime(rows["date"], errors="raise").dt.strftime("%Y-%m-%d")
        rows["ticker"] = rows["ticker"].astype(str)
        return rows.loc[:, ["date", "ticker", "prediction"]]

    return pd.DataFrame(columns=["date", "ticker", "prediction"])


def build_prediction_panel(
    provider: str,
    model: str,
    examples: list[FilingExample],
    use_past_context: bool,
    output_path: Path,
    rows_output_path: Path,
    resume: bool,
    save_every: int,
    batch_size: int,
    verbose: bool,
) -> pd.DataFrame:
    """Infer predictions, checkpoint progress, and return the wide panel."""

    existing_rows = load_existing_predictions(rows_output_path, output_path) if resume else pd.DataFrame(
        columns=["date", "ticker", "prediction"]
    )
    processed_keys = set(zip(existing_rows["date"], existing_rows["ticker"])) if not existing_rows.empty else set()
    print(
        f"Resume={resume} | existing_predictions={len(processed_keys)} | "
        f"save_every={save_every} | batch_size={batch_size}",
        flush=True,
    )

    pending_examples = [
        example
        for example in examples
        if (example.filing_date, example.yahoo_ticker) not in processed_keys
    ]
    print(
        f"Pending examples: {len(pending_examples)} / {len(examples)}",
        flush=True,
    )
    if not pending_examples:
        print("No pending examples to process", flush=True)
        return rows_to_panel(existing_rows)

    print(f"Loading backend: provider={provider} model={model}", flush=True)
    backend = load_backend(provider, model)
    print("Backend loaded", flush=True)
    if provider == "hf":
        attn_implementation = getattr(backend, "_attn_implementation", "unknown")
        print(f"HF attention implementation: {attn_implementation}", flush=True)
    buffered_rows: list[dict[str, object]] = []
    all_rows = existing_rows.copy()
    effective_batch_size = batch_size if provider in {"hf", "vllm"} else 1
    for batch_start in range(0, len(pending_examples), effective_batch_size):
        batch_examples = pending_examples[batch_start : batch_start + effective_batch_size]
        if provider in {"hf", "vllm"}:
            batch_results = infer_batch(
                provider,
                backend,
                model,
                batch_examples,
                use_past_context,
                verbose=verbose,
            )
        else:
            batch_results = []
            for example in batch_examples:
                try:
                    parsed_prediction, raw_output = infer_one(
                        provider,
                        backend,
                        model,
                        example,
                        use_past_context,
                        verbose=verbose,
                    )
                    prediction = float(parsed_prediction)
                except ValueError as exc:
                    prediction = float("nan")
                    raw_output = str(exc)
                batch_results.append((prediction, raw_output))

        for batch_offset, (example, result) in enumerate(zip(batch_examples, batch_results), start=1):
            index = batch_start + batch_offset
            prediction, raw_output = result
            if pd.isna(prediction):
                print(
                    f"[{index}/{len(examples)}] {example.yahoo_ticker} "
                    f"{example.filing_date}: NaN (could not parse model output as risk-change label: {raw_output!r})",
                    flush=True,
                )
            else:
                print(
                    f"[{index}/{len(examples)}] {example.yahoo_ticker} "
                    f"{example.filing_date}: {int(prediction)}",
                    flush=True,
                )
            buffered_rows.append(
                {
                    "date": example.filing_date,
                    "ticker": example.yahoo_ticker,
                    "prediction": prediction,
                }
            )
            processed_keys.add((example.filing_date, example.yahoo_ticker))

        if len(buffered_rows) >= save_every:
            all_rows = pd.concat([all_rows, pd.DataFrame(buffered_rows)], ignore_index=True)
            buffered_rows.clear()
            print(
                f"Checkpointing {len(all_rows)} total predictions to "
                f"{rows_output_path} and {output_path}",
                flush=True,
            )
            save_prediction_checkpoints(all_rows, rows_output_path, output_path)

    if buffered_rows:
        all_rows = pd.concat([all_rows, pd.DataFrame(buffered_rows)], ignore_index=True)

    print(
        f"Final checkpoint with {len(all_rows)} total predictions to "
        f"{rows_output_path} and {output_path}",
        flush=True,
    )
    return save_prediction_checkpoints(all_rows, rows_output_path, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "hf", "vllm"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name", default="baseline_strategy")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/financial-reports-sec/large"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("predictions"))
    parser.add_argument(
        "--summaries-path",
        type=Path,
        help="Optional normalized summary parquet to use instead of raw section_1A text",
    )
    parser.add_argument("--splits", nargs="+", default=("train", "test"))
    parser.add_argument("--tickers", nargs="+", help="Optional Yahoo tickers to restrict the run to")
    parser.add_argument("--use-past-context", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip already checkpointed predictions")
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Checkpoint predictions every N newly processed examples",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for local HF or vLLM inference",
    )
    parser.add_argument("--verbose", action="store_true", help="Print raw model outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers_filter = {ticker.upper() for ticker in args.tickers} if args.tickers else None
    print(
        f"Run name: {args.run_name} | provider={args.provider} | model={args.model}",
        flush=True,
    )
    print(
        f"Dataset dir: {args.dataset_dir} | splits={list(args.splits)} | "
        f"tickers={sorted(tickers_filter) if tickers_filter else 'ALL'} | "
        f"use_past_context={args.use_past_context} | "
        f"resume={args.resume} | save_every={args.save_every} | "
        f"batch_size={args.batch_size} | "
        f"verbose={args.verbose}",
        flush=True,
    )
    print("Loading filing examples...", flush=True)
    if args.summaries_path is not None:
        print(f"Using summaries from {args.summaries_path}", flush=True)
        examples = load_summary_examples(args.summaries_path, tuple(args.splits), tickers_filter=tickers_filter)
    else:
        examples = load_examples(args.dataset_dir, tuple(args.splits), tickers_filter=tickers_filter)
    print(f"Loaded {len(examples)} filing examples", flush=True)
    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.predictions_dir / f"{args.run_name}.parquet"
    rows_output_path = args.predictions_dir / f"{args.run_name}.rows.parquet"
    panel = build_prediction_panel(
        args.provider,
        args.model,
        examples,
        use_past_context=args.use_past_context,
        output_path=output_path,
        rows_output_path=rows_output_path,
        resume=args.resume,
        save_every=args.save_every,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
    print(f"Saved predictions to {output_path}", flush=True)
    print(f"Saved row checkpoints to {rows_output_path}", flush=True)
    print(output_path)


if __name__ == "__main__":
    main()

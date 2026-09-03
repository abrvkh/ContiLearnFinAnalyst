#!/usr/bin/env python3
"""Summarize 10-K Item 1A risk sections into short key-risk summaries.

The script reads local SEC dataset shards, extracts ``section_1A`` from each
filing, and writes a concise summary for each filing to mirrored JSONL shards
under ``data/financial-reports-sec/summary`` by default.

Each output row is normalized to one ``(date, ticker)`` pair plus a
``risk_summary`` field capped at 300 words by prompt instruction. Mirrored JSONL
shards are used for incremental checkpointing, and a consolidated parquet is
also written for downstream strategy use.

The same entry point also supports a second-pass cleanup mode that reviews
generated summaries and filters out low-value outputs.

Examples
--------
Generate summaries with OpenAI::

    OPENAI_API_KEY=... uv run python src/data/summarize_risk_sections.py \
        --provider openai \
        --model gpt-4.1-nano

Generate summaries with Hugging Face::

    uv run python src/data/summarize_risk_sections.py \
        --provider hf \
        --model Qwen/Qwen3-8B \
        --batch-size 16

Generate summaries with vLLM::

    uv run python src/data/summarize_risk_sections.py \
        --provider vllm \
        --model Qwen/Qwen3-8B \
        --batch-size 16

Generate summaries for a small subset of filings::

    uv run python src/data/summarize_risk_sections.py \
        --provider vllm \
        --model Qwen/Qwen3-8B \
        --batch-size 16 \
        --splits train \
        --tickers AAPL MSFT NVDA \
        --resume

Clean existing summaries with Hugging Face::

    uv run python src/data/summarize_risk_sections.py \
        --mode clean \
        --provider hf \
        --model Qwen/Qwen3-8B \
        --batch-size 16 \
        --resume

Clean existing summaries with vLLM::

    uv run python src/data/summarize_risk_sections.py \
        --mode clean \
        --provider vllm \
        --model Qwen/Qwen3-8B \
        --batch-size 16 \
        --resume

Run baseline on the filtered summaries::

    uv run python src/strategies/baseline.py \
        --provider vllm \
        --model Qwen/Qwen3-8B \
        --batch-size 16 \
        --summaries-path data/financial-reports-sec/summary/summaries.filtered.parquet

Use ``--resume`` in generate mode to skip filings that already have checkpointed
JSONL output under ``data/financial-reports-sec/summary/<split>/<shard>.jsonl``.
Use ``--resume`` in clean mode to skip rows that are already present in the
review parquet outputs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
You are reading only the Item 1A Risk Factors section of a 10-K, not the full filing.

Extract only the key risks.
Focus on the most material concrete risks, not boilerplate language.
Do not add interpretation about stock direction.
Do not include headings, introductions, conclusions, or text from other 10-K sections.
Return only short bullet points.
Keep the full output under 300 words."""

USER_PROMPT = """Summarize the following Item 1A Risk Factors section as short bullet points only.

Requirements:
- Include only the key risks.
- Use 3 to 8 bullets.
- Each bullet should be one concise sentence fragment.
- No intro sentence.
- No conclusion.
- No section titles.
- No quotes.
- No commentary beyond the risks themselves.
- Keep the full output under 300 words.

Risk Factors section:
{current_risk}"""

INTRO_PATTERNS = (
    "here are the key risks",
    "based on the provided",
    "the following risk factors",
    "the following is a list",
    "these risk factors could",
)

REVIEW_SYSTEM_PROMPT = """You are reviewing a generated summary of a 10-K Item 1A Risk Factors section.

Output exactly one word as the first and only token of your answer:
KEEP
RESUMMARIZE
DROP

Return KEEP only if the summary is mostly useful, specific, and readable as-is.
Return RESUMMARIZE if it contains meaningful risk content but needs rewriting because of filler, repetition, chain-of-thought, formatting problems, or partial contamination.
Return DROP if it is dominated by boilerplate, repetition, truncation, non-risk content, or generic filler.
"""

REVIEW_USER_PROMPT = """Review this risk summary.

Answer with exactly one word only:
KEEP
RESUMMARIZE
DROP

Mark it DROP if any of these dominate:
- repeated sentences or obvious looping
- generic filler or introductions instead of key risks
- text from unrelated sections
- severe truncation or cut-off output
- content too vague to be useful downstream

Summary:
{summary}
"""

RESUMMARIZE_SYSTEM_PROMPT = """You are rewriting a generated summary of a 10-K Item 1A Risk Factors section.

Extract only the useful key risks from the provided summary.
Do not preserve chain-of-thought, filler, repetition, headings, intros, or conclusions.
Return only short bullet points.
Use 3 to 8 bullets.
Keep the full output under 220 words."""

RESUMMARIZE_USER_PROMPT = """Rewrite this summary into clean key-risk bullets only.

Requirements:
- Keep only meaningful risk content.
- Remove any meta reasoning, chain-of-thought, filler, repeated text, or unrelated material.
- Use short bullet points only.
- No intro sentence.
- No conclusion.

Summary to rewrite:
{summary}"""


@dataclass(frozen=True)
class FilingRiskSection:
    split: str
    shard_name: str
    cik: str
    company_name: str
    filing_date: str
    report_date: str
    form: str
    tickers: tuple[str, ...]
    yahoo_tickers: tuple[str, ...]
    current_risk: str


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
    text = text[:cut_idx].strip()
    return text


def iter_risk_sections(
    dataset_dir: Path,
    splits: tuple[str, ...],
    tickers_filter: set[str] | None = None,
) -> list[FilingRiskSection]:
    """Load filing-level risk sections from SEC shards."""

    examples: list[FilingRiskSection] = []
    for split in splits:
        for shard_path in sorted((dataset_dir / split).glob("*.jsonl")):
            with shard_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    yahoo_tickers = tuple(
                        sec_ticker_to_yahoo(str(ticker)) for ticker in record.get("tickers", [])
                    )
                    if tickers_filter is not None:
                        yahoo_tickers = tuple(
                            ticker for ticker in yahoo_tickers if ticker in tickers_filter
                        )
                    if not yahoo_tickers:
                        continue
                    filings = sorted(record.get("filings", []), key=lambda filing: filing["filingDate"])
                    for filing in filings:
                        report = filing.get("report", {})
                        section = report.get("section_1A")
                        if not section:
                            continue
                        current_risk = clean_risk_section_text(section)
                        if not current_risk:
                            continue
                        examples.append(
                            FilingRiskSection(
                                split=split,
                                shard_name=shard_path.name,
                                cik=str(record.get("cik", "")),
                                company_name=str(record.get("name", "")),
                                filing_date=str(filing.get("filingDate", "")),
                                report_date=str(filing.get("reportDate", "")),
                                form=str(filing.get("form", "")),
                                tickers=tuple(str(ticker) for ticker in record.get("tickers", [])),
                                yahoo_tickers=yahoo_tickers,
                                current_risk=current_risk,
                            )
                        )
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


def run_openai_with_system(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> str:
    """Call the OpenAI Responses API with an explicit system prompt."""

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
    generation_config.max_new_tokens = 300
    generation_config.max_length = None
    generation_config.do_sample = False
    result = generator(
        full_prompt,
        generation_config=generation_config,
        return_full_text=False,
        clean_up_tokenization_spaces=False,
    )
    return str(result[0]["generated_text"]).strip()


def run_hf_batch(
    generator: object,
    system_prompt: str,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    """Run a local Hugging Face causal LM on a batch of prompts."""

    full_prompts = [build_hf_prompt(generator, system_prompt, prompt) for prompt in prompts]
    generation_config = copy.deepcopy(generator.model.generation_config)
    generation_config.max_new_tokens = max_new_tokens
    generation_config.max_length = None
    generation_config.do_sample = False
    result = generator(
        full_prompts,
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


def run_hf_with_system(
    generator: object,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
) -> str:
    """Run a local Hugging Face model with an explicit system prompt."""

    full_prompt = build_hf_prompt(generator, system_prompt, user_prompt)
    generation_config = copy.deepcopy(generator.model.generation_config)
    generation_config.max_new_tokens = max_new_tokens
    generation_config.max_length = None
    generation_config.do_sample = False
    result = generator(
        full_prompt,
        generation_config=generation_config,
        return_full_text=False,
        clean_up_tokenization_spaces=False,
    )
    return str(result[0]["generated_text"]).strip()


def parse_review_label(raw: str) -> str | None:
    """Extract a review label from imperfect model output."""

    without_think = re.sub(r"<think>.*?</think>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
    upper = without_think.strip().upper()
    if not upper:
        return None

    lines = [line.strip() for line in upper.splitlines() if line.strip()]
    for line in lines:
        match = re.fullmatch(r"(KEEP|RESUMMARIZE|DROP)\W*", line)
        if match:
            return str(match.group(1))

    tokens = re.findall(r"\b(KEEP|RESUMMARIZE|DROP)\b", upper)
    if len(set(tokens)) == 1:
        return tokens[0]
    return None


def run_vllm_batch(
    engine: object,
    system_prompt: str,
    user_prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    """Run a local vLLM model on a batch of prompts."""

    try:
        from vllm import SamplingParams  # type: ignore
    except ImportError as exc:
        raise RuntimeError("vllm is required for provider=vllm") from exc

    full_prompts = [f"{system_prompt}\n\n{prompt}" for prompt in user_prompts]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )
    outputs = engine.generate(full_prompts, sampling_params)
    raw_by_prompt: dict[str, str] = {}
    for output in outputs:
        text = output.outputs[0].text if output.outputs else ""
        raw_by_prompt[str(output.prompt)] = str(text).strip()
    return [raw_by_prompt.get(prompt, "") for prompt in full_prompts]


def run_vllm(engine: object, prompt: str) -> str:
    """Run a local vLLM model for one prompt."""

    return run_vllm_batch(engine, SYSTEM_PROMPT, [prompt], max_new_tokens=300)[0]


def run_vllm_with_system(
    engine: object,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
) -> str:
    """Run a local vLLM model with an explicit system prompt."""

    return run_vllm_batch(engine, system_prompt, [user_prompt], max_new_tokens=max_new_tokens)[0]


def _normalize_summary_line(line: str) -> str:
    """Strip common lead-in boilerplate and normalize one summary line."""

    text = line.strip()
    text = re.sub(r"^[\-\*\u2022]+\s*", "", text)
    lower = text.lower()
    if any(pattern in lower for pattern in INTRO_PATTERNS):
        return ""
    return text.strip(" :;-")


def _dedup_within_line(line: str) -> str:
    """Remove repeated sentence-like fragments within one bullet line."""

    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|;\s+|\s+•\s+", line)
        if part.strip()
    ]
    if len(parts) <= 1:
        return line

    unique_parts: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = re.sub(r"\s+", " ", part).strip(" .").lower()
        if len(key) < 20:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique_parts.append(part.rstrip("."))

    if not unique_parts:
        return line
    return "; ".join(unique_parts)


def clean_generated_summary(text: str) -> str:
    """Normalize summaries to concise unique bullet points."""

    if not text.strip():
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = [line for line in text.split("\n") if line.strip()]

    normalized_lines: list[str] = []
    for line in raw_lines:
        normalized = _normalize_summary_line(line)
        if not normalized:
            continue
        normalized_lines.append(normalized)

    if not normalized_lines:
        sentence_candidates = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", text)
            if part.strip()
        ]
        normalized_lines = [
            normalized
            for normalized in (_normalize_summary_line(part) for part in sentence_candidates)
            if normalized
        ]

    unique_lines: list[str] = []
    seen: set[str] = set()
    for line in normalized_lines:
        line = _dedup_within_line(line)
        dedup_key = re.sub(r"\s+", " ", line).strip(" .").lower()
        if len(dedup_key) < 20:
            continue
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        unique_lines.append(line.rstrip("."))

    if not unique_lines:
        return ""

    return "\n".join(f"- {line}" for line in unique_lines[:8])


def summarize_one(
    provider: str,
    backend: object,
    model: str,
    example: FilingRiskSection,
) -> str:
    """Generate one risk-section summary."""

    prompt = USER_PROMPT.format(current_risk=example.current_risk)
    if provider == "openai":
        raw = run_openai(backend, model, prompt)
    elif provider == "hf":
        raw = run_hf(backend, prompt)
    elif provider == "vllm":
        raw = run_vllm(backend, prompt)
    else:
        raise ValueError(f"unsupported provider: {provider}")
    cleaned = clean_generated_summary(raw)
    if cleaned:
        return cleaned
    return raw.strip()


def summarize_batch(
    provider: str,
    backend: object,
    model: str,
    examples: list[FilingRiskSection],
) -> list[str]:
    """Generate a batch of risk-section summaries."""

    prompts = [USER_PROMPT.format(current_risk=example.current_risk) for example in examples]
    if provider == "openai":
        return [summarize_one(provider, backend, model, example) for example in examples]
    if provider == "hf":
        raw_outputs = run_hf_batch(backend, SYSTEM_PROMPT, prompts, max_new_tokens=300)
    elif provider == "vllm":
        raw_outputs = run_vllm_batch(backend, SYSTEM_PROMPT, prompts, max_new_tokens=300)
    else:
        raise ValueError(f"unsupported provider: {provider}")

    cleaned_outputs: list[str] = []
    for raw in raw_outputs:
        cleaned = clean_generated_summary(raw)
        cleaned_outputs.append(cleaned if cleaned else raw.strip())
    return cleaned_outputs


def rewrite_summary(provider: str, backend: object, model: str, summary: str) -> str:
    """Rewrite a noisy but useful summary into clean bullet points."""

    prompt = RESUMMARIZE_USER_PROMPT.format(summary=summary)
    if provider == "openai":
        raw = run_openai_with_system(backend, model, RESUMMARIZE_SYSTEM_PROMPT, prompt)
    elif provider == "hf":
        raw = run_hf_with_system(backend, RESUMMARIZE_SYSTEM_PROMPT, prompt, max_new_tokens=220)
    elif provider == "vllm":
        raw = run_vllm_with_system(backend, RESUMMARIZE_SYSTEM_PROMPT, prompt, max_new_tokens=220)
    else:
        raise ValueError(f"unsupported provider: {provider}")
    cleaned = clean_generated_summary(raw)
    if cleaned:
        return cleaned
    return raw.strip()


def rewrite_summaries(provider: str, backend: object, model: str, summaries: list[str]) -> list[str]:
    """Rewrite a batch of noisy summaries into clean bullet points."""

    prompts = [RESUMMARIZE_USER_PROMPT.format(summary=summary) for summary in summaries]
    if provider == "openai":
        return [rewrite_summary(provider, backend, model, summary) for summary in summaries]
    if provider == "hf":
        raw_outputs = run_hf_batch(backend, RESUMMARIZE_SYSTEM_PROMPT, prompts, max_new_tokens=220)
    elif provider == "vllm":
        raw_outputs = run_vllm_batch(backend, RESUMMARIZE_SYSTEM_PROMPT, prompts, max_new_tokens=220)
    else:
        raise ValueError(f"unsupported provider: {provider}")

    cleaned_outputs: list[str] = []
    for raw in raw_outputs:
        cleaned = clean_generated_summary(raw)
        cleaned_outputs.append(cleaned if cleaned else raw.strip())
    return cleaned_outputs


def review_summary(provider: str, backend: object, model: str, summary: str) -> tuple[str, str]:
    """Review one generated summary and classify it as KEEP, RESUMMARIZE, or DROP."""

    prompt = REVIEW_USER_PROMPT.format(summary=summary)
    if provider == "openai":
        raw = run_openai_with_system(backend, model, REVIEW_SYSTEM_PROMPT, prompt)
    elif provider == "hf":
        raw = run_hf_with_system(backend, REVIEW_SYSTEM_PROMPT, prompt, max_new_tokens=512)
    elif provider == "vllm":
        raw = run_vllm_with_system(backend, REVIEW_SYSTEM_PROMPT, prompt, max_new_tokens=512)
    else:
        raise ValueError(f"unsupported provider: {provider}")

    label = parse_review_label(raw)
    if label is not None:
        return label, raw
    raise ValueError(f"could not parse summary review label: {raw!r}")


def review_summaries(provider: str, backend: object, model: str, summaries: list[str]) -> list[tuple[str | None, str]]:
    """Review a batch of generated summaries."""

    prompts = [REVIEW_USER_PROMPT.format(summary=summary) for summary in summaries]
    if provider == "openai":
        return [review_summary(provider, backend, model, summary) for summary in summaries]
    if provider == "hf":
        raw_outputs = run_hf_batch(backend, REVIEW_SYSTEM_PROMPT, prompts, max_new_tokens=512)
    elif provider == "vllm":
        raw_outputs = run_vllm_batch(backend, REVIEW_SYSTEM_PROMPT, prompts, max_new_tokens=512)
    else:
        raise ValueError(f"unsupported provider: {provider}")

    return [(parse_review_label(raw), raw) for raw in raw_outputs]


def output_shard_path(output_dir: Path, split: str, shard_name: str) -> Path:
    """Resolve the mirrored JSONL output path for one source shard."""

    return output_dir / split / shard_name


REVIEW_KEY_COLUMNS = (
    "date",
    "ticker",
    "cik",
    "company_name",
    "report_date",
    "form",
    "split",
    "shard",
    "summary",
)


def build_review_row_id(record: dict[str, object]) -> str:
    """Build a stable identifier for one summary row."""

    payload = {column: str(record.get(column, "")) for column in REVIEW_KEY_COLUMNS}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def attach_review_row_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach stable per-row review ids to a summary frame."""

    enriched = frame.copy()
    enriched["review_row_id"] = [
        build_review_row_id({column: row[column] for column in REVIEW_KEY_COLUMNS})
        for _, row in enriched.iterrows()
    ]
    return enriched


def load_processed_keys(path: Path) -> set[tuple[str, str]]:
    """Load existing (cik, filing_date) pairs for resume."""

    if not path.exists():
        return set()
    processed: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            processed.add((str(record["cik"]), str(record["date"])))
    return processed


def append_records(path: Path, records: list[dict[str, object]]) -> None:
    """Append JSONL records to one shard output."""

    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def rebuild_consolidated_parquet(output_dir: Path, parquet_path: Path) -> None:
    """Rebuild a consolidated parquet from checkpoint JSONL shards."""

    rows: list[dict[str, object]] = []
    for path in sorted(output_dir.glob("*/*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "cik",
                "company_name",
                "report_date",
                "form",
                "summary",
                "split",
                "shard",
                "filing_url",
            ]
        )
    frame.to_parquet(parquet_path, index=False)


def load_existing_reviews(review_path: Path) -> pd.DataFrame:
    """Load existing summary review labels for resume."""

    if not review_path.exists():
        return pd.DataFrame(columns=["review_row_id", "action", "review_label", "cleaned_summary"])
    frame = pd.read_parquet(review_path)
    if frame.empty:
        return pd.DataFrame(columns=["review_row_id", "action", "review_label", "cleaned_summary"])
    if "review_row_id" not in frame.columns:
        print(
            f"Existing review parquet at {review_path} uses a legacy schema; ignoring it for resume.",
            flush=True,
        )
        return pd.DataFrame(columns=["review_row_id", "action", "review_label", "cleaned_summary"])
    frame = frame.loc[:, ["review_row_id", "action", "review_label", "cleaned_summary"]].copy()
    frame["review_row_id"] = frame["review_row_id"].astype(str)
    frame["action"] = frame["action"].astype(str)
    frame["cleaned_summary"] = frame["cleaned_summary"].fillna("").astype(str)
    return frame


def write_review_outputs(
    source_frame: pd.DataFrame,
    review_frame: pd.DataFrame,
    review_path: Path,
    filtered_path: Path,
    rejected_path: Path,
) -> None:
    """Persist review labels and materialize kept/rejected summary parquets."""

    review_path.parent.mkdir(parents=True, exist_ok=True)
    if not review_frame.empty:
        review_frame = review_frame.drop_duplicates(subset=["review_row_id"], keep="last").copy()
    review_frame.to_parquet(review_path, index=False)

    merged = source_frame.merge(review_frame, on=["review_row_id"], how="left", validate="many_to_one")
    merged["summary"] = merged["cleaned_summary"].where(
        merged["cleaned_summary"].astype(str).str.strip() != "",
        merged["summary"],
    )
    kept = merged[merged["action"].isin(["KEEP", "RESUMMARIZE"])].copy()
    rejected = merged[merged["action"] == "DROP"].copy()

    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(filtered_path, index=False)
    rejected.to_parquet(rejected_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "clean"), default="generate")
    parser.add_argument("--provider", choices=("openai", "hf", "vllm"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/financial-reports-sec/large"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/financial-reports-sec/summary"),
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=Path("data/financial-reports-sec/summary/summaries.parquet"),
    )
    parser.add_argument(
        "--review-path",
        type=Path,
        default=Path("data/financial-reports-sec/summary/summaries.review.parquet"),
    )
    parser.add_argument(
        "--filtered-path",
        type=Path,
        default=Path("data/financial-reports-sec/summary/summaries.filtered.parquet"),
    )
    parser.add_argument(
        "--rejected-path",
        type=Path,
        default=Path("data/financial-reports-sec/summary/summaries.rejected.parquet"),
    )
    parser.add_argument("--splits", nargs="+", default=("train", "test"))
    parser.add_argument("--tickers", nargs="+", help="Optional Yahoo tickers to restrict the run to")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from existing summary/review outputs (default: enabled)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume and recompute from scratch",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Append summary records every N newly processed filings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation mode when using local inference backends",
    )
    parser.add_argument("--verbose", action="store_true", help="Print raw generated summaries")
    return parser.parse_args()


def run_generate_mode(args: argparse.Namespace) -> None:
    tickers_filter = {ticker.upper() for ticker in args.tickers} if args.tickers else None
    print(
        f"mode=generate | provider={args.provider} | model={args.model} | "
        f"output_dir={args.output_dir}",
        flush=True,
    )
    print(
        f"dataset_dir={args.dataset_dir} | splits={list(args.splits)} | "
        f"tickers={sorted(tickers_filter) if tickers_filter else 'ALL'} | "
        f"resume={args.resume} | save_every={args.save_every} | batch_size={args.batch_size} | "
        f"verbose={args.verbose} | "
        f"parquet_path={args.parquet_path}",
        flush=True,
    )

    print("Loading risk sections...", flush=True)
    examples = iter_risk_sections(args.dataset_dir, tuple(args.splits), tickers_filter=tickers_filter)
    print(f"Loaded {len(examples)} filing risk sections", flush=True)

    processed_by_shard: dict[Path, set[tuple[str, str]]] = {}
    pending_examples: list[FilingRiskSection] = []
    for example in examples:
        shard_path = output_shard_path(args.output_dir, example.split, example.shard_name)
        if shard_path not in processed_by_shard:
            processed_by_shard[shard_path] = load_processed_keys(shard_path) if args.resume else set()
        if (example.cik, example.filing_date) in processed_by_shard[shard_path]:
            continue
        pending_examples.append(example)

    print(f"Pending filings: {len(pending_examples)} / {len(examples)}", flush=True)
    if not pending_examples:
        print("No pending filings to summarize", flush=True)
        return

    print(f"Loading backend: provider={args.provider} model={args.model}", flush=True)
    backend = load_backend(args.provider, args.model)
    print("Backend loaded", flush=True)
    if args.provider == "hf":
        attn_implementation = getattr(backend, "_attn_implementation", "unknown")
        print(f"HF attention implementation: {attn_implementation}", flush=True)

    buffered_by_shard: dict[Path, list[dict[str, object]]] = {}
    effective_batch_size = args.batch_size if args.provider in {"hf", "vllm"} else 1
    for batch_start in range(0, len(pending_examples), effective_batch_size):
        batch_examples = pending_examples[batch_start : batch_start + effective_batch_size]
        summaries = summarize_batch(args.provider, backend, args.model, batch_examples)
        for batch_offset, (example, summary) in enumerate(zip(batch_examples, summaries), start=1):
            index = batch_start + batch_offset
            if args.verbose:
                print(
                    f"Summary for {example.yahoo_tickers or example.tickers} "
                    f"{example.filing_date}: {summary!r}",
                    flush=True,
                )

            shard_path = output_shard_path(args.output_dir, example.split, example.shard_name)
            records = buffered_by_shard.setdefault(shard_path, [])
            for ticker in example.yahoo_tickers:
                records.append(
                    {
                        "date": example.filing_date,
                        "ticker": ticker,
                        "cik": example.cik,
                        "company_name": example.company_name,
                        "report_date": example.report_date,
                        "form": example.form,
                        "summary": summary,
                        "split": example.split,
                        "shard": example.shard_name,
                        "filing_url": None,
                    }
                )
                processed_by_shard.setdefault(shard_path, set()).add((example.cik, example.filing_date))
            print(
                f"[{index}/{len(pending_examples)}] {example.yahoo_tickers or example.tickers} "
                f"{example.filing_date}: summarized",
                flush=True,
            )

        buffered_count = sum(len(records) for records in buffered_by_shard.values())
        if buffered_count >= args.save_every:
            for path, records in buffered_by_shard.items():
                append_records(path, records)
            buffered_by_shard.clear()
            print(
                f"Checkpointed {min(batch_start + len(batch_examples), len(pending_examples))} "
                f"new summaries to {args.output_dir}",
                flush=True,
            )

    for path, records in buffered_by_shard.items():
        append_records(path, records)
    rebuild_consolidated_parquet(args.output_dir, args.parquet_path)
    print(f"Saved summaries under {args.output_dir}", flush=True)
    print(f"Saved consolidated parquet to {args.parquet_path}", flush=True)


def run_clean_mode(args: argparse.Namespace) -> None:
    """Run second-pass KEEP/DROP review over generated summaries."""

    tickers_filter = {ticker.upper() for ticker in args.tickers} if args.tickers else None
    source = pd.read_parquet(args.parquet_path)
    if source.empty:
        raise ValueError(f"no summaries found at {args.parquet_path}")
    source["date"] = source["date"].astype(str)
    source["ticker"] = source["ticker"].astype(str)
    source["summary"] = source["summary"].astype(str)
    if tickers_filter is not None:
        source = source[source["ticker"].isin(tickers_filter)].copy()
    source = source.sort_values(["ticker", "date"]).reset_index(drop=True)
    source = attach_review_row_ids(source)
    unique_source = source.drop_duplicates(subset=["review_row_id"], keep="first").copy()

    existing_reviews = load_existing_reviews(args.review_path) if args.resume else pd.DataFrame(
        columns=["review_row_id", "action", "review_label", "cleaned_summary"]
    )
    processed_keys = set(existing_reviews["review_row_id"]) if not existing_reviews.empty else set()
    pending = unique_source.loc[
        ~unique_source["review_row_id"].isin(processed_keys)
    ].copy()

    print(
        f"mode=clean | provider={args.provider} | model={args.model} | "
        f"summaries_path={args.parquet_path}",
        flush=True,
    )
    print(
        f"resume={args.resume} | existing_reviews={len(processed_keys)} | "
        f"pending={len(pending)} | save_every={args.save_every} | "
        f"filtered_path={args.filtered_path}",
        flush=True,
    )

    if pending.empty:
        print("No pending summaries to review", flush=True)
        write_review_outputs(unique_source, existing_reviews, args.review_path, args.filtered_path, args.rejected_path)
        return

    print(f"Loading backend: provider={args.provider} model={args.model}", flush=True)
    backend = load_backend(args.provider, args.model)
    print("Backend loaded", flush=True)
    if args.provider == "hf":
        attn_implementation = getattr(backend, "_attn_implementation", "unknown")
        print(f"HF attention implementation: {attn_implementation}", flush=True)
    buffered_rows: list[dict[str, object]] = []
    all_reviews = existing_reviews.copy()
    effective_batch_size = args.batch_size if args.provider in {"hf", "vllm"} else 1

    for batch_start in range(0, len(pending), effective_batch_size):
        batch = pending.iloc[batch_start : batch_start + effective_batch_size].copy()
        rows = list(batch.itertuples(index=False))
        summaries = [str(row.summary) for row in rows]
        reviews = review_summaries(args.provider, backend, args.model, summaries)

        rewrite_positions: list[int] = []
        rewrite_inputs: list[str] = []
        actions: list[str] = []
        raw_labels: list[str] = []
        cleaned_summaries = [""] * len(rows)

        for offset, (row, review_result) in enumerate(zip(rows, reviews), start=1):
            label, raw_label = review_result
            action = label or "DROP"
            if label is None:
                print(
                    f"Unparseable review for {row.ticker} {row.date}: {raw_label!r}",
                    flush=True,
                )
            elif action == "RESUMMARIZE":
                rewrite_positions.append(offset - 1)
                rewrite_inputs.append(str(row.summary))

            actions.append(action)
            raw_labels.append(raw_label)

        if rewrite_inputs:
            rewritten = rewrite_summaries(args.provider, backend, args.model, rewrite_inputs)
            for position, cleaned_summary in zip(rewrite_positions, rewritten):
                cleaned_summaries[position] = cleaned_summary

        for offset, row in enumerate(rows, start=1):
            index = batch_start + offset
            action = actions[offset - 1]
            raw_label = raw_labels[offset - 1]
            cleaned_summary = cleaned_summaries[offset - 1]
            if args.verbose:
                print(f"Review for {row.ticker} {row.date}: {raw_label!r}", flush=True)
                if cleaned_summary:
                    print(f"Rewritten summary for {row.ticker} {row.date}: {cleaned_summary!r}", flush=True)
            buffered_rows.append(
                {
                    "review_row_id": str(row.review_row_id),
                    "action": action,
                    "review_label": raw_label,
                    "cleaned_summary": cleaned_summary,
                }
            )
            print(
                f"[{index}/{len(pending)}] {row.ticker} {row.date}: {action}",
                flush=True,
            )

        if len(buffered_rows) >= args.save_every:
            all_reviews = pd.concat([all_reviews, pd.DataFrame(buffered_rows)], ignore_index=True)
            buffered_rows.clear()
            write_review_outputs(unique_source, all_reviews, args.review_path, args.filtered_path, args.rejected_path)
            print(f"Checkpointed reviews to {args.review_path}", flush=True)

    if buffered_rows:
        all_reviews = pd.concat([all_reviews, pd.DataFrame(buffered_rows)], ignore_index=True)

    write_review_outputs(unique_source, all_reviews, args.review_path, args.filtered_path, args.rejected_path)
    print(f"Saved review parquet to {args.review_path}", flush=True)
    print(f"Saved filtered summaries to {args.filtered_path}", flush=True)
    print(f"Saved rejected summaries to {args.rejected_path}", flush=True)


def main() -> None:
    args = parse_args()
    if args.mode == "generate":
        run_generate_mode(args)
        return
    run_clean_mode(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Shared helpers for the retrieval agent stack."""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


def load_backend(provider: str, model: str) -> object:
    """Load an LLM backend compatible with the strategy scripts."""

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
        return pipeline("text-generation", **pipeline_kwargs)

    if provider == "vllm":
        try:
            from vllm import LLM  # type: ignore
        except ImportError as exc:
            raise RuntimeError("vllm is required for provider=vllm") from exc
        return LLM(model=model)

    raise ValueError(f"unsupported provider: {provider}")


def run_openai_json(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> str:
    """Call the OpenAI Responses API."""

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return str(response.output_text).strip()


def run_hf_json(generator: object, system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
    """Run a local Hugging Face model."""

    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    generation_config = copy.deepcopy(generator.model.generation_config)
    generation_config.max_new_tokens = max_new_tokens
    generation_config.max_length = None
    result = generator(
        full_prompt,
        do_sample=False,
        generation_config=generation_config,
        return_full_text=False,
        clean_up_tokenization_spaces=False,
    )
    return str(result[0]["generated_text"]).strip()


def run_vllm_json(engine: object, system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
    """Run a local vLLM model."""

    try:
        from vllm import SamplingParams  # type: ignore
    except ImportError as exc:
        raise RuntimeError("vllm is required for provider=vllm") from exc

    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )
    outputs = engine.generate([full_prompt], sampling_params)
    if not outputs or not outputs[0].outputs:
        return ""
    return str(outputs[0].outputs[0].text).strip()


def run_model_text(
    provider: str,
    backend: object,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
) -> str:
    """Run either OpenAI or Hugging Face text generation."""

    if provider == "openai":
        return run_openai_json(backend, model, system_prompt, user_prompt)
    if provider == "hf":
        return run_hf_json(backend, system_prompt, user_prompt, max_new_tokens=max_new_tokens)
    if provider == "vllm":
        return run_vllm_json(backend, system_prompt, user_prompt, max_new_tokens=max_new_tokens)
    raise ValueError(f"unsupported provider: {provider}")


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


@dataclass(frozen=True)
class SearchResult:
    date: str
    ticker: str
    company_name: str
    summary: str
    score: float


def render_search_results(results: list[SearchResult]) -> str:
    """Format search results for the agent prompt."""

    if not results:
        return "No results."
    chunks: list[str] = []
    for index, result in enumerate(results, start=1):
        chunks.append(
            "\n".join(
                [
                    f"Result {index}",
                    f"ticker={result.ticker}",
                    f"date={result.date}",
                    f"company={result.company_name}",
                    f"score={result.score:.4f}",
                    f"summary={result.summary}",
                ]
            )
        )
    return "\n\n".join(chunks)


def load_index_metadata(index_dir: Path) -> dict[str, Any]:
    """Load index metadata from disk."""

    return json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))


def load_sector_map(path: Path) -> dict[str, list[str]]:
    """Load sector -> tickers mapping and normalize ticker casing."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(sector): [str(ticker).upper() for ticker in tickers]
        for sector, tickers in raw.items()
    }


def build_ticker_to_sector(sector_map: dict[str, list[str]]) -> dict[str, str]:
    """Build reverse ticker -> sector lookup."""

    ticker_to_sector: dict[str, str] = {}
    for sector, tickers in sector_map.items():
        for ticker in tickers:
            ticker_to_sector[ticker.upper()] = sector
    return ticker_to_sector

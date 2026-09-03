# ContiLearnFinAnalyst

This repo is a simple workflow for:

- downloading SEC 10-K data from Hugging Face
- downloading and preprocessing return data
- generating LLM-based prediction panels from 10-K risk sections
- evaluating those predictions with a simple PnL script

## Prerequisites

- Python 3.10+
- `uv`
- internet access for:
  - Hugging Face dataset download
  - Yahoo Finance market data download
  - OpenAI API calls if using OpenAI models
  - Hugging Face model download if using local HF models

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Environment Setup

This repo uses a `pyproject.toml` so you can create the environment with `uv`.

From the repo root:

```bash
uv sync
```

For a vLLM-first install, prefer:

```bash
UV_TORCH_BACKEND=auto uv sync
```

Then run scripts with:

```bash
uv run python ...
```

If you want to activate the virtual environment directly:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Credentials

### OpenAI

If you use `strategies/baseline.py` with `--provider openai`, set:

```bash
export OPENAI_API_KEY=your_key_here
```

The baseline strategy uses the official OpenAI Python SDK.

### Hugging Face

For `src/data/build_10k_hf.py`, public dataset download typically works without a token.

For `strategies/baseline.py` with `--provider hf`, public models can often be used without a token, but for gated models or higher rate limits you may need:

```bash
export HF_TOKEN=your_token_here
```

or:

```bash
export HUGGINGFACE_HUB_TOKEN=your_token_here
```

### vLLM

This repo is configured toward a `vllm`-first environment. After refreshing `uv.lock`, install with `UV_TORCH_BACKEND=auto uv sync` and run local generation with `--provider vllm`.

## Workflow

### 1. Download SEC 10-K data

```bash
uv run python src/data/build_10k_hf.py
```

This writes the SEC shards under `data/financial-reports-sec/large`.

### 2. Download returns

Returns only:

```bash
uv run python src/data/build_10k_market_data.py --skip-metadata
```

This writes:

- `data/financial-reports-sec/market_data/prices/*.csv`
- `data/financial-reports-sec/market_data/return.parquet`

### 3. Download sector metadata

Sector data only:

```bash
uv run python src/data/build_10k_market_data.py --skip-prices
```

This writes:

- `data/financial-reports-sec/market_data/equity_metadata.csv`
- `data/financial-reports-sec/market_data/sector_companies.json`

Important: sector labels are current Yahoo classifications, not point-in-time sector labels.

### 4. Preprocess returns

Current preprocessing script:

```bash
uv run python src/data/preprocess_returns.py
```

This writes:

- `data/financial-reports-sec/market_data/returns_demeaned.parquet`

### 5. Run the baseline LLM strategy

OpenAI:

```bash
OPENAI_API_KEY=... uv run python strategies/baseline.py \
  --provider openai \
  --model gpt-4.1-mini
```

Hugging Face:

```bash
uv run python strategies/baseline.py \
  --provider hf \
  --model google/gemma-2-2b-it
```

vLLM:

```bash
uv run python src/strategies/baseline.py \
  --provider vllm \
  --model Qwen/Qwen3-8B
```

This writes:

- `predictions/baseline_strategy.parquet`

You can override the output name with `--run-name`.

Predictions are stored as a wide parquet panel:

- index: filing date
- columns: Yahoo tickers
- values: `-1` or `1`

### 6. Estimate LLM cost

Use built-in OpenAI pricing snapshot:

```bash
uv run python strategies/util.py --model gpt-5.6-terra
```

Use your own prices:

```bash
uv run python strategies/util.py \
  --input-cost-per-1m 2.0 \
  --output-cost-per-1m 12.0
```

By default this uses both `train` and `test` and counts one example per ticker-filing pair with a `section_1A` risk section.

### 7. Evaluate PnL

```bash
uv run python src/eval/run_10k_pnl.py --run-name baseline_strategy
```

This:

- loads `predictions/baseline_strategy.parquet`
- loads `data/financial-reports-sec/market_data/return.parquet`
- aligns dates and tickers
- computes `pred.shift(2).mul(ret).sum(axis=1)`
- prints the annualized Sharpe
- saves a plot next to the prediction file

### Baseline Notes

The current workable baseline setup is:

- Generate risk-section summaries
- Clean them into `data/financial-reports-sec/summary/summaries.filtered.parquet`
- Run `src/strategies/baseline.py` against the filtered summaries instead of raw `section_1A` text
- Evaluate with forward-filled predictions, cross-sectional ranking, and clipped returns

This is not presented as a final research conclusion, but it is a reasonable current pipeline for quick iteration because:

- the filtered summary file is much smaller and cleaner than the raw summary set
- the baseline strategy is easier to run on summaries than on full risk-section text
- clipping returns reduces the impact of obvious return outliers in the market data
- cross-sectional ranking with forward fill gives a more usable daily signal than same-day filing events alone

Two evaluation configurations that currently produce nontrivial results are:

```bash
uv run python src/eval/run_10k_pnl.py \
  --run-name baseline_strategy \
  --ffill-limit 30 \
  --clip-returns 1.0
```

Observed annualized Sharpe on the current local run: about `0.5363`.

This is the simplest decent baseline in the current setup because it uses only:

- forward-filled filing signals
- clipped realized returns

without adding cross-sectional or sector-relative transformations.

```bash
uv run python src/eval/run_10k_pnl.py \
  --run-name baseline_strategy \
  --ffill-limit 30 \
  --postprocess cross-sectional-rank \
  --clip-returns 1.0 \
  --postprocess normalize-predictor
```

Observed annualized Sharpe on the current local run: about `0.5270`.

```bash
uv run python src/eval/run_10k_pnl.py \
  --run-name baseline_strategy \
  --ffill-limit 252 \
  --postprocess cross-sectional-rank \
  --use-sectors \
  --clip-returns 1.0 \
  --postprocess normalize-predictor
```

Observed annualized Sharpe on the current local run: about `0.8912`.

When reproducing these results, make sure the predictions were generated from `summaries.filtered.parquet`, not from the raw filings directly.

### 8. Generate and clean risk summaries

Generate summaries:

```bash
uv run python src/data/summarize_risk_sections.py \
  --mode generate \
  --provider hf \
  --model Qwen/Qwen3-8B \
  --resume
```

`--mode generate` writes or updates:

- Appends checkpoint rows to sharded JSONL files under `data/financial-reports-sec/summary/<split>/<shard>.jsonl`
- Rebuilds and overwrites `data/financial-reports-sec/summary/summaries.parquet` from those JSONL shards
- Does not write `summaries.review.parquet`, `summaries.filtered.parquet`, or `summaries.rejected.parquet`

Clean or filter existing summaries:

```bash
uv run python src/data/summarize_risk_sections.py \
  --mode clean \
  --provider hf \
  --model Qwen/Qwen3-8B \
  --batch-size 16 \
  --resume
```

`--mode clean` reads `data/financial-reports-sec/summary/summaries.parquet` and writes:

- Overwrites `data/financial-reports-sec/summary/summaries.review.parquet`
- Overwrites `data/financial-reports-sec/summary/summaries.filtered.parquet`
- Overwrites `data/financial-reports-sec/summary/summaries.rejected.parquet`
- Does not modify the sharded JSONL checkpoints
- Does not modify `data/financial-reports-sec/summary/summaries.parquet`

### 9. Build the retrieval index

After `summaries.filtered.parquet` exists:

```bash
bash scripts/build_agent_index.sh
```

This builds a local Chroma index under:

- `data/financial-reports-sec/summary/index`

### 10. Run the retrieval agent

Once summaries are generated, cleaned, and indexed:

```bash
bash scripts/run_retrieval_agent.sh \
  --provider hf \
  --model Qwen/Qwen3-8B \
  --ticker AAPL \
  --year 2020 \
  --verbose
```

The run order is:

1. Generate summaries
2. Clean summaries
3. Build the agent index
4. Run the agent

One agentic run does the following:

- loads the local Chroma index over `summaries.filtered.parquet`
- builds a task-specific starting prompt for one `(ticker, year)` pair
- lets the model iterate with search actions, optional sector-peer lookup, and a final stop action
- enforces a hard date cutoff so the agent cannot query future information beyond the target year
- when the agent emits `{"action":"final_answer"}`, the environment runs one last prompt over the gathered evidence and returns exactly `UP` or `DOWN`

What you will see on stdout:

- with `--verbose`, each step's JSON action
- with `--verbose`, the retrieved search results or sector-peer payload for that step
- the final `UP` or `DOWN` prediction
- the saved trajectory path

What one run writes to disk:

- `data/financial-reports-sec/summary/agent_trajectory.json` by default

That trajectory file contains:

- target `ticker`
- target `year`
- hard stop date
- starting prompt
- every tool/action step
- retrieved evidence
- final prediction

## Main Files

- `src/data/build_10k_hf.py`
- `src/data/build_10k_market_data.py`
- `src/data/preprocess_returns.py`
- `src/data/summarize_risk_sections.py`
- `src/data/check_sec_effect.py`
- `src/agent/build_index.py`
- `src/agent/search.py`
- `src/agent/run_agent.py`
- `src/eval/run_10k_pnl.py`
- `strategies/baseline.py`
- `strategies/util.py`

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

If you use `src/strategies/baseline.py` with `--provider openai`, set:

```bash
export OPENAI_API_KEY=your_key_here
```

The baseline strategy uses the official OpenAI Python SDK.

### Hugging Face

For `src/data/build_10k_hf.py`, public dataset download typically works without a token.

For `src/strategies/baseline.py` with `--provider hf`, public models can often be used without a token, but for gated models or higher rate limits you may need:

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

### 5. Generate and clean risk summaries

Cleaning is critical: the strategy workflow is intended to run on
`summaries.filtered.parquet`, not directly on noisy raw filings.

Generate summaries:

```bash
uv run python src/data/summarize_risk_sections.py \
  --mode generate \
  --provider hf \
  --model Qwen/Qwen3-8B \
  --resume
```

This creates or updates `data/financial-reports-sec/summary/summaries.parquet`
and sharded JSONL checkpoints. Then clean/filter them:

```bash
uv run python src/data/summarize_risk_sections.py \
  --mode clean \
  --provider hf \
  --model Qwen/Qwen3-8B \
  --batch-size 16 \
  --resume
```

This writes `summaries.review.parquet`, `summaries.filtered.parquet` (the
strategy input), and `summaries.rejected.parquet`.

### 6. Run the baseline LLM strategy

`src/strategies/baseline.py` is a fixed-prompt workflow: for each filing,
the model receives either the current risk summary alone or the current and
previous-year summaries, then returns `NEW` or `NO_NEW`. Those labels are
converted directly to `-1` or `1`. 

The stored baseline variants are:

- `baseline_no_past_context`: current filing only.
- `baseline_with_past_context`: current filing plus the previous-year filing,
  enabled by `--use-past-context`.

OpenAI:

```bash
OPENAI_API_KEY=... uv run python src/strategies/baseline.py \
  --provider openai \
  --model gpt-4.1-mini
```

Hugging Face:

```bash
uv run python src/strategies/baseline.py \
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

- `predictions/baseline_strategy/baseline_strategy.parquet`

You can override the output name with `--run-name`.

Predictions are stored as a wide parquet panel:

- index: filing date
- columns: Yahoo tickers
- values: `-1` or `1`

### 7. Run one strategy-agent strategy

`src/strategy_agent/run_strategy_agent.py` is the structured strategy layer.
Unlike the fixed baseline, it expects a strategy JSON defining the hypothesis,
analysis steps, labels, and prediction rule.

For example:

```json
{
  "system_prompt": "You are a financial analyst.",
  "hypothesis": "New company-specific risks may predict downside.",
  "steps": [{
    "stocks": ["TARGET_STOCK"],
    "years": ["TARGET_YEAR", "TARGET_YEAR-1"],
    "question": "Did the current filing introduce a material new risk?",
    "output_labels": ["YES", "NO", "UNCLEAR"],
    "label_descriptions": {
      "YES": "A material new risk is explicitly present.",
      "NO": "No material new risk is identified.",
      "UNCLEAR": "The summaries are insufficient to decide."
    }
  }],
  "prediction_rule": {
    "instructions": "DOWN for YES, UP for NO, otherwise abstain.",
    "output_labels": ["DOWN", "UP", "NO_POSITION"],
    "parse_map": {"DOWN": -1, "UP": 1, "NO_POSITION": 0}
  }
}
```

`stocks` may contain `TARGET_STOCK` and/or `SECTOR_PEERS`; years may be
`TARGET_YEAR`, `TARGET_YEAR-N`, or a literal year. Every label needs a
description and a prediction label needs a `parse_map` value of `-1`, `0`, `1`,
or `null`.

Run it with:

```bash
uv run python src/strategy_agent/run_strategy_agent.py \
  --provider openai \
  --model gpt-4.1-mini \
  --predictions-dir predictions/my_strategy \
  --run-name my_strategy \
  --strategy-path predictions/my_strategy/strategy.json \
  --summaries-path data/financial-reports-sec/summary/summaries.filtered.parquet \
  --sector-map-path data/financial-reports-sec/market_data/sector_companies.json
```

With a local vLLM model:

```bash
uv run python src/strategy_agent/run_strategy_agent.py \
  --provider vllm \
  --model Qwen/Qwen3-8B \
  --predictions-dir predictions/my_strategy_vllm \
  --run-name my_strategy_vllm \
  --strategy-path predictions/my_strategy/strategy.json \
  --summaries-path data/financial-reports-sec/summary/summaries.filtered.parquet \
  --sector-map-path data/financial-reports-sec/market_data/sector_companies.json
```

This writes the prediction panel, row-level predictions, model logs, and
metrics. The `agent_*` runs use this path; they are not baseline runs.

### 8. Evaluate PnL

```bash
uv run python src/eval/run_10k_pnl.py --run-name baseline_strategy
```

This:

- loads `predictions/baseline_strategy/baseline_strategy.parquet`
- loads `data/financial-reports-sec/market_data/return.parquet`
- aligns dates and tickers
- computes `predictions.shift(2).mul(returns).sum(axis=1)`
- prints the annualized Sharpe
- saves a plot next to the prediction file

The main evaluation options are:

- No forward-fill: only the filing-date signal is held.
- `--ffill-limit 30`: hold a signal for up to 30 trading days.
- `--ffill-limit 252`: hold it for up to roughly one trading year.
- `--postprocess cross-sectional-rank`: demean and scale signals across all
  tickers on each date.
- `--use-sectors` with cross-sectional ranking: perform that normalization
  within sectors using `sector_companies.json`; without it, ranking is across
  the whole universe.
- `--postprocess normalize-predictor`: apply each ticker's causal rolling
  252-day volatility normalization and clip the result.

These options can materially change the signal and Sharpe, so record the exact
command with every metric. For example, a 30-day forward-filled evaluation is:

```bash
uv run python src/eval/run_10k_pnl.py \
  --run-name agent_single_risk_assessment \
  --predictions-dir predictions/agent_single_risk_assessment \
  --ffill-limit 30
```

Add cross-sectional normalization, without a sector file:
```bash
uv run python src/eval/run_10k_pnl.py \
  --run-name agent_single_risk_assessment \
  --predictions-dir predictions/agent_single_risk_assessment \
  --ffill-limit 30 \
  --postprocess cross-sectional-rank \
  --postprocess normalize-predictor
```
Some example runs to give a sense of the performance: 

| Run/configuration | Sharpe | t-stat | Always-long correlation |
|---|---:|---:|---:|
| `agent_single_risk_assessment`, ffill 30 | -0.251 | -1.319 | -0.030 |
| Same, ffill 252 | -0.704 | -3.703 | 0.041 |
| Same + cross-sectional, ffill 30 | -0.418 | -2.198 | -0.047 |
| Same + cross-sectional, ffill 252 | -1.045 | -5.501 | 0.000 |
| `agent_compare_two_years`, ffill 30 | -0.331 | -1.744 | -0.498 |
| Same + cross-sectional, ffill 30 | 0.152 | 0.801 | -0.053 |
| Same, ffill 252 | -1.162 | -6.116 | -0.833 |
| Same + cross-sectional, ffill 252 | 0.571 | 3.006 | -0.031 |
| `baseline_no_past_context` | 1.472 | 7.744 | 0.797 |
| `baseline_with_past_context` | 0.398 | 2.094 | 0.000 |

### 9. Propose and iterate strategies

`propose_strategy.py` generates an initial strategy, runs it, inspects its
performance, and optionally generates replacements. Each iteration is stored
under the predictions directory.

```bash
OPENAI_API_KEY=... uv run python src/strategy_agent/propose_strategy.py \
  --provider openai \
  --model gpt-4.1-mini \
  --predictions-dir predictions/openai_luna_new \
  --run-name openai_luna_new \
  --max-iterations 10
```

With a local vLLM model:

```bash
uv run python src/strategy_agent/propose_strategy.py \
  --provider vllm \
  --model Qwen/Qwen3-8B \
  --predictions-dir predictions/qwen3_proposals \
  --run-name qwen3_proposals \
  --max-iterations 10
```

The generated strategy is in `iteration_000/strategy.json`; later iterations
contain replacements selected after diagnostic inspections. The complete
proposal history is in `proposal_trajectory.jsonl`. Use
`src/strategy_agent/overview.md` for a shorter proposal-loop reference.

The proposal loop works as follows:

1. Two variants of an initial prompt that instructs the model to generate a hypothesis: i) `initial_prompt` discovers the two seed strategies and records their primary metrics in
   `proposal_trajectory.jsonl` as initial context, ii) 
   `initial_prompt_v2` does not pass those examples into the model prompt; it
   starts from its standalone hypothesis ideas. 
2. The model proposes an initial strategy JSON.
3. The loop runs that strategy through `run_strategy_agent.py` and evaluates
   the resulting predictions.
4. After each iteration, the model chooses one action:
   - `inspect_general_performance`: inspect aggregate error rate, UP-rate bias,
     and Sharpe.
   - `inspect_per_year`: inspect the same diagnostics for one filing year.
   - `inspect_per_sector`: inspect the diagnostics for one sector.
   - `propose_strategy`: replace the current strategy using the diagnostics.
   - `stop`: finish and keep the latest strategy.
   In principle, you can extend this into something stronger. 
5. The loop repeats until the requested iteration count is reached or the
   model chooses `stop`. It allows at most 10 inspection actions between
   strategy proposals.

The inspection tools use the saved final prediction logs and realized returns;
they do not perform new web searches. Each iteration directory contains the
strategy, predictions, logs, plots, and metrics. `proposal_trajectory.jsonl`
records the initial proposal, model action responses, inspections, accepted
replacements, and stop events, which makes the proposal process auditable.

The decision prompt expects one JSON action, for example:

```json
{"action": "inspect_per_year", "target_year": 2020}
```

Inspection results are fed back into the next decision prompt. A replacement
strategy is accepted only after it passes `StrategySpec` validation.

This is separate from `contilearn/simple_reflection`. The proposal agent has
its own performance-inspection actions and does not call the experimental
reflector. The simple-reflection workflow independently asks a model to inspect
logs and reflect on a strategy; it is not part of the normal proposal command.

The proposal agent does not introduce another prediction format. It generates
or revises the same strategy JSON consumed by `run_strategy_agent.py`, runs it,
and uses its metrics and logs to decide whether to inspect, replace, or stop:

```text
baseline.py                    fixed prompt -> predictions
strategy JSON + runner         structured strategy -> predictions
propose_strategy.py            strategy JSON -> runner -> metrics -> revision
```

## Main Files

- `src/data/build_10k_hf.py`
- `src/data/build_10k_market_data.py`
- `src/data/preprocess_returns.py`
- `src/data/summarize_risk_sections.py`
- `src/data/check_sec_effect.py`
- `src/strategy_agent/propose_strategy.py`
- `src/strategy_agent/run_strategy_agent.py`
- `src/strategy_agent/strategy.py`
- `src/eval/run_10k_pnl.py`
- `src/strategies/baseline.py`

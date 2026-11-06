# Strategy agent overview

The proposal loop generates a strategy JSON, runs a backtest, inspects metrics
and logs, and proposes an improved strategy. It can also inspect the existing
example strategies with the `see_examples` action.

## Start the agent loop

From the repository root:

```bash
nohup uv run python src/strategy_agent/propose_strategy.py \
  --provider hf \
  --model Qwen/Qwen3-8B \
  --predictions-dir predictions/agent_loop \
  --run-name agent_loop \
  --max-iterations 10 \
  > predictions/agent_loop.log 2>&1 &
```

For OpenAI:

```bash
nohup uv run python src/strategy_agent/propose_strategy.py \
  --provider openai \
  --model gpt-4.1-mini \
  --predictions-dir predictions/agent_loop \
  --run-name agent_loop \
  --max-iterations 10 \
  > predictions/agent_loop.log 2>&1 &
```

Set `OPENAI_API_KEY` before an OpenAI run. Iterations are written under:

```text
predictions/agent_loop/iteration_001/
predictions/agent_loop/iteration_002/
```

## Monitor a run

```bash
tail -f predictions/agent_loop.log
```

```bash
pgrep -af propose_strategy
```

## Quick test

```bash
uv run python src/strategy_agent/propose_strategy.py \
  --provider hf \
  --model Qwen/Qwen3-8B \
  --predictions-dir predictions/agent_loop_test \
  --run-name agent_loop_test \
  --max-rows 10 \
  --max-iterations 2
```

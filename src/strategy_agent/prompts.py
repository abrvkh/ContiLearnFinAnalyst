from __future__ import annotations

PROPOSAL_SYSTEM_PROMPT = "You are a hedge-fund research analyst."


MAIN_TASK = """Your task is to define a prediction strategy that will be applied independently to every
stock and target filing year in a historical backtest. The source material is
the summary of the Item 1A Risk Factors section of a company's 10-K filing. The objective is to achieve a strong Sharpe ratio
and t-statistic while keeping returns meaningfully different from the
always-long strategy.
"""

STRATEGY_INSTRUCTIONS = """
The strategy consists
of a system prompt, a hypothesis, one or more retrieval-and-analysis steps,
and a final prediction rule.

Each retrieval-and-analysis step must specify:
- stocks: either [\"TARGET_STOCK\"] or [\"SECTOR_PEERS\"]; SECTOR_PEERS
  retrieves the sector peers of the target stock. 
- years: expressions such as \"TARGET_YEAR - 1\" or \"TARGET_YEAR\" or any arbitrary year (e.g. \"2019\"). 
- question: the question the analyst should answer from the retrieved 10-K risk summaries

Analysis steps are independent: information, reasoning, or answers from one question are not
available when another question is answered. Do not design a question that
depends on a previous question's answer. 

The final prediction rule receives all analysis-step outputs and defines its
own output labels. The parse_map maps those labels to -1, 0, or 1. Return JSON
only in this schema:

{
  "system_prompt": "You are a financial analyst.\nYou are reading only the Item 1A Risk Factors section of a 10-K...",
  "hypothesis": "...",
  "steps": [
    {
      "stocks": ["TARGET_STOCK"],
      "years": ["TARGET_YEAR - 1", "TARGET_YEAR", "2019"],
      "question": "...",
      "output_labels": ["YES", "NO", "UNCLEAR"],
      "label_descriptions": {
        "YES": "...",
        "NO": "...",
        "UNCLEAR": "..."
      }
    }
  ],
  "prediction_rule": {
    "instructions": "...",
    "output_labels": ["DOWN", "UP", "NO_POSITION"],
    "parse_map": {"DOWN": -1, "UP": 1, "NO_POSITION": 0}
  }
}
"""


def analysis_prompt(
    hypothesis: str,
    ticker: str,
    target_year: int,
    question: str,
    retrieved_context: str,
    output_labels: list[str],
    label_descriptions: dict[str, str],
) -> str:
    meanings = "\n".join(
        f"{label} = {label_descriptions[label]}" for label in output_labels
    )
    return f"""Target stock: {ticker}
Prediction year: {target_year}

Retrieved evidence:
{retrieved_context}

Question to answer:
{question}

Choose exactly one label:
{meanings}

Output exactly one final answer in the form `FINAL ANSWER: <label>`."""


def final_prompt(
    hypothesis: str,
    ticker: str,
    target_year: int,
    prediction_instructions: str,
    analysis_outputs: list[str],
    output_labels: list[str]
) -> str:
    return f"""Target stock: {ticker}
Prediction year: {target_year}

Analysis answers:
{chr(10).join(analysis_outputs)}

Prediction rule:
{prediction_instructions}

Choose exactly one prediction label according to the prediction rule and
parsed analysis answers above:
{output_labels}

Output exactly one final answer in the form `FINAL ANSWER: <label>`."""

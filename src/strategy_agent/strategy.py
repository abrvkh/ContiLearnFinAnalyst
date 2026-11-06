from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

def _signal(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}.signal must be -1, 0, or 1") from exc
    if result not in {-1, 0, 1}:
        raise ValueError(f"{name}.signal must be -1, 0, or 1")
    return result


@dataclass(frozen=True)
class StrategyStep:
    stocks: tuple[str, ...]
    years: tuple[str, ...]
    question: str
    output_labels: tuple[str, ...]
    label_descriptions: dict[str, str]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StrategyStep":
        stocks = tuple(str(item) for item in value.get("stocks", []))
        if not stocks or any(item not in {"TARGET_STOCK", "SECTOR_PEERS"} for item in stocks):
            raise ValueError("step.stocks must contain target and/or sector_peers")
        years = tuple(str(item).replace(" ", "") for item in value.get("years", []))
        if not years:
            raise ValueError("step.years must contain at least one year expression")
        for expression in years:
            if expression == "TARGET_YEAR" or expression.startswith("TARGET_YEAR-"):
                continue
            try:
                int(expression)
            except ValueError as exc:
                raise ValueError(f"unsupported year expression: {expression}") from exc
        question = str(value.get("question", "")).strip()
        if not question:
            raise ValueError("step.question is required")
        labels = tuple(str(item) for item in value.get("output_labels", []))
        if not labels:
            raise ValueError("step.output_labels is required")
        descriptions = {
            str(key): str(item)
            for key, item in value.get("label_descriptions", {}).items()
        }
        if set(descriptions) != set(labels):
            raise ValueError("step.label_descriptions must describe every output label")
        return cls(stocks, years, question, labels, descriptions)


@dataclass(frozen=True)
class StrategySpec:
    system_prompt: str
    hypothesis: str
    steps: tuple[StrategyStep, ...]
    prediction_instructions: str
    prediction_labels: tuple[str, ...]
    parse_map: dict[str, int | None]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StrategySpec":
        system_prompt = str(value.get("system_prompt", "")).strip()
        hypothesis = str(value.get("hypothesis", "")).strip()
        if not hypothesis:
            raise ValueError("hypothesis is required")
        steps = tuple(StrategyStep.from_dict(item) for item in value.get("steps", []))
        if not steps:
            raise ValueError("at least one retrieval-and-analysis step is required")
        prediction = value.get("prediction_rule", {})
        instructions = str(prediction.get("instructions", "")).strip()
        if not instructions:
            raise ValueError("prediction_rule.instructions is required")
        labels = tuple(str(item) for item in prediction.get("output_labels", []))
        parse_map = {
            str(key): (None if value is None else _signal(value, f"parse_map[{key!r}]"))
            for key, value in prediction.get("parse_map", {}).items()
        }
        if not labels or set(labels) != set(parse_map):
            raise ValueError("prediction_rule.output_labels and parse_map must contain the same labels")
        return cls(system_prompt, hypothesis, steps, instructions, labels, parse_map)

    def to_json(self) -> str:
        return json.dumps(
            {
                "system_prompt": self.system_prompt,
                "hypothesis": self.hypothesis,
                "steps": [
                    {
                        "stocks": list(step.stocks),
                        "years": list(step.years),
                        "question": step.question,
                        "output_labels": list(step.output_labels),
                        "label_descriptions": step.label_descriptions,
                    }
                    for step in self.steps
                ],
                "prediction_rule": {
                    "instructions": self.prediction_instructions,
                    "output_labels": list(self.prediction_labels),
                    "parse_map": self.parse_map,
                },
            },
            indent=2,
        )

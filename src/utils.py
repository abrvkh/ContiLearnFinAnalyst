"""Small shared utilities used across the project."""

from __future__ import annotations

import json
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from plain or lightly wrapped model output.

    Models sometimes surround otherwise valid JSON with prose or Markdown.
    Prefer strict parsing, then fall back to the outermost JSON object.
    """

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])

    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value

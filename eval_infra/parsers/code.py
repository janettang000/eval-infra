from __future__ import annotations

import re

from eval_infra.parsers.base import Parser


def _strip_trailing(code: str) -> str:
    """Strip trailing blank lines and whitespace."""
    lines = code.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


class PythonCodeParser(Parser):
    """Extract a Python function body from model output.

    For HumanEval-style tasks where the prompt already contains the function
    signature and docstring, and the model should complete the body.

    Strategies:
    1. Extract from ```python ... ``` fenced code blocks
    2. Extract from ``` ... ``` generic code blocks
    3. Use the raw text as-is (model may have output bare code)
    """

    def parse(self, text: str) -> str | None:
        if not text or not text.strip():
            return None

        # Strategy 1: fenced python block
        matches = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if matches:
            return _strip_trailing(matches[0])

        # Strategy 2: generic fenced block
        matches = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
        if matches:
            return _strip_trailing(matches[0])

        # Strategy 3: return raw text (strip leading/trailing blank lines)
        return _strip_trailing(text)

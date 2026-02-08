from __future__ import annotations

import re

from eval_infra.parsers.base import Parser


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
            return matches[0].rstrip()

        # Strategy 2: generic fenced block
        matches = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
        if matches:
            return matches[0].rstrip()

        # Strategy 3: return raw text (strip leading/trailing blank lines)
        return text.strip()

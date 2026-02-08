from __future__ import annotations

import re

from eval_infra.parsers.base import Parser

CHOICE_LETTERS = ("A", "B", "C", "D")


class MultiChoiceParser(Parser):
    """Extract a multiple-choice answer (A/B/C/D) from generated text.

    Tries multiple strategies in order:
    1. Explicit "The answer is (X)" pattern
    2. "\\boxed{X}" pattern
    3. Standalone letter at the end of the text
    4. First capital letter A-D on its own line
    """

    def parse(self, text: str) -> str | None:
        text_stripped = text.strip()

        # Strategy 1: "the answer is (X)" or "the answer is X"
        match = re.findall(
            r"(?:the\s+answer\s+is|answer:\s*)\s*\(?([A-Da-d])\)?",
            text_stripped, re.IGNORECASE,
        )
        if match:
            return match[-1].upper()

        # Strategy 2: \boxed{X}
        match = re.findall(r"\\boxed\{([A-Da-d])\}", text_stripped)
        if match:
            return match[-1].upper()

        # Strategy 3: last standalone letter A-D (possibly with period/paren)
        match = re.findall(r"(?:^|\n)\s*\(?([A-D])\)?[\.\):]?\s*$", text_stripped, re.MULTILINE)
        if match:
            return match[-1].upper()

        # Strategy 4: last occurrence of a standalone A-D
        match = re.findall(r"(?<![a-zA-Z])([A-D])(?![a-zA-Z])", text_stripped)
        if match:
            return match[-1].upper()

        return None

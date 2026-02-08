from __future__ import annotations

import re

from eval_infra.parsers.base import Parser


class NumericParser(Parser):
    """Extract a numeric answer from generated text.

    Tries multiple strategies in order:
    1. #### <number> (GSM8K ground-truth format)
    2. \\boxed{<number>} (common in model outputs)
    3. Last number on the last line that starts with a pattern like "the answer is"
    4. Last standalone number in the text
    """

    def parse(self, text: str) -> str | None:
        # Strategy 1: #### delimiter (GSM8K style)
        match = re.findall(r"####\s*([+-]?[\d,]+\.?\d*)", text)
        if match:
            return match[-1].replace(",", "")

        # Strategy 2: \boxed{...}
        from eval_infra.parsers.boxed import BoxedParser
        boxed = BoxedParser().parse(text)
        if boxed is not None:
            # Clean to pure number if possible
            cleaned = boxed.replace(",", "").replace("\\$", "").replace("$", "").strip()
            if re.fullmatch(r"[+-]?\d+\.?\d*", cleaned):
                return cleaned
            return boxed

        # Strategy 3: "the answer is <number>" pattern
        answer_patterns = re.findall(
            r"(?:the\s+answer\s+is|=)\s*\$?\s*([+-]?[\d,]+\.?\d*)",
            text, re.IGNORECASE,
        )
        if answer_patterns:
            return answer_patterns[-1].replace(",", "")

        # Strategy 4: last number in text
        numbers = re.findall(r"(?<![a-zA-Z])([+-]?\d[\d,]*\.?\d*)(?![a-zA-Z])", text)
        if numbers:
            return numbers[-1].replace(",", "")

        return None


class GSM8KGroundTruthParser(Parser):
    """Extract the number after #### from GSM8K ground-truth answer strings."""

    def parse(self, text: str) -> str | None:
        match = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
        if match:
            return match.group(1).replace(",", "")
        return None

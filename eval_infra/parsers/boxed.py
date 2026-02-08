from __future__ import annotations

import re

from eval_infra.parsers.base import Parser


class BoxedParser(Parser):
    """Extract the last \\boxed{...} answer from generated text."""

    def parse(self, text: str) -> str | None:
        # Find all \boxed{...} occurrences, handling nested braces
        results = []
        i = 0
        while i < len(text):
            idx = text.find("\\boxed{", i)
            if idx == -1:
                break
            # Find matching closing brace
            depth = 0
            start = idx + len("\\boxed{")
            for j in range(start, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    if depth == 0:
                        results.append(text[start:j])
                        i = j + 1
                        break
                    depth -= 1
            else:
                break
            if not results:
                i = start

        if results:
            return results[-1].strip()  # Return the last \boxed{} match
        return None

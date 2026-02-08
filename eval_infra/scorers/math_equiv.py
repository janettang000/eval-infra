from __future__ import annotations

import re

from eval_infra.scorers.base import Scorer


class MathEquivalenceScorer(Scorer):
    """Score math answers using math_verify for equivalence checking.

    Falls back to normalized string comparison when math_verify cannot
    parse the expressions (e.g. symbolic answers, text, coordinates).
    """

    def score(self, predicted: str, expected: str) -> bool:
        # Try math_verify first
        try:
            from math_verify import parse, verify
            parsed_pred = parse(predicted)
            parsed_exp = parse(expected)
            # Only trust verify when both sides parsed to non-empty results
            if parsed_pred and parsed_exp:
                return verify(parsed_pred, parsed_exp)
        except Exception:
            pass

        # Fall back to normalized string comparison
        return self._normalize(predicted) == self._normalize(expected)

    @staticmethod
    def _normalize(s: str) -> str:
        s = s.strip()
        # Remove surrounding $ signs
        s = s.strip("$").strip()
        # Remove \text{} wrappers
        s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
        # Remove \left and \right (decorative sizing)
        s = re.sub(r"\\left\s*", "", s)
        s = re.sub(r"\\right\s*", "", s)
        # Remove spaces inside parentheses/brackets: "( x )" -> "(x)"
        s = re.sub(r"\(\s+", "(", s)
        s = re.sub(r"\s+\)", ")", s)
        s = re.sub(r"\[\s+", "[", s)
        s = re.sub(r"\s+\]", "]", s)
        # Normalize whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s

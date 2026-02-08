from __future__ import annotations

from eval_infra.scorers.base import Scorer


class MathEquivalenceScorer(Scorer):
    """Score math answers using math_verify for equivalence checking."""

    def score(self, predicted: str, expected: str) -> bool:
        try:
            from math_verify import parse, verify
            parsed_pred = parse(predicted)
            parsed_exp = parse(expected)
            return verify(parsed_pred, parsed_exp)
        except Exception:
            # Fall back to exact string match (normalized)
            return self._normalize(predicted) == self._normalize(expected)

    @staticmethod
    def _normalize(s: str) -> str:
        s = s.strip()
        # Remove surrounding $ signs and whitespace
        s = s.strip("$").strip()
        # Remove \text{} wrappers
        import re
        s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
        return s

from __future__ import annotations

import re

from eval_infra.scorers.base import Scorer


class NumericScorer(Scorer):
    """Score by comparing numeric values with tolerance for float comparison."""

    def score(self, predicted: str, expected: str) -> bool:
        try:
            p = self._to_number(predicted)
            e = self._to_number(expected)
            if p is None or e is None:
                return predicted.strip() == expected.strip()
            # Integer comparison when both are ints
            if isinstance(p, int) and isinstance(e, int):
                return p == e
            return abs(p - e) < 1e-6
        except Exception:
            return predicted.strip() == expected.strip()

    @staticmethod
    def _to_number(s: str) -> int | float | None:
        s = s.strip().replace(",", "").replace("$", "").replace("%", "")
        s = re.sub(r"\\text\{[^}]*\}", "", s).strip()
        if not s:
            return None
        try:
            if "." in s:
                return float(s)
            return int(s)
        except ValueError:
            return None

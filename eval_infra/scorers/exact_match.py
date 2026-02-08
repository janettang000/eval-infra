from __future__ import annotations

from eval_infra.scorers.base import Scorer


class ExactMatchScorer(Scorer):
    """Case-insensitive exact string match after stripping whitespace."""

    def score(self, predicted: str, expected: str) -> bool:
        return predicted.strip().lower() == expected.strip().lower()

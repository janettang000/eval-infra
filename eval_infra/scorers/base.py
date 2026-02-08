from abc import ABC, abstractmethod


class Scorer(ABC):
    @abstractmethod
    def score(self, predicted: str, expected: str) -> bool:
        """Return True if predicted answer is equivalent to expected."""
        ...

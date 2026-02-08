from abc import ABC, abstractmethod


class Parser(ABC):
    @abstractmethod
    def parse(self, text: str) -> str | None:
        """Extract an answer from generated text. Returns None if no answer found."""
        ...

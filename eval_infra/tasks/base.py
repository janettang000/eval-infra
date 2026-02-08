from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from eval_infra.parsers.base import Parser
from eval_infra.scorers.base import Scorer


@dataclass
class Sample:
    id: str
    prompt: str | list[dict[str, str]]
    expected_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    """Abstract base class for evaluation benchmarks.

    Subclasses must define ``name``, ``parser``, ``scorer`` and implement
    ``load_dataset`` and ``format_prompt``.
    """

    name: str
    parser: Parser
    scorer: Scorer

    @abstractmethod
    def load_dataset(self, max_samples: int | None = None) -> list[Sample]:
        """Load benchmark samples, optionally capped at *max_samples*."""
        ...

    @abstractmethod
    def format_prompt(self, sample: Sample) -> str | list[dict[str, str]]:
        """Format a sample into a prompt string or chat messages list."""
        ...

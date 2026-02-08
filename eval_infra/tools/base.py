from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Execute the tool and return a string result."""
        ...

    def schema(self) -> dict[str, Any]:
        """Return the tool schema for inclusion in prompts."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

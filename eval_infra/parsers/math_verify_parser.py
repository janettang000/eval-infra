from __future__ import annotations

from eval_infra.parsers.base import Parser
from eval_infra.parsers.boxed import BoxedParser


class MathVerifyParser(Parser):
    """Parse math answers using math_verify.parse(), falling back to BoxedParser."""

    def __init__(self):
        self._boxed = BoxedParser()

    def parse(self, text: str) -> str | None:
        # First try boxed extraction
        boxed = self._boxed.parse(text)
        if boxed is not None:
            return boxed

        # Fall back to math_verify's parser on the full text
        try:
            from math_verify import parse as mv_parse
            result = mv_parse(text)
            if result is not None:
                return str(result)
        except Exception:
            pass

        return None

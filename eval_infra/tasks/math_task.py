from __future__ import annotations

from datasets import load_dataset

from eval_infra.parsers.boxed import BoxedParser
from eval_infra.parsers.math_verify_parser import MathVerifyParser
from eval_infra.scorers.math_equiv import MathEquivalenceScorer
from eval_infra.tasks.base import Sample, Task

MATH_SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following problem step by step. "
    "Show your work clearly, then put your final answer inside \\boxed{}."
)


class MathTask(Task):
    name = "math"

    def __init__(self, parser: str = "math_verify", level: str | None = None):
        if parser == "boxed":
            self.parser = BoxedParser()
        else:
            self.parser = MathVerifyParser()
        self.scorer = MathEquivalenceScorer()
        self.level = level  # e.g. "4,5" to filter Level 4-5 only

    def load_dataset(self, max_samples: int | None = None) -> list[Sample]:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        level_filter = set(self.level.split(",")) if self.level else None

        samples: list[Sample] = []
        for i, row in enumerate(ds):
            if level_filter:
                row_level = str(row.get("level", ""))
                # Match e.g. "Level 4" against filter {"4", "5"}
                if not any(lv in row_level for lv in level_filter):
                    continue
            if max_samples is not None and len(samples) >= max_samples:
                break
            answer = row.get("answer") or BoxedParser().parse(row["solution"]) or ""
            samples.append(Sample(
                id=row.get("unique_id", str(i)),
                prompt=row["problem"],
                expected_answer=answer,
                metadata={
                    "type": row.get("subject", "unknown"),
                    "level": row.get("level", "unknown"),
                },
            ))
        return samples

    def format_prompt(self, sample: Sample) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": sample.prompt},
        ]

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

    def __init__(self, parser: str = "math_verify"):
        if parser == "boxed":
            self.parser = BoxedParser()
        else:
            self.parser = MathVerifyParser()
        self.scorer = MathEquivalenceScorer()

    def load_dataset(self, max_samples: int | None = None) -> list[Sample]:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

        samples: list[Sample] = []
        for i, row in enumerate(ds):
            if max_samples is not None and i >= max_samples:
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

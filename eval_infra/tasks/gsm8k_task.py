from __future__ import annotations

from datasets import load_dataset

from eval_infra.parsers.numeric import GSM8KGroundTruthParser, NumericParser
from eval_infra.scorers.numeric import NumericScorer
from eval_infra.tasks.base import Sample, Task

GSM8K_SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following problem step by step. "
    "After your reasoning, write your final numeric answer on a new line after '#### '. "
    "For example: #### 42"
)


class GSM8KTask(Task):
    name = "gsm8k"

    def __init__(self, **kwargs):
        self.parser = NumericParser()
        self.scorer = NumericScorer()
        self._gt_parser = GSM8KGroundTruthParser()

    def load_dataset(self, max_samples: int | None = None) -> list[Sample]:
        ds = load_dataset("openai/gsm8k", "main", split="test")

        samples: list[Sample] = []
        for i, row in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break
            answer = self._gt_parser.parse(row["answer"]) or ""
            samples.append(Sample(
                id=str(i),
                prompt=row["question"],
                expected_answer=answer,
                metadata={"full_solution": row["answer"]},
            ))
        return samples

    def format_prompt(self, sample: Sample) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
            {"role": "user", "content": sample.prompt},
        ]

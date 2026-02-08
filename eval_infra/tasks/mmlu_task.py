from __future__ import annotations

from datasets import load_dataset

from eval_infra.parsers.multichoice import MultiChoiceParser
from eval_infra.scorers.exact_match import ExactMatchScorer
from eval_infra.tasks.base import Sample, Task

CHOICE_LETTERS = ("A", "B", "C", "D")

MMLU_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the following multiple-choice question. "
    "Think step by step, then give your final answer as a single letter (A, B, C, or D)."
)


class MMLUTask(Task):
    name = "mmlu"

    def __init__(self, subject: str = "abstract_algebra", **kwargs):
        self.subject = subject
        self.parser = MultiChoiceParser()
        self.scorer = ExactMatchScorer()

    def load_dataset(self, max_samples: int | None = None) -> list[Sample]:
        ds = load_dataset("cais/mmlu", self.subject, split="test")

        samples: list[Sample] = []
        for i, row in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break
            answer_idx = row["answer"]
            answer_letter = CHOICE_LETTERS[answer_idx]
            choices_text = "\n".join(
                f"{CHOICE_LETTERS[j]}. {row['choices'][j]}" for j in range(len(row["choices"]))
            )
            samples.append(Sample(
                id=f"{self.subject}/{i}",
                prompt=f"{row['question']}\n\n{choices_text}",
                expected_answer=answer_letter,
                metadata={
                    "type": row.get("subject", self.subject),
                    "choices": row["choices"],
                    "answer_idx": answer_idx,
                },
            ))
        return samples

    def format_prompt(self, sample: Sample) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": MMLU_SYSTEM_PROMPT},
            {"role": "user", "content": sample.prompt},
        ]

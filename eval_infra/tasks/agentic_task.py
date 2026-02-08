from __future__ import annotations

import json
from typing import Any

from datasets import load_dataset

from eval_infra.parsers.boxed import BoxedParser
from eval_infra.parsers.math_verify_parser import MathVerifyParser
from eval_infra.scorers.math_equiv import MathEquivalenceScorer
from eval_infra.tasks.base import Sample, Task
from eval_infra.tools.base import Tool

AGENTIC_SYSTEM_PROMPT = """\
You are a helpful math assistant with access to tools. Solve the problem step by step.
You may use tools by outputting a tool call in the following format:

```tool_call
{{"name": "<tool_name>", "arguments": {{<args>}}}}
```

Available tools:
{tool_descriptions}

When you have the final answer, put it inside \\boxed{{}}.
"""


class AgenticMathTask(Task):
    name = "agentic_math"

    def __init__(self, tools: list[Tool] | None = None):
        self.parser = MathVerifyParser()
        self.scorer = MathEquivalenceScorer()
        self.tools = tools or []

    def load_dataset(self, max_samples: int | None = None) -> list[Sample]:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

        # Select harder problems (Level 4-5) that benefit from tool use
        samples: list[Sample] = []
        for i, row in enumerate(ds):
            level = row.get("level", "")
            if "4" not in str(level) and "5" not in str(level):
                continue
            if max_samples is not None and len(samples) >= max_samples:
                break
            answer = row.get("answer") or BoxedParser().parse(row["solution"]) or ""
            samples.append(Sample(
                id=row.get("unique_id", str(i)),
                prompt=row["problem"],
                expected_answer=answer,
                metadata={"type": row.get("subject", "unknown"), "level": level},
            ))
        return samples

    def format_prompt(self, sample: Sample) -> list[dict[str, str]]:
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in self.tools
        )
        system = AGENTIC_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions or "(none)")
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": sample.prompt},
        ]

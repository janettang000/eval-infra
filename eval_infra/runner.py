from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from eval_infra.engine import Engine
from eval_infra.tasks.base import Sample, Task
from eval_infra.tools.base import Tool


@dataclass
class SampleResult:
    id: str
    expected: str
    predicted: str | None
    raw_output: str
    correct: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    task: str
    model: str
    accuracy: float
    total: int
    correct: int
    per_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    samples: list[SampleResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "model": self.model,
            "accuracy": self.accuracy,
            "total": self.total,
            "correct": self.correct,
            "per_category": self.per_category,
            "elapsed_seconds": self.elapsed_seconds,
            "samples": [asdict(s) for s in self.samples],
        }


class Runner:
    def __init__(self, engine: Engine, task: Task):
        self.engine = engine
        self.task = task

    def run(
        self,
        max_samples: int | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 2048,
        batch_size: int = 64,
    ) -> EvalResult:
        """Run standard batch evaluation."""
        samples = self.task.load_dataset(max_samples)
        prompts = [self.task.format_prompt(s) for s in samples]

        start = time.time()

        # Generate in batches
        all_outputs: list[str] = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            if isinstance(batch[0], list):
                outputs = self.engine.chat(batch, temperature=temperature, max_new_tokens=max_new_tokens)
            else:
                outputs = self.engine.generate(batch, temperature=temperature, max_new_tokens=max_new_tokens)
            all_outputs.extend(outputs)

        elapsed = time.time() - start

        # Parse and score
        results: list[SampleResult] = []
        category_stats: dict[str, dict[str, int]] = {}

        for sample, raw_output in zip(samples, all_outputs):
            predicted = self.task.parser.parse(raw_output)
            correct = False
            if predicted is not None:
                correct = self.task.scorer.score(predicted, sample.expected_answer)

            results.append(SampleResult(
                id=sample.id,
                expected=sample.expected_answer,
                predicted=predicted,
                raw_output=raw_output,
                correct=correct,
                metadata=sample.metadata,
            ))

            # Track per-category stats
            category = sample.metadata.get("type") or sample.metadata.get("category", "unknown")
            if category not in category_stats:
                category_stats[category] = {"total": 0, "correct": 0}
            category_stats[category]["total"] += 1
            if correct:
                category_stats[category]["correct"] += 1

        total = len(results)
        correct_count = sum(1 for r in results if r.correct)

        per_category = {
            cat: {**stats, "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0}
            for cat, stats in category_stats.items()
        }

        return EvalResult(
            task=self.task.name,
            model=self.engine.model_path,
            accuracy=correct_count / total if total else 0.0,
            total=total,
            correct=correct_count,
            per_category=per_category,
            samples=results,
            elapsed_seconds=elapsed,
        )


class AgenticRunner:
    """Multi-turn runner for agentic evaluations with tool use."""

    def __init__(self, engine: Engine, task: Task, tools: list[Tool]):
        self.engine = engine
        self.task = task
        self.tools = {t.name: t for t in tools}

    def run(
        self,
        max_samples: int | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 2048,
        max_turns: int = 5,
    ) -> EvalResult:
        samples = self.task.load_dataset(max_samples)
        start = time.time()

        results: list[SampleResult] = []
        category_stats: dict[str, dict[str, int]] = {}

        for sample in samples:
            result = self._run_sample(sample, temperature, max_new_tokens, max_turns)
            results.append(result)

            category = sample.metadata.get("type") or sample.metadata.get("category", "unknown")
            if category not in category_stats:
                category_stats[category] = {"total": 0, "correct": 0}
            category_stats[category]["total"] += 1
            if result.correct:
                category_stats[category]["correct"] += 1

        elapsed = time.time() - start
        total = len(results)
        correct_count = sum(1 for r in results if r.correct)

        per_category = {
            cat: {**stats, "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0}
            for cat, stats in category_stats.items()
        }

        return EvalResult(
            task=self.task.name,
            model=self.engine.model_path,
            accuracy=correct_count / total if total else 0.0,
            total=total,
            correct=correct_count,
            per_category=per_category,
            samples=results,
            elapsed_seconds=elapsed,
        )

    def _run_sample(
        self,
        sample: Sample,
        temperature: float,
        max_new_tokens: int,
        max_turns: int,
    ) -> SampleResult:
        messages = self.task.format_prompt(sample)
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        full_output_parts: list[str] = []

        for _turn in range(max_turns):
            outputs = self.engine.chat([messages], temperature=temperature, max_new_tokens=max_new_tokens)
            response = outputs[0]
            full_output_parts.append(response)
            messages.append({"role": "assistant", "content": response})

            # Try to parse tool calls from response
            tool_calls = self._extract_tool_calls(response)
            if not tool_calls:
                break  # No tool calls = model is done

            # Execute tools and append results
            tool_results: list[str] = []
            for call in tool_calls:
                tool_name = call.get("name", "")
                tool_args = call.get("arguments", {})
                if tool_name in self.tools:
                    try:
                        result = self.tools[tool_name].execute(**tool_args)
                    except Exception as e:
                        result = f"Error: {e}"
                else:
                    result = f"Error: Unknown tool '{tool_name}'"
                tool_results.append(f"Tool '{tool_name}' result: {result}")

            messages.append({"role": "user", "content": "\n".join(tool_results)})

        raw_output = "\n---\n".join(full_output_parts)
        predicted = self.task.parser.parse(raw_output)
        correct = False
        if predicted is not None:
            correct = self.task.scorer.score(predicted, sample.expected_answer)

        return SampleResult(
            id=sample.id,
            expected=sample.expected_answer,
            predicted=predicted,
            raw_output=raw_output,
            correct=correct,
            metadata=sample.metadata,
        )

    def _extract_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Extract tool calls from model output. Expects JSON blocks with tool_call markers."""
        calls = []
        # Look for ```tool_call ... ``` blocks
        import re
        pattern = r"```tool_call\s*\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                call = json.loads(match.strip())
                if "name" in call:
                    calls.append(call)
            except json.JSONDecodeError:
                continue
        return calls

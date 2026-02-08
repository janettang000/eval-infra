from __future__ import annotations

import re

from datasets import load_dataset

from eval_infra.parsers.code import PythonCodeParser
from eval_infra.scorers.code_execution import CodeExecutionScorer
from eval_infra.tasks.base import Sample, Task

HUMANEVAL_SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Complete the following function. Return only the code, no explanations."
)


class HumanEvalTask(Task):
    """HumanEval benchmark: code generation with test-based scoring.

    Unlike parsing-only evals, scoring requires code execution:
    1. Model generates a function completion
    2. Parser extracts the code
    3. Runner assembles: prompt (signature+docstring) + completion + test harness
    4. Scorer executes the assembled program in a sandbox
    """

    name = "humaneval"

    STOP_SEQUENCES = ["\nclass ", "\nif __name__"]

    def __init__(self, timeout: int = 10, **kwargs):
        self.parser = PythonCodeParser()
        self.scorer = CodeExecutionScorer(timeout=timeout)

    def load_dataset(self, max_samples: int | None = None) -> list[Sample]:
        ds = load_dataset("openai/openai_humaneval", split="test")

        samples: list[Sample] = []
        for i, row in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break
            samples.append(Sample(
                id=row["task_id"],
                prompt=row["prompt"],
                expected_answer="",  # Not used — scoring is via execution
                metadata={
                    "entry_point": row["entry_point"],
                    "test": row["test"],
                    "canonical_solution": row["canonical_solution"],
                    "prompt_code": row["prompt"],
                },
            ))
        return samples

    def format_prompt(self, sample: Sample) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": HUMANEVAL_SYSTEM_PROMPT},
            {"role": "user", "content": sample.prompt},
        ]

    def assemble_program(self, sample: Sample, completion: str) -> str:
        """Assemble the full executable program from prompt + completion + tests."""
        prompt_code = sample.metadata["prompt_code"]
        test_code = sample.metadata["test"]
        entry_point = sample.metadata["entry_point"]

        # Extract preamble from prompt (everything before the target function def)
        preamble, _ = _split_at_function(prompt_code, entry_point)

        if _contains_function_def(completion, entry_point):
            # Model gave full function — prepend prompt preamble (imports + helpers)
            # Trim any extra top-level defs after the target function
            full_function = preamble + _trim_after_function(completion, entry_point)
        else:
            # Model gave just the body — indent and append to prompt
            indented = _ensure_indented(completion)
            full_function = prompt_code + indented

        # Assemble: function + test harness + invocation
        program = full_function + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
        return program


def _split_at_function(code: str, func_name: str) -> tuple[str, str]:
    """Split code into preamble (everything before the target function) and the rest."""
    lines = code.split("\n")
    pattern = re.compile(rf"^def\s+{re.escape(func_name)}\s*\(")
    for i, line in enumerate(lines):
        if pattern.match(line):
            preamble = "\n".join(lines[:i])
            if preamble.strip():
                preamble += "\n"
            else:
                preamble = ""
            rest = "\n".join(lines[i:])
            return preamble, rest
    return "", code


def _trim_after_function(code: str, func_name: str) -> str:
    """Keep only the target function and any code before the next top-level def/class."""
    lines = code.split("\n")
    # Find where the target function starts
    func_start = None
    for i, line in enumerate(lines):
        if re.match(rf"^def\s+{re.escape(func_name)}\s*\(", line):
            func_start = i
            break
    if func_start is None:
        return code
    # Find where the next top-level def/class starts (after the target function)
    for i in range(func_start + 1, len(lines)):
        if re.match(r"^(def |class )", lines[i]):
            return "\n".join(lines[:i]).rstrip()
    return code


def _contains_function_def(code: str, func_name: str) -> bool:
    """Check if the code contains a def for the given function name."""
    return bool(re.search(rf"^\s*def\s+{re.escape(func_name)}\s*\(", code, re.MULTILINE))


def _ensure_indented(code: str) -> str:
    """Ensure code lines are indented (for appending as function body)."""
    lines = code.split("\n")
    result = []
    for line in lines:
        if line.strip() == "":
            result.append("")
        elif not line.startswith((" ", "\t")):
            result.append("    " + line)
        else:
            result.append(line)
    return "\n".join(result)

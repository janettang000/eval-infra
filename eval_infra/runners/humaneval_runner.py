from __future__ import annotations

import time
from typing import Any

from eval_infra.engine import Engine
from eval_infra.runner import EvalResult, Runner, SampleResult
from eval_infra.scorers.code_execution import CodeExecutionScorer
from eval_infra.tasks.humaneval_task import HumanEvalTask


class HumanEvalRunner(Runner):
    """Runner for HumanEval-style code generation benchmarks.

    Extends the standard Runner with a code-execution scoring step:
      generate -> parse (extract code) -> assemble (prompt + code + tests) -> execute -> check

    This is architecturally different from parsing-only evals because scoring
    requires running untrusted code in a sandboxed subprocess.
    """

    def __init__(self, engine: Engine, task: HumanEvalTask):
        super().__init__(engine, task)
        self.task: HumanEvalTask = task

    def run(
        self,
        max_samples: int | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 2048,
        batch_size: int = 64,
    ) -> EvalResult:
        samples = self.task.load_dataset(max_samples)
        prompts = [self.task.format_prompt(s) for s in samples]

        start = time.time()

        # Batch generate
        all_outputs: list[str] = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            if isinstance(batch[0], list):
                outputs = self.engine.chat(
                    batch, temperature=temperature, max_new_tokens=max_new_tokens,
                    stop=HumanEvalTask.STOP_SEQUENCES,
                )
            else:
                outputs = self.engine.generate(
                    batch, temperature=temperature, max_new_tokens=max_new_tokens,
                    stop=HumanEvalTask.STOP_SEQUENCES,
                )
            all_outputs.extend(outputs)

        elapsed = time.time() - start

        # Parse, assemble, and execute
        scorer = self.task.scorer
        assert isinstance(scorer, CodeExecutionScorer)

        results: list[SampleResult] = []
        for sample, raw_output in zip(samples, all_outputs):
            # Step 1: Parse — extract code from model output
            extracted_code = self.task.parser.parse(raw_output)

            # Step 2: Assemble — combine prompt + completion + test harness
            if extracted_code is not None:
                program = self.task.assemble_program(sample, extracted_code)
                exec_details = scorer.execute_with_details(program)
                correct = exec_details["passed"]
            else:
                program = None
                exec_details = {"passed": False, "stdout": "", "stderr": "No code extracted", "returncode": -1}
                correct = False

            results.append(SampleResult(
                id=sample.id,
                expected=sample.metadata.get("canonical_solution", "")[:200],
                predicted=extracted_code,
                raw_output=raw_output,
                correct=correct,
                metadata={
                    **sample.metadata,
                    "exec_stdout": exec_details["stdout"],
                    "exec_stderr": exec_details["stderr"],
                    "exec_returncode": exec_details["returncode"],
                    "assembled_program": (program or "")[:3000],
                },
            ))

        total = len(results)
        correct_count = sum(1 for r in results if r.correct)

        return EvalResult(
            task=self.task.name,
            model=self.engine.model_path,
            accuracy=correct_count / total if total else 0.0,
            total=total,
            correct=correct_count,
            per_category={},
            samples=results,
            elapsed_seconds=elapsed,
        )

from __future__ import annotations

import subprocess
import tempfile
from typing import Any

from eval_infra.scorers.base import Scorer


class CodeExecutionScorer(Scorer):
    """Score by executing code in a sandboxed subprocess.

    For HumanEval-style tasks, the predicted string is the full program
    (function + test harness) to execute. If it exits with code 0 and
    no assertion errors, the answer is correct.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def score(self, predicted: str, expected: str) -> bool:
        """Execute `predicted` as a Python script. `expected` is unused (tests are baked in)."""
        if not predicted or not predicted.strip():
            return False

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=True
        ) as f:
            f.write(predicted)
            f.flush()
            try:
                result = subprocess.run(
                    ["python3", "-u", f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                return result.returncode == 0
            except subprocess.TimeoutExpired:
                return False
            except Exception:
                return False

    def execute_with_details(self, code: str) -> dict[str, Any]:
        """Execute code and return full details (for debugging)."""
        if not code or not code.strip():
            return {"passed": False, "stdout": "", "stderr": "Empty code", "returncode": -1}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=True
        ) as f:
            f.write(code)
            f.flush()
            try:
                result = subprocess.run(
                    ["python3", "-u", f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                return {
                    "passed": result.returncode == 0,
                    "stdout": result.stdout[:2000],
                    "stderr": result.stderr[:2000],
                    "returncode": result.returncode,
                }
            except subprocess.TimeoutExpired:
                return {"passed": False, "stdout": "", "stderr": "Timeout", "returncode": -1}
            except Exception as e:
                return {"passed": False, "stdout": "", "stderr": str(e), "returncode": -1}

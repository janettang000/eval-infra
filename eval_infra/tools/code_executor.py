from __future__ import annotations

import subprocess
import tempfile
from typing import Any

from eval_infra.tools.base import Tool


class PythonExecutor(Tool):
    name = "python"
    description = "Execute Python code and return the output. Code runs in an isolated subprocess with a timeout."
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"}
        },
        "required": ["code"],
    }

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def execute(self, code: str = "", **kwargs: Any) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
            f.write(code)
            f.flush()
            try:
                result = subprocess.run(
                    ["python3", f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                output = result.stdout
                if result.stderr:
                    output += f"\nSTDERR:\n{result.stderr}"
                if result.returncode != 0:
                    output = f"Exit code: {result.returncode}\n{output}"
                return output.strip() or "(no output)"
            except subprocess.TimeoutExpired:
                return f"Error: Code execution timed out after {self.timeout} seconds"
            except Exception as e:
                return f"Error executing code: {e}"

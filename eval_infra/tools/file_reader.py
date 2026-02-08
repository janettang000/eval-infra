from __future__ import annotations

import os
from typing import Any

from eval_infra.tools.base import Tool


class FileReader(Tool):
    name = "file_reader"
    description = "Read the contents of a file. Only files within the allowed directory can be read."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read"}
        },
        "required": ["path"],
    }

    def __init__(self, allowed_dir: str = "/workspace"):
        self.allowed_dir = os.path.abspath(allowed_dir)

    def execute(self, path: str = "", **kwargs: Any) -> str:
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(self.allowed_dir):
            return f"Error: Access denied. File must be within {self.allowed_dir}"
        try:
            with open(abs_path) as f:
                content = f.read(100_000)  # Cap at 100KB
            return content
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except Exception as e:
            return f"Error reading file: {e}"

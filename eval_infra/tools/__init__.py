from .calculator import Calculator
from .code_executor import PythonExecutor
from .file_reader import FileReader

TOOL_REGISTRY: dict[str, type] = {
    "calculator": Calculator,
    "python": PythonExecutor,
    "file_reader": FileReader,
}

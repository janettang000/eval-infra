from .math_task import MathTask
from .agentic_task import AgenticMathTask

TASK_REGISTRY: dict[str, type] = {
    "math": MathTask,
    "agentic_math": AgenticMathTask,
}

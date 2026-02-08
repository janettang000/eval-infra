from .math_task import MathTask
from .agentic_task import AgenticMathTask
from .gsm8k_task import GSM8KTask
from .mmlu_task import MMLUTask

TASK_REGISTRY: dict[str, type] = {
    "math": MathTask,
    "agentic_math": AgenticMathTask,
    "gsm8k": GSM8KTask,
    "mmlu": MMLUTask,
}

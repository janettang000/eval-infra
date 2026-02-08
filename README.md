# eval-infra

A modular evaluation framework for open-source LLMs using [sglang](https://github.com/sgl-project/sglang) as the inference backend.

## Quick Start

```bash
# 1. Run inside the sglang Docker container
sudo docker run -d --gpus all \
    --shm-size 32g --ipc=host \
    -v /path/to/eval-infra:/workspace/eval-infra \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --name eval-dev lmsysorg/sglang:latest sleep infinity
sudo docker exec -it eval-dev bash

# 2. Install extra dependencies
pip install "math-verify[antlr4_13_2]" datasets

# 3. Run an evaluation
cd /workspace/eval-infra
python -m eval_infra --model Qwen/Qwen2.5-7B-Instruct --task math --max-samples 100 --output results.json
```

## Supported Benchmarks

| Task          | CLI `--task` | Parser             | Scorer                | Type           |
|---------------|--------------|--------------------|-----------------------|----------------|
| MATH-500      | `math`       | BoxedParser / MathVerifyParser | MathEquivalenceScorer | Parsing-only |
| GSM8K         | `gsm8k`      | NumericParser      | NumericScorer         | Parsing-only   |
| MMLU          | `mmlu`       | MultiChoiceParser  | ExactMatchScorer      | Parsing-only   |
| HumanEval     | `humaneval`  | PythonCodeParser   | CodeExecutionScorer   | Code execution |
| Agentic MATH  | `agentic_math` | MathVerifyParser | MathEquivalenceScorer | Multi-turn agentic |

## CLI Usage

```bash
# MATH benchmark (choose parser: boxed or math_verify)
python -m eval_infra --model Qwen/Qwen2.5-7B-Instruct --task math --parser math_verify --output results/math.json

# GSM8K
python -m eval_infra --model Qwen/Qwen2.5-7B-Instruct --task gsm8k --max-samples 200

# MMLU (specify subject)
python -m eval_infra --model Qwen/Qwen2.5-7B-Instruct --task mmlu --subject abstract_algebra

# HumanEval (code execution)
python -m eval_infra --model Qwen/Qwen2.5-7B-Instruct --task humaneval --max-tokens 1024

# Agentic eval with tools
python -m eval_infra --model Qwen/Qwen2.5-7B-Instruct --task agentic_math --tools calculator,python --max-turns 5
```

### Key Options

| Flag             | Default       | Description                              |
|------------------|---------------|------------------------------------------|
| `--model`        | *(required)*  | HuggingFace model path or local path     |
| `--task`         | *(required)*  | Benchmark task name (see table above)    |
| `--max-samples`  | all           | Cap number of samples to evaluate        |
| `--temperature`  | 0.0           | Sampling temperature                     |
| `--max-tokens`   | 2048          | Max new tokens per generation            |
| `--batch-size`   | 64            | Batch size for inference                 |
| `--output`       | stdout        | Output JSON file path                    |
| `--parser`       | math_verify   | Parser for MATH task (boxed/math_verify) |
| `--tp`           | 1             | Tensor parallelism degree                |

## Architecture

```
eval_infra/
    engine.py           # sglang inference wrapper (generate + chat)
    runner.py           # Runner (batch) + AgenticRunner (multi-turn)
    cli.py              # CLI entry point
    tasks/
        base.py         # Task ABC + Sample dataclass
        math_task.py    # MATH-500 benchmark
        gsm8k_task.py   # GSM8K benchmark
        mmlu_task.py    # MMLU benchmark
        humaneval_task.py  # HumanEval benchmark
        agentic_task.py # Agentic MATH (multi-turn with tools)
    parsers/
        base.py         # Parser ABC
        boxed.py        # \boxed{} extraction (nested braces)
        math_verify_parser.py  # math-verify based extraction
        numeric.py      # Numeric answer extraction (GSM8K)
        multichoice.py  # Multiple choice A/B/C/D extraction
        code.py         # Python code block extraction
    scorers/
        base.py         # Scorer ABC
        math_equiv.py   # Math equivalence (math-verify + fallback)
        numeric.py      # Numeric comparison with tolerance
        exact_match.py  # Case-insensitive exact match
        code_execution.py  # Sandboxed subprocess execution
    runners/
        humaneval_runner.py  # Extended runner for code execution scoring
    tools/
        base.py         # Tool ABC
        calculator.py   # Safe math expression evaluator
        code_executor.py  # Python code execution (subprocess)
        file_reader.py  # Sandboxed file reader
```

### Key Abstractions

- **`Task`** — Defines a benchmark: loads dataset, formats prompts, bundles a parser and scorer.
- **`Parser`** — Extracts the answer from model output (e.g., `\boxed{}`, last number, code block).
- **`Scorer`** — Checks if the predicted answer matches the expected answer.
- **`Engine`** — Thin wrapper around `sglang.Engine` with `generate()` (string prompts) and `chat()` (message lists with auto chat-template application).
- **`Runner`** — Orchestrates batch eval: load -> prompt -> generate -> parse -> score -> aggregate.

The Runner is fully generic and contains zero benchmark-specific logic. All task-specific behavior lives in the Task/Parser/Scorer implementations.

## Adding a New Benchmark

To add a new benchmark (e.g., TriviaQA), create these files:

### 1. `eval_infra/parsers/triviaqa_parser.py`
```python
from eval_infra.parsers.base import Parser

class TriviaQAParser(Parser):
    def parse(self, text: str) -> str | None:
        # Extract the answer from model output
        ...
```

### 2. `eval_infra/scorers/` *(optional — reuse ExactMatchScorer if sufficient)*

### 3. `eval_infra/tasks/triviaqa_task.py`
```python
from datasets import load_dataset
from eval_infra.parsers.triviaqa_parser import TriviaQAParser
from eval_infra.scorers.exact_match import ExactMatchScorer
from eval_infra.tasks.base import Sample, Task

class TriviaQATask(Task):
    name = "triviaqa"

    def __init__(self):
        self.parser = TriviaQAParser()
        self.scorer = ExactMatchScorer()

    def load_dataset(self, max_samples=None) -> list[Sample]:
        ds = load_dataset("trivia_qa", "rc", split="validation")
        ...

    def format_prompt(self, sample) -> list[dict[str, str]]:
        return [{"role": "user", "content": f"Answer this: {sample.prompt}"}]
```

### 4. Register in `eval_infra/tasks/__init__.py`
```python
from .triviaqa_task import TriviaQATask
TASK_REGISTRY["triviaqa"] = TriviaQATask
```

That's it. No changes to `runner.py`, `engine.py`, or any core files.

## Output Format

Results JSON contains both aggregate metrics and per-sample details:

```json
{
  "task": "math",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "accuracy": 0.74,
  "total": 200,
  "correct": 148,
  "per_category": {"Algebra": {"total": 40, "correct": 35, "accuracy": 0.875}, ...},
  "elapsed_seconds": 12.3,
  "samples": [
    {
      "id": "test/algebra/123.json",
      "expected": "42",
      "predicted": "42",
      "raw_output": "Let me solve this step by step...",
      "correct": true,
      "metadata": {"type": "Algebra", "level": 3}
    }
  ]
}
```

## Dependencies

- `sglang[all]` — Inference backend (pre-installed in sglang Docker image)
- `datasets` — HuggingFace dataset loading
- `math-verify[antlr4_13_2]` — Math answer extraction and equivalence checking
- `transformers` — Tokenizer for chat template application

# eval-infra

A modular evaluation framework for open-source LLMs using [sglang](https://github.com/sgl-project/sglang) as the inference backend. Supports 4 benchmarks, multi-turn agentic evaluation with tool use, and models up to 32B on a single GPU.

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
# MATH-500 (full or filtered by difficulty level)
python -m eval_infra --model Qwen/Qwen2.5-7B-Instruct --task math --output results/math.json
python -m eval_infra --model Qwen/Qwen2.5-7B-Instruct --task math --level "4,5"

# GSM8K
python -m eval_infra --model Qwen/Qwen2.5-7B-Instruct --task gsm8k --max-samples 200

# MMLU (single or multiple subjects, comma-separated)
python -m eval_infra --model Qwen/Qwen2.5-7B-Instruct --task mmlu --subject "abstract_algebra,high_school_mathematics"

# HumanEval (code execution)
python -m eval_infra --model Qwen/Qwen2.5-7B-Instruct --task humaneval --max-tokens 1024

# Agentic eval with tools
python -m eval_infra --model Qwen/Qwen2.5-7B-Instruct --task agentic_math --tools "calculator,python" --max-turns 5

# Swap model with zero code changes
python -m eval_infra --model microsoft/Phi-3.5-mini-instruct --task gsm8k --max-samples 200
```

### CLI Options

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
| `--level`        | all           | MATH difficulty filter, comma-separated (e.g. "4,5") |
| `--subject`      | abstract_algebra | MMLU subject(s), comma-separated      |
| `--tools`        | *(none)*      | Agentic tools: calculator, python, file_reader |
| `--max-turns`    | 5             | Max conversation turns for agentic eval  |
| `--tp`           | 1             | Tensor parallelism degree                |

## Architecture

```
eval_infra/
    engine.py           # sglang wrapper: generate() for strings, chat() for message lists
    runner.py           # Runner (batch) + AgenticRunner (multi-turn with tool use)
    cli.py              # CLI entry point
    tasks/              # Task definitions: dataset loading, prompt formatting, parser/scorer binding
        math_task.py, gsm8k_task.py, mmlu_task.py, humaneval_task.py, agentic_task.py
    parsers/            # Answer extraction: \boxed{}, numeric, A/B/C/D, code blocks
    scorers/            # Answer checking: math equivalence, numeric, exact match, code execution
    runners/
        humaneval_runner.py  # Extended runner for code assembly + execution scoring
    tools/              # Agentic tools: calculator, python executor, file reader
```

### Design

- **`Task`** — Loads dataset, formats prompts, bundles a parser and scorer.
- **`Parser`** — Extracts the answer from model output (e.g., `\boxed{}`, last number, code block).
- **`Scorer`** — Checks if predicted answer matches expected answer.
- **`Engine`** — Thin wrapper around `sglang.Engine`. `generate()` takes string prompts; `chat()` takes message lists and auto-applies the model's chat template.
- **`Runner`** — Orchestrates batch eval: load → prompt → generate → parse → score → aggregate. Contains zero benchmark-specific logic.
- **`AgenticRunner`** — Multi-turn loop: generate → extract tool calls → execute → append results → repeat.

## Adding a New Benchmark

Create a task, parser (optional — reuse existing if sufficient), and register:

```python
# eval_infra/tasks/triviaqa_task.py
class TriviaQATask(Task):
    name = "triviaqa"
    def __init__(self):
        self.parser = TriviaQAParser()
        self.scorer = ExactMatchScorer()
    def load_dataset(self, max_samples=None) -> list[Sample]: ...
    def format_prompt(self, sample) -> list[dict[str, str]]: ...

# eval_infra/tasks/__init__.py
TASK_REGISTRY["triviaqa"] = TriviaQATask
```

No changes to `runner.py`, `engine.py`, or any core files.

## Output Format

```json
{
  "task": "math",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "accuracy": 0.74,
  "total": 200,
  "correct": 148,
  "per_category": {"Algebra": {"total": 40, "correct": 35, "accuracy": 0.875}},
  "elapsed_seconds": 12.3,
  "samples": [{"id": "...", "expected": "42", "predicted": "42", "raw_output": "...", "correct": true, "metadata": {...}}]
}
```

## Dependencies

- `sglang[all]` — Inference backend (pre-installed in sglang Docker image)
- `datasets` — HuggingFace dataset loading
- `math-verify[antlr4_13_2]` — Math answer extraction and equivalence checking
- `transformers` — Tokenizer for chat template application

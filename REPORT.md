# Benchmark Results Report

**Model:** Qwen/Qwen2.5-7B-Instruct
**Hardware:** NVIDIA H200 (141GB VRAM)
**Inference Backend:** sglang (offline batch mode)
**Decoding:** Greedy (temperature=0.0)
**Prompt style:** Zero-shot chat (system + user messages)

## Summary

| Benchmark              | Our Score | Reference Score | Diff  | Status |
|------------------------|-----------|-----------------|-------|--------|
| MATH-500 (200 samples) | 74.0%    | 75.5%           | -1.5% | OK     |
| GSM8K (200 samples)    | 93.0%    | 91.6%           | +1.4% | OK     |
| MMLU (5 subjects, 939) | 84.0%    | 75.4% (MMLU-redux) | +8.6% | OK (see notes)  |
| HumanEval (164 full)   | 81.1%    | 84.8%           | -3.7%  | OK     |
| Agentic MATH (Level 4-5, 262) | 56.1% | 61.1% (standard) | -5.0% | See notes |

## Detailed Results

### MATH-500

- **Accuracy:** 148/200 = 74.0%
- **Reference:** 75.5% (source: [Qwen2.5-LLM blog post](https://qwen.ai/blog?id=qwen2.5-llm), performance table for Qwen2.5-7B-Instruct)
- **Delta:** -1.5% (within expected variance)
- **Parser:** MathVerifyParser (boxed extraction + math-verify fallback)
- **Scorer:** MathEquivalenceScorer (math-verify symbolic checking + normalized string fallback)
- **Per-category:** Algebra 87.5%, Prealgebra 93.3%, Number Theory 72.0%, Intermediate Algebra 52.9%, Precalculus 60.7%, Geometry 56.3%, Counting & Probability 68.0%
- **Notes:** Evaluated on 200-sample subset of MATH-500. The reference uses 4-shot prompting while ours is zero-shot. Despite this, the -1.5% gap is within sampling variance, suggesting the chat-style prompt is effective for this model.

### GSM8K

- **Accuracy:** 186/200 = 93.0%
- **Reference:** 91.6% (source: [Qwen2.5-LLM blog post](https://qwen.ai/blog?id=qwen2.5-llm), performance table for Qwen2.5-7B-Instruct)
- **Delta:** +1.4% (within expected variance)
- **Parser:** NumericParser (4 extraction strategies: ####, \boxed{}, "the answer is", last number)
- **Scorer:** NumericScorer (numeric comparison with float tolerance)
- **Notes:** The reference uses 4-shot prompting; ours is zero-shot. All 14 failures were model reasoning errors, not parser/scorer bugs. Our score slightly exceeds reference, likely due to sampling variance on 200 samples.

### MMLU (5 subjects)

- **Aggregate accuracy:** 789/939 = 84.0%
- **Reference:** 75.4% MMLU-redux aggregate (source: [Qwen2.5-LLM blog post](https://qwen.ai/blog?id=qwen2.5-llm), performance table for Qwen2.5-7B-Instruct)
- **Delta:** +8.6% vs aggregate MMLU-redux
- **Parser:** MultiChoiceParser (4 extraction strategies for A/B/C/D answers)
- **Scorer:** ExactMatchScorer (case-insensitive)
- **Per-subject breakdown:**

| Subject | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| abstract_algebra | 73 | 100 | 73.0% |
| high_school_mathematics | 246 | 270 | 91.1% |
| high_school_computer_science | 91 | 100 | 91.0% |
| high_school_us_history | 165 | 204 | 80.9% |
| clinical_knowledge | 214 | 265 | 80.8% |
| **Aggregate** | **789** | **939** | **84.0%** |

- **Notes:** Our 84.0% aggregate exceeds the 75.4% MMLU-redux reference because our 5 subjects skew toward areas where Qwen2.5-7B excels (math and CS at 91%), while the reference is an aggregate across all 57 MMLU subjects including weaker areas. The reference also uses MMLU-redux (a cleaned variant) with 5-shot prompting, while ours uses standard `cais/mmlu` with zero-shot chat. The `--subject` flag accepts comma-separated subjects (e.g., `--subject "abstract_algebra,clinical_knowledge"`) for multi-subject evaluation with automatic per-subject breakdown.

### HumanEval

- **Accuracy (pass@1):** 133/164 = 81.1%
- **Reference:** 84.8% (source: [Qwen2.5-LLM blog post](https://qwen.ai/blog?id=qwen2.5-llm), performance table for Qwen2.5-7B-Instruct)
- **Delta:** -3.7%
- **Parser:** PythonCodeParser (fenced code block extraction)
- **Scorer:** CodeExecutionScorer (sandboxed subprocess, 10s timeout)
- **Prompt:** Simple "Complete the following function. Return only the code, no explanations." with raw HumanEval prompt as user message (EvalPlus-style). Stop sequences: `["\nclass ", "\nif __name__"]`.
- **Assembly:** Detects whether model gave complete function (with `def`) or just body. When model gives full function, prepends preamble from prompt (imports + helper functions). When model gives body only, appends indented body to prompt code.
- **Failure analysis:** 31 failures: 27 model logic errors (assertion failures), 1 NameError, 1 timeout, 2 other runtime errors. No SyntaxError or import-related failures.
- **Remaining gap:** The -3.7% gap vs reference is likely due to chat mode vs completion mode (reference uses infill/completion format where the model continues from the function signature, avoiding any prompt interpretation overhead).

## Cross-Model Compatibility

To demonstrate the framework is model-agnostic, we ran the same GSM8K benchmark with a non-Qwen model using **zero code changes** — only the `--model` CLI argument differs.

### GSM8K — Phi-3.5-mini-instruct

- **Model:** microsoft/Phi-3.5-mini-instruct (3.8B parameters)
- **Accuracy:** 178/200 = 89.0%
- **Reference:** 87.4% (source: [Phi-3.5 Technical Report](https://arxiv.org/abs/2404.14219), Table 3)
- **Delta:** +1.6% (within expected variance)

**Command comparison (only the model name changes):**
```bash
# Qwen run:
python -m eval_infra --model Qwen/Qwen2.5-7B-Instruct --task gsm8k ...
# Phi run:
python -m eval_infra --model microsoft/Phi-3.5-mini-instruct --task gsm8k ...
```

**Key takeaway:** The framework handles different model architectures (Qwen2, Phi-3), tokenizers, and chat templates transparently via sglang + HuggingFace transformers. No task code, parser, scorer, or runner required any modification.

| Model | Family | Params | GSM8K | Reference | Delta |
|-------|--------|--------|-------|-----------|-------|
| Qwen/Qwen2.5-7B-Instruct | Qwen2 | 7B | 93.0% (n=200) | 91.6% | +1.4% |
| microsoft/Phi-3.5-mini-instruct | Phi-3 | 3.8B | 89.0% (n=200) | 87.4% | +1.6% |
| Qwen/Qwen2.5-32B-Instruct | Qwen2 | 32B | 96.0% (n=100) | 95.2% | +0.8% |

### GSM8K — Qwen2.5-32B-Instruct (large model, single GPU)

- **Model:** Qwen/Qwen2.5-32B-Instruct (32B parameters)
- **Accuracy:** 96/100 = 96.0%
- **Reference:** 95.2% (source: [Qwen2.5-LLM blog post](https://qwen.ai/blog?id=qwen2.5-llm), performance table for Qwen2.5-32B-Instruct)
- **Delta:** +0.8% (within expected variance)
- **GPU memory:** ~63 GB for model weights (bfloat16) on a single NVIDIA H200 (141 GB VRAM), leaving ~78 GB for KV cache and activations
- **Special flags needed:** None — loaded with default settings (`dtype=auto` resolves to bfloat16, `tp_size=1`)
- **Inference time:** 20.2s for 100 samples (vs 3.5s for 7B on 200 samples)
- **Notes:** Demonstrates the framework supports models up to 32B on a single GPU with zero code changes. The 32B model scores 3 percentage points higher than the 7B on GSM8K, as expected.

## Agentic Evaluation (Tool Use)

The framework supports multi-turn agentic evaluation where the model can call tools (calculator, Python executor) during problem solving. We compare standard single-turn math evaluation against agentic multi-turn evaluation on the same MATH-500 Level 4-5 subset (262 problems).

### MATH-500 Level 4-5 — Standard vs Agentic

| Mode | Accuracy | Tools Used | Time |
|------|----------|------------|------|
| Standard (single-turn) | 160/262 = 61.1% | N/A | 5.5s (batched) |
| Agentic (multi-turn, calculator+python) | 147/262 = 56.1% | 6/262 samples | 882.1s (sequential) |

**Per-category comparison:**

| Category | Standard | Agentic | Delta |
|----------|----------|---------|-------|
| Algebra | 51/60 = 85.0% | 49/60 = 81.7% | -3.3% |
| Counting & Probability | 15/25 = 60.0% | 15/25 = 60.0% | 0.0% |
| Geometry | 11/23 = 47.8% | 10/23 = 43.5% | -4.3% |
| Intermediate Algebra | 21/59 = 35.6% | 17/59 = 28.8% | -6.8% |
| Number Theory | 20/31 = 64.5% | 19/31 = 61.3% | -3.2% |
| Prealgebra | 29/39 = 74.4% | 29/39 = 74.4% | 0.0% |
| Precalculus | 13/25 = 52.0% | 8/25 = 32.0% | -20.0% |

**Key findings:**
- The 7B model rarely used tools (only 6/262 samples contained `tool_call` markers), preferring to solve problems directly.
- The agentic prompt overhead (tool descriptions, format instructions) slightly degraded performance (-5.0% overall) without meaningful tool utilization.
- Sequential per-sample processing (required for multi-turn) was ~160x slower than batched single-turn inference.
- This suggests that effective agentic math evaluation requires larger models (e.g., 70B+) that can reliably follow tool-use formatting instructions and benefit from computation offloading.

**Architecture:** The `AgenticRunner` implements a multi-turn conversation loop: generate → extract `tool_call` blocks → execute tools → append results → repeat (up to `--max-turns` turns). Tool calls use a fenced code block format (```` ```tool_call ... ``` ````). Available tools are injected via CLI (`--tools "calculator,python"`).

```bash
# Standard evaluation on Level 4-5:
python -m eval_infra --model Qwen/Qwen2.5-7B-Instruct --task math --level "4,5"
# Agentic evaluation on Level 4-5:
python -m eval_infra --model Qwen/Qwen2.5-7B-Instruct --task agentic_math --tools "calculator,python" --max-turns 5
```

## Results Files

All results are stored in `results/` with full per-sample details:

| File | Samples | Size |
|------|---------|------|
| `results/math_results.json` | 200 | 368KB |
| `results/gsm8k_results.json` | 200 | 283KB |
| `results/mmlu_results.json` | 939 | — |
| `results/humaneval_results.json` | 164 | 519KB |
| `results/gsm8k_phi_results.json` | 200 | — |
| `results/gsm8k_32b_results.json` | 100 | — |
| `results/math_level45_results.json` | 262 | — |
| `results/agentic_math_results.json` | 262 | — |

Each JSON file contains per-sample `raw_output`, `predicted`, `expected`, `correct`, and task-specific `metadata` (category, level, execution traces, etc.).

## References

- Qwen Team. "Qwen2.5 Technical Report." arXiv:2412.15115, December 2024. https://arxiv.org/abs/2412.15115
- Qwen2.5-LLM blog post. https://qwen.ai/blog?id=qwen2.5-llm
- Qwen2.5-7B-Instruct model card. https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

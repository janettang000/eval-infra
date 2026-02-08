# Benchmark Results Report

**Model:** Qwen/Qwen2.5-7B-Instruct
**Hardware:** NVIDIA H200 (141GB VRAM), single GPU
**Inference Backend:** sglang (offline batch mode)
**Decoding:** Greedy (temperature=0.0)
**Prompt style:** Zero-shot chat (system + user messages)

## Summary

| Benchmark | Score | Reference | Delta | Notes |
|-----------|-------|-----------|-------|-------|
| MATH-500 (n=200) | 74.0% | 75.5% | -1.5% | Within variance |
| GSM8K (n=200) | 93.0% | 91.6% | +1.4% | Within variance |
| MMLU (5 subjects, n=939) | 84.0% | 75.4% (MMLU-redux) | +8.6% | Subject selection skew (see below) |
| HumanEval (n=164) | 81.1% | 84.8% | -3.7% | Chat vs completion mode gap |

All reference scores from the [Qwen2.5-LLM blog post](https://qwen.ai/blog?id=qwen2.5-llm). References use few-shot prompting; ours is zero-shot chat.

## Detailed Results

### MATH-500

- **Accuracy:** 148/200 = 74.0% (reference: 75.5%, 4-shot)
- **Per-category:** Algebra 87.5%, Prealgebra 93.3%, Number Theory 72.0%, Intermediate Algebra 52.9%, Precalculus 60.7%, Geometry 56.3%, Counting & Probability 68.0%
- The -1.5% gap is within sampling variance for 200 samples, suggesting zero-shot chat is effective for this model.

### GSM8K

- **Accuracy:** 186/200 = 93.0% (reference: 91.6%, 4-shot)
- All 14 failures were model reasoning errors, not parser/scorer bugs.

### MMLU (5 subjects)

- **Aggregate:** 789/939 = 84.0% (reference: 75.4% MMLU-redux, 5-shot)

| Subject | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| abstract_algebra | 73 | 100 | 73.0% |
| high_school_mathematics | 246 | 270 | 91.1% |
| high_school_computer_science | 91 | 100 | 91.0% |
| high_school_us_history | 165 | 204 | 80.9% |
| clinical_knowledge | 214 | 265 | 80.8% |

- Our 84.0% exceeds the 75.4% reference because our 5 subjects skew toward areas where Qwen2.5-7B excels (math and CS at 91%), while the reference aggregates all 57 MMLU subjects.

### HumanEval

- **Accuracy (pass@1):** 133/164 = 81.1% (reference: 84.8%)
- **Prompt:** "Complete the following function. Return only the code, no explanations." Stop sequences: `["\nclass ", "\nif __name__"]`.
- **Assembly:** Detects whether model gave complete function or just body. Preserves imports and helper functions from the original HumanEval prompt.
- **Failure analysis:** 31 failures — 27 logic errors, 1 NameError, 1 timeout, 2 runtime errors. Zero SyntaxError or import failures.
- The -3.7% gap is likely due to chat mode vs completion mode (reference uses infill format where the model continues directly from the function signature).

## Cross-Model Compatibility

The framework is model-agnostic — only the `--model` flag changes. No task, parser, scorer, or runner code was modified.

| Model | Params | GSM8K | Reference | Delta |
|-------|--------|-------|-----------|-------|
| Qwen/Qwen2.5-7B-Instruct | 7B | 93.0% (n=200) | 91.6% | +1.4% |
| microsoft/Phi-3.5-mini-instruct | 3.8B | 89.0% (n=200) | 87.4% | +1.6% |
| Qwen/Qwen2.5-32B-Instruct | 32B | 96.0% (n=100) | 95.2% | +0.8% |

- **Phi-3.5** reference: [Phi-3.5 Technical Report](https://arxiv.org/abs/2404.14219), Table 3.
- **32B model:** ~63 GB VRAM (bfloat16) on a single H200, no special flags needed. Demonstrates the framework supports models up to 32B on one GPU.

## Agentic Evaluation (Tool Use)

Comparison of standard single-turn vs agentic multi-turn evaluation on the same MATH-500 Level 4-5 subset (262 problems).

| Mode | Accuracy | Tools Used | Time |
|------|----------|------------|------|
| Standard (single-turn, batched) | 160/262 = 61.1% | N/A | 5.5s |
| Agentic (multi-turn, calculator+python) | 147/262 = 56.1% | 6/262 samples | 882s |

**Per-category comparison:**

| Category | Standard | Agentic | Delta |
|----------|----------|---------|-------|
| Algebra | 85.0% | 81.7% | -3.3% |
| Counting & Probability | 60.0% | 60.0% | 0.0% |
| Geometry | 47.8% | 43.5% | -4.3% |
| Intermediate Algebra | 35.6% | 28.8% | -6.8% |
| Number Theory | 64.5% | 61.3% | -3.2% |
| Prealgebra | 74.4% | 74.4% | 0.0% |
| Precalculus | 52.0% | 32.0% | -20.0% |

**Findings:**
- The 7B model rarely used tools (6/262 samples), preferring to solve problems directly.
- The agentic prompt overhead (tool descriptions, format instructions) degraded performance by 5.0% without meaningful tool utilization.
- This suggests effective tool-augmented math evaluation requires larger models that can reliably follow tool-calling conventions.

## References

- Qwen2.5 Technical Report: https://arxiv.org/abs/2412.15115
- Qwen2.5-LLM blog post: https://qwen.ai/blog?id=qwen2.5-llm
- Phi-3.5 Technical Report: https://arxiv.org/abs/2404.14219

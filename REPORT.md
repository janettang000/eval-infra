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
| MMLU abstract_algebra  | 74.0%    | 75.4% (MMLU-redux) | -1.4% | OK (see notes) |
| HumanEval (164 full)   | 81.1%    | 84.8%           | -3.7%  | OK     |

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

### MMLU (abstract_algebra)

- **Accuracy:** 74/100 = 74.0%
- **Reference:** 75.4% MMLU-redux aggregate (source: [Qwen2.5-LLM blog post](https://qwen.ai/blog?id=qwen2.5-llm), performance table for Qwen2.5-7B-Instruct)
- **Delta:** -1.4% vs aggregate MMLU-redux
- **Parser:** MultiChoiceParser (4 extraction strategies for A/B/C/D answers)
- **Scorer:** ExactMatchScorer (case-insensitive)
- **Notes:** Direct comparison is approximate: (1) our evaluation covers only the abstract_algebra subject (100 samples) while the reference is the aggregate across all MMLU subjects; (2) the reference uses MMLU-redux (a cleaned version) while we use standard `cais/mmlu`; (3) the reference uses 5-shot prompting while ours is zero-shot.

### HumanEval

- **Accuracy (pass@1):** 133/164 = 81.1%
- **Reference:** 84.8% (source: [Qwen2.5-LLM blog post](https://qwen.ai/blog?id=qwen2.5-llm), performance table for Qwen2.5-7B-Instruct)
- **Delta:** -3.7%
- **Parser:** PythonCodeParser (fenced code block extraction)
- **Scorer:** CodeExecutionScorer (sandboxed subprocess, 10s timeout)
- **Prompt:** Simple "Complete the following function. Return only the code, no explanations." with raw HumanEval prompt as user message (EvalPlus-style). Stop sequences: `["\nclass ", "\nif __name__"]`.
- **Assembly:** Detects whether model gave complete function (with `def`) or just body. When model gives full function, prepends preamble from prompt (imports + helper functions). When model gives body only, appends indented body to prompt code.
- **Failure analysis:** 31 failures: 27 model logic errors (assertion failures), 1 NameError, 1 timeout, 2 other runtime errors. All SyntaxError and import-related failures from the previous version (62.8%) are resolved.
- **Previous score:** 62.8% (103/164) — improved by fixing prompt format, import handling, and helper function preservation.
- **Remaining gap:** The -3.7% gap vs reference is likely due to chat mode vs completion mode (reference uses infill/completion format where the model continues from the function signature, avoiding any prompt interpretation overhead).

## Results Files

All results are stored in `results/` with full per-sample details:

| File | Samples | Size |
|------|---------|------|
| `results/math_results.json` | 200 | 368KB |
| `results/gsm8k_results.json` | 200 | 283KB |
| `results/mmlu_results.json` | 100 | 180KB |
| `results/humaneval_results.json` | 164 | 519KB |

Each JSON file contains per-sample `raw_output`, `predicted`, `expected`, `correct`, and task-specific `metadata` (category, level, execution traces, etc.).

## References

- Qwen Team. "Qwen2.5 Technical Report." arXiv:2412.15115, December 2024. https://arxiv.org/abs/2412.15115
- Qwen2.5-LLM blog post. https://qwen.ai/blog?id=qwen2.5-llm
- Qwen2.5-7B-Instruct model card. https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

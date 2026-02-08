from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Eval Infrastructure — run LLM benchmarks with sglang",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model path or local path")
    parser.add_argument("--task", required=True, choices=["math", "agentic_math"], help="Benchmark task to run")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max new tokens per generation")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--parser", type=str, default="math_verify", choices=["boxed", "math_verify"],
                        help="Parser to use for answer extraction")

    # Agentic options
    parser.add_argument("--max-turns", type=int, default=5, help="Max turns for agentic eval")
    parser.add_argument("--tools", type=str, default="", help="Comma-separated tool names (calculator,python,file_reader)")

    # Engine options
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")

    args = parser.parse_args(argv)

    from eval_infra.engine import Engine
    from eval_infra.runner import AgenticRunner, Runner
    from eval_infra.tasks import TASK_REGISTRY
    from eval_infra.tools import TOOL_REGISTRY

    # Build engine
    engine_kwargs = {}
    if args.tp > 1:
        engine_kwargs["tp_size"] = args.tp
    engine = Engine(args.model, **engine_kwargs)

    try:
        if args.task == "agentic_math":
            # Build tools
            tool_names = [t.strip() for t in args.tools.split(",") if t.strip()]
            tools = [TOOL_REGISTRY[name]() for name in tool_names if name in TOOL_REGISTRY]

            task_cls = TASK_REGISTRY[args.task]
            task = task_cls(tools=tools)

            runner = AgenticRunner(engine, task, tools)
            result = runner.run(
                max_samples=args.max_samples,
                temperature=args.temperature,
                max_new_tokens=args.max_tokens,
                max_turns=args.max_turns,
            )
        else:
            task_cls = TASK_REGISTRY[args.task]
            task = task_cls(parser=args.parser)

            runner = Runner(engine, task)
            result = runner.run(
                max_samples=args.max_samples,
                temperature=args.temperature,
                max_new_tokens=args.max_tokens,
                batch_size=args.batch_size,
            )

        # Output results
        result_dict = result.to_dict()

        print(f"\n{'='*60}")
        print(f"Task: {result.task} | Model: {result.model}")
        print(f"Accuracy: {result.correct}/{result.total} = {result.accuracy:.4f}")
        print(f"Time: {result.elapsed_seconds:.1f}s")
        if result.per_category:
            print(f"\nPer-category results:")
            for cat, stats in sorted(result.per_category.items()):
                print(f"  {cat}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")
        print(f"{'='*60}\n")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result_dict, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            json.dump(result_dict, sys.stdout, indent=2)
            print()

    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()

"""
CLI entry point for Vision Analysis Benchmark.

Commands:
    va-bench run    -- Benchmark models on COCO val2017
    va-bench list   -- Show available models and specs
    va-bench score  -- Compute VA v1 Scores from collected results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_run(args: argparse.Namespace) -> None:
    """Run benchmarks on one or more models."""
    from .benchmark import benchmark_model
    from .models import list_models, get_spec
    from .output import save_result

    if args.all:
        model_keys = list_models()
    elif args.models:
        model_keys = args.models
    else:
        print("Error: specify --models or --all")
        sys.exit(1)

    print(f"Will benchmark {len(model_keys)} model(s)")
    print(f"  COCO dir: {args.coco_dir}")
    print(f"  Output:   {args.output_dir}")
    print(f"  Device:   {args.device}")

    for key in model_keys:
        try:
            result = benchmark_model(
                model_key=key,
                coco_dir=args.coco_dir,
                device=args.device,
                verbose=not args.quiet,
            )
            filepath = save_result(result, args.output_dir)
            print(f"\nSaved: {filepath}")
        except Exception as e:
            print(f"\nError benchmarking {key}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            continue

    print(f"\nDone. Results in {args.output_dir}/")


def cmd_list(args: argparse.Namespace) -> None:
    """List available models."""
    from .models import MODEL_REGISTRY

    print(f"\n{'Key':<16} {'Display Name':<16} {'Family':<10} {'Params(M)':<10} "
          f"{'GFLOPs':<8} {'Input':<6} {'Weights'}")
    print("-" * 90)

    for key in sorted(MODEL_REGISTRY.keys()):
        s = MODEL_REGISTRY[key]
        params = f"{s.paper_params_m:.1f}" if s.paper_params_m > 0 else "?"
        flops = f"{s.paper_flops_g:.1f}" if s.paper_flops_g > 0 else "?"
        print(f"{s.key:<16} {s.display_name:<16} {s.family:<10} {params:<10} "
              f"{flops:<8} {s.input_size:<6} {s.weight_file}")

    print(f"\n{len(MODEL_REGISTRY)} models available")


def cmd_score(args: argparse.Namespace) -> None:
    """Compute VA v1 Scores from benchmark results."""
    from .scoring import compute_va_v1_scores, load_results, save_scores

    results = load_results(args.results_dir)
    if not results:
        print(f"No benchmark JSON files found in {args.results_dir}")
        sys.exit(1)

    print(f"Loaded {len(results)} benchmark result(s)")

    scores = compute_va_v1_scores(results)

    va_scores = scores["va_v1_scores"]
    if not va_scores:
        print("\nNo models qualify for VA v1 Score (need both RTX 5080 and RPi5 benchmarks)")
        if scores["skipped"]:
            print(f"Models with only one hardware: {', '.join(scores['skipped'])}")
        sys.exit(0)

    # Print results
    print(f"\n{'Model':<20} {'VA v1 Score':<12} {'mAP@50':<8} {'mAP@50-95':<10} "
          f"{'mAP_small':<10} {'FPS 5080':<10} {'FPS RPi5':<10} {'mAP/GFLOP':<10}")
    print("-" * 100)

    for model_id, data in va_scores.items():
        c = data["components"]
        print(
            f"{model_id:<20} {data['composite']:<12} "
            f"{c['mAP_50']['raw']:<8.1f} "
            f"{c['mAP_50_95']['raw']:<10.1f} "
            f"{c['mAP_small']['raw']:<10.1f} "
            f"{c['fps_rtx5080']['raw']:<10.1f} "
            f"{c['fps_rpi5']['raw']:<10.1f} "
            f"{c['mAP_per_gflop']['raw']:<10.2f}"
        )

    if scores["skipped"]:
        print(f"\nSkipped (missing hardware): {', '.join(scores['skipped'])}")

    # Save
    output_path = Path(args.results_dir) / "va_v1_scores.json"
    save_scores(scores, output_path)
    print(f"\nScores saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="va-bench",
        description="Vision Analysis Benchmark — powers visionanalysis.org",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Benchmark models on COCO val2017")
    run_parser.add_argument(
        "--models", nargs="+", help="Model keys to benchmark (e.g. yolov9t yolox-s)"
    )
    run_parser.add_argument("--all", action="store_true", help="Benchmark all models")
    run_parser.add_argument(
        "--coco-dir", type=str, required=True,
        help="Path to COCO directory (with annotations/ and images/val2017/)",
    )
    run_parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Output directory for result JSONs (default: ./results)",
    )
    run_parser.add_argument("--device", type=str, default="auto", help="Device (default: auto)")
    run_parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    run_parser.add_argument("--debug", action="store_true", help="Print full tracebacks on error")

    # --- list ---
    subparsers.add_parser("list", help="List available models and specs")

    # --- score ---
    score_parser = subparsers.add_parser("score", help="Compute VA v1 Scores")
    score_parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Directory containing benchmark result JSONs",
    )

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "score":
        cmd_score(args)


if __name__ == "__main__":
    main()

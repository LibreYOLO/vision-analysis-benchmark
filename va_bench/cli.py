"""
CLI entry point for Vision Analysis Benchmark.

Commands:
    va-bench run    -- Benchmark models on COCO val2017
    va-bench list   -- Show available models and specs
"""

from __future__ import annotations

import argparse
import os
import sys

# Per-source NMS IoU defaults for `va-bench run` when --iou is not given.
# libreyolo keeps the historical harness default; ultralytics uses their
# shipped predict default per the eval protocol (PLAN_ultralytics_source.md).
DEFAULT_IOU = {"libreyolo": 0.6, "ultralytics": 0.7}


def _spec_source(key: str) -> str:
    """Return the registry source for a key, or 'libreyolo' if unknown.

    Unknown keys are left for the per-model benchmark loop to report.
    """
    from .models import MODEL_REGISTRY

    spec = MODEL_REGISTRY.get(key)
    return spec.source if spec is not None else "libreyolo"


def cmd_run(args: argparse.Namespace) -> None:
    """Run benchmarks on one or more models."""
    from .benchmark import benchmark_model, run_ultralytics_benchmark
    from .models import list_models
    from .output import save_result

    if args.format in ("onnx", "tensorrt") and not args.weights_dir:
        print(f"Error: --weights-dir is required when --format {args.format}")
        sys.exit(1)

    if args.all:
        model_keys = list_models()
    elif args.models:
        model_keys = args.models
    else:
        print("Error: specify --models or --all")
        sys.exit(1)

    uly_keys = [k for k in model_keys if _spec_source(k) == "ultralytics"]
    uly_python = args.uly_python or os.environ.get("VA_ULY_PYTHON")
    if uly_keys:
        if not uly_python:
            print(
                f"Error: ultralytics-source model(s) requested "
                f"({', '.join(uly_keys)}) but no driver Python configured. "
                f"Pass --uly-python <path to .venv-ultralytics python> or set "
                f"the VA_ULY_PYTHON environment variable."
            )
            sys.exit(1)
        if args.format != "pytorch":
            print(
                f"Error: ultralytics-source models only support --format pytorch "
                f"(got --format {args.format})."
            )
            sys.exit(1)

    print(f"Will benchmark {len(model_keys)} model(s)")
    print(f"  Format:   {args.format}")
    print(f"  COCO dir: {args.coco_dir}")
    print(f"  Output:   {args.output_dir}")
    print(f"  Device:   {args.device}")
    if args.format in ("onnx", "tensorrt"):
        print(f"  Weights:  {args.weights_dir}")
    if uly_keys:
        print(f"  Uly driver python: {uly_python}")

    for key in model_keys:
        try:
            source = _spec_source(key)
            iou = args.iou if args.iou is not None else DEFAULT_IOU[source]
            if source == "ultralytics":
                result = run_ultralytics_benchmark(
                    model_key=key,
                    coco_dir=args.coco_dir,
                    uly_python=uly_python,
                    weights_dir=args.weights_dir,
                    device=args.device,
                    conf=args.conf,
                    iou=iou,
                    max_det=args.max_det,
                    limit=args.limit,
                    verbose=not args.quiet,
                )
            else:
                result = benchmark_model(
                    model_key=key,
                    coco_dir=args.coco_dir,
                    fmt=args.format,
                    weights_dir=args.weights_dir,
                    device=args.device,
                    conf=args.conf,
                    iou=iou,
                    max_det=args.max_det,
                    limit=args.limit,
                    verbose=not args.quiet,
                    precision=args.precision,
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

    print(f"\n{'Key':<16} {'Display Name':<16} {'Family':<10} {'Source':<12} "
          f"{'Params(M)':<10} {'GFLOPs':<8} {'Input':<6} {'Weights'}")
    print("-" * 103)

    for key in sorted(MODEL_REGISTRY.keys()):
        s = MODEL_REGISTRY[key]
        params = f"{s.paper_params_m:.1f}" if s.paper_params_m > 0 else "?"
        flops = f"{s.paper_flops_g:.1f}" if s.paper_flops_g > 0 else "?"
        print(f"{s.key:<16} {s.display_name:<16} {s.family:<10} {s.source:<12} "
              f"{params:<10} {flops:<8} {s.input_size:<6} {s.weight_file}")

    print(f"\n{len(MODEL_REGISTRY)} models available")


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
    run_parser.add_argument(
        "--format", choices=["pytorch", "onnx", "tensorrt"], default="pytorch",
        help="Backend format (default: pytorch)",
    )
    run_parser.add_argument(
        "--weights-dir", type=str, default=None,
        help="Directory with user-supplied .onnx / .engine weights "
             "(required with --format onnx or --format tensorrt)",
    )
    run_parser.add_argument(
        "--conf", type=float, default=0.001,
        help="Confidence threshold recorded in the submission (default: 0.001)",
    )
    run_parser.add_argument(
        "--iou", type=float, default=None,
        help="IoU threshold recorded in the submission "
             "(default: 0.6 for libreyolo models, 0.7 for ultralytics models)",
    )
    run_parser.add_argument(
        "--max-det", type=int, default=300,
        help="Maximum detections per image recorded in the submission (default: 300)",
    )
    run_parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N val2017 images (dev/CPU subset; "
             "NOT a valid full-val2017 submission). Default: all images.",
    )
    run_parser.add_argument(
        "--precision", type=str, default="fp16", choices=["fp16", "fp32"],
        help="TensorRT engine precision label recorded in the submission (default: fp16)",
    )
    run_parser.add_argument(
        "--uly-python", type=str, default=None,
        help="Path to the ultralytics driver venv python (.venv-ultralytics). "
             "Required when benchmarking ultralytics-source models; the "
             "VA_ULY_PYTHON environment variable is honored as a fallback.",
    )
    run_parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    run_parser.add_argument("--debug", action="store_true", help="Print full tracebacks on error")

    # --- list ---
    subparsers.add_parser("list", help="List available models and specs")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()

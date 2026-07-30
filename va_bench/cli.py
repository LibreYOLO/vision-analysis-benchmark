"""
CLI entry point for Vision Analysis Benchmark.

Commands:
    va-bench run    -- Benchmark models on COCO val2017
    va-bench list   -- Show available models and specs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_LEADING_DASH_DATASET = "__VA_BENCH_LEADING_DASH_DATASET__"


def _protect_leading_dash_dataset_names(argv: list[str]) -> list[str]:
    """Keep argparse from treating RF100-VL names like ``-grccs`` as flags."""
    protected = list(argv)
    try:
        start = protected.index("--datasets") + 1
    except ValueError:
        return protected
    for index in range(start, len(protected)):
        value = protected[index]
        if value == "-h" or value.startswith("--"):
            break
        if value.startswith("-"):
            protected[index] = _LEADING_DASH_DATASET + value
    return protected


def _restore_leading_dash_dataset_names(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    return [
        value.removeprefix(_LEADING_DASH_DATASET)
        if value.startswith(_LEADING_DASH_DATASET)
        else value
        for value in values
    ]


def cmd_run(args: argparse.Namespace) -> None:
    """Run benchmarks on one or more models."""
    from .benchmark import benchmark_model
    from .models import list_models
    from .output import save_result
    from .provenance import harness_git_info, reconstruct_command

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

    print(f"Will benchmark {len(model_keys)} model(s)")
    print(f"  Format:   {args.format}")
    print(f"  COCO dir: {args.coco_dir}")
    print(f"  Output:   {args.output_dir}")
    print(f"  Device:   {args.device}")
    if args.format in ("onnx", "tensorrt"):
        print(f"  Weights:  {args.weights_dir}")

    # Invocation-level provenance, shared by every model in this run.
    harness = harness_git_info()
    invocation = {
        "harness_commit": harness["commit"],
        "harness_dirty": harness["dirty"],
        "argv": sys.argv[1:],
        "command": reconstruct_command(sys.argv[1:]),
    }

    for key in model_keys:
        try:
            result = benchmark_model(
                model_key=key,
                coco_dir=args.coco_dir,
                fmt=args.format,
                weights_dir=args.weights_dir,
                device=args.device,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                limit=args.limit,
                verbose=not args.quiet,
                precision=args.precision,
                dataset_id=args.dataset_id,
                dataset_revision=args.dataset_revision,
            )
            result.setdefault("repro", {}).update(invocation)
            filepath = save_result(result, args.output_dir)
            print(f"\nSaved: {filepath}")
        except Exception as e:
            print(f"\nError benchmarking {key}: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            continue

    print(f"\nDone. Results in {args.output_dir}/")


def cmd_train_bench(args: argparse.Namespace) -> None:
    """Measure training throughput (img/s -> sec/epoch -> $/epoch) per config."""
    from .provenance import harness_git_info, reconstruct_command
    from .train_throughput import benchmark_train_throughput, save_train_result

    if args.all:
        from .models import list_models

        model_keys = list_models()
    elif args.models:
        model_keys = args.models
    else:
        print("Error: specify --models or --all")
        sys.exit(1)

    harness = harness_git_info()
    invocation = {
        "harness_commit": harness["commit"],
        "harness_dirty": harness["dirty"],
        "argv": sys.argv[1:],
        "command": reconstruct_command(sys.argv[1:]),
    }

    for key in model_keys:
        try:
            result = benchmark_train_throughput(
                key,
                data=args.data,
                device=args.device,
                batch=args.batch,
                imgsz=args.imgsz,
                warmup_epochs=args.warmup_epochs,
                measure_epochs=args.measure_epochs,
                workers=args.workers,
                amp=args.amp,
                amp_dtype=args.amp_dtype,
                nbs=args.nbs,
                dollars_per_hour=args.dollars_per_hour,
                rig_label=args.rig_label,
                provider=args.provider,
                verbose=not args.quiet,
            )
            result.setdefault("repro", {}).update(invocation)
            path = save_train_result(result, args.output_dir)
            print(f"Saved: {path}")
        except Exception as e:
            print(f"\nError train-benchmarking {key}: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            continue

    print(f"\nDone. Results in {args.output_dir}/")


def cmd_rf100vl(args: argparse.Namespace) -> None:
    """Evaluate models across the RF100-VL datasets and emit submissions."""
    from .output import save_result
    from .provenance import harness_git_info, reconstruct_command
    from .rf100vl import benchmark_model_rf100vl, download_datasets

    if args.download:
        download_datasets(args.data_dir, subset=args.subset, verbose=not args.quiet)

    if args.all:
        from .models import list_models

        model_keys = list_models()
    elif args.models:
        model_keys = args.models
    else:
        if args.download:
            print("Datasets downloaded. Specify --models or --all to evaluate.")
            return
        print("Error: specify --models or --all")
        sys.exit(1)

    print(f"Will evaluate {len(model_keys)} model(s) on RF100-VL")
    print(f"  Format:       {args.format}")
    print(f"  Data dir:     {args.data_dir}")
    print(f"  Weights root: {args.weights_root or '(none; requires --allow-pretrained)'}")
    print(f"  Output:       {args.output_dir}")

    harness = harness_git_info()
    invocation = {
        "harness_commit": harness["commit"],
        "harness_dirty": harness["dirty"],
        "argv": sys.argv[1:],
        "command": reconstruct_command(sys.argv[1:]),
    }

    failed: list[str] = []
    for key in model_keys:
        try:
            result = benchmark_model_rf100vl(
                model_key=key,
                data_dir=args.data_dir,
                fmt=args.format,
                weights_root=args.weights_root,
                device=args.device,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                split=args.split,
                limit=args.limit,
                limit_datasets=args.limit_datasets,
                allow_pretrained=args.allow_pretrained,
                versions_path=args.versions,
                recipe_path=args.recipe,
                per_dataset_dir=args.per_dataset_dir,
                verbose=not args.quiet,
            )
            result.setdefault("repro", {}).update(invocation)
            filepath = save_result(result, args.output_dir)
            print(f"\nSaved: {filepath}")
        except Exception as e:
            failed.append(key)
            print(f"\nError on RF100-VL for {key}: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            continue

    if failed:
        print(f"\nRF100-VL evaluation failed for {len(failed)} model(s): {', '.join(failed)}")
        raise SystemExit(1)
    print(f"\nDone. Results in {args.output_dir}/")


def cmd_rf100vl_train(args: argparse.Namespace) -> None:
    """Train one fine-tuned checkpoint per RF100-VL dataset."""
    from .rf100vl_train import orchestrate_training

    summary = orchestrate_training(
        model_key=args.model,
        data_dir=args.data_dir,
        weights_root=args.weights_root,
        recipe_path=args.recipe,
        gpus=[part.strip() for part in args.gpus.split(",") if part.strip()],
        datasets=args.datasets,
        limit_datasets=args.limit_datasets,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        timeout_hours=args.timeout_hours,
        runs_root=args.runs_root,
        state_root=args.state_root,
        smoke_epochs=args.smoke_epochs,
        force=args.force,
    )
    print(
        "RF100-VL training complete: "
        f"{len(summary['completed'])} completed, "
        f"{len(summary['skipped_done'])} already done, "
        f"{len(summary['failed'])} failed, "
        f"{len(summary.get('active_running', []))} still running"
    )
    print(f"Summary: {Path(summary['rerun_file']).parent / 'summary.json'}")
    if summary["failed"]:
        print(f"Rerun list: {summary['rerun_file']}")
    if summary.get("active_running"):
        print("Active datasets: " + ", ".join(summary["active_running"]))
    if summary["failed"] or summary.get("active_running"):
        raise SystemExit(1)


def cmd_list(args: argparse.Namespace) -> None:
    """List available models."""
    from .models import MODEL_REGISTRY

    print(
        f"\n{'Key':<16} {'Display Name':<16} {'Family':<10} {'Params(M)':<10} "
        f"{'GFLOPs':<8} {'Input':<6} {'Weights'}"
    )
    print("-" * 90)

    for key in sorted(MODEL_REGISTRY.keys()):
        s = MODEL_REGISTRY[key]
        params = f"{s.paper_params_m:.1f}" if s.paper_params_m > 0 else "?"
        flops = f"{s.paper_flops_g:.1f}" if s.paper_flops_g > 0 else "?"
        print(
            f"{s.key:<16} {s.display_name:<16} {s.family:<10} {params:<10} "
            f"{flops:<8} {s.input_size:<6} {s.weight_file}"
        )

    print(f"\n{len(MODEL_REGISTRY)} models available")


def main(argv: list[str] | None = None) -> None:
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
        "--coco-dir",
        type=str,
        required=True,
        help="Path to COCO directory (with annotations/ and images/val2017/)",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for result JSONs (default: ./results)",
    )
    run_parser.add_argument("--device", type=str, default="auto", help="Device (default: auto)")
    run_parser.add_argument(
        "--format",
        choices=["pytorch", "onnx", "tensorrt"],
        default="pytorch",
        help="Backend format (default: pytorch)",
    )
    run_parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="Directory with user-supplied .onnx / .engine weights "
        "(required with --format onnx or --format tensorrt)",
    )
    run_parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold recorded in the submission (default: 0.001)",
    )
    run_parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold recorded in the submission (default: 0.6)",
    )
    run_parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum detections per image recorded in the submission (default: 300)",
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N val2017 images (dev/CPU subset; "
        "NOT a valid full-val2017 submission). Default: all images.",
    )
    run_parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="TensorRT engine precision label recorded in the submission (default: fp16)",
    )
    run_parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="HF dataset label recorded in repro.dataset.hf_dataset "
        "(default: LibreYOLO/coco-val2017-mini500). The verifiable identity "
        "is the computed image_id_sha256, not this label.",
    )
    run_parser.add_argument(
        "--dataset-revision",
        type=str,
        default=None,
        help="HF dataset revision recorded in repro.dataset.hf_revision (default: main)",
    )
    run_parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    run_parser.add_argument("--debug", action="store_true", help="Print full tracebacks on error")

    # --- train-bench ---
    tb = subparsers.add_parser(
        "train-bench",
        help="Benchmark training throughput (img/s, $/epoch) on a COCO subset",
    )
    tb.add_argument("--models", nargs="+", help="Model keys (e.g. yolov9t yolov9s)")
    tb.add_argument("--all", action="store_true", help="Benchmark all models")
    tb.add_argument(
        "--data", type=str, default="coco1000", help="Dataset yaml/name (default: coco1000)"
    )
    tb.add_argument("--device", type=str, default="auto", help="Device (default: auto)")
    tb.add_argument("--batch", type=int, default=16, help="Micro-batch (default: 16)")
    tb.add_argument(
        "--imgsz", type=int, default=None, help="Input size (default: model's native size)"
    )
    tb.add_argument(
        "--warmup-epochs",
        type=int,
        default=1,
        help="Leading epochs discarded as warmup (default: 1)",
    )
    tb.add_argument(
        "--measure-epochs", type=int, default=3, help="Steady-state epochs averaged (default: 3)"
    )
    tb.add_argument("--workers", type=int, default=8, help="Dataloader workers (default: 8)")
    tb.add_argument(
        "--amp", action="store_true", help="Use the family's AMP path (native fast precision)"
    )
    tb.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="CUDA AMP dtype when --amp is set (default: float16)",
    )
    tb.add_argument(
        "--nbs",
        type=int,
        default=None,
        help="Effective batch for gradient accumulation (default: none)",
    )
    tb.add_argument(
        "--dollars-per-hour",
        type=float,
        default=None,
        help="Rental price of this config; enables $/epoch projection",
    )
    tb.add_argument(
        "--rig-label", type=str, default=None, help="Label for the GPU+host box, e.g. home-5070ti"
    )
    tb.add_argument(
        "--provider", type=str, default="local", help="Where it ran (local, modal, runpod, ...)"
    )
    tb.add_argument(
        "--output-dir",
        type=str,
        default="./results_train",
        help="Output dir for result JSONs (default: ./results_train)",
    )
    tb.add_argument("--quiet", action="store_true", help="Suppress progress output")
    tb.add_argument("--debug", action="store_true", help="Print full tracebacks on error")

    # --- rf100vl ---
    rf = subparsers.add_parser(
        "rf100vl",
        help="Evaluate models across the RF100-VL datasets (fine-tuned protocol)",
    )
    rf.add_argument("--models", nargs="+", help="Model keys (e.g. yolov9t)")
    rf.add_argument("--all", action="store_true", help="Evaluate all models")
    rf.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory of RF100-VL datasets (one COCO-format sub-folder each)",
    )
    rf.add_argument(
        "--download",
        action="store_true",
        help="Download the datasets into --data-dir first (pip install rf100vl + ROBOFLOW_API_KEY)",
    )
    rf.add_argument(
        "--subset",
        choices=["rf100vl", "rf20vl", "rf100vl-fsod", "rf20vl-fsod"],
        default="rf100vl",
        help="Which RF100-VL release to download (default: rf100vl)",
    )
    rf.add_argument(
        "--format",
        choices=["pytorch", "onnx"],
        default="pytorch",
        help="Backend format (default: pytorch)",
    )
    rf.add_argument(
        "--weights-root",
        type=str,
        default=None,
        help="Root of per-dataset fine-tuned checkpoints: "
        "<root>/<dataset>/<weight file>. Datasets without one are skipped.",
    )
    rf.add_argument(
        "--versions",
        type=str,
        default=None,
        help="Explicit versions.json lock (default: <data-dir>/versions.json)",
    )
    rf.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Training recipe JSON to hash into the submission. By default, "
        "the evaluator reads recipe hashes from per-dataset stats.json files.",
    )
    rf.add_argument(
        "--per-dataset-dir",
        type=str,
        default=None,
        help="Atomic per-dataset result directory used for evaluation resume "
        "(default: <data-dir>/.va-bench/eval/<model>/<format>/<split>)",
    )
    rf.add_argument(
        "--allow-pretrained",
        action="store_true",
        help="Force COCO-pretrained registry weights when no --weights-root "
        "(smoke tests / open-vocab only; NOT submittable)",
    )
    rf.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to score (default: test, per RF100-VL protocol)",
    )
    rf.add_argument("--device", type=str, default="auto", help="Device (default: auto)")
    rf.add_argument(
        "--conf", type=float, default=0.001, help="Confidence threshold (default: 0.001)"
    )
    rf.add_argument(
        "--iou", type=float, default=0.65, help="IoU threshold for NMS (protocol default: 0.65)"
    )
    rf.add_argument(
        "--max-det", type=int, default=500, help="Max detections per image (protocol default: 500)"
    )
    rf.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max images per dataset (smoke run; NOT submittable)",
    )
    rf.add_argument(
        "--limit-datasets",
        type=int,
        default=None,
        help="Evaluate only the first N datasets (smoke run; NOT submittable)",
    )
    rf.add_argument(
        "--output-dir",
        type=str,
        default="./results_rf100vl",
        help="Output dir for result JSONs (default: ./results_rf100vl)",
    )
    rf.add_argument("--quiet", action="store_true", help="Suppress progress output")
    rf.add_argument("--debug", action="store_true", help="Print full tracebacks on error")

    # --- rf100vl-train ---
    rft = subparsers.add_parser(
        "rf100vl-train",
        help="Train one fine-tuned checkpoint per RF100-VL dataset",
    )
    rft.add_argument("--model", required=True, help="One model registry key")
    rft.add_argument(
        "--data-dir",
        required=True,
        help="Version-locked RF100-VL root containing versions.json",
    )
    rft.add_argument(
        "--weights-root",
        required=True,
        help="Output root: <root>/<dataset>/<model weight file>",
    )
    rft.add_argument(
        "--recipe",
        default=None,
        help="Versioned family recipe JSON (default: packaged recipe)",
    )
    rft.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated physical GPU ids; one child process per GPU",
    )
    rft.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset names. Scheduling is always by name, never index.",
    )
    rft.add_argument(
        "--limit-datasets",
        type=int,
        default=None,
        help="Only the first N selected names (smoke scheduling)",
    )
    rft.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based deterministic box shard (default: 0)",
    )
    rft.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of deterministic box shards (default: 1)",
    )
    rft.add_argument(
        "--timeout-hours",
        type=float,
        default=6.0,
        help="Per-dataset wall-clock timeout (default: 6 hours)",
    )
    rft.add_argument(
        "--runs-root",
        default=None,
        help="LibreYOLO run root (default: <weights-root>/.runs/<model>)",
    )
    rft.add_argument(
        "--state-root",
        default=None,
        help="Atomic status/log root (default: <weights-root>/.state/<model>)",
    )
    rft.add_argument(
        "--smoke-epochs",
        type=int,
        default=None,
        help="Override epochs for local plumbing checks; marks stats non-protocol",
    )
    rft.add_argument(
        "--force",
        action="store_true",
        help="Re-enter datasets already marked done (resume safety still applies)",
    )

    # --- list ---
    subparsers.add_parser("list", help="List available models and specs")

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(_protect_leading_dash_dataset_names(raw_argv))
    if hasattr(args, "datasets"):
        args.datasets = _restore_leading_dash_dataset_names(args.datasets)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "train-bench":
        cmd_train_bench(args)
    elif args.command == "rf100vl":
        cmd_rf100vl(args)
    elif args.command == "rf100vl-train":
        cmd_rf100vl_train(args)
    elif args.command == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()

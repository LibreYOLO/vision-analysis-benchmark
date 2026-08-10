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
                datasets=args.datasets,
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

    state_root = args.state_root or str(
        Path(args.weights_root) / ".state" / args.model
    )
    syncer, sync_run_id = _make_syncer(args, state_root)
    summary = orchestrate_training(
        on_dataset_complete=(syncer.notify if syncer else None),
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
        jobs_per_gpu=args.jobs_per_gpu,
        runs_root=args.runs_root,
        state_root=args.state_root,
        smoke_epochs=args.smoke_epochs,
        force=args.force,
        keep_cache=args.keep_cache,
    )
    _stop_syncer(syncer)
    print(
        "RF100-VL training complete: "
        f"{len(summary['completed'])} completed, "
        f"{len(summary['skipped_done'])} already done, "
        f"{len(summary['failed'])} failed, "
        f"{len(summary.get('interrupted', []))} interrupted, "
        f"{len(summary.get('active_running', []))} still running"
    )
    print(f"Summary: {Path(summary['rerun_file']).parent / 'summary.json'}")
    if summary["failed"]:
        print(f"Rerun list: {summary['rerun_file']}")
    if summary.get("interrupted"):
        print(
            "Interrupted datasets (re-run the same command to resume): "
            + ", ".join(summary["interrupted"])
        )
    if summary.get("active_running"):
        print("Active datasets: " + ", ".join(summary["active_running"]))
    if summary["failed"] or summary.get("interrupted") or summary.get("active_running"):
        raise SystemExit(1)


def cmd_rf100vl_preflight(args: argparse.Namespace) -> None:
    """Validate every campaign precondition; exit non-zero on any failure."""
    from .rf100vl_preflight import has_failure, render, run_preflight

    checks = run_preflight(
        model_key=args.model,
        data_dir=Path(args.data_dir),
        weights_root=Path(args.weights_root),
        recipe=args.recipe,
    )
    print(render(checks))
    if has_failure(checks):
        raise SystemExit(1)


def cmd_rf100vl_report(args: argparse.Namespace) -> None:
    """Render submission JSONs as markdown for humans."""
    from .rf100vl_report import build_leaderboard, build_report, load_submissions

    submissions = load_submissions(Path(args.submission))
    if not submissions:
        print(f"No RF100-VL submissions found at {args.submission}")
        raise SystemExit(1)
    weights_root = Path(args.weights_root) if args.weights_root else None
    if len(submissions) == 1 and not args.leaderboard:
        text = build_report(submissions[0], weights_root=weights_root)
    else:
        roots = None
        if weights_root is not None:
            roots = {
                str(s.get("model", {}).get("id", "?")): weights_root
                for s in submissions
            }
        text = build_leaderboard(submissions, weights_roots=roots)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    print(text)


def cmd_rf100vl_gpu_report(args: argparse.Namespace) -> None:
    """Post-mortem: where the GPU-hours went, worst waste first."""
    from .gpu_trace import (
        render_efficiency_report,
        run_dirs_from_state,
        write_dataset_traces,
    )

    state_root = args.state_root or str(
        Path(args.weights_root) / ".state" / args.model
    )
    traces = write_dataset_traces(
        Path(state_root) / "gpu",
        run_dirs_from_state(state_root),
        dollars_per_hour=args.dollars_per_hour,
    )
    report = render_efficiency_report(traces)
    print(report)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Wrote {args.output}")


def _stop_syncer(syncer) -> None:
    """Drain the final upload and report what the syncer managed."""
    if syncer is None:
        return
    print("auto-sync: draining final upload...", flush=True)
    syncer.stop()
    message = (
        f"auto-sync: {syncer.syncs} syncs, {syncer.uploaded} files uploaded, "
        f"{syncer.failures} failures"
    )
    if syncer.last_error:
        message += f" (last error: {syncer.last_error})"
    print(message, flush=True)


def _make_syncer(args: argparse.Namespace, state_root: str):
    """Build the background artifact syncer, or None when it is not wanted.

    Shared by both training verbs so neither can drift into uploading with
    different semantics than the other.
    """
    repo = getattr(args, "sync_repo", None)
    if not repo:
        return None, None

    from huggingface_hub import get_token

    from .artifacts import (
        BackgroundSyncer,
        build_manifest,
        collect_artifacts,
        default_run_id,
        write_manifest,
    )

    if get_token() is None:
        raise SystemExit(
            "--sync-repo needs Hugging Face credentials. Set $HF_TOKEN or run "
            "`hf auth login`. Refusing to train for hours and only then "
            "discover the upload cannot work."
        )

    recipe_path = getattr(args, "recipe", None)
    if recipe_path is None:
        try:
            from .models import get_spec
            from .rf100vl_train import recipe_path_for_family

            recipe_path = recipe_path_for_family(get_spec(args.model).family)
        except Exception:
            recipe_path = None

    seed = build_manifest(
        model_key=args.model,
        run_id="PENDING",
        state_dir=Path(state_root),
        data_dir=args.data_dir,
        recipe_path=recipe_path,
    )
    run_id = getattr(args, "sync_run_id", None) or default_run_id(seed)

    def collect_now() -> list:
        manifest = build_manifest(
            model_key=args.model,
            run_id=run_id,
            state_dir=Path(state_root),
            data_dir=args.data_dir,
            recipe_path=recipe_path,
        )
        write_manifest(Path(state_root), manifest)
        return collect_artifacts(
            model_key=args.model,
            run_id=run_id,
            weights_root=args.weights_root,
            data_dir=args.data_dir,
            recipe_path=recipe_path,
            tier=getattr(args, "sync_tier", "results"),
        )

    syncer = BackgroundSyncer(
        collect=collect_now,
        repo=repo,
        min_interval_seconds=getattr(args, "sync_interval_seconds", 60.0),
        log=lambda message: print(message, flush=True),
    )
    syncer.start()
    print(f"auto-sync -> {repo} run-id {run_id}")
    return syncer, run_id


def cmd_rf100vl_campaign(args: argparse.Namespace) -> None:
    """Preflight, train, evaluate, and report with one set of arguments."""
    from .output import save_result
    from .provenance import harness_git_info, reconstruct_command
    from .rf100vl import benchmark_model_rf100vl
    from .rf100vl_preflight import has_failure, render, run_preflight
    from .rf100vl_report import build_report
    from .artifacts import (
        BackgroundSyncer,
        build_manifest,
        collect_artifacts,
        default_run_id,
        write_manifest,
    )
    from .gpu_trace import GpuSampler, render_efficiency_report, run_dirs_from_state, write_dataset_traces
    from .rf100vl_train import orchestrate_training

    if not args.skip_preflight:
        checks = run_preflight(
            model_key=args.model,
            data_dir=Path(args.data_dir),
            weights_root=Path(args.weights_root),
            recipe=args.recipe,
        )
        print(render(checks))
        if has_failure(checks):
            raise SystemExit(1)
        print()

    state_root = args.state_root or str(
        Path(args.weights_root) / ".state" / args.model
    )
    print(
        f"Monitor with: va-bench rf100vl-dash --state-root {state_root} "
        f"--data-dir {args.data_dir}"
    )
    # Profile hint next to the monitor line on purpose. An hour of py-spy /
    # ps / log forensics once produced a confidently wrong answer that
    # `libreyolo profile run` corrected in 52 seconds; the campaign should
    # surface the right tool before anyone invents a worse one.
    print(
        "Profile a slow dataset with: libreyolo profile phases "
        "(or `libreyolo profile run`) on one representative data yaml\n"
    )

    # Read-only NVML sampling alongside the campaign. It can only add an
    # artifact, never take a paid run down: start() returns False and the
    # campaign proceeds untelemetered if NVML is missing.
    sampler = None
    if not args.no_gpu_trace:
        sampler = GpuSampler(Path(state_root) / "gpu", Path(state_root))
        if not sampler.start():
            print(f"GPU telemetry off: {sampler.error}")
            sampler = None

    syncer, sync_run_id = _make_syncer(args, state_root)

    summary = orchestrate_training(
        on_dataset_complete=(syncer.notify if syncer else None),
        model_key=args.model,
        data_dir=args.data_dir,
        weights_root=args.weights_root,
        recipe_path=args.recipe,
        gpus=[part.strip() for part in args.gpus.split(",") if part.strip()],
        datasets=args.datasets,
        limit_datasets=args.limit_datasets,
        shard_index=0,
        num_shards=1,
        timeout_hours=args.timeout_hours,
        jobs_per_gpu=args.jobs_per_gpu,
        runs_root=args.runs_root,
        state_root=args.state_root,
        smoke_epochs=args.smoke_epochs,
        force=args.force,
    )
    # Stop the sampler and write its per-dataset traces BEFORE the final sync,
    # or the telemetry lands on disk after the last upload and never leaves the
    # box. Ordering here is the whole difference between shipping the traces
    # and silently dropping them.
    if sampler is not None:
        sampler.stop()
        try:
            traces = write_dataset_traces(
                Path(state_root) / "gpu",
                run_dirs_from_state(state_root),
                dollars_per_hour=args.dollars_per_hour,
            )
            if traces:
                report_path = Path(state_root) / "gpu_efficiency.md"
                report_path.write_text(
                    render_efficiency_report(traces), encoding="utf-8"
                )
                print(f"GPU telemetry: {len(traces)} datasets -> {report_path}")
        except Exception as exc:  # never fail a finished campaign on telemetry
            print(f"GPU telemetry post-processing failed: {exc}")

    _stop_syncer(syncer)

    print(
        f"Training: {len(summary['completed'])} completed, "
        f"{len(summary['skipped_done'])} already done, "
        f"{len(summary['failed'])} failed, "
        f"{len(summary.get('interrupted', []))} interrupted"
    )
    if summary.get("interrupted"):
        print(
            "Not evaluating: the run was interrupted. Re-run the same command "
            "to resume; finished datasets are skipped and interrupted ones "
            "continue from their last epoch checkpoint."
        )
        raise SystemExit(1)
    if summary["failed"] or summary.get("active_running"):
        print(
            "Not evaluating: resolve the failures (rerun list: "
            f"{summary['rerun_file']}) and run the same command again; "
            "completed datasets resume instantly."
        )
        raise SystemExit(1)

    harness = harness_git_info()
    result = benchmark_model_rf100vl(
        model_key=args.model,
        data_dir=args.data_dir,
        weights_root=args.weights_root,
        datasets=args.datasets,
        limit_datasets=args.limit_datasets,
        recipe_path=args.recipe,
        verbose=not args.quiet,
    )
    result.setdefault("repro", {}).update(
        {
            "harness_commit": harness["commit"],
            "harness_dirty": harness["dirty"],
            "argv": sys.argv[1:],
            "command": reconstruct_command(sys.argv[1:]),
        }
    )
    filepath = save_result(result, args.output_dir)
    report = build_report(result, weights_root=Path(args.weights_root))
    report_path = Path(filepath).with_suffix(".md")
    report_path.write_text(report, encoding="utf-8")
    print(f"\nSubmission: {filepath}\nReport:     {report_path}\n")
    print(report)


def cmd_rf100vl_dash(args: argparse.Namespace) -> None:
    """Serve the live campaign dashboard."""
    from .rf100vl_dash import serve

    serve(
        state_root=Path(args.state_root),
        host=args.host,
        port=args.port,
        open_browser=args.open,
        data_dir=Path(args.data_dir) if getattr(args, "data_dir", "") else None,
    )


def cmd_sync_artifacts(args: argparse.Namespace) -> None:
    """Upload campaign artifacts. Run after EACH dataset, not just at the end."""
    import os
    from pathlib import Path

    from .artifacts import (
        build_manifest,
        default_run_id,
        collect_artifacts,
        upload_artifacts,
        write_manifest,
    )

    # Write the provenance manifest BEFORE collecting, so it uploads with the
    # rest rather than as an afterthought a reader has to go hunting for.
    state_dir = Path(args.weights_root) / ".state" / args.model
    # Resolve the PACKAGED recipe when none was passed. Omitting --recipe is the
    # normal case, and a manifest that then says nothing about the recipe is
    # exactly the manifest you did not want.
    recipe_path = args.recipe or None
    if recipe_path is None:
        try:
            from .models import get_spec
            from .rf100vl_train import recipe_path_for_family

            recipe_path = recipe_path_for_family(get_spec(args.model).family)
        except Exception:
            recipe_path = None
    run_id = args.run_id or "PENDING"
    manifest = build_manifest(
        model_key=args.model,
        run_id=run_id,
        state_dir=state_dir,
        data_dir=args.data_dir or None,
        recipe_path=recipe_path,
    )
    if not args.run_id:
        run_id = default_run_id(manifest)
        manifest["run_id"] = run_id
        print(f"run-id (derived): {run_id}")
    args.run_id = run_id
    if state_dir.is_dir():
        write_manifest(state_dir, manifest)
        commits = {
            entry["package"]: entry.get("commit", "?")[:12]
            for entry in manifest["packages"]
        }
        print(f"manifest: {commits}")

    # Pass the RESOLVED recipe_path, not args.recipe: the fallback above was
    # being computed for the manifest and then discarded here, so a sync
    # without --recipe wrote a manifest naming a recipe and uploaded no recipe.
    items = collect_artifacts(
        model_key=args.model,
        run_id=args.run_id,
        weights_root=args.weights_root,
        eval_dir=args.eval_dir or None,
        submissions_dir=args.submissions or None,
        data_dir=args.data_dir or None,
        recipe_path=recipe_path,
        tier=args.tier,
        report=lambda message: print(message, flush=True),
    )
    total = sum(path.stat().st_size for path, _ in items)
    print(f"{len(items)} files, {total / 1e6:.1f} MB, tier={args.tier}")
    if args.dry_run:
        for path, repo_path in items:
            print(f"  {repo_path}  ({path.stat().st_size / 1e3:.0f} KB)")
        return
    if not items:
        print("nothing to sync")
        return

    # Credential resolution is huggingface_hub's job: an explicit token, then
    # $HF_TOKEN, then the file written by `hf auth login` (under HF_HOME, not
    # ~/.config). Re-implementing it only produces wrong answers.
    from huggingface_hub import get_token

    if get_token() is None:
        raise SystemExit(
            "no Hugging Face token found. Set $HF_TOKEN, or run `hf auth login`. "
            "For a campaign, prefer a fine-grained token scoped to "
            f"{args.repo!r} with write access only."
        )

    # Refuse to write into someone else's run id. Silently landing on an
    # existing run's paths is worse than an error: same-size files are skipped,
    # so you would read the OLD campaign's numbers under the new run's name.
    if not args.append:
        try:
            from huggingface_hub import HfApi

            prefix = f"{args.model}/{run_id}/"
            existing = [
                name
                for name in HfApi().list_repo_files(args.repo, repo_type="dataset")
                if name.startswith(prefix)
            ]
        except Exception:
            existing = []
        if existing:
            raise SystemExit(
                f"run id {run_id!r} already has {len(existing)} files in "
                f"{args.repo}. Pass --append to add to it deliberately, or use a "
                "different --run-id. Reusing one silently mixes two campaigns."
            )

    result = upload_artifacts(
        items, repo=args.repo, token=None, private=args.private,
        progress=lambda line: print(f"  {line}", flush=True),
    )
    print(f"\nuploaded {result['uploaded']}, skipped {result['skipped']} already present")
    print(f"https://huggingface.co/datasets/{args.repo}/tree/main/{args.model}/{args.run_id}")


def cmd_rescore(args: argparse.Namespace) -> None:
    """Recompute metrics from saved detections, with no GPU and no model."""
    import json as _json
    from pathlib import Path

    from .artifacts import rescore_from_predictions

    report = rescore_from_predictions(
        eval_root=args.eval_dir,
        data_dir=args.data_dir,
        split=args.split,
        max_det=args.max_det,
        fingerprint_prefix=args.fingerprint,
        verify=not args.no_verify,
        progress=lambda line: print(f"  {line}", flush=True),
    )
    Path(args.output).write_text(_json.dumps(report, indent=2), encoding="utf-8")

    print(f"\ndatasets scored : {report['num_datasets']}")
    print(f"mean AP50:95    : {report['mean_mAP_50_95']:.4f}")
    print(f"mean AP50       : {report['mean_mAP_50']:.4f}")
    if not report["is_full_benchmark"]:
        print(f"NOTE: {report['num_datasets']} datasets, not 100. This is a SUBSET and "
              f"is NOT a protocol-conformant RF100-VL result.")
    if report["mismatches"]:
        print(f"\nMISMATCH against recorded metrics: {len(report['mismatches'])}")
        for row in report["mismatches"][:10]:
            print(f"  {row['dataset']}: recorded={row['recorded']:.6f} "
                  f"rescored={row['rescored']:.6f}")
        raise SystemExit(1)
    print("\nmatches the metrics recorded at evaluation time")


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
        "--datasets",
        nargs="+",
        default=None,
        help="Evaluate only these datasets by name (partial run; NOT submittable). "
        "For the dataset literally named -grccs write --datasets=-grccs.",
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
        help="Comma-separated physical GPU ids",
    )
    rft.add_argument("--sync-repo", default=None, help="Auto-upload artifacts here as datasets finish")
    rft.add_argument("--sync-run-id", default=None)
    rft.add_argument("--sync-tier", default="results", choices=("results", "checkpoints", "all"))
    rft.add_argument("--sync-interval-seconds", type=float, default=60.0)
    rft.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=1,
        help="Concurrent trainings per GPU. Each is an ordinary independent "
        "run at the recipe's batch, so results are unchanged; this only "
        "fills a card that one small model cannot.",
    )
    rft.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset names. Scheduling is always by name, never index, "
        "and runs in alphabetical order regardless of the order given here.",
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

    # --- rf100vl-gpu-report ---
    rg = subparsers.add_parser(
        "rf100vl-gpu-report",
        help="Post-mortem GPU efficiency per dataset from captured telemetry",
    )
    rg.add_argument("--model", required=True)
    rg.add_argument("--weights-root", required=True)
    rg.add_argument("--state-root", default=None)
    rg.add_argument("--dollars-per-hour", type=float, default=None)
    rg.add_argument("--output", default=None, help="Also write the markdown here")

    # --- rf100vl-preflight ---
    rp = subparsers.add_parser(
        "rf100vl-preflight",
        help="Validate a box before a campaign: libreyolo, data, recipe, GPU, disk",
    )
    rp.add_argument("--model", required=True, help="One model registry key")
    rp.add_argument("--data-dir", required=True, help="Version-locked RF100-VL root")
    rp.add_argument("--weights-root", required=True, help="Campaign output root")
    rp.add_argument("--recipe", default=None, help="Recipe JSON (default: packaged)")

    # --- rf100vl-report ---
    rr = subparsers.add_parser(
        "rf100vl-report",
        help="Render submission JSONs as markdown (report or leaderboard)",
    )
    rr.add_argument(
        "--submission",
        required=True,
        help="One submission JSON, or a directory of them (renders a leaderboard)",
    )
    rr.add_argument(
        "--weights-root",
        default=None,
        help="Campaign weights root; adds train cost from per-dataset stats.json",
    )
    rr.add_argument(
        "--leaderboard",
        action="store_true",
        help="Force the leaderboard table even for a single submission",
    )
    rr.add_argument("--output", default=None, help="Also write the markdown here")

    # --- rf100vl-campaign ---
    rc = subparsers.add_parser(
        "rf100vl-campaign",
        help="One command: preflight, train, evaluate, report (protocol defaults)",
    )
    rc.add_argument("--model", required=True, help="One model registry key")
    rc.add_argument("--data-dir", required=True, help="Version-locked RF100-VL root")
    rc.add_argument("--weights-root", required=True, help="Campaign output root")
    rc.add_argument("--gpus", default="0", help='GPU ids, e.g. "0,1,2,3" (default: 0)')
    rc.add_argument("--recipe", default=None, help="Recipe JSON (default: packaged)")
    rc.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset subset (result is then NOT submittable)",
    )
    rc.add_argument("--limit-datasets", type=int, default=None)
    rc.add_argument("--timeout-hours", type=float, default=6.0)
    rc.add_argument("--runs-root", default=None)
    rc.add_argument("--state-root", default=None)
    rc.add_argument("--smoke-epochs", type=int, default=None)
    rc.add_argument("--force", action="store_true")
    rc.add_argument("--output-dir", default="./results_rf100vl")
    rc.add_argument("--skip-preflight", action="store_true")
    rc.add_argument(
        "--sync-repo",
        default=None,
        help="Hugging Face dataset repo to auto-upload artifacts to as each "
        "dataset finishes. Needs $HF_TOKEN. Off when unset.",
    )
    rc.add_argument("--sync-run-id", default=None, help="Defaults to a derived id")
    rc.add_argument("--sync-tier", default="results", choices=("results", "checkpoints", "all"))
    rc.add_argument(
        "--sync-interval-seconds",
        type=float,
        default=60.0,
        help="Do not sync more often than this, so a burst of completions "
        "produces one upload rather than many",
    )
    rc.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=1,
        help="Concurrent trainings per GPU. Each is an ordinary independent "
        "run at the recipe's batch, so results are unchanged; this only "
        "fills a card that one small model cannot.",
    )
    rc.add_argument(
        "--no-gpu-trace",
        action="store_true",
        help="Disable GPU telemetry capture (on by default; ~4.4 MB per campaign)",
    )
    rc.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep each dataset's post-resize .npy cache and its last.pt once "
        "the dataset finishes. Off by default: across 100 datasets those are "
        "the two largest consumers on a campaign box and neither is read again "
        "after a dataset is done.",
    )
    rc.add_argument(
        "--dollars-per-hour",
        type=float,
        default=None,
        help="Box price, to attribute spend per dataset in the efficiency report",
    )
    rc.add_argument("--quiet", action="store_true")

    # --- rf100vl-dash ---
    rd = subparsers.add_parser(
        "rf100vl-dash",
        help="Live web dashboard for an RF100-VL training campaign (read-only)",
    )
    rd.add_argument(
        "--state-root",
        required=True,
        help="Campaign state dir: <weights-root>/.state (all models) or "
        "<weights-root>/.state/<model> (one model)",
    )
    rd.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default 127.0.0.1; on a rented box keep the default "
        "and use an SSH tunnel: ssh -L 8877:127.0.0.1:8877 <box>)",
    )
    rd.add_argument("--port", type=int, default=8877, help="Port (default: 8877)")
    rd.add_argument("--open", action="store_true", help="Open the browser")
    rd.add_argument(
        "--data-dir",
        default="",
        help="RF100-VL root. Optional, but without it the dashboard only knows "
        "the size of datasets it has already launched, so queued datasets show "
        "no image count and the ETA is size-blind for most of a campaign",
    )

    # --- sync-artifacts ---
    sa = subparsers.add_parser(
        "sync-artifacts",
        help="Upload campaign artifacts to a HuggingFace dataset repo (run after each dataset)",
    )
    sa.add_argument("--model", required=True, help="One model registry key")
    sa.add_argument(
        "--run-id",
        default=None,
        help="Campaign run id. Omit to derive one from the date plus the code "
        "and recipe identity, which cannot silently collide with another run.",
    )
    sa.add_argument(
        "--append",
        action="store_true",
        help="Allow writing into a run id that already exists in the repo",
    )
    sa.add_argument("--weights-root", required=True)
    sa.add_argument("--eval-dir", default="", help="Per-dataset eval result dir")
    sa.add_argument("--submissions", default="", help="Directory of submission JSONs")
    sa.add_argument("--data-dir", default="", help="Source of versions.json")
    sa.add_argument("--recipe", default="", help="Recipe JSON to preserve")
    sa.add_argument("--repo", default="LibreYOLO/rf100-vl-results")
    sa.add_argument("--tier", choices=("results", "checkpoints", "all"), default="results",
                    help="results (~239MB/model), checkpoints (+2.5GB), all (+30GB)")
    sa.add_argument("--private", action="store_true")
    sa.add_argument("--dry-run", action="store_true")

    # --- rescore ---
    rs = subparsers.add_parser(
        "rescore",
        help="Re-score from saved predictions with no GPU, model, or rented box",
    )
    rs.add_argument("--eval-dir", required=True, help="Dir of per-dataset prediction dumps")
    rs.add_argument("--data-dir", required=True, help="RF100-VL root, for ground truth")
    rs.add_argument("--split", default="test")
    rs.add_argument("--max-det", type=int, default=500)
    rs.add_argument("--fingerprint", default="",
                    help="Fingerprint prefix, when a dataset has several dumps")
    rs.add_argument("--output", default="rescore.json")
    rs.add_argument("--no-verify", action="store_true",
                    help="Skip comparing against the metrics recorded at evaluation time")

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
    elif args.command == "rf100vl-dash":
        cmd_rf100vl_dash(args)
    elif args.command == "rf100vl-gpu-report":
        cmd_rf100vl_gpu_report(args)
    elif args.command == "rf100vl-preflight":
        cmd_rf100vl_preflight(args)
    elif args.command == "rf100vl-report":
        cmd_rf100vl_report(args)
    elif args.command == "rf100vl-campaign":
        cmd_rf100vl_campaign(args)
    elif args.command == "sync-artifacts":
        cmd_sync_artifacts(args)
    elif args.command == "rescore":
        cmd_rescore(args)
    elif args.command == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()

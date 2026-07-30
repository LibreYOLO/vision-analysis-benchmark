# Output Checklist

Review the emitted JSON before treating the run as publishable.

## Required checks

- `schema_version == "va.submission.v1"`
- `submission_id` is present
- `created_at` is present
- `model.id` matches the requested registry key
- `config.input_size` matches the actual runtime input size
- `runtime.format` matches the requested backend
- `runtime.provider` and `runtime.device` reflect the real execution path
- `hardware.id` is present
- `benchmark.libreyolo_commit` is present, or `unknown` if unavailable

## Reproducibility (`repro`) checks

- `repro.harness_commit` is a real commit, not `unknown`
- `repro.harness_dirty` is `false` for a publishable run (a `true` here means the
  harness working tree had uncommitted changes; re-run from a clean checkout)
- `repro.command` is the exact command you ran, and `repro.argv` matches it
- `repro.dataset.image_id_sha256` is present. For a canonical mini500 run it must
  match the known mini500 fingerprint; for a full-val2017 run it reflects all 5000 ids
- `repro.weights.sha256` is set for ONNX / TensorRT runs (the actual artifact hash);
  `null` is expected for LibreYOLO-managed `.pt` files
- For ONNX / TensorRT, `repro.weights.export_manifest` is present when a
  `<weights>.json` sidecar was supplied, and `repro.weights.onnx_opset` is populated

## Sanity checks

- No negative timing, throughput, or model-stats values
- `timing.total_ms.mean` and `throughput.fps_mean` look internally consistent
- ONNX output uses the expected backend/provider, not a silent CPU fallback
- If the run is intended for publication, the filename is the harness-generated one from `save_result()`

## Useful commands

```bash
python -m json.tool results/<file>.json | sed -n '1,220p'
pytest -q
```

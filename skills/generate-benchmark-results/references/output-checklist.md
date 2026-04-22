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

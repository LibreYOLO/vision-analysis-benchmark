# Vision Analysis Benchmark

Benchmark harness for generating submission JSONs consumed by `vision-analysis`.

## Project Structure

- `va_bench/` contains the CLI, benchmark loop, model registry, timing, hardware detection, and output assembly.
- `tests/` contains the schema and regression tests for the current harness behavior.
- `skills/` contains repo-local Claude skills for common harness tasks.

## Core Files

- `va_bench/cli.py` — `va-bench run` and `va-bench list`
- `va_bench/benchmark.py` — PyTorch and ONNX benchmark paths
- `va_bench/models.py` — supported model registry
- `va_bench/output.py` — submission JSON assembly and filename generation
- `README.md` — current support matrix and scope limits

## Rules

- This repo produces benchmark submissions; it does not own website ingestion or scoring.
- Do not claim backend or hardware support in `README.md` unless the code path exists and can be exercised honestly.
- Keep output compatible with `va.submission.v1`.

## Validation

```bash
pytest -q
```

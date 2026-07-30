# RF100-VL setup-phase findings

Date: 2026-07-30. Scope: prove the campaign can actually run before spending
credits. Companions: `RF100VL_INVESTIGATION.md` and `PLAN_rf100vl_execution.md`
in the libreyolo repo, and PR #666 (`skills/run-rf100vl-benchmark`).

Status: pipeline verified end to end on one dataset locally; compute not yet
rented; canonical dataset download in progress.

## 1. Split coverage: the assumption holds

Every one of the 100 datasets carries all three splits. Measured by reading
only the `dataset_name` column out of the HF mirror's parquet shards with
DuckDB range reads (a few MB of traffic instead of the 42 GB of images):

| split | datasets | images |
|---|---|---|
| train | 100 | 115,777 |
| valid | 100 | 33,137 |
| test  | 100 | 14,237 |

The canonical download re-proves this independently: `_dataset_materialized`
raises unless all three `_annotations.coco.json` files exist after cleaning, so
a missing split fails the download loudly rather than surfacing later as a
scoring anomaly.

Two operational facts fall out.

**Three test splits are smaller than the batch size**: `invoice-processing`
(12 images), `thermal-cheetah` (13), `circuit-voltages` (14); their valid
splits are 20/25/26. Any `drop_last=True` on an eval loader scores these on
nothing while reporting a number. This is trap #12 and it needs an assertion,
not an assumption. Smallest train split is 72 images (4 batches at 16),
median 705, max 8,791.

**`-grccs` is real** and its leading dash is confirmed in both the mirror and
the live workspace listing. Address datasets by name with `--datasets=-grccs`.

**Reconciled** (2026-07-30, by counting the full canonical download): the
benchmark's released splits hold **163,151 images and 1,353,434 annotations**,
which is exactly what the HF mirror also holds. The paper's 164,149 / 1,355,491
and Roboflow's API agree with each other but count roughly a thousand images
that sit in the Universe projects outside any released split, so no download
contains them. Neither copy is lossy. All 100 datasets verified to carry all
three splits with a readable annotation file, and 564 classes in total.

## 2. Bug found and fixed: Windows MAX_PATH in the result cache

`va-bench rf100vl` aborted with `[Errno 2] No such file or directory` naming a
temp file inside a directory that plainly existed.

Cause: the per-dataset cache path is `<data_dir>/.va-bench/eval/<model>/<fmt>/
<split>/<dataset>/<64-hex-fingerprint>.json`, and `atomic_write_json` appends
`.<pid>.tmp`. That reached 286 characters, past Windows' 260-char `MAX_PATH`.
`mkdir` succeeded (short enough), only the file write failed, which is why the
error blamed a missing directory.

The quieter half of the bug mattered more: `Path.exists()` returns `False`
rather than raising for an over-length path, so `_load_cached_dataset_result`
would have **silently missed every cached result and re-evaluated**, defeating
resume without any error at all.

Fixed with a `long_path()` shim (`\\?\` prefix, UNC-aware) applied to the
writer, the reader, and the cache-hit check. Verified by re-running the exact
command that failed; 78/78 harness tests still pass. Linux is unaffected, but
the documented workflow is local-first on Windows.

## 3. PR #666 contradicts the harness on early stopping

The skill's protocol table states `Epoch budget | cap 100, early stopping
patience 20`. Three sources disagree:

- `va_bench/recipes/rf100vl/*.json` ship `"patience": 0`
- `rf100vl_train.py` *enforces* `patience: 0` in its `protocol_required` check,
  so a patience-20 run is stamped non-conformant
- `PLAN_rf100vl_execution.md` A3 says early stopping disabled

`trainer.py` gates early stop on `self.config.patience > 0`, so `0` means
disabled. The skill is the outlier and should be corrected before merge.

## 4. PR #666 promises artifacts the harness does not write

The skill lists "per-dataset raw predictions (COCO detections)" among kept
artifacts and pitches the reproducibility story as *"anyone can then rescore
from the JSONs with pycocotools, no GPU needed"*. `_predictions_for_split`
builds detections in memory, passes them to the evaluator, and drops them.

That claim is currently unbacked. Persisting them is the right fix, but it is a
real cost decision (~500 detections per image at conf 0.001, across 14,237 test
images per model), so it needs an owner ruling rather than a silent default.

## 5. Design: the eval cache lives inside `--data-dir`

Default cache root is `<data_dir>/.va-bench/...`, i.e. harness state written
into the dataset directory. That conflicts with the intended architecture,
where datasets become a shared, potentially read-only mount (see §7). A
`per_dataset_dir` override already exists; the fix is to default it under
`weights_root`. Left unchanged because it relocates existing caches.

## 6. What is verified working

- **maxDets 500 is genuinely closed.** `metrics/max_det: 500.0` flows through
  during-training validation; the submission records
  `eval.maxDets: [1, 10, 100, 500]`; the capability guard records
  `eval_max_det: 500` beside `default_eval_max_det: 100`, proving normal users'
  defaults are untouched. Both contract tests the plan demanded exist and pass:
  GT-as-predictions scores AP 1.0 at 500, and a synthetic dense case proves
  AP@500 strictly exceeds AP@100.
- **Full loop runs**: `rf100vl-train` → `rf100vl` → `va.submission.v1`, with
  per-dataset `stats.json`, atomic status files, run signatures, rerun list,
  and a repro block carrying recipe sha, dataset-version sha, and harness commit.
- **The submission validator does its job.** The pilot was correctly rejected
  with four reasons: non-protocol training run, training metadata mismatch,
  1 dataset instead of 100, and a dirty harness tree.
- **The upstream package index bug is real and already worked around.**
  `DatasetList.__init__` sorts `self.projects` but builds `self.datasets` from
  the unsorted constructor argument, so `[i]` is workspace-API order, not
  alphabetical, despite the docstring claiming otherwise. The harness keys by
  name throughout. Worth reporting upstream.

## 7. Decisions taken

**Docker: lean image, datasets staged separately.** `deploy/Dockerfile` on
`pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` — cu128 is mandatory because
the primary pool is RTX 5090 (sm_120), which a cu124 image cannot run at all.
libreyolo is pinned by commit, never branch, so two boxes in one campaign cannot
disagree about what they measured. `deploy/verify_image.py` fails the *build* on
a wrong ref and re-runs with `--require-gpu` on the box. Built by GitHub Actions
and pushed to GHCR, because the image is ~10 GB and a home uplink is slower than
a runner building it inside GitHub's network.

Datasets are not baked in: 40 GB would turn a ~30s cold start into a
multi-minute pull on every uncached host, for data that stages from a CDN at
~100 MB/s anyway.

**Proposed: host a rebuilt RF100-VL tarball on HF under the LibreYOLO org.**
Apache 2.0 permits it. It removes the Roboflow key from every box, and removes
the risk the investigation flagged, that Universe datasets get revised in place
mid-campaign. Must be built from the *package* download, not the HF mirror.

**The HF mirror is not a canonical substitute.** Its `category_id` is already
per-dataset 0-based contiguous and identical to `original_category_ids`, so it
matches the package's cleaning — but it carries **no class names**. Fine for
pipeline smoke tests, unusable for the zero-shot track or for publication.

## 7b. Bugs found on the rented box

Both would have been baked into the Dockerfile had it been frozen before a real
box ran the sequence, which is why the onstart-first order was worth the detour.

- **PEP 668 blocks pip in the base image.** `pytorch/pytorch:2.11.0-cuda12.8-*`
  sits on Ubuntu 24.04, whose system python is externally-managed, so
  `pip install` refuses outright. The bootstrap then died ten lines later as
  `ModuleNotFoundError: No module named 'libreyolo'`, blaming the wrong thing.
  Fixed with `PIP_BREAK_SYSTEM_PACKAGES=1` in both onstart and Dockerfile.
- **Git Bash rewrites absolute remote paths.** `--log /root/setup2.log` reached
  the box as `C:/Program Files/Git/root/setup2.log`. Needs `MSYS_NO_PATHCONV=1`,
  and because that disables conversion both ways, local script paths must then
  be given in Windows form. Recorded in the vast-launch skill.

Verified on the box: RTX 5090 reports capability (12, 0), `sm_120` is in
`torch.cuda.get_arch_list()`, a CUDA matmul runs, and `eval_max_det` is present
on a clean install of the pinned commit. Bootstrap takes 48 seconds.

## 7c. Provenance bug: `libreyolo.__version__` reports the INSTALLED version

`libreyolo/__init__.py` sets `__version__ = version("libreyolo")`, which reads
installed distribution metadata rather than the source tree. Running the
protocol worktree through `PYTHONPATH` therefore stamps `1.3.0.dev0` into every
submission while executing `1.4.0` code; the box, with a clean install of the
same commit, correctly reports `1.4.0`.

`libreyolo_commit` is derived from git and stays correct, so provenance is
recoverable, but the version string is untrustworthy under any editable-install
or PYTHONPATH shadowing. Campaign runs must use a clean install, not a path
override.

## 7d. License inventory: all 100 datasets are MIT

Full per-dataset inventory in `RF100VL_LICENSE_INVENTORY.md`, machine-readable
at `va_bench/data/rf100vl_licenses.json`, regenerable via
`deploy/license_inventory.py`. 100/100 MIT, 100/100 public, zero blocked. The
API license field was cross-validated against the `License:` line inside the
downloaded exports with 0 mismatches.

See §1 for the corrected image-count reconciliation: the mirror is NOT lossy,
and an earlier claim here that it was has been retracted.

Measured download rates (both ours, both real): **2.2 MB/s / ~203 s per dataset
/ ~5.6 h** on a home connection, versus **16.3 MB/s / ~27-39 s per dataset /
~45-65 min** on the rented box. Most of that time is server-side zip generation,
not transfer, so a faster local link does not fix it. The redistribution plan
(`deploy/hf_dataset/`) is what removes it, along with the API key requirement
and the unpinned-version risk.

## 8. Outside this repo

- **Vast auth**: the CLI writes its 2FA session key to
  `~/.config/vastai/vast_tfa_key` and then prefers it over the API key for
  every request. On expiry the server returns 404 "Session expired", but the
  CLI's own recovery only fires on 401 "Invalid user key", so the stale file
  poisons every command *including `tfa login` itself* and a healthy account
  looks dead. Fixed in the `vast-launch` skill: stale-session detection and
  retry, plus optional TOTP-seed self-refresh.
- **libreyolo main worktree**: `libreyolo/validation/detection_validator.py`
  contains a literal `…1908 tokens truncated…` marker where ~130 lines were,
  and imports a `validation/contracts.py` that does not exist, so
  `import libreyolo` fails from the shared venv. Unrelated to RF100-VL; all
  work here runs against the clean `rf100vl-protocol` worktree, which is also
  the build the protocol requires.

# RF100-VL setup session: decisions, changes, numbers

Date: 2026-07-30. Signal only. Everything here was measured or executed, not estimated,
unless explicitly marked as a projection.

---

## 1. What was shipped

**Dataset published: https://huggingface.co/datasets/LibreYOLO/rf100-vl**
Public, 100 per-dataset `.tar`, 43.0 GB, plus `README.md`, `NOTICE`, `licenses.json`,
`versions.json`. Byte-identical to what the official `rf100vl` package produces.

---

## 2. Changes to LibreYOLO

**None.** No LibreYOLO source was modified this session.

All work ran against the pre-existing `rf100vl-protocol` branch (commit
`19b9321ab11450f9d9569ca059e2346267a51aca`), which already carries `eval_max_det` and
`amp_dtype`. Two LibreYOLO defects were *found* and are recorded in §7.

---

## 3. Changes to the harness (`vision-analysis-benchmark`, branch `rf100vl-harness`)

### Code fixes

| file | change | why |
|---|---|---|
| `va_bench/rf100vl_data.py` | added `long_path()`; writer and reader use it | Windows `MAX_PATH` bug, §7.1 |
| `va_bench/rf100vl.py` | cache-hit check uses `os.path.exists(long_path(...))` | `Path.exists()` returns False past MAX_PATH, silently defeating resume |
| `va_bench/rf100vl.py` | `save_predictions=True` param, `_write_predictions()`, `PREDICTIONS_SCHEMA` | raw COCO detections were built then discarded |
| `va_bench/rf100vl.py` | predictions stored as 4 fields only, coords rounded 2dp, **scores full precision** | 81% of the file was pycocotools-injected padding, §7.2 |
| `va_bench/rf100vl.py` | `predictions_file` recorded in each per-dataset record | so the submission points at its own evidence |

Tests: 78/78 still pass after every change.

### New files

```
deploy/Dockerfile                     campaign image, pinned by commit
deploy/freeze_torch.py                pins the base image's torch before any pip install
deploy/verify_image.py                fails the BUILD on a wrong ref; --require-gpu on the box
deploy/vast/onstart.sh                bootstrap proven on a real box
deploy/license_inventory.py           per-dataset license audit
deploy/hf_dataset/README.md           the dataset card
deploy/hf_dataset/build_notice.py     generates NOTICE from the license inventory
deploy/hf_dataset/publish.py          tar -> upload -> delete, resumable, refuses partial
deploy/hf_dataset/audit_dataset.py    full integrity audit of a materialised tree
deploy/hf_dataset/verify_published.py round-trip: HF -> unpack -> SHA-256 vs source
deploy/hf_dataset/sync_artifacts.py   incremental campaign-artifact upload, tiered
deploy/hf_dataset/rescore_offline.py  re-score from predictions, no GPU/model/box
va_bench/data/rf100vl_licenses.json   100 records: slug, URL, license, verdict
.github/workflows/benchmark-image.yml builds the image in CI, pushes to GHCR
RF100VL_SETUP_FINDINGS.md             the bug/finding log
RF100VL_LICENSE_INVENTORY.md          license reasoning and method
```

**Not done:** `sync_artifacts.py` and `rescore_offline.py` are standalone scripts, not
CLI verbs, and have no tests. The Dockerfile has never been built (no Docker locally).

---

## 4. Changes to skills

### `vast-launch` (personal skill)

- **Self-healing auth.** Every CLI call detects the expired-session signature, moves the
  stale key aside, retries on the API key. `tfa-login` clears it first, so it can no
  longer fail against its own stale state.
- **Auto-TOTP.** A base32 seed at `~/.config/vastai/vast_totp_seed` makes the session
  self-refreshing, no human. TOTP implementation verified against RFC 6238 vectors
  (287082, 081804). **Seed not yet saved by the owner.**
- **New `auth` command** that diagnoses and repairs; `preflight` previously misreported a
  good key as rejected because `show user` is itself 2FA-gated.
- **`--max-transfer-cost` (default $0.01/GB) and `--gb-moved`.** Bandwidth filter, §7.5.
- **New section "Picking a box: how the money actually works"**, and gotchas 7b (Git Bash
  rewrites remote paths), 7c (PEP 668), 7d (pin the image's torch).
- **New `reference/storage-and-volumes.md`**: volumes, network volumes, real pricing, and
  the units the docs omit.

### `run-rf100vl-benchmark` (PR #666) — committed `d072d119`, **NOT pushed**

- Added the HF fast path with URL, unpack loop, and `HF_HUB_ENABLE_HF_TRANSFER=1`.
- Kept the `rf100vl` package as the canonical rebuild/verify route.
- **Corrected**: "early stopping patience 20" to `patience: 0`, disabled. The recipes,
  `trainer.py`, and `rf100vl_train.py`'s protocol check all say 0; a patience-20 run
  would be stamped non-conformant by our own harness.
- **Corrected**: an HF token does NOT increase download throughput. `hf_transfer` and
  large files do. The token buys the higher authenticated rate limit.
- Pointed the rescore-without-a-GPU claim at where predictions actually land.

---

## 5. Uploading every experiment to HF

### Why

Interruptible boxes are destroyed and take container disk with them. Anything not synced
is gone, and on a 100-dataset campaign that means re-running paid training. Second
reason: **predictions are the durable scientific artifact.** With them, a future
evaluation fix costs a CPU afternoon instead of a re-run. The maxDets 500 gap was exactly
that class of bug.

### What gets uploaded, per experiment (model x run)

Tier `results` (the default):

| artifact | scope | content |
|---|---|---|
| `eval/<dataset>/<fp>.predictions.json.gz` | 100 | raw COCO detections |
| `eval/<dataset>/<fp>.json` | 100 | per-dataset metrics + the input fingerprint |
| `stats/<dataset>.json` | 100 | recipe sha, best epoch, seed, dataset version, wall time, run signature, libreyolo capabilities |
| `runs/<dataset>/primary/train_config.yaml` | 100 | the resolved training config |
| `runs/<dataset>/primary/data.yaml` | 100 | generated dataset config (nc, names, annotation paths) |
| `runs/<dataset>/primary/metrics.jsonl` | 100 | per-epoch metrics |
| `runs/<dataset>/primary/results.csv` | 100 | same, tabular |
| `runs/<dataset>/primary/train.log` | 100 | training log |
| `runs/<dataset>/primary/summary.json`, `status.json` | 100 | run outcome |
| `state/summary.json`, `rerun.json`, `failures.json` | 1 | campaign status, failure list |
| `state/logs/<dataset>.log` | 100 | orchestrator logs |
| `provenance/versions.json` | 1 | dataset version lock |
| `provenance/<family>.json` | 1 | the recipe |
| `submissions/*.json` | 1+ | `va.submission.v1`, carrying libreyolo commit, harness commit, recipe sha, dataset-versions sha |
| `manifest.json` | 1 | file count and bytes |

Tiers `checkpoints` adds `best.pt`; `all` adds `last.pt`. **Never sync `all`.**

### Size, and how it was calculated

Measured, not guessed:

1. Test split totals **14,237 images** across the 100 datasets (counted from the
   annotation files of the full canonical download).
2. Protocol is `conf 0.001`, `max_det 500`. At that threshold the cap saturates:
   **494.7 detections/image measured**. So the size is `500 x images`, independent of
   model quality. A model with catastrophic false positives produces the same size.
3. `14,237 x 500 = 7.12M detections` per model.
4. Gzipped cost per detection, measured on a real dump after the slimming fix:
   **22.3 bytes**.
5. `7.12M x 22.3 B = 159 MB` of predictions per model.
6. Non-prediction artifacts measured at ~820 KB/dataset, so ~80 MB per model.

| | per model | 7 models (flagships) | 17 models (wave 2) |
|---|---|---|---|
| **results tier** | **239 MB** | **1.6 GB** | **4.0 GB** |
| + `best.pt` | 2.5 GB | 17.5 GB | 42.5 GB |
| + full run dirs | 30 GB | 210 GB | 510 GB |

DETR families cap at 300 queries rather than 500, so their predictions are ~40% smaller.

### Recommendation

Always sync `results`: 4 GB for a full wave 2 is free on HF public storage and it is the
entire scientific record. Keep `best.pt` for flagships only. Never sync run dirs, which
are ~30 GB of mostly-duplicate weights.

Sync **after each dataset**, not at the end: an interruptible box preempted mid-campaign
then loses at most one dataset.

### The offline path

`rescore_offline.py` recomputes everything from predictions plus test ground truth
(5 MB for all 100), with no GPU, model, or box. It **verifies against what the box
recorded** and fails loudly on mismatch. Confirmed clean on the pilot.

It refuses to average multiple prediction dumps for the same dataset (a real bug found
while testing: two runs in one directory were being counted as two datasets), and stamps
any result that is not 100 datasets as a SUBSET.

---

## 6. Verified facts about RF100-VL

- **100/100 datasets carry train, valid and test** with readable
  `_annotations.coco.json`. Verified on the canonical download, 0 problems.
- **163,151 images / 1,353,434 annotations / 564 classes** in the released splits
  (115,777 / 33,137 / 14,237). The paper's 164,149 and Roboflow's API count ~1,000
  images that are in no released split, so no download contains them. The HF parquet
  mirror holds exactly 163,151 too, so **the mirror is not lossy** (an earlier claim in
  this session that it was has been retracted).
- **All 100 datasets are MIT**, all 100 projects public. Checked per project via the
  Roboflow API and cross-validated against the `License:` line inside 8 downloaded
  exports: 0 mismatches.
- **Three test splits are below the batch size**: `invoice-processing` 12,
  `thermal-cheetah` 13, `circuit-voltages` 14. `drop_last` on an eval loader would score
  these on nothing. Still needs an explicit assertion.
- **101 degenerate (zero-area) boxes across 22 datasets**, 0.0075% of annotations. They
  are upstream, in Roboflow's data; the package's cleaning only rewrites ids. **Not
  fixed deliberately**: stripping them would make our copy disagree with every published
  RF100-VL number.
- **`-grccs`** exists, leading dash and all. It broke argparse in one of our own scripts
  during this session, exactly as trap #11 predicts.
- **Download speed measured twice**: 2.2 MB/s and ~203 s/dataset from a home line
  (~5.6 h for 100); 16.3 MB/s and ~27-39 s/dataset from a datacenter (~45-65 min). Most
  of that is server-side zip generation, not transfer, so a faster local link does not
  fix it.

---

## 7. Defects found

### 7.1 Harness: Windows MAX_PATH in the result cache — FIXED

Cache path plus atomic `.tmp` suffix reached 286 chars, past the 260 limit. `mkdir`
succeeded and only the write failed, so it surfaced as "No such file or directory" for a
directory that existed. **The quieter half mattered more**: `Path.exists()` returns
`False` rather than raising past MAX_PATH, so resume would have silently re-evaluated
everything. Fixed with a `\\?\` shim on writer, reader and cache-hit check; verified by
re-running the exact failing command.

### 7.2 Harness: 81% of every prediction dump was padding — FIXED

`_predictions_for_split` builds 4 fields. `evaluate_coco` calls pycocotools `loadRes`,
which **mutates the dicts in place**, adding `id`, `iscrowd`, `area`, and a
`segmentation` polygon that re-encodes the box corners. Predictions were saved after
evaluation, so all of it persisted: 74.6 to 22.3 bytes/detection, 531 MB to 159 MB per
model. AP verified unchanged at 0.2352 by both the harness and the offline rescore.

### 7.3 LibreYOLO: `__version__` reports the INSTALLED version, not the running code

`libreyolo/__init__.py` sets `__version__ = version("libreyolo")`, i.e. distribution
metadata. Running the protocol worktree via `PYTHONPATH` stamps `1.3.0.dev0` into
submissions while executing `1.4.0` code; a clean install on the box correctly reported
`1.4.0`. `libreyolo_commit` stays accurate, so provenance is recoverable.
**Campaign runs must use a clean install, never a path override.**

### 7.4 LibreYOLO: no degenerate-box filtering in the detection data path

Nothing in `data/labels.py`, `dataset.py` or `utils.py` handles zero-area boxes.
Empirically **yolov9t is fine**: trained on `human-detection-in-floods` (32 degenerate
boxes) with mosaic at `p=1.0`, reaching mAP50:95 0.3166 in 3 epochs. This clears the
flagship only; the known `ec`-family degenerate-box assertion is untested and its
no-mosaic recipe should stay.

### 7.5 Vast: the expired-session trap — FIXED in the skill

The CLI writes its 2FA session key to `~/.config/vastai/vast_tfa_key` and then **prefers
it over the API key for every request**. On expiry the server returns 404 "Session
expired", but the CLI's recovery only fires on 401 "Invalid user key", so the stale file
poisons every command *including `tfa login` itself*. A healthy account looks dead.

### 7.6 Vast: bandwidth dominated the bill, and I caused it

Instance `46301707` billed **$4.119**, of which GPU time was ~$0.61. The rest was ~84 GB
of transfer (41 GB in, 43 GB out). Chain: the first host pulled its image slowly, so I
raised `--min-down` from 500 to 2000, which cut the pool from 57 offers to 2, and
high-advertised-bandwidth hosts charge the most per GB. The RF100-VL plan specifies a
`<= $0.01/GB` filter that I did not apply. At cheap-transfer rates the same job is ~$0.89.
**Overspend ~$3.25.** Fixed by making the filter a default.

Real cost structure, units derived arithmetically because the docs omit them
(`storage_cost` is $/GB/month, `storage_total_cost` is $/hour for the searched size):

| meter | example rate | billed |
|---|---|---|
| GPU | $0.29/hr | while RUNNING |
| disk | $0.867/GB/month | while the instance EXISTS, including stopped |
| net in / out | $0.0026 to $0.017 /GB | per byte, ~6x spread between hosts at equal GPU price |

### 7.7 Harness: eval cache defaults inside `--data-dir` — NOT fixed, needs a ruling

Default cache root is `<data_dir>/.va-bench/...`, i.e. harness state written into the
dataset directory. Conflicts with datasets being a shared read-only mount. A
`per_dataset_dir` override exists; the fix is to default it under `weights_root`. Left
alone because it relocates existing caches.

### 7.8 Box bootstrap: PEP 668 and Git Bash path rewriting — both FIXED

`pytorch/pytorch:*-cuda12.8-*` sits on Ubuntu 24.04, whose system python is
externally-managed, so `pip install` refuses and the failure surfaces ten lines later as
`ModuleNotFoundError`. And `--log /root/x.log` reached the box as
`C:/Program Files/Git/root/x.log`. Both would have been baked into the Dockerfile had it
been frozen before a real box ran the sequence.

### 7.9 Upstream `rf100vl` package: index order is unstable

`DatasetList.__init__` sorts `self.projects` but builds `self.datasets` from the
**unsorted** constructor argument, so `dataset_list[i]` is workspace-API order despite a
docstring in the same file promising stability. The harness keys by name throughout.
**Worth reporting upstream.**

### 7.10 LibreYOLO main worktree: corrupted file, unrelated to RF100-VL

`libreyolo/validation/detection_validator.py` contains a literal
`...1908 tokens truncated...` marker where ~130 lines were, and imports a
`validation/contracts.py` that does not exist. `import libreyolo` fails from the shared
venv. It sits inside an otherwise healthy 28-file uncommitted change set, so it was
**not reverted**; a copy is in this session's scratchpad. All work here used the clean
`rf100vl-protocol` worktree.

---

## 8. Decisions taken

**Lean Docker image, datasets staged separately.** `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`,
because cu128 is mandatory for RTX 5090 (sm_120) and the vast skill's default cu124 image
cannot launch a kernel on one. libreyolo pinned **by commit, never branch**, so two boxes
in one campaign cannot disagree about what they measured. 40 GB of data is not baked in:
it would turn a ~30 s cold start into a multi-minute pull on every uncached host.

**Republish the dataset rather than re-download per box.** All 100 MIT, so permitted.
Removes the Roboflow key from every box and pins against Universe datasets being revised
in place. Honest saving is ~1 hour per box launch, not the "most of a day" claimed
earlier from a wrong extrapolation.

**Tar, not parquet.** No HF-native format holds a directory tree; they are all
row-oriented. The split-level COCO json would have to be shredded and rebuilt, and that
rebuild is exactly where the id bugs that score 0 mAP come from. Evidence: the authors'
own parquet copy lost the class names entirely. A parquet config can be added alongside
later if the viewer is wanted.

**Ship the degenerate boxes.** Faithfulness beats cleanliness for a redistribution.

**Do not use Vast volumes for the campaign.** They are real (64 verified offers,
~10x cheaper per GB than instance disk) but **physically bound to one machine**. Network
volumes are advertised, documented as "coming soon", and have **zero live offers**.
Pinning to one host throws away the marketplace. Volumes suit a different pattern:
repeatedly returning to the same box.

---

## 9. Open items

| item | state |
|---|---|
| Round-trip verification of published tars | RUNNING locally, incomplete. Until it finishes, "byte-identical" is a claim about the code, not a verified fact |
| Vast credit | **$0.26.** Campaign not funded; wave 2 needs $110-260 |
| TOTP seed | not saved; auth still needs a human every ~7 days |
| Docker image | written, **never built**. Treat as unproven until CI builds it once |
| PR #666 commit `d072d119` | committed, not pushed |
| Harness branch `rf100vl-harness` | never pushed |
| `sync_artifacts.py`, `rescore_offline.py` | standalone, not CLI verbs, no tests |
| `drop_last` assertion for the 3 sub-batch test splits | not written |
| Eval cache location (§7.7) | needs an owner ruling |
| Degenerate boxes vs `ec`/`rfdetr` families | untested |
| Upstream bug report to Roboflow (§7.9) | not filed |
| Ask Roboflow about the redistribution | not asked; not a blocker, licenses already permit it |

# PRD: Extend RF100-VL performance & robustness coverage beyond yolov9

Status: open. Owner: unassigned (written for an agent to execute).
Date: 2026-08-02. Author: campaign-advisor agent, verified against
`libreyolo` dev @ `89a80a9e` (+ PR #681) and `vision-analysis-benchmark`
@ `ad293f9` (branch `rf100vl-harness`).

## Context

Issue [LibreYOLO/libreyolo#674](https://github.com/LibreYOLO/libreyolo/issues/674)
plans RF100-VL campaigns for: **MUST** yolov9 t/s/m/c, rfdetr n/s/m/l,
dfine n/s, ec s/m, deimv2 atto/femto/pico/n; **NICE** yolox nano/tiny,
rtdetrv2 r18, rtmdet t/s, picodet s, yolonas s, yolov9-e2e t, yolov7.

The 2026-08-01 yolov9t campaign produced a set of performance and
robustness improvements. Some are family-agnostic and already cover every
model above; others are wired only for yolov9. This PRD closes the gap.

## What already applies to every family (do NOT redo)

| Improvement | Where | Scope |
|---|---|---|
| OOM -> automatic solo-GPU retry drain phase | harness `rf100vl_train.py` (`ad293f9`) | family-agnostic, orchestrator-level |
| Capture-race classification + recorded eager retry | harness `rf100vl_train.py` (`ad293f9`) | family-agnostic (fires only where graphs run) |
| Hoisted per-epoch validator (no rebuild per epoch) | libreyolo `training/trainer.py` (#677) | all families; no family overrides the validation loop |
| thread_local capture (pin-memory race root fix) | libreyolo #681 | wherever capture happens |
| rfdetr `dense_oom_fallback` (grad-accum, protocol-sanctioned) | rfdetr recipe | rfdetr only, by design |
| Staging, pick_box, dashboard, billing lessons | harness/skill | family-agnostic |

## Verified coverage matrix (the gaps)

Legend: LIB = library capability exists; REC = recipe enables it.

| Family | Train CUDA graphs | Graphed validation | Post-resize cache | Recipe exists |
|---|---|---|---|---|
| yolov9 t/s/m/c | LIB+REC | LIB (SUPPORTS_CUDA_GRAPH) | LIB+REC (`disk`) | yes |
| rfdetr n/s/m/l | LIB only (spec in `rfdetr/trainer.py:899`; recipe does not set `cuda_graph`) | no (model lacks `SUPPORTS_CUDA_GRAPH`) | LIB only (BaseTrainer path; recipe does not set `cache`) | yes |
| dfine n/s | no spec (flag would warn + run eager) | no | LIB only (`dfine/trainer.py:467` calls `enable_image_cache`; recipe does not set `cache`) | yes |
| ec s/m | no spec (inherits DFINETrainer) | no | LIB only (via DFINETrainer) | yes |
| deimv2 atto/femto/pico/n | no spec (inherits DEIMTrainer) | no | LIB only (`deim/trainer.py:404`) | yes |
| yolox nano/tiny | no spec | no | LIB only (BaseTrainer) | yes |
| rtdetrv2 r18 | no spec | no | LIB only | yes |
| rtmdet t/s, picodet s, yolonas s | no spec | no | LIB only | yes |
| yolov9-e2e t | **deliberately excluded** (dual-assignment head is not plain `DDetect`, see `yolo9/trainer.py:144`) | unverified (may inherit `SUPPORTS_CUDA_GRAPH` from LibreYOLO9 — verify clean fallback) | LIB only | **NO — `yolov9-e2e.json` missing** (registry key `yolov9e2e-t` exists, `recipe_path_for_family("yolov9-e2e")` will fail) |
| yolov7 | n/a | n/a | n/a | **NO — not even in the model registry** (issue #674 already flags this) |

## Hard constraints (protocol integrity)

1. **Recipe hash discipline.** `cache` and `cuda_graph` live in the recipe
   `protocol` block, so enabling them CHANGES the recipe SHA-256 and
   invalidates resume for any run already made with the old recipe. Flip
   these flags ONLY before a family's campaign starts, NEVER mid-campaign.
2. **Numerics must be provably unchanged.** `cuda_graph` and `cache` are
   execution details (excluded from the run signature) precisely because
   they are bit-identical. Every newly-enabled flag must re-earn that claim
   per family (see verification below). rfdetr is special: even eager
   training is not bit-reproducible (deformable-attention atomics), so its
   gate is the existing noise-band contract in
   `tests/unit/test_cuda_graph_training.py::TestCudaParityRFDETR`.
3. Harness changes commit straight to `rf100vl-harness`; libreyolo changes
   go via PR to `dev`.

## Work items

### P0 — needed before the corresponding family's campaign

**P0.1 Enable `"cache": "disk"` in the recipes of every campaign family**
(rfdetr, dfine, ec, deimv2; NICE: yolox, rtdetrv2, rtmdet, picodet,
yolonas). The library plumbing exists for all of them (verified above);
only the recipe flag is missing. For each family run the parity smoke
(below) before committing the recipe change. Watch one caveat per
DETR-style family: the cache point is the deterministic pre-augmentation
resize; families with multi-scale collate (dfine/deim resize per batch at
collate time, after dataset load) must still produce identical batches
with cache on/off.

**P0.2 Add the missing `yolov9-e2e.json` recipe.** Copy `yolov9.json`,
audit e2e-specific defaults in the library (`YOLO9E2EConfig` if present),
set `"cuda_graph": false` (training capture is deliberately unsupported
for the dual-assignment head), keep `"cache": "disk"` after the parity
smoke. Without this file the harness cannot even start `yolov9e2e-t`.

**P0.3 Verify yolov9-e2e graphed-validation fallback.** If the e2e model
class inherits `SUPPORTS_CUDA_GRAPH` from LibreYOLO9, training-time
validation will try to capture its eval forward. Confirm it either
captures correctly (parity vs eager on one batch) or falls back cleanly;
if neither, override `SUPPORTS_CUDA_GRAPH = False` on the e2e model
(libreyolo PR).

### P1 — measured wins, enable only if they pay

**P1.1 Measure training `cuda_graph` for rfdetr-n on one RF100-VL-like
dataset** (the spec exists and is parity-gated already). If epoch time
improves >= 5%, set `"cuda_graph": true` in `rfdetr.json` before the
rfdetr campaign. Note rfdetr trains at its own fixed resolution, so the
single-shape graph covers all full batches; last partial batches run
eager by design.

**P1.2 Graphed validation for the DETR families (libreyolo PR).** Add
`SUPPORTS_CUDA_GRAPH = True` to rfdetr (then dfine/deimv2/ec) models with
the same per-family parity gates yolo9 has
(`tests/unit/test_validator_cuda_graph.py` pattern). Validation is
launch-bound for small models, so this is likely the biggest remaining
epoch-time win for the DETR MUST families.

**P1.3 Training capture specs for dfine/deimv2/ec (libreyolo PR,
larger).** Network-only capture like rfdetr's spec (loss/Hungarian stays
eager). Caveat: dfine s+ use multi-scale batches (`base_size_repeat`,
fixed in #679), and the manager captures exactly one shape — only the
base-size batches replay, the rest run eager. Measure on dfine-s first;
skip if the multi-scale mix caps the win below ~5%.

### P2 — nice tier completion

**P2.1 yolov7 registry + recipe wiring** in the harness (issue #674
already scopes this as "needs harness registry+recipe wiring").

## Per-family verification recipe (applies to P0.1, P1.1, P1.2, P1.3)

1. **Parity smoke (numerics):** one rf20vl dataset, 3 epochs, seed 0,
   flag off vs on: loss trajectory and `valid_mAP50_95` must be identical
   (bit-identical for all families except rfdetr, which uses its
   documented noise band). Use `va-bench rf100vl-train --smoke-epochs 3
   --limit-datasets 1` twice and diff stats.
2. **Speed check:** `libreyolo profile run` (or wall-clock per epoch from
   the smoke) with the flag on vs off; record the number in the PR/commit.
3. **Recipe hash:** update the recipe only after 1-2 pass; note in the
   commit that the hash changed and prior runs of that family (if any)
   are invalidated.
4. **Pilot gate:** rf20vl pilot for the family (per issue #674) before
   the full 100 datasets.

## Out of scope

- yolo1/2/3/4 (inference-only ports, no trainers — library project first).
- Changing any protocol field (epochs, batch, precision, selection).
- Inference-time (predict/export) graph coverage beyond validation.

# Results: RF100-VL perf & robustness coverage beyond yolov9

Companion to `prd-rf100vl-perf-coverage.md`. Executed 2026-08-02 on a local
RTX 5070 Ti (16 GB), libreyolo dev @ `89a80a9e`, smoke dataset
`crystal-clean-brain-tumors-mri-dataset` (48 train images), 3 epochs, seed 0.
Raw artifacts: `parity_verdicts.json`, `determinism_controls.json`,
`cache_byte_parity` output and per-arm `metrics.jsonl` files in the local
smoke workspace.

## P0.1 — `"cache": "disk"` enabled for every campaign family: DONE

Recipes changed: rfdetr, dfine, ec, deimv2, yolox, rtdetrv2, rtmdet,
picodet, yolonas (plus the new yolov9-e2e and yolov7 recipes ship with it).
Recipe SHA-256 changes for all of them; no campaign for these families had
started, so nothing is invalidated.

The parity gate was earned in two parts, because the original plan
(bit-identical trajectories) turned out to be unattainable for reasons
unrelated to the cache:

1. **Byte parity of the cache itself (the load-bearing half).** For every
   train image of the smoke dataset, both cache points (`load_image`
   pre-resize, `load_resized_img` post-resize) return arrays byte-identical
   to a fresh decode/resize, on both the cache-fill and the cache-hit
   (`.npy` read-back) paths. The dataset pipeline calls the same loader
   functions whether caching is on or off (`pull_item` does not branch on
   cache mode), so identical arrays imply identical augmentation inputs.
2. **Trajectory smoke within the family's own noise band.** Off vs on
   3-epoch runs differ for every family — but an off vs off2 control
   (same recipe, same seed, cache off both times) shows the same or larger
   divergence. None of these families is run-to-run reproducible on CUDA
   (deformable-attention/`grid_sample` and interpolate backwards use
   atomics; tiny-dataset training amplifies one-ulp differences into
   percent-level mAP swings by epoch 3). The cache arm is indistinguishable
   from a rerun:

   | family | off vs off2 (max rel diff) | off vs on (max rel diff) |
   |---|---|---|
   | dfine | 1.3e-01 | 4.6e-02 |
   | ec | 1.3e-01 | 1.1e-01 |
   | deimv2 | 1.7e-01 | 9.0e-02 |
   | yolov9-e2e | 8.9e-01 | 1.0e+00 |
   | yolox | 7.2e-01 | 6.9e-01 |
   | rtdetrv2 | 1.0e+00 | 1.0e+00 |
   | rtmdet | 6.1e-01 | 5.3e-01 |
   | picodet | 1.0e+00 | 4.0e-01 |
   | yolonas | 1.0e+00 | 1.0e+00 |
   | yolov7 | 1.0e+00 | 1.0e+00 |

   (Worst key is typically `metrics/mAP_small`-style near-zero metrics,
   where relative diffs saturate. Epoch-1 train-loss relative differences
   are 1e-5..5e-3 in both comparisons.) This is the same noise-band
   contract the PRD already sanctioned for rfdetr, now measured per family.

Caveat honored: dfine/deim multi-scale collate happens after the dataset
load, so the cache point (deterministic pre-augmentation resize) feeds it
identically either way — covered by (1).

## P0.2 — `yolov9-e2e.json` recipe: DONE

Copied from `yolov9.json`. Audited at the pinned commit: `YOLO9E2EConfig`
subclasses `YOLO9Config` and overrides only the run name, so the
hyperparameters are identical by construction. `"cuda_graph": false`
(training capture requires the plain DDetect head; the e2e dual-assignment
head is deliberately excluded in `yolo9/trainer.py`), `"cache": "disk"`
after the parity gate above. Unit tests updated: the family is no longer
excluded from the recipe-coverage test, and a new test pins
`cuda_graph: false`.

## P0.3 — yolov9-e2e graphed-validation fallback: VERIFIED SAFE

The e2e model inherits `SUPPORTS_CUDA_GRAPH = True` from LibreYOLO9 and the
inherited capture path is proven correct, not just harmless:
`tests/unit/test_cuda_graph_detr_families.py` (libreyolo PR) captures the
e2e eval forward on CUDA and asserts bit-identical replay vs eager at both
the model level and the validator wiring level (up-front capture,
replay-only loop). CPU fallback also covered. No override needed.

## P1.1 — training `cuda_graph` for rfdetr-n: MEASURED, KEEP OFF

Steady-state training-step throughput (libreyolo window profiler, 32
measured steps after 16 warmup, 3 trials per arm) at the campaign per-step
config (batch 4, fp32, adamw, workers 0, the recipe's multi-scale
augmentation): eager 15.5/17.8/17.4 img/s, graphed 16.4/16.7/16.4 img/s —
**-2.4% with the graph on** (RTX 5070 Ti). Capture itself worked (logged at
shape 4x3x416x416, all three trials), but the recipe trains multi-scale
(`multi_scale` + `expanded_scales`), so only base-shape batches replay and
the rest run eager — precisely the dilution the PRD flagged. Both arms are
host/launch-bound, yet the single-shape replay does not pay for its
overhead here. Decision per the >= 5% rule: `rfdetr.json` keeps
`cuda_graph` unset. Re-measure on the campaign GPU class only if its
host/GPU balance differs a lot. Raw numbers: `rfdetr_graph_timing.json`.

## P1.2 — graphed validation for DETR families: DONE (libreyolo PR)

`SUPPORTS_CUDA_GRAPH = True` added to LibreRFDETR, LibreDFINE, LibreDEIMv2,
LibreEC with per-family parity gates in
`tests/unit/test_cuda_graph_detr_families.py`: 20 tests, all passing on
CUDA hardware — opt-in surface (CPU), clean CPU fallback, model-level
capture/replay bit-parity, and validator-level wiring bit-parity for all
five families (the four DETR ones plus yolov9-e2e).

## P1.3 — training capture specs for dfine/deimv2/ec: INFEASIBLE BY DESIGN

The rfdetr spec works because the LWDETR detect forward never reads
targets. The D-FINE decoder line (dfine, deimv2 via DEIM, ec via
DFINETrainer) passes `targets` into the network forward to build
contrastive-denoising query groups (`get_contrastive_denoising_training_group`
in `dfine/decoder.py`), whose query count varies with per-image GT counts.
That violates both capture-contract requirements (target-free forward,
static shapes). A network-only `CudaGraphTrainSpec` cannot express this;
enabling it would require restructuring denoising (fixed-size dn padding
outside the graph), which is library surgery out of scope here. Skipped,
per the PRD's own skip clause.

## P2.1 — yolov7 registry + recipe wiring: DONE

`yolov7` (single variant `b`, `LibreYOLO7b.pt`) added to the model
registry; `yolov7.json` recipe audited from `YOLOv7Config` at the pinned
commit (YOLOX SimOTA pipeline with v7's overrides: momentum 0.937, warmup
3, EMA 0.9999; `yoloxwarmcos`, mosaic+mixup on, degrees 10, shear 2). No
`cuda_graph` flag (no capture spec, no `SUPPORTS_CUDA_GRAPH` opt-in yet).
`cache: disk` passed the same two-part parity gate (byte parity + noise
band). Trained end-to-end in the smoke (3 epochs, both arms).

## Incidental harness fixes

- `build_train_kwargs` now passes `allow_experimental=True` for rtmdet and
  picodet (their trainers hard-refuse without it; issue #674 scopes both
  as campaign families). Same opt-in ec already had. Without this, the
  families could not start at all.
- `cmd_rf100vl_train` defined `state_root` after first use (`NameError`
  on any single-model train invocation with the artifact syncer). Fixed by
  hoisting the assignment, mirroring `cmd_rf100vl_campaign`.
- yolonas weight auto-download crashed on Windows consoles with cp1252
  encoding (cosmetic, environment-specific; worked with
  `PYTHONIOENCODING=utf-8`). No harness change; campaign boxes are Linux.

## Follow-ups before each family's campaign

- rf20vl pilot per family (issue #674 gate) — unchanged.
- If the rfdetr P1.1 number on the campaign GPU class differs materially
  from the local one, re-measure there before flipping the flag.
- yolov9-e2e t: registry key `yolov9e2e-t` + new recipe are ready; smoke
  passed end-to-end.

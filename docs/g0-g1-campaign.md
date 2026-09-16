# G0/G1 detection campaign

Inventory source: LibreYOLO `6a0ccc3a0e579948011d5a55200309c4082e61e7` (dev, includes TinyFormer).
Groups classify families; this campaign selects the detect task only.

60 variants across 13 families. 58 have public checkpoint endpoints; P2 t/s need locally COCO-trained checkpoints.

| Group | Family | Sizes | COCO checkpoint |
| --- | --- | --- | --- |
| g0 | yolo9 | t, s, m, c | Public download endpoint verified |
| g0 | rfdetr | n, s, m, l | Public download endpoint verified |
| g1 | yolo9_e2e | t, s, m, c | Public download endpoint verified |
| g1 | yolo9_p2 | t, s | Local trained weights required |
| g1 | ec | s, m, l, x | Public download endpoint verified |
| g1 | rtdetr | r18, r34, r50, r50m, r101, l, x | Public download endpoint verified |
| g1 | rtdetrv2 | r18, r34, r50, r50m, r101 | Public download endpoint verified |
| g1 | rtdetrv4 | s, m, l, x | Public download endpoint verified |
| g1 | dfine | n, s, m, l, x | Public download endpoint verified |
| g1 | deim | n, s, m, l, x | Public download endpoint verified |
| g1 | deimv2 | atto, femto, pico, n, s, m, l, x | Public download endpoint verified |
| g1 | tinyformer | s, m, l, x, xl | Public download endpoint verified |
| g1 | yolonas | s, m, l | Public download endpoint verified |

## Setup

Pin the library, not a moving dev branch. v1.5.0 does not include TinyFormer.

```bash
python -m pip install "libreyolo @ git+https://github.com/LibreYOLO/libreyolo.git@6a0ccc3a0e579948011d5a55200309c4082e61e7"
va-bench list --groups g0 g1
```

Use a clean harness checkout containing this change and record its exact commit.
Keep the same PyTorch/CUDA environment for the whole Spark campaign. Start by
repeating YOLO9-T and RF-DETR-S so the new library pin has anchor measurements.

## Public checkpoint batch (58 variants)

```bash
va-bench run --models \
  yolov9t yolov9s yolov9m yolov9c rfdetr-n rfdetr-s rfdetr-m rfdetr-l \
  yolov9e2e-t yolov9e2e-s yolov9e2e-m yolov9e2e-c ec-s ec-m ec-l ec-x \
  rtdetr-r18 rtdetr-r34 rtdetr-r50 rtdetr-r50m rtdetr-r101 rtdetr-l rtdetr-x rtdetrv2-r18 \
  rtdetrv2-r34 rtdetrv2-r50 rtdetrv2-r50m rtdetrv2-r101 rtdetrv4-s rtdetrv4-m rtdetrv4-l rtdetrv4-x \
  dfine-n dfine-s dfine-m dfine-l dfine-x deim-n deim-s deim-m \
  deim-l deim-x deimv2-atto deimv2-femto deimv2-pico deimv2-n deimv2-s deimv2-m \
  deimv2-l deimv2-x tinyformer-s tinyformer-m tinyformer-l tinyformer-x tinyformer-xl yolonas-s \
  yolonas-m yolonas-l \
  --coco-dir /path/to/coco-mini500 --output-dir results/spark-g0-g1 \
  --device cuda --format pytorch \
  --dataset-id LibreYOLO/coco-val2017-mini500 \
  --dataset-revision 6c5d8d10dab92cf6f2cd476e1d5509ab5786cb88
```

No --limit. Keep conf=0.001, IoU=0.6, max_det=300 and each model's native input
size. Run on an otherwise idle Spark. Inspect every model failure; a sweep exits
nonzero if any model failed, even though remaining models are attempted.

## Full inventory including P2

```bash
va-bench run --groups g0 g1 --coco-dir /path/to/coco-mini500 --device cuda
```

This requests all 60 variants and explicitly reports P2 missing-checkpoint errors
unless its canonical files exist locally. For a separate P2 run:

```bash
va-bench run --models yolov9p2-t yolov9p2-s --weights-dir /path/to/coco-trained-p2 \
  --coco-dir /path/to/coco-mini500 --device cuda
```

Do not use the 10-class VisDrone preview or transferred/randomly initialized P2
heads as COCO benchmarks. The harness checks checkpoint family, detect task,
size, 80-class count, and COCO class names/order. Training evidence for supplied
weights remains the contributor's responsibility.

## Evidence and limits

- All 60 native variants passed CPU preprocessing, forward, and postprocessing
  at native input resolution with initialized weights on this pinned library.
- Published YOLO9-E2E-T and TinyFormer-S checkpoints completed the harness on
  two actual COCO images, including evaluation and weight hashing. Smoke outputs
  were not submitted as benchmark measurements.
- Public Hugging Face checkpoint endpoints were checked for 55 variants and
  the library's Deci CDN endpoints for the three YOLO-NAS variants.
- Full COCO accuracy, all 58 published checkpoint loads, and Spark/CUDA execution
  have not been independently reproduced here. These are the contributor run.
- YOLO-NAS checkpoints retain Deci's weight terms. TinyFormer checkpoints have
  Apache-2.0 plus DINOv3 terms. The harness links existing library downloads; it
  does not redistribute weights or relabel their licenses.
- Registration of P2/TinyFormer is for inference; no RF100-VL fine-tuning recipe
  is claimed by this change.

## Submission

The corresponding website catalogue change must land before new-family JSONs
validate. Keep harness JSONs unchanged, with hashes and commands intact. Full
COCO accuracy references remain separate from mini500 latency runs; never splice
metrics from different runs without a separate provenance contract.

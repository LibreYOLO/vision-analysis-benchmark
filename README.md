# Vision Analysis Benchmark

Benchmarking suite that powers [visionanalysis.org](https://visionanalysis.org). Runs object detection models on COCO val2017, measures timing/accuracy/memory, and outputs JSON that the website consumes directly.

Models are loaded through [LibreYOLO](https://github.com/visionanalysis/libreyolo). Currently supports YOLOX, YOLOv9, and RF-DETR.

## Setup

```
pip install -e .
```

You need COCO val2017 somewhere on disk:
```
coco/
├── annotations/
│   └── instances_val2017.json
└── images/
    └── val2017/
        └── *.jpg
```

## Usage

Run a benchmark:
```
va-bench run --models yolov9t yolox-s --coco-dir /path/to/coco
```

Run all models:
```
va-bench run --all --coco-dir /path/to/coco --output-dir ./results
```

See what's available:
```
va-bench list
```

Compute VA v1 Scores (needs both RTX 5080 and RPi5 results for each model):
```
va-bench score --results-dir ./results
```

## Output

Each run produces a JSON file in the output directory (`./results` by default). The JSON matches the `RawBenchmark` schema from the website and includes accuracy (all 12 COCO metrics), per-phase timing, FPS, memory usage, and hardware/software info.

## VA v1 Score

The composite score (0-100) ranks models across 6 metrics: mAP@50, mAP@50-95, mAP_small, FPS on RTX 5080, FPS on RPi5, and mAP/GFLOP. Each metric is min-max normalized across all qualifying models, then averaged. A model needs benchmarks on both hardware platforms to get a score.

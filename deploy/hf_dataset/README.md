---
license: mit
task_categories:
  - object-detection
tags:
  - rf100-vl
  - object-detection
  - benchmark
size_categories:
  - 100K<n<1M
---

# RF100-VL, packaged for one download

**This is a temporary redistribution of Roboflow's RF100-VL benchmark, published so that benchmark runs can fetch the data in one step instead of a hundred. It is not an original work, and if Roboflow asks us to take it down we will, immediately and without argument.**

Everything here comes from [Roboflow 100-VL](https://rf100-vl.org). The benchmark, the datasets, the splits and the cleaning logic are theirs. We changed none of it.

## Why this exists

RF100-VL is 100 separate projects on Roboflow Universe. The official `rf100vl` package downloads them one at a time, and each download waits on Roboflow generating that project's zip on demand. We measured it twice while setting up our own benchmark runs:

| | throughput | per dataset | all 100 |
|---|---|---|---|
| home connection | 2.2 MB/s | ~203 s | ~5.6 hours |
| datacenter GPU box | 16.3 MB/s | ~27 s | ~45 minutes |

Neither number is Roboflow's fault. Most of that time is server-side zip generation, not transfer. But it is time you pay again on every machine you start, and if you are renting GPUs by the hour it comes out of the same budget as the training.

Pulling these tars instead takes a few minutes, and it needs no Roboflow API key.

There is a second reason. The `rf100vl` package always resolves the *latest* version of each Universe project, and some projects have been revised in place since the paper. Two runs a month apart can therefore score different data while both calling it RF100-VL. The `versions.json` in this repo pins the exact version of every project in the archive, so a number produced against it can be reproduced later.

## What is in here

One `.tar` per dataset, 100 of them, plus:

- `versions.json`, the exact Roboflow project version behind each dataset
- `NOTICE`, every dataset with its license, uploader credit and Universe URL
- `licenses.json`, the same thing machine-readable

Each tar unpacks to the layout the official package produces:

```
<dataset>/
  train/  _annotations.coco.json + images
  valid/  _annotations.coco.json + images
  test/   _annotations.coco.json + images
  README.dataset.txt      (Roboflow's, carries the license and credit)
  README.roboflow.txt
```

100 datasets, 564 classes, 7 domains. Counted from the annotation files in this
archive: **163,151 images** (115,777 train, 33,137 valid, 14,237 test) and
1,353,434 annotations.

The paper quotes 164,149 images and 1,355,491 annotations, and Roboflow's API
reports the same image total. The gap is about a thousand images that exist in
the projects but are not in any released split, so they never arrive in a
download. Nothing is missing here: every one of the 100 datasets was verified to
carry all three splits with a readable `_annotations.coco.json`. We mention it
only so the numbers reconcile if you compare them against the paper.

## This copy is cleaned, and that matters

The images and boxes are untouched. The category numbering is not, because the official package rewrites it on download and we ran that same code:

- the dummy supercategory at id 0 is removed
- every remaining category id shifts down by one, giving 0-based contiguous ids
- annotation ids start at 1

This is not our idea and not optional. A raw export from the Universe website keeps the dummy class and the original numbering, and scoring those predictions against the benchmark's ground truth gives close to zero mAP. Roboflow's own evaluation notes warn about it. Because we ran that cleaning, this archive is a derivative work, which the licenses below permit.

## Licensing

Every one of the 100 datasets is **MIT**, and every project is public. We checked them individually rather than relying on a blanket statement: the license came from each project's Roboflow API record, and we confirmed it matched the `License:` line inside the downloaded export itself.

MIT allows redistribution and derivatives as long as the copyright notices travel with the work, so we kept Roboflow's per-dataset `README.dataset.txt` inside every tar and listed all 100 in `NOTICE`. Do not strip those files.

Licenses are set by the people who uploaded each project and can be changed later. The inventory here is a snapshot taken on 2026-07-30 against the pinned versions.

## Getting it

```bash
pip install "huggingface_hub[hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1     # Rust downloader, saturates the link
```

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="LibreYOLO/rf100-vl", repo_type="dataset",
                  local_dir="rf100-vl", max_workers=8)
```

For one dataset:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="LibreYOLO/rf100-vl", filename="aerial-cows.tar",
                repo_type="dataset", local_dir="rf100-vl")
```

Then unpack the tars in place:

```bash
cd rf100-vl && for f in *.tar; do tar xf "$f" && rm "$f"; done
```

### Why tar and not parquet

Because the point of this repo is a byte-identical copy of the layout the
official package writes, and parquet cannot express a directory tree. Tar-based
image datasets are a normal shape on the Hub, which is what WebDataset is.

The tradeoff is real, so it is worth stating: 100 tars download quickly and
unpack straight into the layout every RF100-VL tool already expects, but they
get no dataset viewer. Shipping the 164,149 images as loose files would give a
viewer and a file browser at the cost of 164,149 separate requests, which is
considerably slower.

If you want the parquet shape instead, the authors already publish one at
[`probicheaux/rf100-vl`](https://huggingface.co/datasets/probicheaux/rf100-vl).

## If you are running the benchmark

Use the test split, score with pycocotools at **maxDets 500**, and average across the 100 datasets without weighting. The 500 matters: the default of 100 truncates detections on the crowded images and quietly costs you accuracy. Do not report a YOLO toolkit's own mAP, which runs several points high against pycocotools on these datasets.

## Credit

RF100-VL is by Peter Robicheaux, Matvei Popov, Anish Madan, Isaac Robinson, Joseph Nelson, Deva Ramanan and Neehar Peri, at Roboflow and Carnegie Mellon University.

```bibtex
@article{robicheaux2025roboflow100vl,
  title   = {Roboflow100-VL: A Multi-Domain Object Detection Benchmark for Vision-Language Models},
  author  = {Robicheaux, Peter and Popov, Matvei and Madan, Anish and Robinson, Isaac
             and Nelson, Joseph and Ramanan, Deva and Peri, Neehar},
  journal = {arXiv preprint arXiv:2505.20612},
  year    = {2025}
}
```

Go to [rf100-vl.org](https://rf100-vl.org) and the [official repository](https://github.com/roboflow/rf100-vl) for the benchmark itself. The authors also publish a flattened parquet copy at [`probicheaux/rf100-vl`](https://huggingface.co/datasets/probicheaux/rf100-vl). That one is a different shape to this: it merges all 100 datasets into single splits and carries no class names, so it does not drop into a per-dataset training loop.

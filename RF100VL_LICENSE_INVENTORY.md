# RF100-VL per-dataset license inventory

Date: 2026-07-30. Machine-readable source of truth:
`va_bench/data/rf100vl_licenses.json` (100 records). Regenerate with
`deploy/license_inventory.py`.

## Verdict

**All 100 datasets are MIT. All 100 projects are public. Nothing is blocked.**

| licenses found | count |
|---|---|
| MIT | 100 |

| redistribution verdict | count |
|---|---|
| OK - permissive | 100 |
| NC / No-Derivatives / unknown | 0 |

A cleaned, re-derived copy of the full benchmark can be republished under the
LibreYOLO org, provided the MIT copyright notices travel with it.

## Why this was checked rather than assumed

The rf100-vl README states Apache 2.0, but that covers **the benchmark repo**.
The datasets themselves are 100 separate Roboflow Universe projects, each
carrying a license chosen by its uploader, so the blanket claim is not the
thing to gate redistribution on. Had even one been CC BY-ND, republishing a
cleaned copy would have been forbidden outright, because the id-remapping the
`rf100vl` package performs makes our copy a derivative work.

## Method and verification

Licenses come from `project.license` on the public Roboflow API
(`GET https://api.roboflow.com/rf100-vl/<slug>`), one call per project.

That field was **cross-validated against the `README.dataset.txt` that ships
inside each downloaded export** for the 8 datasets materialised at the time:
**0 mismatches**. The API and the shipped export agree, so the API is a sound
proxy for the full 100 without downloading 40 GB first.

Datasets are addressed by name throughout, never by package index: the upstream
`DatasetList` sorts `self.projects` but builds `self.datasets` from the
unsorted constructor argument, so positional access is workspace-API order.

## Independent corroboration

The paper's first author (Peter Robicheaux, Roboflow) already redistributes the
complete benchmark on HuggingFace as `probicheaux/rf100-vl` — 42 GB, Apache 2.0
tag, ungated, no API key. Roboflow publishing a full copy themselves is a clear
signal that redistribution is intended, independent of our license reading.

## The image-count discrepancy, corrected

An earlier revision of this document claimed the HF parquet mirror was lossy
because the Roboflow API totals 164,149 images against the mirror's 163,151.
**That was wrong.** The full canonical download of all 100 datasets was then
counted directly from the annotation files and gives:

| | images | annotations |
|---|---|---|
| canonical download (measured) | **163,151** | 1,353,434 |
| HF parquet mirror | 163,151 | - |
| paper / Roboflow API | 164,149 | 1,355,491 |

The canonical download and the mirror agree exactly. The ~1,000-image gap is
between the *released splits* and the projects' total image counts: those images
exist in the Universe projects but are in no released split, so no download of
any kind contains them. Nothing is missing from either copy.

The mirror remains unsuitable for our purposes, but for the reason that was
always true and is sufficient on its own: it flattens the 100 datasets into
single splits and carries no class names.

## Obligations to honour when republishing

MIT is permissive but not obligation-free. The republished archive must ship:

1. Every per-dataset `README.dataset.txt` (carries the MIT notice, the
   uploader credit, and the Universe URL) — do not strip these.
2. A top-level `NOTICE` enumerating all 100 datasets with license and source
   URL, generated from `rf100vl_licenses.json`.
3. The `versions.json` lock, so the archive is pinned to specific dataset
   versions rather than a moving target.
4. A statement that the copy is cleaned (dummy class 0 removed, category ids
   shifted to 0-based contiguous, annotation ids from 1), i.e. a derivative.

## Open item

Licenses can be changed by uploaders. This inventory is a snapshot dated
2026-07-30 against the locked versions; re-run `deploy/license_inventory.py`
before any republish and diff against the committed JSON.

Asking Roboflow for explicit blessing remains worthwhile goodwill, but on the
evidence above it is not a blocker for datasets whose licenses already permit
redistribution — which, here, is all of them.

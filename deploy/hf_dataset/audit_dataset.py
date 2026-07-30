"""Full integrity audit of a materialised RF100-VL tree, before it is trusted.

Written because a subtle data fault found after a full campaign costs hundreds
of dollars and a republish, while finding it here costs minutes. Every check
below corresponds to a way a benchmark run could produce a wrong number rather
than an obvious crash.

FAIL means do not publish or benchmark. WARN means known-benign in COCO data
(boxes clipped at the image edge, images with no objects) but reported so the
counts are visible rather than assumed.

Usage: python audit_dataset.py /root/rf100-vl [--workers 64] [--skip-decode]
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageFile

# Do NOT let PIL silently tolerate truncation here: detecting it is the point.
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = None  # huge aerial/medical images are legitimate

SPLITS = ("train", "valid", "test")
ANNOTATION = "_annotations.coco.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def audit_dataset(args) -> dict:
    root, name, decode = args
    dataset_dir = Path(root) / name
    fails: list[str] = []
    warns: list[str] = []
    stats = {"images": 0, "annotations": 0, "empty_images": 0, "clipped_boxes": 0}
    category_signature = {}

    for split in SPLITS:
        split_dir = dataset_dir / split
        ann_path = split_dir / ANNOTATION
        if not ann_path.exists():
            fails.append(f"{split}: missing {ANNOTATION}")
            continue
        try:
            payload = json.loads(ann_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            fails.append(f"{split}: unreadable annotations ({type(exc).__name__})")
            continue

        images = payload.get("images") or []
        annotations = payload.get("annotations") or []
        categories = payload.get("categories") or []
        if not images:
            fails.append(f"{split}: zero images")
            continue
        if not categories:
            fails.append(f"{split}: zero categories")
            continue
        if not annotations:
            fails.append(f"{split}: zero annotations")

        # --- categories: the cleaned contract ------------------------------
        cat_ids = [c["id"] for c in categories]
        if sorted(cat_ids) != list(range(len(cat_ids))):
            fails.append(f"{split}: category ids not 0-based contiguous: {sorted(cat_ids)[:8]}")
        names = [c.get("name") for c in categories]
        if len(set(names)) != len(names):
            dupes = [n for n, c in Counter(names).items() if c > 1]
            fails.append(f"{split}: duplicate category names {dupes[:5]}")
        # The uncleaned export carries a dummy class named after the project.
        supers = {c.get("supercategory") for c in categories}
        for bad in supers & set(names):
            fails.append(f"{split}: dummy supercategory class {bad!r} still present (uncleaned)")
        category_signature[split] = tuple(sorted((c["id"], c.get("name")) for c in categories))

        # --- referential integrity -----------------------------------------
        by_id = {}
        file_names = Counter()
        for image in images:
            by_id[image["id"]] = image
            file_names[image["file_name"]] += 1
        if len(by_id) != len(images):
            fails.append(f"{split}: duplicate image ids")
        for fname, count in file_names.items():
            if count > 1:
                fails.append(f"{split}: duplicate file_name {fname!r} x{count}")

        ann_ids = [a["id"] for a in annotations]
        if len(set(ann_ids)) != len(ann_ids):
            fails.append(f"{split}: duplicate annotation ids")
        if annotations and min(ann_ids) < 1:
            fails.append(f"{split}: annotation ids start at {min(ann_ids)}, expected >= 1")

        valid_cat_ids = set(cat_ids)
        with_objects = set()
        for ann in annotations:
            image = by_id.get(ann["image_id"])
            if image is None:
                fails.append(f"{split}: annotation {ann['id']} references missing image "
                             f"{ann['image_id']}")
                continue
            with_objects.add(ann["image_id"])
            if ann["category_id"] not in valid_cat_ids:
                fails.append(f"{split}: annotation {ann['id']} has unknown category "
                             f"{ann['category_id']}")
            box = ann.get("bbox")
            if not box or len(box) != 4:
                fails.append(f"{split}: annotation {ann['id']} has malformed bbox")
                continue
            x, y, w, h = (float(v) for v in box)
            if not all(math.isfinite(v) for v in (x, y, w, h)):
                fails.append(f"{split}: annotation {ann['id']} has non-finite bbox")
                continue
            if w <= 0 or h <= 0:
                # A degenerate box is a real hazard: several augmentation paths
                # assert on them mid-training.
                fails.append(f"{split}: annotation {ann['id']} has degenerate bbox {w}x{h}")
                continue
            W, H = image["width"], image["height"]
            if x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1:
                stats["clipped_boxes"] += 1

        stats["empty_images"] += len(images) - len(with_objects)
        stats["images"] += len(images)
        stats["annotations"] += len(annotations)

        # --- files on disk match the manifest -------------------------------
        on_disk = {p.name for p in split_dir.iterdir()
                   if p.suffix.lower() in IMAGE_SUFFIXES}
        listed = set(file_names)
        for missing in sorted(listed - on_disk)[:5]:
            fails.append(f"{split}: listed image not on disk: {missing}")
        orphans = on_disk - listed
        if orphans:
            warns.append(f"{split}: {len(orphans)} image files on disk not in annotations")

        if decode:
            for image in images:
                path = split_dir / image["file_name"]
                if not path.exists():
                    continue
                try:
                    with Image.open(path) as handle:
                        size = handle.size
                        handle.load()  # full decode catches truncation
                except Exception as exc:
                    fails.append(f"{split}: unreadable image {image['file_name']} "
                                 f"({type(exc).__name__})")
                    continue
                if size != (image["width"], image["height"]):
                    fails.append(f"{split}: {image['file_name']} is {size} but annotations "
                                 f"say {(image['width'], image['height'])}")

    # --- categories must agree across splits --------------------------------
    if len(category_signature) == len(SPLITS) and len(set(category_signature.values())) != 1:
        detail = {s: len(v) for s, v in category_signature.items()}
        fails.append(f"category sets differ across splits (sizes {detail}); nc would be ambiguous")

    if not (dataset_dir / "README.dataset.txt").exists():
        fails.append("missing README.dataset.txt (license attribution)")

    return {"dataset": name, "fails": fails, "warns": warns, "stats": stats,
            "num_classes": len(category_signature.get("train", ()))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--workers", type=int, default=min(64, (os.cpu_count() or 8)))
    parser.add_argument("--skip-decode", action="store_true",
                        help="skip full image decode (much faster, misses truncation)")
    args = parser.parse_args()

    root = Path(args.root)
    names = sorted(p.name for p in root.iterdir()
                   if p.is_dir() and (p / "train").exists())
    print(f"auditing {len(names)} datasets with {args.workers} workers "
          f"(decode={'off' if args.skip_decode else 'on'})", flush=True)

    payload = [(str(root), n, not args.skip_decode) for n in names]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(audit_dataset, payload), start=1):
            results.append(result)
            mark = "FAIL" if result["fails"] else ("warn" if result["warns"] else "ok")
            if result["fails"] or i % 20 == 0:
                print(f"[{i}/{len(names)}] {result['dataset']}: {mark}", flush=True)

    Path("audit_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    totals = Counter()
    for result in results:
        for key, value in result["stats"].items():
            totals[key] += value
    failed = [r for r in results if r["fails"]]
    warned = [r for r in results if r["warns"]]

    print("\n=== totals ===")
    print(f"  datasets     : {len(results)}")
    print(f"  images       : {totals['images']:,}")
    print(f"  annotations  : {totals['annotations']:,}")
    print(f"  classes      : {sum(r['num_classes'] for r in results):,}")
    print(f"  images with no objects : {totals['empty_images']:,}")
    print(f"  boxes past the edge    : {totals['clipped_boxes']:,}")

    print(f"\n=== datasets with FAILURES: {len(failed)} ===")
    for result in failed:
        print(f"  {result['dataset']}:")
        for problem in result["fails"][:6]:
            print(f"      {problem}")
        if len(result["fails"]) > 6:
            print(f"      ... and {len(result['fails']) - 6} more")

    print(f"\n=== datasets with warnings: {len(warned)} ===")
    for result in warned[:10]:
        print(f"  {result['dataset']}: {'; '.join(result['warns'][:2])}")

    print("\nAUDIT_FAILED" if failed else "\nAUDIT_CLEAN")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

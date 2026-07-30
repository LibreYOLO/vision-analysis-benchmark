"""Tar each RF100-VL dataset and upload it to a HuggingFace dataset repo.

Runs on the machine that did the download, so 40 GB never crosses a home link.

Tars and uploads ONE dataset at a time, deleting each tar after it lands: the
download box has roughly 45 GB free against a ~40 GB dataset, so there is no
room to stage the whole archive at once. Peak extra space is therefore the size
of the largest single dataset (~3.4 GB), not the total.

Resumable. Files already present in the repo are skipped, so an interrupted run
picks up where it stopped.

Plain .tar, not .tar.gz: the payload is already-compressed JPEG, so gzip would
spend CPU to save nothing.
"""

import argparse
import json
import os
import sys
import tarfile
import time
from pathlib import Path

from huggingface_hub import HfApi

TOKEN_FILE = Path.home() / ".config" / "huggingface" / "token"


def read_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text(encoding="utf-8").strip()
    raise SystemExit(f"no HF token in $HF_TOKEN or {TOKEN_FILE}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/root/rf100-vl")
    parser.add_argument("--repo", default="LibreYOLO/rf100-vl")
    parser.add_argument("--assets", default="/root/assets",
                        help="Directory holding README.md, NOTICE, licenses.json")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    assets = Path(args.assets)

    lock = json.loads((data_dir / "versions.json").read_text(encoding="utf-8"))
    names = sorted(lock["datasets"])
    downloaded = set(lock.get("downloaded", []))
    missing = [n for n in names if n not in downloaded]

    # Checked before the token is read, so completeness can be validated on a
    # box that has no credentials on it yet.
    if args.dry_run:
        for name in ("README.md", "NOTICE", "licenses.json"):
            print(f"  asset {name}: {'present' if (assets / name).exists() else 'MISSING'}")
        print(f"  datasets downloaded: {len(downloaded)}/{len(names)}")
        if missing:
            print(f"  NOT READY, {len(missing)} missing, e.g. {', '.join(missing[:5])}")
            return 1
        print(f"READY: would publish {len(names)} datasets to {args.repo}")
        return 0

    if missing:
        raise SystemExit(
            f"{len(missing)} datasets are not downloaded yet, refusing to publish a "
            f"partial archive: {', '.join(missing[:5])}..."
        )

    api = HfApi(token=read_token())
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    existing = set(api.list_repo_files(args.repo, repo_type="dataset"))

    # Metadata first, so a partially uploaded repo still explains what it is and
    # under what terms.
    for name, source in (
        ("README.md", assets / "README.md"),
        ("NOTICE", assets / "NOTICE"),
        ("licenses.json", assets / "licenses.json"),
        ("versions.json", data_dir / "versions.json"),
    ):
        if not source.exists():
            print(f"  skip {name}: {source} not found")
            continue
        api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=name,
            repo_id=args.repo,
            repo_type="dataset",
        )
        print(f"  uploaded {name}", flush=True)

    for index, name in enumerate(names, start=1):
        target = f"{name}.tar"
        if target in existing:
            print(f"[{index}/{len(names)}] {name}: already in repo", flush=True)
            continue
        source_dir = data_dir / name
        tar_path = data_dir / f".{name}.tar.staging"
        started = time.time()
        with tarfile.open(tar_path, "w") as tar:
            tar.add(source_dir, arcname=name)
        size = tar_path.stat().st_size
        api.upload_file(
            path_or_fileobj=str(tar_path),
            path_in_repo=target,
            repo_id=args.repo,
            repo_type="dataset",
        )
        tar_path.unlink()
        elapsed = time.time() - started
        print(
            f"[{index}/{len(names)}] {name}: {size / 1e6:.0f} MB in {elapsed:.0f}s "
            f"({size / elapsed / 1e6:.1f} MB/s)",
            flush=True,
        )

    print("PUBLISH_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

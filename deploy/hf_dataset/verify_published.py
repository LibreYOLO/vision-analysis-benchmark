"""Round-trip check: does what HuggingFace serves match what we uploaded?

The dataset audit proves the SOURCE tree is sound. This proves the published
copy is byte-identical to it, which is the separate failure mode: a truncated
upload, a corrupted tar, or a file that silently landed empty.

Downloads each requested tar from the Hub into a scratch dir, unpacks it, and
compares SHA-256 of every file against the local source tree. Cleans up as it
goes so it runs on a box with little free disk.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)).replace("\\", "/"): sha256(p)
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="LibreYOLO/rf100-vl")
    parser.add_argument("--source", default="/root/rf100-vl")
    parser.add_argument("--scratch", default="/root/verify")
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()

    source_root = Path(args.source)
    scratch = Path(args.scratch)
    failures = []

    for name in args.datasets:
        scratch.mkdir(parents=True, exist_ok=True)
        print(f"--- {name}", flush=True)
        try:
            tar_path = Path(hf_hub_download(
                repo_id=args.repo, filename=f"{name}.tar", repo_type="dataset",
                local_dir=str(scratch / "dl"),
            ))
            extract_to = scratch / "x"
            extract_to.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path) as tar:
                tar.extractall(extract_to)

            got = tree_hashes(extract_to / name)
            want = tree_hashes(source_root / name)

            missing = sorted(set(want) - set(got))
            extra = sorted(set(got) - set(want))
            differing = sorted(k for k in set(got) & set(want) if got[k] != want[k])

            if missing or extra or differing:
                failures.append(name)
                print(f"  MISMATCH files={len(want)} missing={len(missing)} "
                      f"extra={len(extra)} differing={len(differing)}")
                for k in (missing[:3] + extra[:3] + differing[:3]):
                    print(f"    {k}")
            else:
                print(f"  OK {len(want)} files, all SHA-256 identical", flush=True)
        except Exception as exc:
            failures.append(name)
            print(f"  ERROR {type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    print(f"\nverified {len(args.datasets)} datasets, {len(failures)} failed")
    if failures:
        print("FAILED:", ", ".join(failures))
        print("VERIFY_FAILED")
        return 1
    print("VERIFY_CLEAN")
    return 0


if __name__ == "__main__":
    sys.exit(main())

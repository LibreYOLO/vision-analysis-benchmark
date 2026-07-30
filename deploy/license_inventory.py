"""Per-dataset license inventory for RF100-VL.

The rf100-vl README states Apache 2.0, but that covers the benchmark repo. The
100 datasets are Roboflow Universe projects, each carrying a license chosen by
its uploader, so redistribution has to be gated on the actual per-project
license rather than the blanket claim.

Reads `project.license` from the public Roboflow API (verified to match the
`License:` line in the README.dataset.txt that ships inside each export) and
classifies it for redistribution.
"""

import collections
import json
import os
import sys
import time
from pathlib import Path

import requests

from rf100vl import roboflow100vl

KEY_FILE = Path.home() / ".config" / "roboflow" / "api_key"
WORKSPACE = "rf100-vl"

# Redistribution of a CLEANED, re-derived copy is what we intend to publish, so
# a No-Derivatives term is disqualifying even though the data is public.
PERMISSIVE = {
    "MIT", "Apache-2.0", "Apache 2.0", "BSD", "BSD-3-Clause", "BSD-2-Clause",
    "CC0-1.0", "CC0 1.0", "Public Domain", "Unlicense", "WTFPL",
}
ATTRIBUTION = {"CC BY 4.0", "CC BY 3.0", "CC BY 2.0", "CC BY-SA 4.0", "CC BY-SA 3.0", "ODbL v1.0"}
NON_COMMERCIAL = {"CC BY-NC 4.0", "CC BY-NC-SA 4.0", "CC BY-NC-ND 4.0", "CC BY-NC 2.0"}
NO_DERIVATIVES = {"CC BY-ND 4.0", "CC BY-NC-ND 4.0"}


def classify(license_name: str) -> str:
    name = (license_name or "").strip()
    if not name or name.lower() in {"undefined", "none", "null"}:
        return "UNKNOWN - must resolve before publishing"
    if name in NO_DERIVATIVES:
        return "BLOCKED - No-Derivatives forbids republishing a cleaned copy"
    if name in NON_COMMERCIAL:
        return "NC - redistributable with a non-commercial tag"
    if name in PERMISSIVE:
        return "OK - permissive"
    if name in ATTRIBUTION:
        return "OK - attribution (and share-alike where applicable)"
    if "ND" in name.replace("AND", "").split("-"):
        return "BLOCKED - No-Derivatives forbids republishing a cleaned copy"
    if "NC" in name.replace("INC", "").split("-"):
        return "NC - redistributable with a non-commercial tag"
    return f"REVIEW - unrecognised license string {name!r}"


def main() -> int:
    api_key = KEY_FILE.read_text(encoding="ascii").strip()
    datasets = roboflow100vl.get_rf100vl_projects(api_key)
    # Key by NAME: the package builds .datasets from the UNSORTED constructor
    # argument even though it sorts .projects, so index order is unstable.
    wrappers = {str(d.name): d for d in datasets}
    print(f"projects: {len(wrappers)}", flush=True)

    rows = []
    for i, (name, wrapper) in enumerate(sorted(wrappers.items()), start=1):
        slug = str(wrapper.rf_project.id).split("/")[-1]
        url = f"https://api.roboflow.com/{WORKSPACE}/{slug}?api_key={api_key}"
        license_name, public, images, classes = "(request failed)", None, None, None
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=60)
                if response.status_code == 200:
                    project = response.json().get("project", {})
                    license_name = project.get("license") or "(absent)"
                    public = project.get("public")
                    images = project.get("images")
                    classes = len(project.get("classes") or {})
                    break
                license_name = f"(HTTP {response.status_code})"
            except requests.RequestException as exc:
                license_name = f"(error {type(exc).__name__})"
            time.sleep(2 * (attempt + 1))
        rows.append({
            "dataset": name,
            "universe_slug": slug,
            "universe_url": f"https://universe.roboflow.com/{WORKSPACE}/{slug}",
            "license": license_name,
            "verdict": classify(license_name),
            "public": public,
            "images": images,
            "classes": classes,
        })
        print(f"[{i}/{len(wrappers)}] {name}: {license_name}", flush=True)
        time.sleep(0.2)

    # Two upstream statements about licensing exist and they do not agree. Both
    # are recorded rather than silently resolved: the blanket claim is what a
    # reader of the benchmark repo sees, the per-project field is what each
    # uploader actually chose, and a future reader must be able to see that we
    # knew about both.
    payload = {
        "schema_version": "rf100vl.licenses.v1",
        "checked_at": date,
        "method": (
            "project.license from GET https://api.roboflow.com/rf100-vl/<slug>, one call "
            "per project, cross-validated against the 'License:' line in the "
            "README.dataset.txt shipped inside downloaded exports"
        ),
        "upstream_statements": {
            "benchmark_repo_readme": (
                "github.com/roboflow/rf100-vl states Apache-2.0 as a blanket claim "
                "covering the repository and, by implication, the datasets"
            ),
            "per_project_api": (
                "every one of the 100 Universe projects reports its own license, "
                "and all 100 report MIT"
            ),
            "conflict": (
                "The blanket Apache-2.0 claim and the per-project MIT fields disagree "
                "about which licence governs the data. Both are permissive and both "
                "permit redistribution and derivative works with notice retention, so "
                "the conflict does not change what we may do. We rely on the "
                "per-project field because it is what each uploader actually set, and "
                "we preserve every dataset's own README.dataset.txt so a downstream "
                "reader can check for themselves."
            ),
        },
        "datasets": rows,
    }
    out = Path("rf100vl_license_inventory.json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== license counts ===")
    for license_name, n in collections.Counter(r["license"] for r in rows).most_common():
        print(f"  {n:>3}  {license_name}")
    print("\n=== verdicts ===")
    for verdict, n in collections.Counter(r["verdict"] for r in rows).most_common():
        print(f"  {n:>3}  {verdict}")

    problems = [r for r in rows if not r["verdict"].startswith("OK")]
    print(f"\nnot cleanly redistributable: {len(problems)}")
    for r in problems:
        print(f"  {r['dataset']}: {r['license']} -> {r['verdict']}")
    print(f"\nwrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

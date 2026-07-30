"""Generate the NOTICE file that ships with the redistributed archive.

MIT permits redistribution and derivatives only if the copyright notices travel
with the work. Roboflow's per-dataset README.dataset.txt stays inside each tar;
this adds a single top-level index so the provenance of all 100 is readable
without unpacking anything.
"""

import json
import sys
from pathlib import Path

HEADER = """RF100-VL, redistributed by the LibreYOLO project
================================================

This archive is a temporary redistribution of the RF100-VL benchmark by
Roboflow and Carnegie Mellon University. It is not an original work.

  Benchmark:  https://rf100-vl.org
  Repository: https://github.com/roboflow/rf100-vl
  Paper:      arXiv:2505.20612

The images and bounding boxes are unmodified. Category numbering was rewritten
by Roboflow's own `rf100vl` package during download (dummy class 0 removed,
category ids shifted to 0-based contiguous, annotation ids starting at 1),
which makes this copy a derivative work.

Licensing: two upstream statements exist and they do not agree. Both are
recorded here rather than silently resolved.

  1. The RF100-VL benchmark repository states Apache-2.0 as a blanket claim
     covering the repository and, by implication, the datasets.
  2. Every one of the 100 Roboflow Universe projects carries its own licence
     field set by its uploader, and all 100 of those report MIT.

Both are permissive, and both permit redistribution and derivative works
provided the notices travel with the work, so the disagreement does not change
what may be done with this archive. We rely on the per-project field because it
is what each uploader actually set. Each dataset's own README.dataset.txt is
preserved inside its archive and carries the upstream notice and credit; do not
remove it, and check it yourself if the distinction matters to you.

License inventory taken {date} against the versions pinned in versions.json.
Uploaders can change a licence at any time; this is a snapshot.

Datasets
--------
"""


def main() -> int:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    # Accept both the flat list emitted by earlier revisions and the current
    # object that also carries the upstream-statement record.
    licenses = payload["datasets"] if isinstance(payload, dict) else payload
    date = sys.argv[2] if len(sys.argv) > 2 else "2026-07-30"
    out = Path(sys.argv[3] if len(sys.argv) > 3 else "NOTICE")

    lines = [HEADER.format(date=date)]
    for record in sorted(licenses, key=lambda r: r["dataset"]):
        lines.append(
            f"{record['dataset']}\n"
            f"    license: {record['license']}\n"
            f"    source:  {record['universe_url']}\n"
        )
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} with {len(licenses)} datasets")
    return 0


if __name__ == "__main__":
    sys.exit(main())

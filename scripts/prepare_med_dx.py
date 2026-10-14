#!/usr/bin/env python
"""Install the med-dx corpus into sheldon's data root.

Point `--corpus` (or `MED_DX_CORPUS`) at a built corpus, so several checkouts
share one copy rather than each rebuilding it. Rebuilding is not free: synthesis is two LLM calls per case over
300 cases, and it will not reproduce byte-identically because this endpoint
rejects `temperature=0`, so the cached files are the environment.

The pipeline that produced them is in `scripts/med_dx/` -- fetching the three
datasets and Med-Evidence-2.6k, joining the multiple-choice options, labelling
`question_focus`, synthesising the records with EviDx's synthesiser plus a
leakage guard, and the audits that justify each step. See
`scripts/med_dx/README.md`.

This copies the finished `ehr-<split>.jsonl` files into `data/med-dx/`, so the
benchmark reads one directory and knows nothing about how it was made.

    ./scripts/prepare_med_dx.py                    # from the shared copy
    ./scripts/prepare_med_dx.py --source ~/med-eval --link
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

SPLITS = ("DiagnosisArena", "JAMA", "MedXpertQA-Diag")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default=os.environ.get("MED_DX_CORPUS", "med-dx-corpus"),
        help="where the built corpus lives (default: $MED_DX_CORPUS, else ./med-dx-corpus)",
    )
    ap.add_argument("--dest", default="", help="default: sheldon's data root / med-dx")
    ap.add_argument("--link", action="store_true",
                    help="symlink instead of copy, so a rebuilt corpus is picked up")
    a = ap.parse_args()

    from sheldon.core.io import data_root

    src = Path(a.source)
    dest = Path(a.dest) if a.dest else data_root() / "med-dx"
    dest.mkdir(parents=True, exist_ok=True)

    missing = [s for s in SPLITS if not (src / f"ehr-{s}.jsonl").exists()]
    if missing:
        raise SystemExit(
            f"missing in {src}: {', '.join('ehr-' + m + '.jsonl' for m in missing)}\n"
            f"build them first with scripts/med_dx/ (see its README), or point\n"
            f"--source at a directory that has them."
        )

    for split in SPLITS:
        s, d = src / f"ehr-{split}.jsonl", dest / f"ehr-{split}.jsonl"
        if d.exists() or d.is_symlink():
            d.unlink()
        if a.link:
            d.symlink_to(s.resolve())
        else:
            shutil.copy2(s, d)
        rows = [json.loads(line) for line in d.open(encoding="utf-8")]
        withehr = sum(1 for r in rows if r.get("ehr"))
        withopts = sum(1 for r in rows if r.get("options"))
        print(f"  {split:<18} {len(rows):>4} cases  {withehr} with records  "
              f"{withopts} with options  {'->' if a.link else '=='} {d}")

    print(f"\n  ready: {dest}")


if __name__ == "__main__":
    main()

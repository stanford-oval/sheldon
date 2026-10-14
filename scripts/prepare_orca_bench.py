#!/usr/bin/env python
"""Normalize an ORCA-bench release into data/orca-bench/.

Upstream ships tasks and a telemetry snapshot inside a Docker image rather than
as loose files, so this script has two jobs: pull the snapshot out, and reshape
the task records into the flat form the adapter reads.

    # if you already have the snapshot extracted somewhere
    scripts/prepare_orca_bench.py --input path/to/extracted --out data/orca-bench

    # otherwise pull it out of the published image first
    scripts/prepare_orca_bench.py --docker-image \\
        orcabench/sre-otel-snapshot:data-0418-harbor-template --out data/orca-bench

    # look before you map
    scripts/prepare_orca_bench.py --input path/to/extracted --inspect

Produces:

    data/orca-bench/public.jsonl
      {"id", "report", "difficulty", "root_cause", "symptoms": [str], "meta": {}}
    data/orca-bench/snapshot/{metrics,logs,traces,source}/...

The field mapping is not pinned — this repo has not seen the release. Run with
--inspect, then pass --map, then write the result into DEFAULT_MAP so nobody has
to pass it twice. Dataset is CC BY 4.0; see arXiv:2607.28545.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Read .jsonl / .json / a directory of either."""
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for child in sorted(path.iterdir()):
            if child.suffix in {".json", ".jsonl"}:
                rows.extend(load_rows(child))
        return rows
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    for key in ("data", "rows", "examples", "tasks"):
        if isinstance(obj.get(key), list):
            return obj[key]
    raise SystemExit(f"{path}: could not find a list of rows (keys: {list(obj)[:20]})")


def inspect(rows: list[dict[str, Any]]) -> None:
    print(f"{len(rows)} rows\n")
    freq = Counter(k for r in rows for k in r)
    print(f"{'field':<28} {'present':<9} example")
    for field, count in freq.most_common():
        sample = next((r[field] for r in rows if field in r), None)
        preview = json.dumps(sample, default=str)
        print(f"{field:<28} {count:<9} {preview[:90]}")

DEFAULT_MAP: dict[str, str] = {
    "id": "id",
    "report": "report",
    "difficulty": "difficulty",
    "root_cause": "root_cause",
    "symptoms": "symptoms",
}
OURS = tuple(DEFAULT_MAP)

#: Where each telemetry store tends to live inside the snapshot. Extend freely —
#: the adapter only cares about the four destination directory names.
STORE_HINTS = {
    "metrics": ["metrics", "prometheus", "promdata"],
    "logs": ["logs", "opensearch", "elasticsearch"],
    "traces": ["traces", "jaeger", "tempo"],
    "source": ["source", "src", "services", "repo"],
}


def extract_docker(image: str, dest: Path) -> Path:
    """Copy /app out of the published snapshot image into `dest`."""
    if shutil.which("docker") is None:
        raise SystemExit("docker not found on PATH; extract the snapshot manually and use --input")
    print(f"pulling {image} (this is several GB) ...")
    subprocess.run(["docker", "pull", image], check=True)
    cid = subprocess.run(
        ["docker", "create", image], check=True, capture_output=True, text=True
    ).stdout.strip()
    try:
        dest.mkdir(parents=True, exist_ok=True)
        print(f"copying /app -> {dest}")
        subprocess.run(["docker", "cp", f"{cid}:/app/.", str(dest)], check=True)
    finally:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
    return dest


def stage_snapshot(src: Path, out: Path) -> dict[str, int]:
    """Link each telemetry store into out/snapshot/<store>/."""
    counts: dict[str, int] = {}
    for store, hints in STORE_HINTS.items():
        target = out / "snapshot" / store
        target.mkdir(parents=True, exist_ok=True)
        found = 0
        for hint in hints:
            for candidate in src.rglob(hint):
                if not candidate.is_dir():
                    continue
                for path in candidate.rglob("*"):
                    if not path.is_file():
                        continue
                    link = target / f"{candidate.name}__{path.name}"
                    if not link.exists():
                        try:
                            link.symlink_to(path.resolve())
                        except OSError:
                            shutil.copy2(path, link)
                    found += 1
        counts[store] = found
    return counts


def find_task_rows(src: Path) -> list[dict[str, Any]]:
    """Best-effort hunt for the task records anywhere in the extract."""
    for pattern in ("**/tasks*.json*", "**/orca*bench*.json*", "**/*task*.json*"):
        for path in sorted(src.glob(pattern)):
            try:
                rows = load_rows(path)
            except SystemExit:
                continue
            if rows and isinstance(rows[0], dict):
                print(f"tasks from {path}")
                return rows
    return []


def normalize(rows: list[dict[str, Any]], mapping: dict[str, str]) -> list[dict[str, Any]]:
    src = {ours: mapping.get(ours, ours) for ours in OURS}
    out, skipped = [], 0
    for i, row in enumerate(rows):
        report = str(row.get(src["report"], "")).strip()
        if not report:
            skipped += 1
            continue
        symptoms = row.get(src["symptoms"]) or []
        if isinstance(symptoms, str):
            symptoms = [symptoms]
        out.append(
            {
                "id": str(row.get(src["id"], i)),
                "report": report,
                "difficulty": str(row.get(src["difficulty"], "")),
                "root_cause": str(row.get(src["root_cause"], "")).strip(),
                "symptoms": [str(s) for s in symptoms],
                "meta": {k: v for k, v in row.items() if k not in set(src.values())},
            }
        )
    if skipped:
        print(f"warning: skipped {skipped} rows with no report", file=sys.stderr)
    ungraded = sum(1 for r in out if not r["root_cause"] and not r["symptoms"])
    if ungraded:
        print(
            f"warning: {ungraded}/{len(out)} rows have no ground truth — they will always "
            "score 0. Check --map before running a sweep.",
            file=sys.stderr,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, help="already-extracted snapshot directory")
    ap.add_argument("--docker-image", help="snapshot image to extract first")
    ap.add_argument("--out", type=Path, default=Path("data/orca-bench"))
    ap.add_argument("--map", type=json.loads, default=None, help="JSON: our field -> theirs")
    ap.add_argument("--inspect", action="store_true", help="print the task schema and exit")
    ap.add_argument("--skip-snapshot", action="store_true", help="tasks only, no telemetry")
    args = ap.parse_args()

    if not args.input and not args.docker_image:
        raise SystemExit("pass --input or --docker-image")
    src = args.input or extract_docker(args.docker_image, args.out / ".src")

    rows = find_task_rows(src)
    if not rows:
        raise SystemExit(f"no task records found under {src}; point --input at the right directory")
    if args.inspect:
        inspect(rows)
        return 0

    records = normalize(rows, {**DEFAULT_MAP, **(args.map or {})})
    if not records:
        print("nothing usable — run with --inspect and fix --map", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "public.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in sorted(records, key=lambda r: r["id"]):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(records):>5} tasks -> {path}")

    if not args.skip_snapshot:
        counts = stage_snapshot(src, args.out)
        print("snapshot stores:", ", ".join(f"{k}={v}" for k, v in counts.items()))
        if not any(counts.values()):
            print(
                "warning: no telemetry staged. The probes will return nothing and a run "
                "will measure guessing, not RCA.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Copy SkyDiscover's benchmark tasks into data/skydiscover/ and index them.

    github.com/skydiscover-ai/skydiscover/tree/main/benchmarks

Upstream keeps ~200 tasks as directories, each an `initial_program.*`, a
`config.yaml` whose `prompt.system_message` states the problem, and an
evaluator: a Docker context (`evaluator/Dockerfile` + `evaluate.sh`) or a host
`evaluator.py`. Some multi-variant problems nest one level (`heilbronn_convex/13`).
The benchmark should not have to walk that tree, or know upstream's YAML, at
run time -- so this script walks it once and writes what `load()` reads:

    data/skydiscover/tasks.jsonl
      {"id": "math.heilbronn_convex.13", "family": "math", "task": "heilbronn_convex/13",
       "dir": "math/heilbronn_convex/13", "initial_program": "initial_program.py",
       "evaluator": "evaluator", "language": "python", "suffix": ".py",
       "timeout": 600, "system_message": "...", "source": "<commit>"}
    data/skydiscover/<family>/<task>/...      the task directory, copied verbatim

Task ids are the directory path with `/` replaced by `.`, because a trace is
written to `traces/<task_id>.jsonl`.

    scripts/prepare_skydiscover.py --download                # clone upstream, then prepare
    scripts/prepare_skydiscover.py --source /path/to/skydiscover
    scripts/prepare_skydiscover.py --download --families math,ADRS,qnn_circuit_topology
    scripts/prepare_skydiscover.py --download --datasets     # also run download_dataset.sh

The default families are the ones whose evaluators are self-contained CPU
images. `gpu_mode` and `kernelbench` need a GPU, `frontier-cs-eval` a judge
server, `ale_bench` the ale_bench package, `image_gen` and `prompt_optimization`
LLM-backed evaluators; name them with `--families` if you have the setup, and
the same index format carries them.

Two ADRS tasks (`cloudcast`, `llm_sql`) ship a `download_dataset.sh` that
fetches their data from Hugging Face, and `eplb` names a workload file in its
README; `--datasets` runs those. Without it those three tasks are copied and
listed but their evaluators score every program 0, which `sheldon check` cannot
tell from a real zero -- so run it.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sheldon.core.io import data_root  # noqa: E402

UPSTREAM = "https://github.com/skydiscover-ai/skydiscover.git"
DEFAULT_FAMILIES = ("math", "ADRS", "qnn_circuit_topology")

#: Data a task needs that is not in the checkout. Keyed by task dir relative to
#: `benchmarks/`; a script is run in its own directory, a URL is fetched into
#: the evaluator directory under the given name.
EXTRA_DATA: dict[str, dict[str, str]] = {
    "ADRS/eplb": {
        "url": "https://huggingface.co/datasets/abmfy/eplb-openevolve/resolve/main/expert-load.json",
        "into": "evaluator/expert-load.json",
    },
}

#: Requirement pins upstream leaves loose and a current resolver breaks on.
#: `ADRS/llm_sql` writes into a DataFrame slice in a way pandas 3.0 rejects
#: (`Invalid value ... for dtype 'int64'`), so every program scored 0 there
#: with an unpinned `pandas`; 2.x runs it. Keyed by task dir, value is
#: {loose requirement: pinned replacement}.
REQUIREMENT_PINS: dict[str, dict[str, str]] = {
    "ADRS/llm_sql": {"pandas": "pandas<3"},
}

SUFFIX = {"python": ".py", "cpp": ".cpp", "c++": ".cpp", "text": ".txt", "image": ".txt"}


def _clone(dest: Path) -> Path:
    print(f"cloning {UPSTREAM} -> {dest}", flush=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", UPSTREAM, str(dest)],
        check=True,
    )
    return dest


def _commit(source: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def _is_task(d: Path) -> bool:
    if not (d / "config.yaml").exists():
        return False
    if not any(d.glob("initial_program.*")):
        return False
    return (d / "evaluator").is_dir() or (d / "evaluator.py").exists()


def _find_tasks(bench_root: Path, family: str) -> list[Path]:
    fam = bench_root / family
    if not fam.is_dir():
        return []
    if _is_task(fam):
        return [fam]
    found: list[Path] = []
    for sub in sorted(p for p in fam.iterdir() if p.is_dir()):
        if _is_task(sub):
            found.append(sub)
            continue
        found.extend(p for p in sorted(sub.iterdir()) if p.is_dir() and _is_task(p))
    return found


def _read_config(task_dir: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load((task_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        print(f"  ! {task_dir}: config.yaml unreadable ({e})", file=sys.stderr)
        return {}


def _system_message(task_dir: Path, cfg: dict[str, Any]) -> str:
    msg = (cfg.get("prompt") or {}).get("system_message") or ""
    # Upstream lets this be a path to a file next to the config.
    if msg and "\n" not in msg and len(msg) < 200:
        try:
            if (task_dir / msg).is_file():
                return (task_dir / msg).read_text(encoding="utf-8")
        except OSError:
            pass
    return str(msg)


def _row(bench_root: Path, task_dir: Path, family: str, commit: str) -> dict[str, Any]:
    rel = task_dir.relative_to(bench_root)
    cfg = _read_config(task_dir)
    language = str(cfg.get("language", "python")).lower()
    initial = sorted(task_dir.glob("initial_program.*"))[0].name
    evaluator = "evaluator" if (task_dir / "evaluator").is_dir() else "evaluator.py"
    ev_cfg = cfg.get("evaluator") or {}
    return {
        "id": ".".join(rel.parts),
        "family": family,
        "task": "/".join(rel.parts[1:]) or rel.parts[0],
        "dir": str(rel),
        "initial_program": initial,
        "evaluator": evaluator,
        "language": language,
        "suffix": SUFFIX.get(language, ".py"),
        "timeout": float(ev_cfg.get("timeout") or 600),
        "system_message": _system_message(task_dir, cfg),
        "source": commit,
    }


def _copy_task(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src, dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "outputs", "output"),
    )


def _patch_evaluator(task_dir: Path) -> None:
    """Put `initial_program.py` inside a Docker context that imports it.

    `ADRS/llm_sql/evaluator/evaluator.py` does `from initial_program import
    Evolved` at module level, reaching one directory up -- which exists on a
    host checkout and not in the image, whose build context is `evaluator/`
    alone. The import is a stray: `evaluate()` scores the candidate it is
    handed, and the name it imports is only used by a helper nothing calls.
    Copying the file in makes the import resolve without touching scoring.
    """
    evaluator = task_dir / "evaluator"
    if not evaluator.is_dir():
        return
    initial = next(iter(task_dir.glob("initial_program.*")), None)
    if initial is None or (evaluator / initial.name).exists():
        return
    stem = initial.stem
    for py in evaluator.glob("*.py"):
        text = py.read_text(encoding="utf-8", errors="replace")
        if re.search(rf"^\s*(from|import)\s+{re.escape(stem)}\b", text, re.MULTILINE):
            print(f"  copying {initial.name} into evaluator/ ({py.name} imports it)")
            shutil.copy2(initial, evaluator / initial.name)
            return


def _json_stdout(task_dir: Path) -> None:
    """Make an evaluator that prints a dict repr print JSON instead.

    `qnn_circuit_topology/evaluator/evaluator.py` ends in `print(result)`, which
    writes `{'combined_score': 0.68, ...}` -- single quotes, not JSON. The
    container protocol parses stdout with `json.loads`, so upstream's own
    `container_evaluator` rejects every program there, records no
    `combined_score`, and EvoX searches blind: two 100-iteration runs finished
    with the seed as "best" at iteration 0. Only the printing changes; the dict
    is the same one, so a reader that already tolerated the repr scores alike.
    """
    for py in (task_dir / "evaluator").glob("*.py"):
        text = py.read_text(encoding="utf-8")
        fixed = re.sub(r"^(\s*)print\(result\)\s*$",
                       r"\1print(json.dumps(result, default=float))", text, flags=re.MULTILINE)
        if fixed == text:
            continue
        if not re.search(r"^import json\b", fixed, re.MULTILINE):
            fixed = re.sub(r"^(import \S+)$", r"import json\n\1", fixed, count=1, flags=re.MULTILINE)
        print(f"  {py.name}: printing JSON instead of a dict repr")
        py.write_text(fixed, encoding="utf-8")


def _pin_requirements(task_dir: Path, rel: str) -> None:
    pins = REQUIREMENT_PINS.get(rel)
    req = task_dir / "evaluator" / "requirements.txt"
    if not pins or not req.exists():
        return
    lines = req.read_text(encoding="utf-8").splitlines()
    out = [pins.get(line.strip(), line) for line in lines]
    if out != lines:
        print(f"  pinning requirements: {pins}")
        req.write_text("\n".join(out) + "\n", encoding="utf-8")


def _fetch_datasets(task_dir: Path, rel: str) -> None:
    for script in sorted(task_dir.rglob("download_dataset.sh")):
        print(f"  running {script.relative_to(task_dir)}", flush=True)
        subprocess.run(["bash", script.name], cwd=str(script.parent), check=True)
    extra = EXTRA_DATA.get(rel)
    if extra:
        target = task_dir / extra["into"]
        if not target.exists():
            print(f"  fetching {extra['url']} -> {extra['into']}", flush=True)
            subprocess.run(
                ["curl", "-sSL", "-o", str(target), extra["url"]], check=True,
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="clone upstream into a temp dir first")
    ap.add_argument("--source", type=Path, help="an existing skydiscover checkout")
    ap.add_argument("--out", type=Path, default=None, help="default: data/skydiscover")
    ap.add_argument("--families", default=",".join(DEFAULT_FAMILIES),
                    help="comma-separated family directories under benchmarks/")
    ap.add_argument("--datasets", action="store_true",
                    help="run each task's download_dataset.sh and fetch known extra files")
    args = ap.parse_args()

    out = args.out or (data_root() / "skydiscover")
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    tmp: tempfile.TemporaryDirectory[str] | None = None
    if args.source:
        source = args.source.resolve()
    elif args.download:
        tmp = tempfile.TemporaryDirectory(prefix="skydiscover-")
        source = _clone(Path(tmp.name) / "skydiscover")
    else:
        ap.error("give --download or --source")
        return 2

    bench_root = source / "benchmarks"
    if not bench_root.is_dir():
        print(f"no benchmarks/ under {source}", file=sys.stderr)
        return 1
    commit = _commit(source)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for family in families:
        tasks = _find_tasks(bench_root, family)
        if not tasks:
            print(f"{family}: no tasks found", file=sys.stderr)
            continue
        print(f"{family}: {len(tasks)} task(s)", flush=True)
        for task_dir in tasks:
            rel = task_dir.relative_to(bench_root)
            dst = out / rel
            _copy_task(task_dir, dst)
            _patch_evaluator(dst)
            _json_stdout(dst)
            _pin_requirements(dst, str(rel))
            if args.datasets:
                _fetch_datasets(dst, str(rel))
            rows.append(_row(bench_root, task_dir, family, commit))
            print(f"  {rows[-1]['id']}  [{rows[-1]['evaluator']}, {rows[-1]['language']}]")

    # Tasks another prepare script indexed here (`prepared_by`) are not this
    # script's to drop: it rewrites the file, and they are not in upstream's tree.
    index = out / "tasks.jsonl"
    if index.exists():
        mine = {r["id"] for r in rows}
        for line in index.read_text(encoding="utf-8").splitlines():
            kept = json.loads(line) if line.strip() else {}
            if kept.get("prepared_by") and kept["id"] not in mine:
                rows.append(kept)

    rows.sort(key=lambda r: r["id"])
    with open(out / "tasks.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"\nwrote {len(rows)} tasks to {out / 'tasks.jsonl'} (upstream {commit or 'unknown'})")
    if tmp:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())

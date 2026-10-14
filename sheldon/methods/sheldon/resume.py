"""Continue a finished run instead of starting one.

`-p resume_from=<run id or directory>` opens the round loop on the belief
state another run ended with, so a second $20 answers "what does more of this
buy" rather than "what does a different draw look like".

Why this file and not the LLM cache: replay is not continuation. A cached call
is charged what it cost, so a run with a raised cap does arrive at the old
stopping point having spent the old budget -- but only while every request
matches byte for byte, and the concurrent local loops break that inside round
one. A probe result that carries a shared counter reads differently depending
on which loop got there first, so the text fed back differs from last time, the
next request misses, and a conversation that misses its prefix once never hits
again.

Four things have to come back or the continuation is not one:

  state        the hypotheses, observations, regions and per-round history,
               from `predictions.jsonl`'s `belief` -- the snapshot is already
               written there for every task that finished
  round        the round counter, so the loop opens at 16 and the prompts say
               so; `rounds` is then how many *more* to run
  the oracle   `Benchmark.probe_state`: the objective's call budget continues
               where it stopped, not at zero
  retrievals   the record and its files. Observations cite `r285`; without
               the entry behind it a resumed stage reads a citation of nothing

What deliberately does not come back is the money. `Budget` starts fresh,
because a fresh cap is the whole point -- the first run's spend is already
recorded in its own summary.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from sheldon.core.io import runs_root
from sheldon.core.probes import ProbeResult, Retrieval
from sheldon.methods.sheldon.run import Run
from sheldon.methods.sheldon.state import State


def _run_dir(spec: str) -> Path:
    """A path, a run id, or a tagged prefix -- the last of which is how a
    per-task run is named (`...__<tag>__<hash>`)."""
    p = Path(spec)
    if p.is_dir():
        return p
    root = runs_root()
    direct = root / spec
    if direct.is_dir():
        return direct
    matches = sorted(root.glob(f"*{spec}*"))
    if not matches:
        raise FileNotFoundError(f"resume_from: no run directory matches {spec!r}")
    if len(matches) > 1:
        raise ValueError(f"resume_from: {spec!r} matches {len(matches)} runs: {[m.name for m in matches]}")
    return matches[0]


def _prediction(run_dir: Path, task_id: str) -> dict[str, Any]:
    path = run_dir / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"resume_from: {path} does not exist")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task_id") == task_id:
            if row.get("error"):
                # A row with an error never answered, so there is no state
                # behind it -- resuming onto one would continue a run that
                # did not happen.
                raise ValueError(f"resume_from: {task_id} ended in an error: {row['error']}")
            return row
    raise KeyError(f"resume_from: {run_dir.name} has no row for task {task_id}")


def _retrieval_files(run_dir: Path, task_id: str) -> list[Path]:
    d = run_dir / "retrievals" / task_id
    if not d.is_dir():
        return []
    # r1, r2, ... r1959: numeric order, not the lexicographic order glob gives,
    # or the record is rebuilt with its ids shuffled and `r285` names a
    # different lookup than the observation citing it meant.
    def key(p: Path) -> tuple[int, str]:
        stem = p.stem
        return (int(stem[1:]), stem) if stem[1:].isdigit() else (1 << 30, stem)

    return sorted(d.glob("r*.json"), key=key)


def _restore_retrievals(run: Run, run_dir: Path) -> int:
    """Rebuild `ProbeSet.record` from the files the prior run wrote, and copy
    them into this run's directory so a shell can still read them back."""
    tools = run.ctx.tools
    if tools is None:
        return 0
    files = _retrieval_files(run_dir, run.task.id)
    if not files:
        return 0
    target = getattr(tools, "record_dir", None)
    for path in files:
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        probe = tools.probes.get(entry.get("tool", ""))
        if probe is None:
            # A probe this run does not offer: keep the slot so the numbering
            # stays aligned with the observations' citations, and let the file
            # carry the content.
            probe = next(iter(tools.probes.values()), None)
            if probe is None:
                break
        result = ProbeResult(
            summary=entry.get("summary", ""),
            ok=bool(entry.get("ok", True)),
            evidence=list(entry.get("evidence") or []),
        )
        tools.record.append(Retrieval(entry.get("id", f"r{len(tools.record) + 1}"),
                                      probe, dict(entry.get("arguments") or {}), result))
        if target is not None:
            try:
                Path(target).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, Path(target) / path.name)
            except OSError:
                target = None  # one directory's worth of noise, not one per file
    return len(files)


def _measured(run_dir: Path, task_id: str, probe: str, key: str) -> dict[str, dict[str, Any]]:
    """What the objective probe measured, read back out of the retrievals.

    The prior run never wrote `probe_state` -- it had no reason to -- so the
    per-call files are where its measurements live. Every objective-probe
    entry carries the whole result as evidence, which is exactly what
    `_measured` holds.
    """
    out: dict[str, dict[str, Any]] = {}
    for path in _retrieval_files(run_dir, task_id):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if entry.get("tool") != probe or not entry.get("ok", True):
            continue
        for item in entry.get("evidence") or []:
            if isinstance(item, dict) and item.get(key) and not item.get("error"):
                out[str(item[key])] = item
    return out


def apply(run: Run) -> int:
    """Restore `resume_from` onto this run. Returns the round it left off at."""
    spec = str(run.param("resume_from", "") or "").strip()
    if not spec:
        return 0
    run_dir = _run_dir(spec)
    row = _prediction(run_dir, run.task.id)
    snapshot = row.get("belief") or {}
    if not snapshot.get("hypotheses"):
        raise ValueError(f"resume_from: {run_dir.name} has no belief state for {run.task.id}")
    run.state = State.restore(snapshot)
    rounds = int(row.get("rounds") or snapshot.get("round") or len(run.state.best_by_round))

    benchmark = run.ctx.benchmark
    probe = str(getattr(benchmark, "objective_probe", "") or "")
    key = "smiles" if probe == "dock_molecule" else ""
    measured: dict[str, dict[str, Any]] = {}
    if probe and key and hasattr(benchmark, "restore_probe_state"):
        measured = _measured(run_dir, run.task.id, probe, key)
        benchmark.restore_probe_state(run.task, measured)
    n_retrievals = _restore_retrievals(run, run_dir)

    run.resumed = rounds
    run.ctx.trace.log(
        "resume",
        run_dir=str(run_dir),
        round=rounds,
        hypotheses=len(run.state.hypotheses),
        observations=len(run.state.observations),
        best=run.state.best(),
        measured=len(measured),
        retrievals=n_retrievals,
        prior_cost_usd=round(float((row.get("usage") or {}).get("cost_usd") or 0.0), 4),
    )
    return rounds

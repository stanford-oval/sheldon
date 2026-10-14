"""End-to-end offline smoke test.

Builds three synthetic ARC tasks on disk, runs the full `sheldon` loop
against a scripted `FakeLLM`, and asserts the pipeline solved them. No API key,
no network, a few seconds. Run it before every real run -- a wiring bug caught
here is free, and the same bug caught in a live sweep is not.

The script drives every stage the loop has, not only the happy path. The
scripted model opens with two wrong hypotheses. The local loop builds each one
as a file, runs it on the demonstrations, and files what it showed. Revise
shelves one and refines the other into the right rule; round 2 tests the
refinement; round 3 has nothing new to test, so the loop stops on its own rule
rather than the round cap; finalize submits the best artifact's path, which is
realized into the answer. Both modes run it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from sheldon.core.config import Config
from sheldon.core.llm import FakeLLM
from sheldon.methods.sheldon.schemas import (
    SUBMIT_SCHEMAS,
    EvidenceOut,
    ExploreOut,
    FinalOut,
    HypothesisDraft,
    LocalOut,
    MergeDraft,
    MergeOut,
    ObservationDraft,
    PlanOut,
    RegionDraft,
    ReviseOp,
    ReviseOut,
    SelectOut,
    VerdictDraft,
)
from sheldon.runner import run_experiment

# The hidden rule the synthetic tasks follow, and the program that implements it.
PROGRAM = "def transform(grid):\n    return grid[::-1]\n"
REVISED_STATEMENT = "Rows are reversed top to bottom."
#: What the scripted model opens with: two wrong accounts, so the smoke drives a
#: real test -> revise cycle rather than only the happy path.
WRONG_PROGRAM = "def transform(grid):\n    return grid\n"
WRONG_STATEMENT = "The grid is returned unchanged."
#: Wrong along the other axis: mirrored, but left-to-right. Refining it into
#: the right rule is a genuine repair, not a reword.
MIRROR_PROGRAM = "def transform(grid):\n    return [row[::-1] for row in grid]\n"
MIRROR_STATEMENT = "Each row is reversed left to right."
PROGRAMS = {
    WRONG_STATEMENT: WRONG_PROGRAM,
    MIRROR_STATEMENT: MIRROR_PROGRAM,
    REVISED_STATEMENT: PROGRAM,
}
_PAIRS = [
    ([[1, 2], [3, 4]], [[3, 4], [1, 2]]),
    ([[5, 6], [7, 8]], [[7, 8], [5, 6]]),
    ([[1, 0], [0, 1]], [[0, 1], [1, 0]]),
    ([[9, 9], [1, 2]], [[1, 2], [9, 9]]),
]


def write_synthetic_arc(root: Path) -> Path:
    """Three tasks whose rule is 'reverse the row order'."""
    d = root / "arc-agi-1" / "evaluation"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        demos = [{"input": a, "output": b} for a, b in _PAIRS[: 3 + (i % 2)]]
        # three rows, so a test grid can never coincide with a two-row demo pair
        test_in = [[i + 1, i + 2], [i + 3, i + 4], [i + 5, i + 6]]
        payload = {
            "train": demos,
            "test": [{"input": test_in, "output": test_in[::-1]}],
        }
        (d / f"synthetic_{i}.json").write_text(json.dumps(payload), encoding="utf-8")
    return root / "arc-agi-1"


_HYP = re.compile(r"^- \[(h\d+)\] (.+?)  \((pending|tested|shelved)", re.M)
_LOCAL = re.compile(r"^LOCAL (h\d+) (\S+): reproduces (\d+) of (\d+)", re.M)
_ARTIFACT = re.compile(r"^  - artifact: (\S+)$", re.M)
_STATEMENT = re.compile(r"^---\n(.*?)\n---$", re.M | re.S)
_DECIDE = re.compile(r"Decide every one of: (.+?)\.$", re.M)
#: A session turn names hypotheses by id and does not repeat what the
#: conversation already holds; a real model remembers, this one keeps a table.
#: Every task in the smoke opens the same way, so ids mean the same claim.
_SEEN: dict[str, str] = {}


def _path(hid: str, tag: str) -> str:
    return str(Path(tempfile.gettempdir()) / f"sheldon-hd-smoke-{tag}-{hid}.py")


def _snippet(hid: str, statement: str, path: str) -> str:
    program = PROGRAMS[statement]
    return "\n".join([
        f"open({path!r}, 'w').write({program!r})",
        "ns = {}",
        f"exec(open({path!r}).read(), ns)",
        "ok = [ns['transform'](d['input']) == d['output'] for d in demos]",
        f"print('LOCAL {hid} {path}: reproduces %d of %d' % (sum(ok), len(ok)))",
    ])


def scripted(prompt: str, schema: Any) -> Any:
    _SEEN.update({hid: st for hid, st, _s in _HYP.findall(prompt)})
    if schema is PlanOut:
        m = _STATEMENT.search(prompt)
        return PlanOut(situation=m.group(1) if m else "")
    if schema is MergeOut:
        return MergeOut(merges=[MergeDraft(space="h", members=["flip", "identity"], name="symmetry",
                                           axis="geometry", description="any reflection or none",
                                           why="one of them reproduces no pair")])
    if schema is ExploreOut:
        if "**Current hypotheses**" in prompt:
            return ExploreOut()
        return ExploreOut(
            hypothesis_regions=[RegionDraft(axis="geometry", name="flip", description="a reflection"),
                                RegionDraft(axis="geometry", name="identity", description="no change")],
            observation_regions=[RegionDraft(axis="evidence", name="candidate runs",
                                             description="a candidate run on the given pairs")],
            hypotheses=[HypothesisDraft(statement=WRONG_STATEMENT, rationale="simplest", region="identity"),
                        HypothesisDraft(statement=MIRROR_STATEMENT, rationale="mirrored", region="flip")],
        )
    if schema is LocalOut:
        m = _LOCAL.search(prompt)
        if not m:
            return LocalOut(notes="nothing was run")
        hid, path, got, total = m.groups()
        rel = "supports" if got == total else "refutes"
        return LocalOut(
            file_path=path, notes=f"reproduces {got} of {total}", done=True,
            observations=[ObservationDraft(
                content=f"The artifact of {hid} reproduces {got} of {total} given pairs.",
                region="candidate runs", verdicts=[VerdictDraft(hypothesis=hid, relation=rel, why="run")],
            )],
        )
    if schema is EvidenceOut:
        return EvidenceOut(summary="the local runs already separate them")
    if schema is SelectOut:
        return SelectOut(chosen=[hid for hid, _s, st in _HYP.findall(prompt) if st == "pending"])
    if schema is ReviseOut:
        ops = []
        m = _DECIDE.search(prompt)
        if m:
            pairs = [(hid.strip(), _SEEN.get(hid.strip(), "")) for hid in m.group(1).split(",")]
        else:
            pairs = [(hid, st) for hid, st, status in _HYP.findall(prompt) if status == "tested"]
        for hid, statement in pairs:
            if statement == MIRROR_STATEMENT:
                ops.append(ReviseOp(hypothesis=hid, op="refine", statement=REVISED_STATEMENT,
                                    rationale="mirrored along the other axis"))
            elif statement == WRONG_STATEMENT:
                ops.append(ReviseOp(hypothesis=hid, op="shelve", rationale="reproduces no pair"))
            else:
                ops.append(ReviseOp(hypothesis=hid, op="keep", rationale="reproduces every pair"))
        return ReviseOut(ops=ops)
    if schema is FinalOut:
        m = _ARTIFACT.search(prompt.split("**Hypotheses, best first**", 1)[-1])
        return FinalOut(file_path=m.group(1) if m else "", confidence=0.9)
    return "ok"


def scripted_tools_for(tag: str):
    """The local stage runs its candidate first, then every stage submits.
    `tag` keeps concurrent runs' artifact files apart."""
    def responder(prompt: str, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
        names = {t.get("name") for t in tools}
        if "submit_local" in names and "run_python" in names and "**Hypothesis under test**" in prompt:
            hyps = [(h, s) for h, s, _st in _HYP.findall(prompt) if s in PROGRAMS]
            if hyps:
                hid, statement = hyps[0]
                return {"name": "run_python", "input": {"code": _snippet(hid, statement, _path(hid, tag))}}
        for name, schema in SUBMIT_SCHEMAS.items():
            if name in names:
                return {"name": name, "input": scripted(prompt, schema).model_dump()}
        return None
    return responder


#: A ready-made tool responder for tests that need only one run at a time.
scripted_tools = scripted_tools_for(f"shared-{os.getpid()}")


async def run(mode: str = "staged", verbose: bool = False) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        write_synthetic_arc(Path(tmp))
        cfg = Config(
            benchmark="arc-agi-1", method="sheldon", split="eval",
            max_rounds=5, max_calls=None, max_usd=None, concurrency=3,
            cache=False, resume=False, tag=f"smoke-{mode}",
            params={"mode": mode, "max_regions": 2,
                    "benchmark_opts": {"root": str(Path(tmp) / "arc-agi-1"), "allow_code_exec": True}},
        )
        os.environ["SHELDON_RUNS"] = str(Path(tmp) / "runs")
        try:
            llm = FakeLLM(responder=scripted, tool_responder=scripted_tools_for(f"{mode}-{os.getpid()}"))
            summary = await run_experiment(cfg, llm=llm, echo=verbose)
            summary["max_rounds"] = cfg.max_rounds
            # The hypothesis map opens with two regions and the cap is two, so
            # every task's round 2 must have merged them before exploring.
            traces = list((Path(tmp) / "runs").glob("*/traces/*.jsonl"))
            summary["merged"] = sum(
                1 for t in traces for line in t.read_text(encoding="utf-8").splitlines()
                if '"merge.done"' in line and '"merged": ["h: flip + identity -> symmetry' in line
            )
            return summary
        finally:
            os.environ.pop("SHELDON_RUNS", None)


def main(verbose: bool = False) -> int:
    ok = True
    for mode in ("staged", "session"):
        hd = asyncio.run(run(mode, verbose))
        # Round 3 has nothing new to test: stopping there is the loop's own rule.
        hd_ok = (hd["n"] == 3 and hd["score"] == 1.0 and hd["errors"] == 0 and hd["mean_rounds"] == 3
                 and hd["merged"] == 3)
        print(f"[{'PASS' if hd_ok else 'FAIL'}] smoke sheldon ({mode}): n={hd['n']} "
              f"score={hd['score']} errors={hd['errors']} rounds={hd['mean_rounds']} merged={hd['merged']}")
        if not hd_ok:
            print(json.dumps(hd, indent=2, default=str))
        ok = ok and hd_ok
    return 0 if ok else 1

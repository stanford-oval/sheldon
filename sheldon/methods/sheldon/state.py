"""The run's state: hypotheses, observations, and the regions both sit in.

Deliberately small. A hypothesis is a claim, the
reason it was proposed, where it sits, and what testing it produced; an
observation is something another reader could check, with a verdict per
hypothesis it was gathered to tell apart. Nothing here calls a model.

A hypothesis is in one of three states:

    pending   waiting for a local test: new, refined, repaired or revived
    tested    tested locally, and live -- kept rounds carry it without retest
    shelved   set aside; recoverable, never deleted

A refinement is a sibling with `parent` set, never an edit: the trace must show
every claim in the form it was actually tested.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

PENDING, TESTED, SHELVED = "pending", "tested", "shelved"
RELATIONS = ("supports", "refutes", "neutral")


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


@dataclass
class Region:
    axis: str
    name: str
    description: str = ""
    #: the regions this one was merged from, and the round it was made
    merged_from: list[str] = field(default_factory=list)
    round: int = 0


@dataclass
class Hypothesis:
    id: str
    statement: str
    rationale: str = ""
    region: str = ""
    born: int = 0
    parent: str = ""
    status: str = PENDING
    #: the last round anything happened to it: born, tested, shelved, revived
    touched: int = 0
    #: why it was shelved, or what revise asked a repair to fix
    reason: str = ""
    repair: str = ""
    #: the best artifact found by the local loop, and every score it measured
    artifact: str = ""
    scores: list[float] = field(default_factory=list)
    #: every number the objective reported for the best-scoring artifact: the
    #: combined score hides which parts of the task a claim wins on
    metrics: dict[str, float] = field(default_factory=dict)
    #: what the local loop reported, in its own words
    notes: str = ""

    @property
    def best(self) -> float | None:
        return max(self.scores) if self.scores else None


@dataclass
class Observation:
    id: str
    content: str
    region: str = ""
    sources: list[str] = field(default_factory=list)
    round: int = 0
    origin: str = ""
    #: hypothesis id -> {"relation", "why"}
    verdicts: dict[str, dict[str, str]] = field(default_factory=dict)


class State:
    def __init__(self) -> None:
        self.hypotheses: dict[str, Hypothesis] = {}
        self.observations: dict[str, Observation] = {}
        #: region name -> Region; one namespace per space
        self.h_regions: dict[str, Region] = {}
        self.o_regions: dict[str, Region] = {}
        #: regions per space, or None for no cap; set by the stage from its param
        self.region_limit: int | None = None
        #: space -> {normalized retired name: the live region it merged into}
        self.aliases: dict[str, dict[str, str]] = {"h": {}, "o": {}}
        #: every add, merge and refusal, in order: the map's history
        self.region_log: list[dict[str, Any]] = []
        #: best score seen at the end of each round, for the stagnation rule
        self.best_by_round: list[float | None] = []
        self.obs_by_round: list[int] = []
        #: the task's starting program and what it scored, measured before round 1
        self.seed_path: str = ""
        self.seed_score: float | None = None
        #: round -> ids of the hypotheses whose local test ended in it
        self.tested_by_round: dict[int, list[str]] = {}

    # -- regions ---------------------------------------------------------------

    def regions(self, space: str) -> dict[str, Region]:
        return self.h_regions if space == "h" else self.o_regions

    def full(self, space: str) -> bool:
        return self.region_limit is not None and len(self.regions(space)) >= self.region_limit

    def add_region(self, space: str, axis: str, name: str, description: str = "",
                   round_i: int = 0) -> str:
        """Add a region. Returns `added`, `folded` (one of that name exists --
        the name is the identity, a near-duplicate folds), `full` (the map is
        at `region_limit`; only a merge makes room), or `empty`."""
        if not name.strip():
            return "empty"
        if self.resolve_region(space, name) in self.regions(space):
            return "folded"
        if self.full(space):
            self.region_log.append({"round": round_i, "op": "refuse", "space": space, "name": name.strip()})
            return "full"
        self.regions(space)[name.strip()] = Region(axis.strip() or "general", name.strip(),
                                                   description.strip(), round=round_i)
        self.region_log.append({"round": round_i, "op": "add", "space": space, "name": name.strip()})
        return "added"

    def merge_regions(self, space: str, members: list[str], name: str, axis: str,
                      description: str, why: str, round_i: int) -> tuple[bool, str]:
        """Fold two or more regions into one, moving every hypothesis or
        observation in them. The merged region records what it came from, and
        the retired names stay as aliases, so a later draft naming one lands
        here rather than recreating it."""
        regions = self.regions(space)
        live = list(dict.fromkeys(self.resolve_region(space, m) for m in members if m.strip()))
        missing = [m for m in live if m not in regions]
        if missing:
            return False, f"not on the map: {', '.join(missing)}"
        name = name.strip()
        if not name:
            return False, "the merged region needs a name"
        target = self.resolve_region(space, name)
        if target in regions and target not in live:
            live.append(target)  # an existing target absorbs the members
        if len(live) < 2:
            return False, "a merge needs two or more distinct regions"
        name = target if target in regions else name
        origin: list[str] = []
        for m in live:
            origin += regions[m].merged_from or [m]
        merged = Region(axis.strip() or regions[live[0]].axis, name, description.strip(),
                        merged_from=list(dict.fromkeys(origin)), round=round_i)
        for m in live:
            if m != name:
                del regions[m]
        regions[name] = merged
        items = self.hypotheses.values() if space == "h" else self.observations.values()
        moved = 0
        for item in items:
            if item.region in live and item.region != name:
                item.region, moved = name, moved + 1
        aliases = self.aliases.setdefault(space, {})
        for old, new in list(aliases.items()):
            if new in live:
                aliases[old] = name
        for m in live:
            if m != name:
                aliases[_key(m)] = name
        self.region_log.append({"round": round_i, "op": "merge", "space": space, "name": name,
                                "members": live, "moved": moved, "why": why.strip()})
        return True, f"{' + '.join(live)} -> {name} ({moved} moved)"

    def replace_axis(self, space: str, old: str, new: str) -> int:
        """A superset axis absorbs an old one: its regions move under it."""
        moved = 0
        for r in self.regions(space).values():
            if _key(r.axis) == _key(old):
                r.axis = new
                moved += 1
        return moved

    def resolve_region(self, space: str, name: str) -> str:
        """The live region a name refers to: itself, a case-insensitive match,
        or what it was merged into. An unknown name comes back as given."""
        k = _key(name)
        seen: set[str] = set()
        while k in self.aliases.get(space, {}) and k not in seen:
            seen.add(k)
            k = _key(self.aliases[space][k])
        for r in self.regions(space):
            if _key(r) == k:
                return r
        return name.strip()

    def place(self, space: str, name: str, round_i: int) -> str:
        """The region an item sits in: an existing one, else a new one under
        `general` while the map has room, else nothing -- an undeclared name
        must not open a region past the cap."""
        if not name.strip():
            return ""
        live = self.resolve_region(space, name)
        if live in self.regions(space):
            return live
        return name.strip() if self.add_region(space, "general", name, "", round_i) == "added" else ""

    def unplaced(self, space: str) -> list[str]:
        """Region names refused at the cap and never since merged or added."""
        wanted = [e["name"] for e in self.region_log if e["op"] == "refuse" and e["space"] == space]
        return [n for n in dict.fromkeys(wanted) if self.resolve_region(space, n) not in self.regions(space)]

    # -- hypotheses ------------------------------------------------------------

    def _next(self, prefix: str, items: dict[str, Any]) -> str:
        return f"{prefix}{len(items) + 1}"

    def find(self, statement: str) -> Hypothesis | None:
        k = _key(statement)
        return next((h for h in self.hypotheses.values() if _key(h.statement) == k), None)

    def add(self, statement: str, rationale: str, region: str, round_i: int, parent: str = "") -> Hypothesis | None:
        """A new pending hypothesis, or None when the same claim is already held."""
        if not statement.strip() or self.find(statement) is not None:
            return None
        h = Hypothesis(
            id=self._next("h", self.hypotheses), statement=statement.strip(),
            rationale=rationale.strip(), region=self.place("h", region, round_i),
            born=round_i, parent=parent, touched=round_i,
        )
        self.hypotheses[h.id] = h
        return h

    def refine(self, h: Hypothesis, statement: str, rationale: str, round_i: int) -> Hypothesis | None:
        """A sibling that inherits the region; the parent is left as it was."""
        return self.add(statement, rationale, h.region, round_i, parent=h.id)

    def shelve(self, h: Hypothesis, reason: str, round_i: int) -> None:
        h.status, h.reason, h.touched = SHELVED, reason, round_i

    def revive(self, h: Hypothesis, round_i: int) -> None:
        h.status, h.touched = PENDING, round_i

    def tested(self, h: Hypothesis, round_i: int) -> None:
        h.status, h.touched, h.repair = TESTED, round_i, ""
        ids = self.tested_by_round.setdefault(round_i, [])
        if h.id not in ids:
            ids.append(h.id)

    def request_repair(self, h: Hypothesis, reason: str, round_i: int) -> None:
        h.status, h.repair, h.touched = PENDING, reason, round_i

    def of(self, status: str) -> list[Hypothesis]:
        return [h for h in self.hypotheses.values() if h.status == status]

    def eligible(self, round_i: int, cooldown: int, stagnating: bool) -> list[Hypothesis]:
        """Shelved hypotheses explore may revive: untouched for `cooldown`
        rounds, or any of them once the run has stopped making progress."""
        return [h for h in self.of(SHELVED) if stagnating or round_i - h.touched >= cooldown]

    def ranked(self) -> list[Hypothesis]:
        """Every hypothesis ever tested, best first: score, then evidence."""
        def key(h: Hypothesis) -> tuple[float, int]:
            sup, ref = self.counts(h.id)
            return (h.best if h.best is not None else float("-inf"), sup - ref)
        tested = [h for h in self.hypotheses.values() if h.status != PENDING or h.scores]
        return sorted(tested, key=key, reverse=True)

    def lineage_best(self, h: Hypothesis) -> float | None:
        """The best score of `h` or any hypothesis it was refined from: what a
        revision has to beat to count as progress."""
        values, cur, seen = [], h, set()
        while cur is not None and cur.id not in seen:
            seen.add(cur.id)
            if cur.best is not None:
                values.append(cur.best)
            cur = self.hypotheses.get(cur.parent)
        return max(values) if values else None

    def lineage_artifact(self, h: Hypothesis) -> tuple[str, str]:
        """The artifact a hypothesis starts from: its own, else the nearest
        ancestor's it was refined from, else nothing. With the id it came from."""
        cur, seen = h, set()
        while cur is not None and cur.id not in seen:
            seen.add(cur.id)
            if cur.artifact:
                return cur.artifact, cur.id
            cur = self.hypotheses.get(cur.parent)
        return "", ""

    def leader(self) -> Hypothesis | None:
        """The hypothesis holding the run's best score, if any has one."""
        scored = [h for h in self.hypotheses.values() if h.best is not None and h.artifact]
        return max(scored, key=lambda h: h.best) if scored else None

    def best(self) -> float | None:
        values = [h.best for h in self.hypotheses.values() if h.best is not None]
        return max(values) if values else None

    # -- observations ----------------------------------------------------------

    def observe(
        self, content: str, region: str, sources: list[str], round_i: int, origin: str,
        verdicts: list[dict[str, str]] | None = None,
    ) -> Observation | None:
        """File an observation; an exact repeat folds into the existing one."""
        if not content.strip():
            return None
        k = _key(content)
        o = next((x for x in self.observations.values() if _key(x.content) == k), None)
        if o is None:
            o = Observation(
                id=self._next("o", self.observations), content=content.strip(),
                region=self.place("o", region, round_i), round=round_i, origin=origin,
            )
            self.observations[o.id] = o
        o.sources = list(dict.fromkeys([*o.sources, *sources]))
        for v in verdicts or []:
            hid, rel = v.get("hypothesis", ""), v.get("relation", "")
            if hid in self.hypotheses and rel in RELATIONS:
                o.verdicts[hid] = {"relation": rel, "why": v.get("why", "")}
        return o

    def counts(self, hid: str) -> tuple[int, int]:
        sup = sum(1 for o in self.observations.values() if o.verdicts.get(hid, {}).get("relation") == "supports")
        ref = sum(1 for o in self.observations.values() if o.verdicts.get(hid, {}).get("relation") == "refutes")
        return sup, ref

    # -- progress --------------------------------------------------------------

    def close_round(self) -> None:
        self.best_by_round.append(self.best())
        self.obs_by_round.append(len(self.observations))

    def stagnating(self, window: int) -> bool:
        """No new best score and no new observation over the last `window`
        rounds. What unlocks the shelf early."""
        if window <= 0 or len(self.best_by_round) <= window:
            return False
        then_best, now_best = self.best_by_round[-window - 1], self.best_by_round[-1]
        better = now_best is not None and (then_best is None or now_best > then_best)
        grew = self.obs_by_round[-1] > self.obs_by_round[-window - 1]
        return not better and not grew

    # -- rendering -------------------------------------------------------------

    def render_regions(self, space: str) -> str:
        regions = self.regions(space)
        if not regions:
            return "(no regions declared)"
        by_axis: dict[str, list[Region]] = {}
        for r in regions.values():
            by_axis.setdefault(r.axis, []).append(r)
        lines = [self.render_fill(space)]
        for axis, rs in by_axis.items():
            lines.append(f"- axis `{axis}`  [{self._axis_stats(space, rs)}]")
            for r in rs:
                origin = (f"  (merged from {', '.join(f'`{m}`' for m in r.merged_from)} in round {r.round})"
                          if r.merged_from else "")
                lines.append(f"  - `{r.name}`: {r.description}  [{self._region_stats(space, r.name)}]{origin}")
        unplaced = self.unplaced(space)
        if unplaced:
            lines.append("- refused at the cap, their items unplaced: " + ", ".join(f"`{n}`" for n in unplaced))
        return "\n".join(lines)

    def render_fill(self, space: str) -> str:
        n = len(self.regions(space))
        if self.region_limit is None:
            return f"{n} regions"
        state = "FULL: a new region needs a merge first" if n >= self.region_limit else "room for more"
        return f"{n} of {self.region_limit} regions ({state})"

    def _axis_stats(self, space: str, regions: list[Region]) -> str:
        names = {r.name for r in regions}
        if space == "o":
            n = sum(1 for o in self.observations.values() if o.region in names)
            return f"{len(regions)} regions, {n} observations"
        hs = [h for h in self.hypotheses.values() if h.region in names]
        tested = [h for h in hs if h.status != PENDING or h.scores]
        return f"{len(regions)} regions, {len(hs)} hypotheses, {len(tested)} tested"

    def _region_stats(self, space: str, name: str) -> str:
        if space == "o":
            n = sum(1 for o in self.observations.values() if o.region == name)
            return f"{n} observations"
        hs = [h for h in self.hypotheses.values() if h.region == name]
        tested = [h for h in hs if h.status != PENDING or h.scores]
        best = [h.best for h in hs if h.best is not None]
        stats = f"{len(hs)} hypotheses, {len(tested)} tested"
        return stats + (f", best {max(best):.4g}" if best else "") if hs else "UNTRIED"

    def render_hypothesis(self, h: Hypothesis) -> str:
        sup, ref = self.counts(h.id)
        where = f"region `{h.region}`" if h.region else "unplaced"
        lines = [f"- [{h.id}] {h.statement}  ({h.status}, {where})"]
        if h.rationale:
            lines.append(f"  - rationale: {h.rationale}")
        if h.parent:
            lines.append(f"  - refines {h.parent}")
        if h.scores:
            lines.append(f"  - scores: {', '.join(f'{s:.4g}' for s in h.scores)} (best {h.best:.4g})")
        if sup or ref:
            lines.append(f"  - evidence: {sup} supporting, {ref} refuting")
        if h.metrics:
            shown = ", ".join(f"{k}={v:.4g}" for k, v in sorted(h.metrics.items()))
            lines.append(f"  - metrics of its best artifact: {shown}")
        if h.artifact:
            lines.append(f"  - artifact: {h.artifact}")
        if h.notes:
            lines.append(f"  - local test notes: {h.notes}")
        if h.repair:
            lines.append(f"  - repair requested: {h.repair}")
        if h.status == SHELVED and h.reason:
            lines.append(f"  - shelved (round {h.touched}): {h.reason}")
        return "\n".join(lines)

    def distinct_on_parts(self, rel: float = 0.02) -> list[Hypothesis]:
        """Shelved hypotheses whose metrics differ from the leader's by more
        than `rel` on some part of the task. Which direction is better is not
        assumed here: the stage reading it knows what each metric means. A
        claim that loses overall and wins somewhere is the ingredient a
        combination needs, and shelving it hides it from every later stage."""
        lead = self.leader()
        if lead is None or not lead.metrics:
            return []
        out = []
        for h in self.of(SHELVED):
            if not h.metrics or not h.artifact:
                continue
            for k, v in h.metrics.items():
                base = lead.metrics.get(k)
                if base is not None and abs(v - base) > rel * max(abs(base), 1e-9):
                    out.append(h)
                    break
        return out

    def render_hypotheses(self, statuses: tuple[str, ...] = (PENDING, TESTED)) -> str:
        hs = [h for h in self.hypotheses.values() if h.status in statuses]
        return "\n".join(self.render_hypothesis(h) for h in hs) or "(none)"

    def render_observation(self, o: Observation) -> str:
        where = f"region `{o.region}`" if o.region else "unplaced"
        lines = [f"- [{o.id}] {o.content}  [{where}, sources {', '.join(o.sources) or '-'}]"]
        for hid, v in o.verdicts.items():
            lines.append(f"  - {v['relation']} {hid}: {v['why']}")
        return "\n".join(lines)

    def render_evidence(self, hids: list[str]) -> str:
        """For each hypothesis, the observations that support or refute it,
        each with the verdict's reason and the observation itself."""
        blocks = []
        for hid in hids:
            h = self.hypotheses.get(hid)
            if h is None:
                continue
            best = f", best {h.best:.4g}" if h.best is not None else ""
            lines = [f"- [{hid}] {h.statement}  ({h.status}{best})"]
            for rel in ("supports", "refutes"):
                for o in self.observations.values():
                    v = o.verdicts.get(hid)
                    if v and v.get("relation") == rel:
                        why = f" -- {v['why']}" if v.get("why") else ""
                        lines.append(f"  - {rel} [{o.id}]: {o.content}{why}")
            if len(lines) == 1:
                lines.append("  - no observation supports or refutes it yet")
            blocks.append("\n".join(lines))
        return "\n".join(blocks) or "(none)"

    def render_observations(self) -> str:
        return "\n".join(self.render_observation(o) for o in self.observations.values()) or "(none yet)"

    # -- persistence -----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "hypotheses": [asdict(h) for h in self.hypotheses.values()],
            "observations": [asdict(o) for o in self.observations.values()],
            "h_regions": [asdict(r) for r in self.h_regions.values()],
            "o_regions": [asdict(r) for r in self.o_regions.values()],
            "seed": {"path": self.seed_path, "score": self.seed_score},
            "aliases": self.aliases,
            "region_log": self.region_log,
            "best_by_round": self.best_by_round,
            "obs_by_round": self.obs_by_round,
            # Which round tested what is how explore and revise are shown the
            # evidence for last round's hypotheses. Left out of the snapshot it
            # would come back empty and a resumed run would open blind to the
            # round it is answering.
            "tested_by_round": {str(k): v for k, v in self.tested_by_round.items()},
        }

    @classmethod
    def restore(cls, snap: dict[str, Any]) -> State:
        """The state a `snapshot()` recorded, rebuilt.

        Read defensively on purpose: a snapshot written before a field existed
        is still a resumable run, and a field this version does not know is not
        a reason to refuse one. Ids come back as they were -- `_next` counts
        entries, so h1..h99 continues at h100 -- and that is what keeps the
        observations' verdicts, which name hypotheses by id, pointing at the
        same claims.
        """
        s = cls()
        for r in snap.get("h_regions") or []:
            s.h_regions[r["name"]] = Region(**{k: v for k, v in r.items() if k in Region.__annotations__})
        for r in snap.get("o_regions") or []:
            s.o_regions[r["name"]] = Region(**{k: v for k, v in r.items() if k in Region.__annotations__})
        for h in snap.get("hypotheses") or []:
            s.hypotheses[h["id"]] = Hypothesis(**{k: v for k, v in h.items() if k in Hypothesis.__annotations__})
        for o in snap.get("observations") or []:
            s.observations[o["id"]] = Observation(**{k: v for k, v in o.items() if k in Observation.__annotations__})
        seed = snap.get("seed") or {}
        s.seed_path, s.seed_score = seed.get("path", ""), seed.get("score")
        for space, table in (snap.get("aliases") or {}).items():
            s.aliases[space] = {str(k): str(v) for k, v in (table or {}).items()}
        s.region_log = [dict(e) for e in (snap.get("region_log") or [])]
        s.best_by_round = list(snap.get("best_by_round") or [])
        s.obs_by_round = list(snap.get("obs_by_round") or [])
        s.tested_by_round = {int(k): list(v) for k, v in (snap.get("tested_by_round") or {}).items()}
        return s

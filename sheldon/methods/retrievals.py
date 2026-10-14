"""Rendering the run's retrieval record for a prompt.

The raw retrieval record (`ctx.tools.record`) is everything the run looked
up. `render_retrievals` renders it as an index plus as much content as fits,
so a stage can see what has been looked up and cite it by id.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: Leading count in a probe summary -- "5 source hit(s) for ...", "115 logs
#: matched ...". How many things a query returned is the cheapest available
#: proxy for how much of it is worth reading.
_HITS = re.compile(r"^(\d+)\b")


def _hits(summary: str) -> int:
    m = _HITS.match(summary)
    return int(m.group(1)) if m else 1


def _citable(record: list[Any]) -> list[Any]:
    """The retrievals worth showing: the ones that ran and found something.
    "no series matched" is the absence of a citation, not a citation."""
    seen: set[str] = set()
    kept = []
    for entry in record:
        result = entry.result
        summary = str(getattr(result, "summary", ""))
        if not getattr(result, "ok", True) or not summary or summary.startswith("no "):
            continue
        if summary in seen:
            continue
        seen.add(summary)
        kept.append(entry)
    return kept


def has_retrievals(record: list[Any]) -> bool:
    """Whether `render_retrievals` would have anything to show. A prompt that
    guards on the rendered text needs this, because the placeholder for an
    empty record is itself text."""
    return bool(_citable(record))


def render_retrievals(
    record: list[Any],
    limit: int = 26000,
    sample: int = 6,
    broad: int = 25,
    contents: bool = True,
    index_limit: int = 14000,
) -> str:
    """What the run has looked up: an index, and as much content as fits.

    Observations are what a stage made of the evidence; this is the evidence
    itself, so a stage can cite a retrieval by id and read what an earlier
    stage saw. Nothing is summarised for transport:

    * **Every retrieval is indexed**, one line each -- id, tool, the arguments,
      what came back. An index of 200 queries costs a fraction of a transcript
      and is what lets a stage know something exists.
    * **Content goes to the narrow results first.** Five hits means five things
      worth reading; 115 means a fishing expedition.
    * **Anything omitted is still named**, with the arguments that produced it,
      so a stage can run it again rather than cast about for a new query.

    `contents=False` renders the index alone, for stages that need to know what
    exists without paying for it up front.
    """
    kept = _citable(record)
    if not kept:
        return "(nothing has been retrieved yet)"

    lines = [
        "## Everything this run has looked up",
        "",
        "Each line carries the query that produced it, so anything you need in "
        "full can be run again as written.",
        "",
    ]
    # The index goes into every stage's prompt and grows all run, so it needs a
    # ceiling. When one is needed the broad queries go first: "10000 matched
    # '*'" tells a later stage nothing it could act on, while "5 source hit(s)
    # for 'AD_FAILURE|...'" is the whole answer.
    index = [e.index_line() for e in kept]
    if sum(map(len, index)) > index_limit:
        rank = sorted(range(len(kept)), key=lambda i: (_hits(kept[i].result.summary), i))
        room, chosen = index_limit, set()
        for i in rank:
            if room - len(index[i]) < 0:
                break
            room -= len(index[i])
            chosen.add(i)
        elided = len(index) - len(chosen)
        index = [ln for i, ln in enumerate(index) if i in chosen]
        index.append(f"({elided} broader queries omitted from this index.)")
    lines += index
    if not contents:
        return "\n".join(lines)

    # Narrowest first, because that is where the signal is. Ties keep run order,
    # so an unchanged set of retrievals renders identically.
    lines += ["", "## Results, most specific first", ""]
    order = sorted(range(len(kept)), key=lambda i: (_hits(kept[i].result.summary), i))
    total = sum(len(x) for x in lines)
    shown = 0
    for i in order:
        e = kept[i]
        evidence = list(getattr(e.result, "evidence", None) or [])
        take = 1 if _hits(e.result.summary) > broad else sample
        block = [f"- [{e.id}] {e.probe.name} -> {e.result.summary[:200]}"]
        block += [f"    {json.dumps(item, default=str)[:600]}" for item in evidence[:take]]
        text = "\n".join(block)
        if total + len(text) > limit:
            break
        lines.append(text)
        total += len(text)
        shown += 1

    if shown < len(kept):
        lines += [
            "",
            f"({len(kept) - shown} of {len(kept)} are indexed above without their "
            f"contents. Re-run any you need -- the query is given above.)",
        ]
    return "\n".join(lines)

#!/usr/bin/env python
"""Build one corpus over EviDx's three medical benchmarks.

Med-Evidence-2.6k supplies the case text, the gold label and the evidence
annotations, and it is what EviDx itself runs on -- `question_text` is the
string its `main.py` reads as `question`. What it does not carry is the
multiple-choice options, which differ by dataset:

  JAMA             options are embedded in question_text ("What Is Your
                   Diagnosis?" with the choices in the case); the original
                   also has them as opa..opd, which is cleaner to parse.
  MedXpertQA-Text  embedded as "Answer Choices: (A) ... (J)" -- ten options,
                   not four; also present as an options dict in the original.
  DiagnosisArena   NOT embedded. question_text ends at "What is the most
                   likely diagnosis?", so the options must be joined in from
                   the original `Options` / `Right Option` columns.

Joins, verified against the sources: DiagnosisArena and MedXpertQA on `id`,
JAMA on 1-based row index (the original has no id column).

Output is one JSONL per dataset in the shape EviDx's main.py accepts --
`question`, `id`, `dataset`, `options`, `answer_idx` -- plus the gold text and
golden evidence, so the same file serves a simple text baseline and, later, a
full EviDx run and evidence-recall scoring.

    ./build_corpus.py --sample 100 --seed 0
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def as_obj(v):
    """Options arrive as a dict, or as its repr. Tolerate both."""
    if isinstance(v, (dict, list)):
        return v
    s = str(v or "").strip()
    if not s:
        return None
    for parse in (json.loads, ast.literal_eval):
        try:
            return parse(s)
        except (ValueError, SyntaxError):
            continue
    return None


def parse_embedded_choices(text: str) -> dict | None:
    """Pull "Answer Choices: (A) x (B) y ..." out of a question body."""
    m = re.search(r"Answer Choices:\s*(.+)$", text, re.S)
    if not m:
        return None
    body = m.group(1)
    parts = re.findall(r"\(([A-Z])\)\s*([^()]+?)(?=\s*\([A-Z]\)|$)", body, re.S)
    return {k: v.strip() for k, v in parts} or None


def build_diagnosisarena() -> list[dict]:
    ev = load(DATA / "medevidence-DiagnosisArena.jsonl")
    orig = {str(r["id"]): r for r in load(DATA / "diagnosisarena.jsonl")}
    out = []
    for r in ev:
        o = orig.get(str(r["original_id"]))
        opts = as_obj(o.get("Options")) if o else None
        out.append({
            "id": f"DiagnosisArena-{r['original_id']}",
            "dataset": "DiagnosisArena",
            "question": r["question_text"],
            "gold_text": r["ground_truth_text"],
            "options": opts,
            "answer_idx": (o or {}).get("Right Option"),
            "golden_evidence": as_obj(r.get("golden_evidence")) or [],
        })
    return out


def build_jama() -> list[dict]:
    ev = load(DATA / "medevidence-JAMA.jsonl")
    orig = load(DATA / "jama.jsonl")           # joined by 1-based row index
    out = []
    for r in ev:
        i = int(r["original_id"]) - 1
        o = orig[i] if 0 <= i < len(orig) else {}
        opts = {k.upper()[-1]: o.get(f"op{k}") for k in ("a", "b", "c", "d") if o.get(f"op{k}")}
        out.append({
            "id": f"JAMA-{r['original_id']}",
            "dataset": "JAMA",
            "question": r["question_text"],
            "gold_text": r["ground_truth_text"],
            "options": opts or None,
            "answer_idx": o.get("answer_idx"),
            "golden_evidence": as_obj(r.get("golden_evidence")) or [],
        })
    return out


def build_medxpertqa() -> list[dict]:
    ev = load(DATA / "medevidence-MedXpertQA-Text.jsonl")
    orig = {str(r["id"]): r for r in load(DATA / "medxpertqa-text.jsonl")}
    out = []
    for r in ev:
        o = orig.get(str(r["original_id"]), {})
        opts = as_obj(o.get("options")) or parse_embedded_choices(r["question_text"])
        out.append({
            "id": f"MedXpertQA-{r['original_id']}",
            "dataset": "MedXpertQA-Diag",
            "question": r["question_text"],
            "gold_text": r["ground_truth_text"],
            "options": opts,
            "answer_idx": o.get("label"),
            "golden_evidence": as_obj(r.get("golden_evidence")) or [],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=100,
                    help="cases per dataset; EviDx uses a fixed random 100")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import random

    builders = {"DiagnosisArena": build_diagnosisarena,
                "JAMA": build_jama,
                "MedXpertQA-Diag": build_medxpertqa}

    summary = []
    for name, fn in builders.items():
        rows = fn()
        full = ROOT / f"corpus-{name}-full.jsonl"
        with full.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Sampled with a stated seed. EviDx samples 100 per dataset but does not
        # publish which, so ours is reproducible rather than identical to theirs.
        rng = random.Random(a.seed)
        pick = rows if len(rows) <= a.sample else rng.sample(rows, a.sample)
        pick.sort(key=lambda r: r["id"])
        samp = ROOT / f"corpus-{name}.jsonl"
        with samp.open("w", encoding="utf-8") as f:
            for r in pick:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with_opts = sum(1 for r in pick if r["options"])
        with_ans = sum(1 for r in pick if r["answer_idx"])
        with_ev = sum(1 for r in pick if r["golden_evidence"])
        n_opts = sorted({len(r["options"]) for r in pick if r["options"]})
        summary.append((name, len(rows), len(pick), with_opts, with_ans, with_ev, n_opts))

    print(f"  {'dataset':<18}{'full':>6}{'samp':>6}{'opts':>6}{'ans':>6}{'evid':>6}   #options")
    for s in summary:
        print(f"  {s[0]:<18}{s[1]:>6}{s[2]:>6}{s[3]:>6}{s[4]:>6}{s[5]:>6}   {s[6]}")


if __name__ == "__main__":
    main()

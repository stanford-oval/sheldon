#!/usr/bin/env python3
"""Pull Med-Evidence-2.6k -- case text, gold label, and evidence annotations.

This is the corpus EviDx actually runs on: `question_text` is the pre-formatted
case its `main.py` reads as `question`, so the original datasets are needed only
for MC options that are not already embedded in the text (DiagnosisArena's are
not; JAMA's and MedXpertQA's are).

It also publishes the curated MedXpertQA-Diag subset as its MedXpertQA-Text
config -- 234 diagnostic-focused cases, so that filtering does not have to be
reconstructed.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent
DS = "danieez/Med-Evidence-2.6k"
CONFIGS = ["DiagnosisArena", "JAMA", "MedXpertQA-Text"]


def rows(cfg, offset, length, tries=8):
    q = urllib.parse.urlencode({"dataset": DS, "config": cfg, "split": "test",
                                "offset": offset, "length": length})
    url = f"https://datasets-server.huggingface.co/rows?{q}"
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == tries - 1:
                raise
            time.sleep(5 * (i + 1))
    raise RuntimeError("unreachable")


for cfg in CONFIGS:
    out = OUT / f"medevidence-{cfg}.jsonl"
    if out.exists():
        print(f"  {cfg}: already have {sum(1 for _ in out.open())}"); continue
    part = OUT / f"medevidence-{cfg}.partial.jsonl"
    got = [json.loads(l) for l in part.open()] if part.exists() else []
    total, offset = None, len(got)
    while total is None or offset < total:
        d = rows(cfg, offset, 100)
        total = d["num_rows_total"]
        batch = [r["row"] for r in d["rows"]]
        if not batch:
            break
        got += batch; offset += len(batch)
        with part.open("w", encoding="utf-8") as f:
            for r in got:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {cfg}: {offset}/{total}", end="\r", flush=True)
        time.sleep(0.5)
    with out.open("w", encoding="utf-8") as f:
        for r in got:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    part.unlink(missing_ok=True)
    print(f"  {cfg}: {len(got)} rows -> {out.name}                ")

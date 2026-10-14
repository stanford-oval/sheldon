#!/usr/bin/env python3
"""Pull the three EviDx datasets via the HF datasets-server rows API.

No huggingface_hub or pyarrow needed -- the rows endpoint returns JSON, which
avoids adding parquet dependencies to an env we are borrowing.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent
SETS = [
    ("SII-SPIRAL-MED/DiagnosisArena", "default", "test", "diagnosisarena"),
    ("TsinghuaC3I/MedXpertQA",        "Text",    "test", "medxpertqa-text"),
    ("JesseLiu/Jama_challenge",       "default", "test", "jama"),
]

def rows(ds, cfg, split, offset, length, tries=8):
    """One page, with backoff. The rows endpoint rate-limits at 429 partway
    through a large split, so a bare loop dies after a few hundred rows."""
    q = urllib.parse.urlencode({"dataset": ds, "config": cfg, "split": split,
                                "offset": offset, "length": length})
    url = f"https://datasets-server.huggingface.co/rows?{q}"
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == tries - 1:
                raise
            wait = 5 * (i + 1)
            print(f"    429, retrying in {wait}s", end="\r", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")

for ds, cfg, split, name in SETS:
    out = OUT / f"{name}.jsonl"
    if out.exists():
        print(f"  {name}: already have {sum(1 for _ in out.open())} rows"); continue
    part = OUT / f"{name}.partial.jsonl"
    got = [json.loads(l) for l in part.open()] if part.exists() else []
    total, offset = None, len(got)
    if got: print(f"  {name}: resuming from {offset}")
    while total is None or offset < total:
        d = rows(ds, cfg, split, offset, 100)
        total = d["num_rows_total"]
        batch = [r["row"] for r in d["rows"]]
        if not batch: break
        got += batch; offset += len(batch)
        with part.open("w", encoding="utf-8") as f:
            for r in got: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name}: {offset}/{total}", end="\r", flush=True)
        time.sleep(0.5)
    with out.open("w", encoding="utf-8") as f:
        for r in got: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    part.unlink(missing_ok=True)
    print(f"  {name}: {len(got)} rows -> {out.name}          ")

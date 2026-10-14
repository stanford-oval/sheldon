#!/usr/bin/env python
"""Normalize the DOCKSTRING dataset into data/dockstring/.

The upstream release is one 99 MB TSV — `inchikey`, `smiles`, then one column
per protein target holding an AutoDock Vina score (kcal/mol, lower is better),
with blanks where docking failed. That is not what the benchmark needs at run
time, which is three things it cannot afford to recompute per task:

  * the **percentile thresholds** the paper scores against. Feedback Descent's
    claim is "past the 99.9th percentile of 260k compounds", and the quantity
    ranked is the combined objective s = -Vina - 10*(1 - QED), not Vina. QED is
    not in the release, so the thresholds have to be derived here.
  * a **canonical SMILES -> row** index, so a proposed molecule that is already
    in the database is answered from the table instead of by a 20-second dock.
  * **Morgan fingerprints**, so the similarity-search probe is a numpy popcount
    rather than 260k RDKit calls.

Normalizing here rather than in `load()` is the repo rule, and it earns its keep
twice over: QED over 260k molecules is minutes of CPU, and pinning the thresholds
to a file means a rerun months later scores against the same numbers.

    scripts/prepare_dockstring.py --download          # fetch, then prepare
    scripts/prepare_dockstring.py                     # prepare what's already there
    scripts/prepare_dockstring.py --targets PPARG,F2  # only report on these

Writes, all aligned by row index:

    dockstring-dataset.tsv   the raw release, kept as the input of record
    smiles.txt               RDKit-canonical SMILES, one per line
    qed.npy                  float32[N]      QED, NaN where RDKit refused
    scores.npy               float32[N, T]   Vina per target, NaN where missing
    fps.npy                  uint8[N, 128]   packed 1024-bit Morgan (radius 2)
    targets.txt              the T target names, in column order
    targets.json             per target: percentile thresholds of s, and counts
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import urllib.request
from pathlib import Path
from typing import Any

# Permanent figshare URL, same one the dockstring package uses.
DATASET_URL = "https://figshare.com/ndownloader/files/35948138"

#: The percentiles the thresholds are cut at.
#:
#: Feedback Descent's Table 3 has six database rows, and its *labels* are one
#: decade off the quantiles they hold: its "Top 90%" is the real 99th percentile,
#: its "Top 99.9%" -- the row its abstract claims to surpass -- is the real
#: 99.99th. Both ends of the table pin this down, because the "Top 50%" and
#: "Best Molecule" rows match what is computed here exactly, target for target,
#: while every row between them lines up only after the shift. So the grid runs
#: one decade deeper than the paper's, the names here mean what they say, and
#: `benchmark_opts.pass_percentile: p99.99` is the bar the paper actually cleared.
PERCENTILES = (50.0, 90.0, 99.0, 99.9, 99.99, 99.999)

#: Feedback Descent (arXiv:2511.07919) Table 3.
PAPER_TARGETS = ("ADRB1", "PGR", "PPARA", "PPARG", "CDK2", "F2")

FP_BITS = 1024
FP_RADIUS = 2


def check_space(out: Path, molecules: int, targets: int) -> None:
    """Refuse to spend five minutes of CPU on arrays there is no room for.

    The featurization runs before the first byte is written, so a full disk
    surfaces as an OSError after the expensive part rather than before it.
    """
    import shutil  # noqa: PLC0415

    need = molecules * (targets * 4 + FP_BITS // 8 + 4 + 80) + 20_000_000
    free = shutil.disk_usage(out).free
    if free < need:
        raise SystemExit(
            f"{out}: needs about {need / 1e6:.0f} MB for the prepared arrays, "
            f"{free / 1e6:.0f} MB free"
        )


def download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {DATASET_URL} -> {dest}", flush=True)
    urllib.request.urlretrieve(DATASET_URL, filename=dest)
    print(f"  {dest.stat().st_size / 1e6:.0f} MB", flush=True)


# -- per-molecule work, run in a process pool --------------------------------

_gen = None


def _featurize(smiles: str) -> tuple[str, float, bytes]:
    """Canonical SMILES, QED, packed Morgan bits. ("", nan, b"") if RDKit balks.

    Failure is a normal outcome — a handful of the 260k rows do not survive
    sanitization — and it must not take the row's docking scores with it, so
    the caller keeps the row and marks the derived fields missing.
    """
    global _gen
    import numpy as np  # noqa: PLC0415
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem import QED, rdFingerprintGenerator  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    if _gen is None:
        _gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "", float("nan"), b""
        canonical = Chem.MolToSmiles(mol)
        qed = float(QED.qed(mol))
        packed = np.packbits(_gen.GetFingerprintAsNumPy(mol).astype("uint8")).tobytes()
        return canonical, qed, packed
    except Exception:
        return "", float("nan"), b""


def featurize_all(smiles: list[str], jobs: int) -> tuple[list[str], Any, Any]:
    import numpy as np  # noqa: PLC0415

    n = len(smiles)
    canonical: list[str] = []
    qed = np.full(n, np.nan, dtype="float32")
    fps = np.zeros((n, FP_BITS // 8), dtype="uint8")
    empty = bytes(FP_BITS // 8)

    print(f"featurizing {n} molecules on {jobs} processes (QED + Morgan)...", flush=True)
    with mp.Pool(jobs) as pool:
        for i, (canon, q, packed) in enumerate(pool.imap(_featurize, smiles, chunksize=512)):
            canonical.append(canon)
            qed[i] = q
            fps[i] = np.frombuffer(packed or empty, dtype="uint8")
            if (i + 1) % 25000 == 0:
                print(f"  {i + 1}/{n}", flush=True)
    bad = sum(1 for c in canonical if not c)
    print(f"  done; {bad} molecules RDKit could not parse", flush=True)
    return canonical, qed, fps


# -- thresholds --------------------------------------------------------------


def thresholds(scores: Any, qed: Any, targets: list[str]) -> dict[str, Any]:
    """Percentiles of s = -Vina - 10*(1 - QED), per target.

    Computed over the molecules that have both a Vina score for that target and
    a QED — a target the release only scored half of must report the count it
    actually ranked, or "99.9th percentile of 260k" is a claim about a number
    that was never 260k.
    """
    import numpy as np  # noqa: PLC0415

    out: dict[str, Any] = {}
    for j, target in enumerate(targets):
        vina = scores[:, j]
        ok = np.isfinite(vina) & np.isfinite(qed)
        s = -vina[ok] - 10.0 * (1.0 - qed[ok])
        if s.size == 0:
            continue
        cuts = np.percentile(s, PERCENTILES)
        out[target] = {
            "n": int(s.size),
            "thresholds": {
                f"p{p:g}": round(float(c), 4)
                for p, c in zip(PERCENTILES, cuts, strict=True)
            },
            "best": round(float(s.max()), 4),
            "vina_best": round(float(np.nanmin(vina)), 4),
            "vina_mean": round(float(np.nanmean(vina)), 4),
        }
    return out


# -- main --------------------------------------------------------------------


def report(table: dict[str, Any], wanted: list[str]) -> None:
    cols = [f"p{p:g}" for p in PERCENTILES]
    print(f"\nthresholds on s = -vina - 10*(1-QED)\n{'target':<8}{'n':>8}" +
          "".join(f"{c:>10}" for c in cols) + f"{'best':>10}")
    for t in wanted:
        e = table.get(t)
        if not e:
            print(f"{t:<8}   -- not in the release --")
            continue
        print(f"{t:<8}{e['n']:>8}" + "".join(f"{e['thresholds'][c]:>10.3f}" for c in cols) +
              f"{e['best']:>10.3f}")


def recut(out: Path, targets_arg: str) -> int:
    """Recompute targets.json from prepared arrays.

    The percentile grid is the one thing here likely to be revised -- deciding
    which bar a run is scored against should not cost another pass of QED over
    260k molecules.
    """
    import numpy as np  # noqa: PLC0415

    missing = [f for f in ("qed.npy", "scores.npy", "targets.txt") if not (out / f).exists()]
    if missing:
        print(f"{out}: nothing prepared yet ({', '.join(missing)} missing)", file=sys.stderr)
        return 2
    qed = np.load(out / "qed.npy")
    scores = np.load(out / "scores.npy", mmap_mode="r")
    targets = (out / "targets.txt").read_text(encoding="utf-8").split()
    table = thresholds(scores, qed, targets)

    path = out / "targets.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update({
        "source": DATASET_URL,
        "objective": "s = -vina - 10 * (1 - QED)",
        "percentiles": list(PERCENTILES),
        "molecules": int(scores.shape[0]),
        "targets": table,
    })
    payload.setdefault("fingerprint", {"kind": "morgan", "radius": FP_RADIUS, "bits": FP_BITS})
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report(table, [t.strip().upper() for t in targets_arg.split(",") if t.strip()])
    print(f"\nrewrote {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=Path("data/dockstring"))
    ap.add_argument("--input", type=Path, default=None,
                    help="the release TSV (default: --out/dockstring-dataset.tsv)")
    ap.add_argument("--download", action="store_true", help="fetch the release first")
    ap.add_argument("--jobs", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    ap.add_argument("--targets", default=",".join(PAPER_TARGETS),
                    help="which targets to print thresholds for (all are written)")
    ap.add_argument("--thresholds-only", action="store_true",
                    help="recut targets.json from the arrays already prepared, no refeaturizing")
    args = ap.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("needs numpy and rdkit:  uv pip install '.[dockstring]'", file=sys.stderr)
        return 2

    if args.thresholds_only:
        return recut(args.out, args.targets)

    tsv = args.input or (args.out / "dockstring-dataset.tsv")
    if args.download or not tsv.exists():
        download(tsv)
    if not tsv.exists():
        print(f"missing {tsv}; pass --download", file=sys.stderr)
        return 2

    print(f"reading {tsv}", flush=True)
    with open(tsv, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    if header[:2] != ["inchikey", "smiles"]:
        print(f"unexpected header {header[:3]}; expected inchikey, smiles, <targets>",
              file=sys.stderr)
        return 2
    targets = header[2:]
    print(f"  {len(rows)} molecules x {len(targets)} targets", flush=True)

    scores = np.full((len(rows), len(targets)), np.nan, dtype="float32")
    for i, row in enumerate(rows):
        for j, cell in enumerate(row[2:]):
            if cell:
                scores[i, j] = float(cell)

    args.out.mkdir(parents=True, exist_ok=True)
    check_space(args.out, len(rows), len(targets))

    canonical, qed, fps = featurize_all([r[1] for r in rows], args.jobs)

    (args.out / "smiles.txt").write_text("\n".join(canonical) + "\n", encoding="utf-8")
    (args.out / "targets.txt").write_text("\n".join(targets) + "\n", encoding="utf-8")
    np.save(args.out / "qed.npy", qed)
    np.save(args.out / "scores.npy", scores)
    np.save(args.out / "fps.npy", fps)

    table = thresholds(scores, qed, targets)
    payload = {
        "source": DATASET_URL,
        "objective": "s = -vina - 10 * (1 - QED)",
        "percentiles": list(PERCENTILES),
        "fingerprint": {"kind": "morgan", "radius": FP_RADIUS, "bits": FP_BITS},
        "molecules": len(rows),
        "targets": table,
    }
    (args.out / "targets.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report(table, [t.strip().upper() for t in args.targets.split(",") if t.strip()])
    print(f"\nwrote {args.out}/  (smiles.txt, qed.npy, scores.npy, fps.npy, targets.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

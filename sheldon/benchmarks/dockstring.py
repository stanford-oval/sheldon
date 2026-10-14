"""DOCKSTRING — design a molecule that binds a protein and still looks like a drug.

    Garcia-Ortegon et al., "DOCKSTRING: Easy Molecular Docking Yields Better
    Benchmarks for Ligand Design", J. Chem. Inf. Model. 62 (2022).
    Apache-2.0, github.com/dockstring/dockstring

The task shape is the one Feedback Descent (Lee, Boen & Finn, arXiv:2511.07919)
scores its molecule experiments on: pick a protein target, write a SMILES string,
and maximise

    s = -Vina - 10 * (1 - QED)

Vina is AutoDock Vina's predicted binding free energy in kcal/mol (more negative
is tighter); QED is RDKit's drug-likeness in [0, 1]. Neither half is optional.
Maximising -Vina alone rewards greasy unsynthesisable blobs, which is the oldest
failure mode in de novo design; the 10x QED penalty is what makes the objective
worth optimising -- a molecule at QED 0.5 forfeits five kcal/mol of apparent
affinity.

The bar is the DOCKSTRING database: 260,155 real compounds docked against all 58
targets. A task passes when the best molecule the method **measured** beats that
target's **99.9th percentile**, and every finer cut stays in the metrics so a
near miss is visible rather than rounded to zero.

Scoring the best molecule found rather than the one handed in is what the paper
does, and what every optimizer it compares against does — a search does not have
a final answer, it has a best result. The distinction is not academic: a
method can dock an excellent molecule and then submit a worse one, and scored
on the submission that task reads as a failure when the search succeeded. Both
numbers are always in
the metrics — `objective` is what scored, `submitted_objective` is what was
handed in, and `finalize_loss` is the gap — because that gap is a real defect in
the method and hiding it would be its own kind of dishonesty.
`benchmark_opts.score_on: submitted` restores the stricter reading.

One thing to know before comparing a number here with the paper's Table 3: its
database rows are labelled one decade off the quantiles they hold. Its "Top 90%"
is the real 99th percentile, and its "Top 99.9%" -- the row the abstract claims
to surpass -- is the real 99.99th. Its "Top 50%" and "Best Molecule" rows match
what `scripts/prepare_dockstring.py` computes exactly, target for target, and
every row between them lines up only after the shift, which is what makes it a
labelling slip rather than a different objective. The names here mean what they
say, so `benchmark_opts.pass_percentile: p99.99` is the bar the paper cleared;
`p99.9`, the default, is one decade easier and leaves a usable signal.

The library cuts both ways, and it is worth knowing which way before reading a
number off this benchmark. `search_similar_molecules` is feedback the paper
gives its optimizer, and it is also a path to a good score that involves no
design at all: walking it to a high-percentile compound already in the
database. `max_library_similarity` and
`in_library` are in every score row so that is visible rather than inferred,
`benchmark_opts.require_novel` refuses to count it, and
`benchmark_opts.dataset_search: false` removes the path.

This is an *optimization* benchmark, not a question-answering one, and that
changes where the difficulty lives. There is no answer to recall and no gold
label to leak -- only an objective that has to be measured. So the docking oracle
is a plain probe handed to `direct`, `best-of-n` and `sheldon` alike: it is the
environment, not a crutch. What is scarce is calls to it. `max_docking_calls`
(default 40) is the per-task budget and it is the whole reason "design a
molecule" does not collapse into "enumerate the library". Setting it to 0 is the
no-oracle control: what the model can do from memory alone.

Docking is real work -- 20 to 60 CPU-seconds a molecule, single-threaded. Three
things keep that affordable, in the order they are tried: a molecule already in
the database is read from the table, a molecule docked before is read from
`.cache/dockstring.db`, and only then does Vina run. Re-scoring a finished run
therefore costs nothing, exactly like the LLM cache. A repeat within one task is
free and does not spend budget; the first evaluation of a molecule spends one
whichever of the three answered it, because a cheap answer is still an answer,
and a replay that could evaluate more molecules than the live run did is not a
replay.

Data comes from `scripts/prepare_dockstring.py`; `sheldon check dockstring` says
what is missing.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from sheldon.benchmarks.base import Benchmark
from sheldon.core.cache import DiskCache, cache_key
from sheldon.core.io import REPO_ROOT
from sheldon.core.probes import CallableProbe, Probe, ProbeResult
from sheldon.core.registry import benchmarks
from sheldon.core.types import Falsification, Prediction, RunContext, Score, Task, Verifier

#: The six targets Feedback Descent reports in its Table 3.
PAPER_TARGETS = ("ADRB1", "PGR", "PPARA", "PPARG", "CDK2", "F2")

#: What a target is, in one clause. Only for the prompt -- a target name alone
#: is a UniProt gene symbol, and half the difficulty of the task is knowing what
#: kind of pocket it is.
TARGET_NOTES = {
    "ADRB1": "beta-1 adrenergic receptor, a class A GPCR with a buried aminergic pocket",
    "PGR": "progesterone receptor, a nuclear receptor with a deep hydrophobic steroid pocket",
    "PPARA": "peroxisome proliferator-activated receptor alpha, a nuclear receptor whose "
             "Y-shaped pocket favours a lipophilic tail and an acidic head",
    "PPARG": "peroxisome proliferator-activated receptor gamma, a large nuclear-receptor "
             "pocket -- the glitazone target",
    "CDK2": "cyclin-dependent kinase 2, an ATP-site kinase with the usual hinge donor/acceptor",
    "F2": "thrombin, a serine protease with a charged S1 pocket and a small hydrophobic S3/S4",
}

#: The paper's starting points: acetamide, pentane, benzene.
SEEDS = (("acetamide", "CC(N)=O"), ("pentane", "CCCCC"), ("benzene", "c1ccccc1"))

#: Weight on the drug-likeness penalty, s = -vina - QED_WEIGHT * (1 - QED).
QED_WEIGHT = 10.0

#: The percentile a task has to beat to count as solved. See the note above on
#: how this lines up with the paper's Table 3; `p99.99` is the paper's own bar.
PASS_PERCENTILE = "p99.9"

#: Vina's own floor. A "score" beyond this is a docking artefact, not a ligand.
VINA_FLOOR = -20.0


def objective(vina: float, qed: float) -> float:
    """s = -Vina - 10 * (1 - QED). The one number this benchmark maximises."""
    return -float(vina) - QED_WEIGHT * (1.0 - float(qed))


# -- the prepared database -------------------------------------------------


class _Dataset:
    """The prepared DOCKSTRING table: thresholds, exact lookups, neighbours.

    Loaded once per data root and shared by every task -- 260k rows is a second
    of I/O and 100 MB of arrays, which is fine once and absurd per task. The big
    arrays are memory-mapped and the SMILES index is built on first use, so a run
    that never calls `search_similar_molecules` never pays for it.
    """

    _open: dict[Path, _Dataset] = {}

    @classmethod
    def open(cls, root: Path) -> _Dataset:
        root = Path(root)
        if root not in cls._open:
            cls._open[root] = cls(root)
        return cls._open[root]

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.meta: dict[str, Any] = {}
        path = self.root / "targets.json"
        if path.exists():
            self.meta = json.loads(path.read_text(encoding="utf-8"))
        self._targets: list[str] | None = None
        self._smiles: list[str] | None = None
        self._index: dict[str, int] | None = None
        self._qed: Any = None
        self._scores: Any = None
        self._fps: Any = None
        self._sorted: dict[str, Any] = {}

    # -- readiness --------------------------------------------------------

    @property
    def ready(self) -> bool:
        return bool(self.meta.get("targets"))

    def missing(self) -> list[str]:
        want = ["targets.json", "smiles.txt", "targets.txt", "qed.npy", "scores.npy", "fps.npy"]
        return [f for f in want if not (self.root / f).exists()]

    def target_names(self) -> list[str]:
        return sorted((self.meta.get("targets") or {}).keys())

    def entry(self, target: str) -> dict[str, Any] | None:
        return (self.meta.get("targets") or {}).get(target)

    # -- arrays, loaded on demand -----------------------------------------

    def _load_arrays(self) -> None:
        if self._scores is not None:
            return
        import numpy as np  # noqa: PLC0415

        self._targets = (self.root / "targets.txt").read_text(encoding="utf-8").split()
        self._qed = np.load(self.root / "qed.npy", mmap_mode="r")
        self._scores = np.load(self.root / "scores.npy", mmap_mode="r")
        self._fps = np.load(self.root / "fps.npy", mmap_mode="r")

    def _load_smiles(self) -> None:
        if self._index is not None:
            return
        self._smiles = (self.root / "smiles.txt").read_text(encoding="utf-8").splitlines()
        # First occurrence wins: canonicalization can collapse two release rows
        # onto one molecule, and the row we answer lookups from must be stable.
        index: dict[str, int] = {}
        for i, smiles in enumerate(self._smiles):
            if smiles and smiles not in index:
                index[smiles] = i
        self._index = index

    def column(self, target: str) -> int | None:
        self._load_arrays()
        assert self._targets is not None
        return self._targets.index(target) if target in self._targets else None

    def lookup(self, smiles: str, target: str) -> tuple[float, float] | None:
        """(vina, qed) straight from the release, or None if it is not in there."""
        self._load_arrays()
        self._load_smiles()
        assert self._index is not None
        row = self._index.get(smiles)
        col = self.column(target)
        if row is None or col is None:
            return None
        vina = float(self._scores[row, col])
        qed = float(self._qed[row])
        if not math.isfinite(vina) or not math.isfinite(qed):
            return None
        return vina, qed

    def _sorted_objective(self, target: str) -> Any:
        """Every database molecule's s for this target, ascending."""
        if target not in self._sorted:
            import numpy as np  # noqa: PLC0415

            self._load_arrays()
            col = self.column(target)
            if col is None:
                self._sorted[target] = np.zeros(0, dtype="float32")
            else:
                vina = np.asarray(self._scores[:, col], dtype="float64")
                qed = np.asarray(self._qed, dtype="float64")
                ok = np.isfinite(vina) & np.isfinite(qed)
                self._sorted[target] = np.sort(-vina[ok] - QED_WEIGHT * (1.0 - qed[ok]))
        return self._sorted[target]

    def percentile(self, target: str, value: float) -> float:
        """Where `value` falls in the database, as a percentage in [0, 100]."""
        import numpy as np  # noqa: PLC0415

        column = self._sorted_objective(target)
        if column.size == 0:
            return float("nan")
        return float(np.searchsorted(column, value, side="left") / column.size * 100.0)

    def neighbours(self, smiles: str, target: str, k: int = 5) -> list[dict[str, Any]]:
        """The k most Tanimoto-similar database molecules, with their scores."""
        import numpy as np  # noqa: PLC0415

        query = _fingerprint(smiles)
        if query is None:
            return []
        self._load_arrays()
        self._load_smiles()
        col = self.column(target)
        if col is None or self._smiles is None:
            return []

        fps = np.asarray(self._fps)
        popcount = _POPCOUNT()
        both = popcount[np.bitwise_and(fps, query)].sum(axis=1, dtype="int32")
        either = popcount[np.bitwise_or(fps, query)].sum(axis=1, dtype="int32")
        tanimoto = both / np.maximum(either, 1)

        # Rank on similarity among molecules this target was actually docked
        # against; a neighbour with no score is not evidence about anything.
        vina = np.asarray(self._scores[:, col], dtype="float64")
        qed = np.asarray(self._qed, dtype="float64")
        tanimoto = np.where(np.isfinite(vina) & np.isfinite(qed), tanimoto, -1.0)
        top = np.argpartition(-tanimoto, min(k, tanimoto.size - 1))[:k]
        top = top[np.argsort(-tanimoto[top])]
        return [
            {
                "smiles": self._smiles[i],
                "similarity": round(float(tanimoto[i]), 3),
                "vina": round(float(vina[i]), 2),
                "qed": round(float(qed[i]), 3),
                "objective": round(objective(vina[i], qed[i]), 3),
            }
            for i in top
            if tanimoto[i] >= 0
        ]

    def max_similarity(self, smiles: str) -> float:
        """Closest database molecule, Tanimoto. 1.0 means it is already in there."""
        import numpy as np  # noqa: PLC0415

        query = _fingerprint(smiles)
        if query is None:
            return float("nan")
        self._load_arrays()
        fps = np.asarray(self._fps)
        popcount = _POPCOUNT()
        both = popcount[np.bitwise_and(fps, query)].sum(axis=1, dtype="int32")
        either = popcount[np.bitwise_or(fps, query)].sum(axis=1, dtype="int32")
        return float((both / np.maximum(either, 1)).max())


_POPCOUNT_TABLE: Any = None


def _POPCOUNT() -> Any:
    """Bits per byte, as a lookup table. Turns Tanimoto over 260k fingerprints
    into two numpy passes instead of 260k RDKit calls."""
    global _POPCOUNT_TABLE
    if _POPCOUNT_TABLE is None:
        import numpy as np  # noqa: PLC0415

        _POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype="uint8")
    return _POPCOUNT_TABLE


# -- chemistry helpers -----------------------------------------------------


def _rdkit() -> Any:
    from rdkit import Chem, RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    return Chem


def canonical(smiles: str, min_heavy_atoms: int = 1) -> str | None:
    """RDKit-canonical SMILES, or None if it is not a molecule.

    Everything downstream is keyed on this: the database index, the docking
    cache, and whether two hypotheses are the same molecule written twice.

    `min_heavy_atoms` is for reading a molecule out of prose. RDKit parses "No"
    as nobelium and "I" as iodine, so a line of English is a valid one-atom
    molecule surprisingly often; nothing a method is trying to submit here is.
    """
    text = (smiles or "").strip()
    if not text:
        return None
    try:
        Chem = _rdkit()
        mol = Chem.MolFromSmiles(text)
        if mol is None or mol.GetNumHeavyAtoms() < min_heavy_atoms:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _fingerprint(smiles: str) -> Any:
    """Packed 1024-bit Morgan bits, shaped to broadcast against `fps.npy`."""
    try:
        import numpy as np  # noqa: PLC0415
        from rdkit.Chem import rdFingerprintGenerator  # noqa: PLC0415

        Chem = _rdkit()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        return np.packbits(gen.GetFingerprintAsNumPy(mol).astype("uint8"))
    except Exception:
        return None


def descriptors(smiles: str) -> dict[str, Any]:
    """QED and the descriptors a medicinal chemist reads first."""
    try:
        from rdkit.Chem import QED, Crippen, Descriptors, rdMolDescriptors  # noqa: PLC0415

        Chem = _rdkit()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"valid": False}
        return {
            "valid": True,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "qed": round(float(QED.qed(mol)), 4),
            "mw": round(float(Descriptors.MolWt(mol)), 2),
            "logp": round(float(Crippen.MolLogP(mol)), 2),
            "tpsa": round(float(rdMolDescriptors.CalcTPSA(mol)), 2),
            "hbd": int(rdMolDescriptors.CalcNumHBD(mol)),
            "hba": int(rdMolDescriptors.CalcNumHBA(mol)),
            "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            "rings": int(rdMolDescriptors.CalcNumRings(mol)),
            "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "heavy_atoms": int(mol.GetNumHeavyAtoms()),
            "formal_charge": int(Chem.GetFormalCharge(mol)),
            "stereocenters": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        }
    except ImportError:
        return {"valid": False, "error": "rdkit is not installed"}
    except Exception as e:
        return {"valid": False, "error": f"{type(e).__name__}: {e}"}


def _ensure_obabel_on_path() -> None:
    """Put the interpreter's own bin directory on PATH if obabel lives there.

    `dockstring` shells out to `obabel`, and the pip route (openbabel-wheel)
    installs it into the venv's bin -- which is on PATH when the venv is
    activated and absent when `.venv/bin/sheldon` is invoked directly. The failure
    is a FileNotFoundError raised a minute into the first dock, which reads like
    a broken install rather than a PATH difference.
    """
    if not sys.executable:
        return
    bindir = Path(sys.executable).parent
    if (bindir / "obabel").exists() and str(bindir) not in os.environ.get("PATH", "").split(":"):
        os.environ["PATH"] = f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"


# -- the oracle ------------------------------------------------------------


class DockingOracle:
    """Evaluate one molecule against one target: database, then cache, then Vina.

    `evaluate` always returns a dict and never raises -- an unparseable SMILES,
    an embedding failure, or a Vina crash are all normal outcomes of asking a
    language model for a molecule, and each one has to come back as a readable
    reason rather than as a traceback that kills the task.
    """

    def __init__(
        self,
        target: str,
        root: Path,
        *,
        cache: bool = True,
        num_cpus: int = 1,
        ph: float = 7.4,
        timeout: float = 600.0,
    ) -> None:
        self.target = target
        self.data = _Dataset.open(root)
        self.num_cpus = num_cpus
        self.ph = ph
        self.timeout = timeout
        self.cache = DiskCache(
            path=os.environ.get("SHELDON_DOCK_CACHE", REPO_ROOT / ".cache" / "dockstring.db"),
            enabled=cache,
        )

    def _key(self, smiles: str) -> str:
        # num_cpus is in the key on purpose: Vina's search is only reproducible
        # thread-for-thread, so a score computed on 8 threads is not the same
        # measurement as one computed on 1.
        return cache_key(
            {
                "kind": "dock",
                "target": self.target,
                "smiles": smiles,
                "ph": self.ph,
                "num_cpus": self.num_cpus,
            }
        )

    async def evaluate(self, smiles: str) -> dict[str, Any]:
        """{smiles, vina, qed, objective, source, error}. Never raises."""
        canon = canonical(smiles)
        if canon is None:
            return {
                "smiles": (smiles or "").strip()[:200],
                "error": "not a valid SMILES string",
                "source": "invalid",
            }

        props = descriptors(canon)
        qed = props.get("qed")
        if qed is None:
            return {"smiles": canon, "error": props.get("error", "QED unavailable"),
                    "source": "invalid"}

        found = self.data.lookup(canon, self.target)
        if found is not None:
            return self._finish(canon, found[0], qed, "dataset")

        hit = self.cache.get(self._key(canon))
        if hit is not None:
            if hit.get("error"):
                return {"smiles": canon, "error": hit["error"], "source": "cache"}
            return self._finish(canon, float(hit["vina"]), qed, "cache")

        vina, error = await self._dock(canon)
        self.cache.put(self._key(canon), {"vina": vina, "error": error}, tag=self.target)
        if error:
            return {"smiles": canon, "error": error, "source": "docked"}
        return self._finish(canon, float(vina), qed, "docked")

    def _finish(self, smiles: str, vina: float, qed: float, source: str) -> dict[str, Any]:
        if vina < VINA_FLOOR:
            # Vina reports these for degenerate poses; treating one as a real
            # affinity is how a benchmark gets "solved" by a molecule nobody
            # could make.
            return {"smiles": smiles, "error": f"implausible Vina score {vina}", "source": source}
        return {
            "smiles": smiles,
            "vina": round(float(vina), 2),
            "qed": round(float(qed), 4),
            "objective": round(objective(vina, qed), 3),
            "source": source,
        }

    async def _dock(self, smiles: str) -> tuple[float | None, str | None]:
        _ensure_obabel_on_path()

        def run() -> tuple[float | None, str | None]:
            try:
                from dockstring import load_target  # noqa: PLC0415

                score, _aux = load_target(self.target).dock(
                    smiles, pH=self.ph, num_cpus=self.num_cpus
                )
                if score is None:
                    return None, "docking produced no score"
                return float(score), None
            except ImportError:
                return None, "the dockstring package is not installed"
            except Exception as e:
                return None, f"{type(e).__name__}: {e}"

        try:
            return await asyncio.wait_for(asyncio.to_thread(run), timeout=self.timeout)
        except TimeoutError:
            return None, f"docking timed out after {self.timeout:.0f}s"


class DockingVerifier:
    """A label-free check: measure the molecule, and hold it to its own claim.

    A hypothesis here is a design plus a number -- "this scaffold reaches s = 11
    against PPARG". Docking settles that, which is what makes it a verifier
    rather than a second opinion. With no claim attached there is nothing to
    refute, so the measurement is reported and the candidate stands.
    """

    def __init__(self, oracle: DockingOracle, tolerance: float = 0.0) -> None:
        self.oracle = oracle
        self.tolerance = tolerance

    async def check(self, candidate: Any) -> Falsification:
        smiles = _candidate_text(candidate)
        if not smiles:
            return Falsification(refuted=False, detail="no SMILES in the candidate", checked=0)

        result = await self.oracle.evaluate(smiles)
        if result.get("error"):
            return Falsification(
                refuted=True,
                counterexamples=[{"kind": "invalid", "smiles": result["smiles"],
                                  "error": result["error"]}],
                detail=f"{result['smiles']}: {result['error']}",
                checked=1,
            )

        claim = _candidate_claim(candidate)
        measured = result["objective"]
        detail = (
            f"{result['smiles']}: vina={result['vina']} qed={result['qed']} "
            f"s={measured} ({result['source']})"
        )
        if claim is None:
            return Falsification(refuted=False, detail=detail, checked=1)
        if measured < claim - self.tolerance:
            return Falsification(
                refuted=True,
                counterexamples=[{
                    "kind": "mismatch", "smiles": result["smiles"], "claimed": claim,
                    "measured": measured, "vina": result["vina"], "qed": result["qed"],
                }],
                detail=f"{detail} -- claimed s >= {claim}",
                checked=1,
            )
        return Falsification(refuted=False, detail=detail, checked=1)


class ObjectiveProbe(Probe):
    """`DockingVerifier` as a decisive probe, so stage 1 measures before arguing.

    Opt-in (`benchmark_opts.oracle`), like every other oracle in this repo, and
    for the same reason: it changes what an attack costs and what it can find,
    so it has to be priceable rather than assumed. It is cheap in practice -- the
    candidate has usually been docked already and the cache answers.
    """

    name = "check_objective"
    description = (
        "Measure a molecule against the target and check it reaches the score claimed for it. "
        "This is the same docking that scores the task: a molecule that falls short here "
        "falls short, full stop."
    )
    decisive = True
    max_strength = 0.98

    def __init__(self, verifier: DockingVerifier) -> None:
        self.verifier = verifier

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "The molecule, as a SMILES string."},
                "claim": {
                    "type": "number",
                    "description": "The objective value claimed for it, if the claim names one.",
                },
            },
            "required": ["smiles"],
            "additionalProperties": False,
        }

    async def run(self, **kwargs: Any) -> ProbeResult:
        falsification = await self.verifier.check(kwargs)
        return ProbeResult(
            summary=falsification.detail or "checked",
            refutes=falsification.refuted,
            evidence=falsification.counterexamples,
            ok=falsification.checked > 0 or falsification.refuted,
        )


def _candidate_text(candidate: Any) -> str | None:
    """Whatever the candidate offered as its molecule, unvalidated.

    Kept separate from `_candidate_smiles` because the two callers want opposite
    things from a candidate that does not parse. `realize` wants None -- an
    unusable candidate must not become the submitted answer. The verifier wants
    the raw string, because "this is not a molecule" is a refutation, and
    dropping it here would report the candidate as having nothing to check.
    """
    if isinstance(candidate, str):
        return candidate or None
    if not isinstance(candidate, dict):
        return None
    for key in ("smiles", "molecule", "answer"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _candidate_smiles(candidate: Any) -> str | None:
    """The candidate's molecule, canonicalized, or None if there isn't one."""
    text = _candidate_text(candidate)
    return parse_smiles(text) if text else None


def _candidate_claim(candidate: Any) -> float | None:
    if not isinstance(candidate, dict):
        return None
    for key in ("claim", "claim_s", "objective", "expected_objective"):
        value = candidate.get(key)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            return float(value)
    return None


# -- answer parsing --------------------------------------------------------

_FENCE = re.compile(r"```(?:smiles|SMILES)?\s*\n?(.*?)```", re.DOTALL)
_LABELLED = re.compile(r"(?:^|\n)\s*(?:final\s+)?smiles\s*[:=]\s*(\S+)", re.IGNORECASE)
_TAGGED = re.compile(r"<smiles>\s*(.*?)\s*</smiles>", re.IGNORECASE | re.DOTALL)
#: SMILES never contain whitespace and use a narrow alphabet. Filtering on it
#: first is what stops "the" and "No" from being read as molecules.
_SMILESISH = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$:/\\%.*]+$")


def parse_smiles(text: str) -> str | None:
    """Pull a molecule out of whatever the model said, canonicalized.

    Ordered by how deliberate the marker is: an explicit tag, then a `SMILES:`
    label, then a fenced block, then the last line. Every candidate has to
    survive RDKit, and an unmarked line has to be more than a single atom, so a
    sentence that happens to parse is dropped rather than docked.
    """
    if isinstance(text, dict):
        text = str(text.get("smiles") or text.get("answer") or "")
    body = str(text or "")
    # (candidate, how many heavy atoms it has to have to be believed)
    candidates: list[tuple[str, int]] = []
    for pattern in (_TAGGED, _LABELLED):
        candidates.extend((m.strip(), 1) for m in pattern.findall(body))
    for block in _FENCE.findall(body):
        candidates.extend((line.strip(), 1) for line in block.strip().splitlines())
    # Last line first: a model that explains itself puts the answer at the end.
    candidates.extend((ln.strip(), 3) for ln in reversed(body.splitlines()) if ln.strip())

    for candidate, floor in candidates:
        token = candidate.strip().strip("`\"'").rstrip(".,;")
        if not token or len(token) > 400 or not _SMILESISH.match(token):
            continue
        canon = canonical(token, min_heavy_atoms=floor)
        if canon:
            return canon
    return None


# -- the benchmark ---------------------------------------------------------


@benchmarks.register("dockstring", "ds")
class Dockstring(Benchmark):
    name = "dockstring"
    splits = ("eval", "all")
    answer_kind = "smiles"
    answer_format = (
        "Give the molecule as one SMILES string on its own line, prefixed `SMILES:`. "
        "No name, no formula, no explanation on that line."
    )
    blurb = "design a molecule that docks well and stays drug-like (DOCKSTRING)"
    objective_probe = "dock_molecule"
    objective_key = "objective"

    def __init__(self, root: Any = None, **opts: Any) -> None:
        super().__init__(root, **opts)
        self.max_docking_calls = int(opts.get("max_docking_calls", 40))
        self.dataset_search = bool(opts.get("dataset_search", True))
        #: The free descriptor probe. `properties: false` with
        #: `dataset_search: false` leaves the docking oracle alone, which is
        #: the ablation asking what the loop is worth without a free half of
        #: the objective and a library of measured neighbours to calibrate on.
        self.properties_probe = bool(opts.get("properties", True))
        self.num_cpus = int(opts.get("dock_num_cpus", 1))
        self.dock_timeout = float(opts.get("dock_timeout", 600.0))
        self.cache_docking = bool(opts.get("dock_cache", True))
        self.pass_percentile = str(opts.get("pass_percentile", PASS_PERCENTILE))
        self.require_novel = bool(opts.get("require_novel", False))
        self.score_on = str(opts.get("score_on", "best"))
        #: task id -> {canonical smiles: measurement}. The probe's budget dict
        #: *is* this, so scoring can see every molecule the method docked and
        #: not just the one it chose to hand in.
        self._measured: dict[str, dict[str, dict[str, Any]]] = {}
        self.data = _Dataset.open(self.root)

    def objective_exhausted(self, task: Task) -> bool:
        limit = self.max_docking_calls
        return bool(limit) and len(self._measured.get(task.id) or {}) >= limit

    def probe_state(self, task: Task) -> Any:
        """The molecules this task has already measured, and what they scored.

        The rationed thing here is `max_docking_calls`, counted as the size of
        `_measured[task.id]`, so a run that comes back without it is handed a
        fresh 1000 and its number does not compare with one that ran straight
        through. Carrying the measurements rather than the count also keeps a
        re-dock free, exactly as it was the first time round.
        """
        return dict(self._measured.get(task.id) or {})

    def restore_probe_state(self, task: Task, state: Any) -> None:
        """Put back what `probe_state` returned.

        In place, never replacing the dict: `probes()` closes over the object
        `setdefault` returned, so rebinding it here would leave the probe
        spending a budget nothing else can see.
        """
        if not isinstance(state, dict):
            return
        seen = self._measured.setdefault(task.id, {})
        for smiles, result in state.items():
            if isinstance(result, dict) and result.get("smiles") and not result.get("error"):
                seen.setdefault(smiles, result)

    # -- data -------------------------------------------------------------

    def _targets(self, split: str) -> list[str]:
        chosen = self.opts.get("targets")
        if isinstance(chosen, str):
            chosen = [t.strip().upper() for t in chosen.split(",") if t.strip()]
        if chosen:
            return [str(t).upper() for t in chosen]
        if split == "all":
            return self.data.target_names() or list(PAPER_TARGETS)
        return list(PAPER_TARGETS)

    def load(self, split: str = "eval", limit: int | None = None, seed: int = 0) -> list[Task]:
        tasks: list[Task] = []
        for target in self._targets(split):
            entry = self.data.entry(target) or {}
            tasks.append(
                Task(
                    id=target,
                    benchmark=self.name,
                    question=self._render(target),
                    payload={"target": target, "seeds": [s for _, s in SEEDS]},
                    # The bar, not a molecule: there is no gold answer to leak
                    # here, only the percentile a submission is measured against.
                    answer={
                        "thresholds": entry.get("thresholds", {}),
                        "n": entry.get("n", 0),
                        "best": entry.get("best"),
                    },
                    meta={"note": TARGET_NOTES.get(target, "")},
                )
            )
        return self.subsample(tasks, limit, seed)

    def _render(self, target: str) -> str:
        note = TARGET_NOTES.get(target)
        seeds = ", ".join(f"{name} (`{smiles}`)" for name, smiles in SEEDS)
        budget = (
            f"You may call `dock_molecule` {self.max_docking_calls} times on this task"
            if self.max_docking_calls
            else "No docking oracle is available on this task -- you get one shot, unmeasured"
        )
        search = (
            "\n- `search_similar_molecules` returns compounds from a 260,000-molecule "
            "screening library that resemble a molecule you name, with their measured "
            "scores. Free, and the fastest way to calibrate what a good score even is."
            if self.dataset_search
            else ""
        )
        # A statement that lists a tool the run does not offer is worse than one
        # that lists none: the agent spends turns calling it and reads the
        # refusal as its own mistake.
        props = (
            "\n- `molecule_properties` is free and returns QED and the usual descriptors, so "
            "the drug-likeness half never needs to cost a docking call."
            if self.properties_probe
            else ""
        )
        return "\n".join(
            [
                f"# Design a ligand for {target}",
                "",
                f"Write a single small molecule, as a SMILES string, that binds **{target}**"
                + (f" -- {note}" if note else "")
                + " as tightly as possible while still looking like a drug.",
                "",
                "It is scored by",
                "",
                "```",
                "    s = -Vina - 10 * (1 - QED)",
                "```",
                "",
                "- **Vina** is AutoDock Vina's predicted binding free energy against "
                f"{target}, in kcal/mol. More negative is tighter; real ligands run from "
                "about -3 to -12.",
                "- **QED** is RDKit's quantitative estimate of drug-likeness, 0 to 1. The "
                "penalty is heavy on purpose: at QED 0.5 a molecule forfeits 5 kcal/mol of "
                "apparent affinity, so a large greasy binder loses to a smaller clean one.",
                "",
                "Higher `s` is better. You are competing against the best of a 260,000-compound "
                "library that was docked against this same target.",
                "",
                "## What you can measure",
                "",
                f"- `dock_molecule` runs the real docking and returns Vina, QED and `s`. "
                "It is the only way to know a molecule's score -- predicting it from structure "
                f"is exactly what nobody can do reliably. {budget}.",
                (props + search).lstrip("\n") or
                "Nothing else: no descriptor calculator and no library lookup, so every "
                "question about a molecule costs a docking call.",
                "",
                f"Somewhere to start, if you want it: {seeds}.",
                "",
                "## Answer",
                "",
                "One SMILES string, on its own line, prefixed `SMILES:`. It must be the "
                "molecule you actually want scored.",
            ]
        )

    # -- tools ------------------------------------------------------------

    def _oracle(self, target: str) -> DockingOracle:
        return DockingOracle(
            target,
            self.root,
            cache=self.cache_docking,
            num_cpus=self.num_cpus,
            timeout=self.dock_timeout,
        )

    def probes(self, task: Task) -> list[Probe]:
        """Docking, descriptors, and the library -- to every method alike.

        The docking oracle is not the ARC oracle in disguise. It does not tell a
        method whether its answer is right; it is the objective the answer is
        made of, and withholding it would leave nothing to optimise. What is
        rationed is how often it can be called, which is the actual currency of
        de novo design, and `max_docking_calls: 0` prices the whole thing.
        """
        target = task.payload["target"]
        oracle = self._oracle(target)
        limit = self.max_docking_calls
        # One budget per task, shared by every probe and by every concurrent
        # attack -- `ProbeSet.view()` shares probe objects, which is what makes
        # a within-round fan-out spend one budget instead of `attacks_k` of them.
        #
        # `pending` is what makes that budget hold under a fan-out. Checking
        # `len(seen)` and then awaiting the dock is a check-then-act across a
        # suspension point: with `attacks_k=3`, three attacks read the same
        # count of 39, all three pass the check, and the task evaluates 42
        # molecules against a cap of 40. Observed at 41/40 on two tasks before
        # this existed. Reserving the slot synchronously, before the first
        # await, is what closes it -- and sharing the in-flight task means two
        # attacks that pick the same molecule pay Vina once, not twice.
        seen: dict[str, dict[str, Any]] = self._measured.setdefault(task.id, {})
        pending: dict[str, asyncio.Task[dict[str, Any]]] = {}

        async def dock_molecule(smiles: str, refutes: bool = False) -> ProbeResult:
            canon = canonical(smiles)
            if canon is None:
                # Not a molecule. Evaluating it costs nothing and no docking
                # happens, so it must not spend a slot -- a model that fumbles a
                # SMILES should get the reason back, not a smaller budget.
                result = await oracle.evaluate(smiles)
            elif canon in seen:
                result = seen[canon]  # already measured: free, and not a new call
            else:
                inflight = pending.get(canon)
                if inflight is None:
                    if len(seen) + len(pending) >= limit:
                        return ProbeResult(
                            summary=f"docking budget exhausted ({limit} molecules evaluated)",
                            ok=False,
                        )
                    inflight = asyncio.ensure_future(oracle.evaluate(smiles))
                    pending[canon] = inflight
                try:
                    result = await inflight
                finally:
                    pending.pop(canon, None)
                if result.get("smiles") and not result.get("error"):
                    seen[result["smiles"]] = result
            if result.get("error"):
                return ProbeResult(summary=f"{result['smiles']}: {result['error']}", ok=False)
            best = max((r["objective"] for r in seen.values()), default=result["objective"])
            return ProbeResult(
                summary=(
                    f"{result['smiles']}\nvina={result['vina']} kcal/mol  qed={result['qed']}  "
                    f"s={result['objective']}  [{result['source']}]\n"
                    f"{len(seen)}/{limit} molecules evaluated; best s so far {best}"
                ),
                refutes=bool(refutes),
                evidence=[result],
            )

        async def molecule_properties(smiles: str) -> ProbeResult:
            props = descriptors(smiles)
            if not props.get("valid"):
                return ProbeResult(
                    summary=props.get("error", f"{smiles!r} is not a valid SMILES"), ok=False
                )
            return ProbeResult(
                summary=json.dumps(props, sort_keys=True),
                evidence=[props],
            )

        async def search_similar_molecules(smiles: str, k: int = 5) -> ProbeResult:
            if not self.data.ready:
                return ProbeResult(summary="the prepared library is not available", ok=False)
            hits = self.data.neighbours(smiles, target, k=max(1, min(int(k or 5), 10)))
            if not hits:
                return ProbeResult(summary="no comparable molecules found", ok=False)
            return ProbeResult(
                summary="\n".join(
                    f"{h['smiles']}  tanimoto={h['similarity']} vina={h['vina']} "
                    f"qed={h['qed']} s={h['objective']}"
                    for h in hits
                ),
                evidence=hits,
            )

        probes: list[Probe] = [*super().probes(task)]
        if limit:
            probes.append(
                CallableProbe(
                    name="dock_molecule",
                    description=(
                        f"Dock a molecule against {target} and return its Vina score, QED and "
                        "combined objective s = -vina - 10*(1-QED). This is the real measurement "
                        "the task is scored on. Budgeted: "
                        f"{limit} distinct molecules for this task, and re-measuring one you "
                        "have already docked is free."
                    ),
                    fn=dock_molecule,
                    input_schema={
                        "type": "object",
                        "properties": {
                            "smiles": {
                                "type": "string",
                                "description": "The molecule to dock, as a SMILES string.",
                            },
                        },
                        "required": ["smiles"],
                        "additionalProperties": False,
                    },
                    # The docking score is the objective itself, so a measurement
                    # that contradicts a claim about it is as close to settled as
                    # this benchmark gets -- short only of the agent misreading it.
                    max_strength=0.95,
                )
            )
        if self.properties_probe:
            probes.append(
                CallableProbe(
                    name="molecule_properties",
                    description=(
                        "Compute QED, molecular weight, logP, TPSA, ring and rotatable-bond "
                        "counts for a SMILES string. Free and instant -- it costs no docking "
                        "call, and it settles the drug-likeness half of the objective exactly."
                    ),
                    fn=molecule_properties,
                    input_schema={
                        "type": "object",
                        "properties": {
                            "smiles": {"type": "string", "description": "The molecule, as SMILES."}
                        },
                        "required": ["smiles"],
                        "additionalProperties": False,
                    },
                    # An exact computation of half the objective. Not decisive on
                    # its own: a poor QED is only fatal once the Vina half is known.
                    max_strength=0.9,
                )
            )
        if self.dataset_search and self.data.ready:
            probes.append(
                CallableProbe(
                    name="search_similar_molecules",
                    description=(
                        f"Find molecules in the 260,000-compound DOCKSTRING library similar to "
                        f"a SMILES string you give, and return their measured scores against "
                        f"{target}. Free. Use it to see what a good score looks like here, and "
                        "which substructures the library says are worth keeping."
                    ),
                    fn=search_similar_molecules,
                    input_schema={
                        "type": "object",
                        "properties": {
                            "smiles": {
                                "type": "string",
                                "description": "The molecule to find neighbours of, as SMILES.",
                            },
                            "k": {
                                "type": "integer",
                                "description": "How many neighbours to return (1-10).",
                            },
                        },
                        "required": ["smiles"],
                        "additionalProperties": False,
                    },
                    # Measured scores for real molecules -- but for *other*
                    # molecules. Evidence about a scaffold, never about the
                    # candidate itself.
                    max_strength=0.7,
                )
            )
        # `oracle_probes` is opt-in and empty by default -- but it has to be
        # reached from here, or `benchmark_opts.oracle` is a switch wired to
        # nothing and the ablation silently measures the default twice.
        return [*probes, *self.oracle_probes(task)]

    def oracle_probes(self, task: Task) -> list[Probe]:
        if not self.opts.get("oracle", False):
            return []
        verifier = self.verifier(task)
        return [ObjectiveProbe(verifier)] if verifier is not None else []

    def verifier(self, task: Task) -> Verifier | None:
        return DockingVerifier(self._oracle(task.payload["target"]))

    def python_context(self, task: Task) -> dict[str, Any]:
        """What the sandbox starts with. RDKit is importable in there; docking is
        not -- Vina takes far longer than the snippet timeout, which is what
        keeps the docking budget meaningful."""
        return {
            "target": task.payload["target"],
            "seed_molecules": task.payload["seeds"],
            "objective": "s = -vina - 10 * (1 - QED)",
        }

    # -- answers ----------------------------------------------------------

    def parse_answer(self, text: str) -> Any:
        return parse_smiles(text)

    def realize(self, task: Task, candidate: dict[str, Any]) -> Any:
        """A hypothesis carrying a molecule yields the answer with no further call."""
        return _candidate_smiles(candidate)

    # -- scoring ----------------------------------------------------------

    async def score(self, task: Task, pred: Prediction, ctx: RunContext) -> Score:
        target = task.payload["target"]
        gold = task.answer or {}
        thresholds = gold.get("thresholds") or {}
        # Always through the parser: `answer` is a string either way, and
        # "SMILES: CCO" is a string that is not a molecule.
        raw = pred.answer if isinstance(pred.answer, str) else str(pred.answer or "")
        smiles = parse_smiles(raw)

        if not smiles:
            return Score(
                task_id=task.id,
                score=0.0,
                metrics={"valid": 0.0},
                detail={"target": target, "reason": "no parseable SMILES in the answer"},
            )

        # A fresh oracle: scoring must not draw on the method's docking budget,
        # and must not top it up either.
        result = await self._oracle(target).evaluate(smiles)
        if result.get("error"):
            return Score(
                task_id=task.id,
                score=0.0,
                metrics={"valid": 0.0},
                detail={"target": target, "smiles": result.get("smiles"),
                        "reason": result["error"]},
            )

        submitted = float(result["objective"])

        # What the paper scores, and what every optimizer baseline it compares
        # against scores, is the best molecule found during the search — not a
        # final answer, because a search does not have one. Scoring the handed-in
        # answer measures something different and much harsher: a method that
        # docks an excellent molecule and submits a worse one reads as a failure
        # when the search actually succeeded. The gap between the two is
        # itself worth reporting — it is a defect in `finalize`, not in the
        # search — so both numbers stay in the metrics whichever one is primary.
        best = self.best_found(task.id)
        if best is not None and best["objective"] > submitted:
            scored, source = best, "best_found"
        else:
            scored, source = result, "submitted"
        if self.score_on == "submitted":
            scored, source = result, "submitted"

        s = float(scored["objective"])
        smiles = scored["smiles"]
        result = scored
        bar = thresholds.get(self.pass_percentile)
        percentile = self.data.percentile(target, s) if self.data.ready else float("nan")
        similarity = self.data.max_similarity(smiles) if self.data.ready else float("nan")

        metrics = {
            "valid": 1.0,
            "objective": round(s, 3),
            "vina": float(result["vina"]),
            "qed": float(result["qed"]),
            # The handed-in answer, always, so the cost of `finalize` picking
            # the wrong molecule stays visible even when it is not what scored.
            "submitted_objective": round(submitted, 3),
            "finalize_loss": round(max(0.0, s - submitted), 3),
            "molecules_measured": float(len(self._measured.get(task.id, {}))),
        }
        if math.isfinite(percentile):
            metrics["percentile"] = round(percentile, 4)
        if math.isfinite(similarity):
            # 1.0 means the submission *is* a library molecule. The paper's claim
            # is about novel chemistry, so a copied answer has to be visible in
            # the table rather than only in the trace.
            metrics["max_library_similarity"] = round(similarity, 3)
            metrics["in_library"] = float(similarity >= 0.999)
        for name, cut in thresholds.items():
            metrics[f"beats_{name}"] = float(s >= cut)

        # A library molecule does not "surpass the library" -- it is a member of
        # it. A model can walk `search_similar_molecules` to a high-percentile
        # compound and never design anything. That is a fair strategy under the paper's
        # feedback design, so it stays a pass by default and shows up as
        # `in_library`; `require_novel` is the switch that stops counting it.
        copied = bool(metrics.get("in_library"))
        passed = bar is not None and s >= bar and not (self.require_novel and copied)

        return Score(
            task_id=task.id,
            score=float(passed),
            metrics=metrics,
            detail={
                "target": target,
                "smiles": result["smiles"],
                "source": result["source"],
                "scored_on": source,
                "submitted_smiles": parse_smiles(raw),
                "thresholds": thresholds,
                "bar": bar,
                "pass_percentile": self.pass_percentile,
                "require_novel": self.require_novel,
                "library_best": gold.get("best"),
            },
        )

    def best_found(self, task_id: str) -> dict[str, Any] | None:
        """The best molecule the oracle measured on this task, or None.

        Populated by `dock_molecule`, so a method that never docked anything has
        nothing here and is scored on what it handed in — which is right: the
        `codex` engine cannot reach `ctx.tools` at all, and its answer is the
        only molecule it ever had.
        """
        measured = self._measured.get(task_id) or {}
        return max(measured.values(), key=lambda r: r["objective"], default=None)

    # -- readiness --------------------------------------------------------

    def check(self) -> list[str]:
        problems: list[str] = []
        try:
            import rdkit  # noqa: F401, PLC0415
        except ImportError:
            problems.append("rdkit is not installed:  uv pip install '.[dockstring]'")
        try:
            import numpy  # noqa: F401, PLC0415
        except ImportError:
            problems.append("numpy is not installed:  uv pip install '.[dockstring]'")

        missing = self.data.missing()
        if missing:
            problems.append(
                f"prepared data missing from {self.root} ({', '.join(missing)}). Build it with:"
                "\n    scripts/prepare_dockstring.py --download"
            )

        if self.max_docking_calls:
            problems.extend(self._check_docking())
        return problems

    @staticmethod
    def _check_docking() -> list[str]:
        """Whether Vina can actually run. Three separate ways for it not to."""
        try:
            from dockstring.utils import check_obabel_install, get_vina_path  # noqa: PLC0415
        except ImportError:
            return [
                "the dockstring package is not installed:  uv pip install '.[dockstring]'"
            ]
        _ensure_obabel_on_path()
        problems: list[str] = []
        try:
            check_obabel_install()
        except Exception as e:
            problems.append(
                f"openbabel is not usable ({type(e).__name__}). dockstring shells out to "
                "`obabel`; install it with `uv pip install openbabel-wheel` (or conda) and "
                "make sure it is on PATH."
            )
        try:
            get_vina_path()
        except Exception as e:
            problems.append(f"AutoDock Vina binary not found: {e}")
        return problems

    def describe(self) -> str:
        problems = self.check()
        if problems:
            return f"{self.name:<18} {self.blurb}  [{problems[0].splitlines()[0]}]"
        n = len(self._targets("eval"))
        molecules = self.data.meta.get("molecules", 0)
        return (
            f"{self.name:<18} {self.blurb}  "
            f"[ready, {n} targets, {molecules} library molecules]"
        )

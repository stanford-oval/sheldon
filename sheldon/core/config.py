"""Resolved experiment configuration.

One `Config` describes exactly one (benchmark, method) cell. Sweeps live in the
CLI, which expands comma-separated benchmarks/methods into a Config per cell.
Every field here either changes the numbers or is recorded in the run manifest,
so `config_hash` is a real identity for "this result".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# Fields that change how fast a run goes but never what it produces. Excluded
# from the hash so resuming with different concurrency still matches.
# base_url is plumbing: the same proxy behind a new hostname produces the same
# numbers, so moving it must not fork every run directory. `provider` is not —
# it changes which SDK and which serving path produced the result.
_NON_SEMANTIC = {"concurrency", "call_concurrency", "cache", "resume", "run_dir",
                 "tag", "notes", "base_url", "trace_prompts"}


@dataclass
class Config:
    benchmark: str = "arc-agi-1"
    method: str = "sheldon"
    split: str = "eval"
    limit: int | None = None
    tasks: str = ""      # comma-separated ids; a debugging lens, not a split
    seed: int = 0

    # model
    engine: str = "toolloop"        # how one reasoning step runs; `sheldon ls` lists them
    provider: str = "auto"          # auto | anthropic | openai
    base_url: str | None = None     # OpenAI-compatible endpoint; defaults to $OPENAI_BASE_URL
    model: str = "gpt-5.6-terra"
    effort: str = "high"
    max_tokens: int = 16000
    thinking: bool = True

    # per-task budget
    max_rounds: int | None = 4      # None: only spend caps end the loop
    max_calls: int | None = 40
    max_usd: float | None = 1.0
    max_seconds: float | None = None   # wall clock per task; None: no limit

    # method-specific knobs, passed through to Method(**params)
    params: dict[str, Any] = field(default_factory=dict)

    # execution (non-semantic)
    concurrency: int = 8       # tasks in flight, summed across cells
    call_concurrency: int = 0  # API calls in flight; 0 = 4x concurrency
    cache: bool = True
    trace_prompts: bool = True   # full prompts in traces; off keeps them small
    resume: bool = True
    tag: str = ""
    notes: str = ""

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> Config:
        raw: dict[str, Any] = {}
        if path:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            base = raw.pop("extends", None)
            if base:
                parent = asdict(cls.load(Path(path).parent / base))
                child = raw.get("params")
                raw = {**parent, **raw}
                # `params` merges with the parent exactly as it merges with a
                # CLI `-p`, and for the same reason: a config that names one
                # knob is refining the base, not replacing it. The shallow
                # merge this replaces dropped the parent's whole params block,
                # which cost every such config its `probes` -- 47 of them, each
                # asking for a shell and a web search and silently getting
                # neither. Nothing else showed it: the other defaults happen to
                # equal their code defaults, so only `available:` in the trace
                # ever said so.
                if child and parent.get("params"):
                    raw["params"] = _merge_params(parent["params"], child)
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        merged = {**raw, **{k: v for k, v in overrides.items() if v is not None}}
        # params merges rather than replaces, so CLI -p flags refine a file.
        # Recursively: `-p benchmark_opts.report_dir=x` arrives nested, and a
        # one-level merge replaced the file's whole `benchmark_opts` with that
        # one key -- the interface, the URLs and the verifier were gone, and
        # the arm silently ran a different benchmark under the same name.
        if "params" in raw and "params" in overrides and overrides["params"]:
            merged["params"] = _merge_params(raw["params"], overrides["params"])
        return cls(**merged)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        payload = {k: v for k, v in self.to_dict().items() if k not in _NON_SEMANTIC}
        if not payload.get("tasks"):
            # Absent unless used, so adding the field left every existing run's
            # id — and its resume — untouched. When it *is* used it belongs in
            # the hash: a run over three named tasks must not resume into the
            # directory of a run over fifty.
            payload.pop("tasks", None)
        if payload.get("max_seconds") is None:
            # Same rule: a wall limit that is not set must not move the id of
            # every run made before the field existed.
            payload.pop("max_seconds", None)
        version = self.method_version
        if version != "1":
            # Only a bumped method joins the payload, so declaring a new version
            # for one method leaves every other run's id — and its resume — alone.
            payload["method_version"] = version
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:10]

    @property
    def method_version(self) -> str:
        """What the method says about its own behaviour. See `Method.version`."""
        try:
            from sheldon.methods import get as get_method  # noqa: PLC0415

            return str(getattr(get_method(self.method), "version", "1") or "1")
        except Exception:
            # An unregistered method still has to produce an id — `--dry-run`
            # and the report both build one for configs they cannot construct.
            return "1"

    @property
    def run_id(self) -> str:
        parts = [self.benchmark, self.method, self.split]
        if self.tag:
            parts.append(self.tag)
        parts.append(self.config_hash)
        return "__".join(parts)


def _merge_params(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """`override` on top of `base`, descending into dicts on both sides.

    A dict override refines the file's dict; anything else replaces it, so
    `-p probes=[]` still clears a list rather than merging into it.
    """
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_params(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    return Config.load(path, **overrides)

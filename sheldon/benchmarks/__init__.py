"""Importing this package is what registers the benchmarks.

Add a benchmark: drop a module in here, decorate the class with
`@benchmarks.register("name")`, and add one import line below.
"""

from sheldon.benchmarks import (  # noqa: F401
    arc_agi,
    dockstring,
    med_dx,
    orca_bench,
    skydiscover,
)
from sheldon.benchmarks.base import Benchmark
from sheldon.core.registry import benchmarks


def get(name: str, **opts) -> Benchmark:
    return benchmarks.get(name)(**opts)


__all__ = ["Benchmark", "benchmarks", "get"]

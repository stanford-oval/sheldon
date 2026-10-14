"""Name -> class lookup, so adding a benchmark or method is one decorator.

Nothing imports benchmarks or methods by module path. `sheldon.benchmarks`
and `sheldon.methods` import their submodules on package import, which is what
populates these registries.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type[T]] = {}
        self._aliases: dict[str, str] = {}

    def register(self, *names: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            primary = names[0] if names else getattr(cls, "name", cls.__name__)
            if primary in self._items:
                raise ValueError(f"duplicate {self.kind} name: {primary}")
            self._items[primary] = cls
            for alias in names[1:]:
                self._aliases[alias] = primary
            return cls

        return deco

    def get(self, name: str) -> type[T]:
        key = self._aliases.get(name, name)
        if key not in self._items:
            raise KeyError(
                f"unknown {self.kind} {name!r}. available: {', '.join(sorted(self._items))}"
            )
        return self._items[key]

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return self._aliases.get(name, name) in self._items


benchmarks: Registry = Registry("benchmark")
methods: Registry = Registry("method")
engines: Registry = Registry("engine")

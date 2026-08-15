"""Addressable graph rewrites, each guarded by a numerical parity check.

The prior project's equivalent was a single script that applied five rewrites
in one pass. That worked for one model. Generalising it needs three changes:

**Each rewrite is separately addressable and separately reportable.** The zoo
needs to record *which* patch made a model compile, because that is the finding
— "this class of export needs its Slice bounds normalised" is a constraint,
while "we ran the patcher" is not.

**Each rewrite declares when it applies.** A patch that fires unconditionally
is a patch that eventually corrupts a graph it was never meant to touch.

**No rewrite is accepted without a parity check.** Every patch here claims to
be semantics-preserving. A patch that silently is not produces a model that
compiles, runs, and is wrong — the single worst outcome the zoo can produce,
because everything downstream looks like success. So the default is: run the
graph before and after on the same inputs, and reject the rewrite if the
outputs move. Where parity cannot be evaluated (unpinned shapes, ops ORT will
not run), that is recorded rather than assumed away.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------


@dataclass
class PatchResult:
    name: str
    applied: bool
    changed: int = 0
    note: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    #: Result of the numerical parity check. None means it could not be run —
    #: which is a reportable state, not a pass.
    parity_ok: bool | None = None
    parity_max_abs: float | None = None
    parity_note: str = ""

    @property
    def trustworthy(self) -> bool:
        """Applied and demonstrated to preserve semantics."""
        return self.applied and self.parity_ok is True


@dataclass
class Patch:
    name: str
    fixes: str
    summary: str
    applies: Callable[[Any], bool]
    run: Callable[[Any], tuple[Any, PatchResult]]
    #: Some rewrites change the graph's *interface* (removing an input that is
    #: folded to a constant, for example). Those cannot be parity-checked by
    #: feeding identical inputs to both graphs, so they declare it.
    changes_signature: bool = False


REGISTRY: dict[str, Patch] = {}


def register(
    name: str,
    *,
    fixes: str,
    summary: str,
    changes_signature: bool = False,
) -> Callable:
    """Decorate a `(model) -> (model, PatchResult)` function into the registry."""

    def decorator(fn: Callable) -> Callable:
        applies = getattr(fn, "_applies", lambda _model: True)
        REGISTRY[name] = Patch(
            name=name,
            fixes=fixes,
            summary=summary,
            applies=applies,
            run=fn,
            changes_signature=changes_signature,
        )
        return fn

    return decorator


def applies_when(predicate: Callable[[Any], bool]) -> Callable:
    """Attach an `applies` predicate to a patch function."""

    def decorator(fn: Callable) -> Callable:
        fn._applies = predicate
        return fn

    return decorator


def available() -> list[Patch]:
    return [REGISTRY[k] for k in sorted(REGISTRY)]


def get(name: str) -> Patch:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown patch {name!r}; available: {', '.join(sorted(REGISTRY))}"
        ) from None


# Importing the modules is what populates the registry.
from zoo.graph.patches import structural  # noqa: E402,F401

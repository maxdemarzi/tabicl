"""Semirings, the algebra both aggregation paths are special cases of.

A semiring is a set with two operations: ``add`` (+) combines *alternatives*, ``mul``
(x) combines *independent parts*. Aggregation over a join is variable elimination in
some semiring -- that is the content of FAQ (Abo Khamis, Ngo, Rudra, PODS 2016), and
it is why counting a query, summing weights over it, and finding the cheapest witness
are the same algorithm with different operators rather than three algorithms.

The practical payoff here is that ``_relational.py``'s roll-up only ever applied
``add``. Combining across a *hop* -- a per-order discount multiplied by that order's
item total, a probability along a path -- needs ``mul``, and had no expression at all
before this module existed.

Two properties are tracked because they decide what a caller may do:

* ``invertible`` -- whether ``add`` has an inverse. Counts and sums can be un-added,
  so a deleted row can be retracted from a maintained aggregate. ``min``/``max``
  cannot: removing the current minimum tells you nothing about the next one. This is
  exactly the line between statistics that survive incremental maintenance and ones
  that need a rebuild.
* ``pandas_agg`` -- the name of the vectorised ``groupby`` reduction implementing
  ``add``, when one exists, so the tree path stays fast instead of calling a Python
  lambda per group.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

__all__ = [
    "Semiring",
    "SUM_PRODUCT",
    "MIN_PLUS",
    "MAX_PLUS",
    "BOOLEAN",
    "BUILTIN_SEMIRINGS",
    "check_semiring_laws",
]


@dataclass(frozen=True)
class Semiring:
    """An algebraic structure ``(K, add, mul, zero, one)``.

    Parameters
    ----------
    name : str
        Identifier, used in generated feature names and error messages.

    zero : Any
        Identity for ``add``, and annihilator for ``mul``. Represents "no derivation".

    one : Any
        Identity for ``mul``. Represents "an empty derivation".

    add : callable
        Combines alternatives. Must be associative and commutative.

    mul : callable
        Combines independent parts. Must be associative and distribute over ``add``.

    invertible : bool, default=False
        Whether ``add`` admits an inverse, i.e. whether a contribution can be
        retracted. Determines whether an aggregate can be maintained under deletes.

    pandas_agg : str, optional
        Name of the equivalent vectorised pandas ``groupby`` reduction, when one
        exists.
    """

    name: str
    zero: Any
    one: Any
    add: Callable[[Any, Any], Any]
    mul: Callable[[Any, Any], Any]
    invertible: bool = False
    pandas_agg: Optional[str] = None

    def sum(self, values: Sequence[Any]) -> Any:
        """Fold ``add`` over ``values``; ``zero`` when empty."""
        total = self.zero
        for value in values:
            total = self.add(total, value)
        return total

    def product(self, values: Sequence[Any]) -> Any:
        """Fold ``mul`` over ``values``; ``one`` when empty."""
        total = self.one
        for value in values:
            total = self.mul(total, value)
        return total


# Counting and summing. The ordinary arithmetic ring, and the only built-in whose
# `add` can be undone -- which is why count/sum/sumsq survive incremental deletes and
# min/max do not.
SUM_PRODUCT = Semiring(
    name="sum_product",
    zero=0.0,
    one=1.0,
    add=lambda x, y: x + y,
    mul=lambda x, y: x * y,
    invertible=True,
    pandas_agg="sum",
)

# Tropical. `add = min` picks the best alternative, `mul = +` accumulates along a
# path, so this answers "cheapest witness" -- shortest chain of intermediaries,
# lowest-cost route between two entities.
MIN_PLUS = Semiring(
    name="min_plus",
    zero=math.inf,
    one=0.0,
    add=min,
    mul=lambda x, y: x + y,
    invertible=False,
    pandas_agg="min",
)

# The dual: widest/bottleneck reasoning, e.g. the strongest chain of relationships.
MAX_PLUS = Semiring(
    name="max_plus",
    zero=-math.inf,
    one=0.0,
    add=max,
    mul=lambda x, y: x + y,
    invertible=False,
    pandas_agg="max",
)

# Reachability. Answers existence rather than quantity, and is what a join degenerates
# to when payloads are dropped.
BOOLEAN = Semiring(
    name="boolean",
    zero=False,
    one=True,
    add=lambda x, y: bool(x) or bool(y),
    mul=lambda x, y: bool(x) and bool(y),
    invertible=False,
    pandas_agg="max",
)

BUILTIN_SEMIRINGS = {s.name: s for s in (SUM_PRODUCT, MIN_PLUS, MAX_PLUS, BOOLEAN)}


def check_semiring_laws(semiring: Semiring, samples: Sequence[Any]) -> None:
    """Verify the semiring axioms on ``samples``, raising on the first violation.

    Worth running on any custom semiring before trusting an aggregate built from it:
    a structure that fails distributivity will still *produce numbers* during variable
    elimination, they will simply be the wrong numbers, and silently so -- the failure
    looks like a plausible feature rather than a crash.
    """
    add, mul, zero, one = semiring.add, semiring.mul, semiring.zero, semiring.one

    def close(a: Any, b: Any) -> bool:
        if isinstance(a, bool) or isinstance(b, bool):
            return bool(a) == bool(b)
        if a == b:
            return True
        if isinstance(a, float) and isinstance(b, float) and math.isinf(a) and math.isinf(b):
            return math.copysign(1, a) == math.copysign(1, b)
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)

    for x in samples:
        if not close(add(x, zero), x):
            raise ValueError(f"{semiring.name}: zero is not an identity for add (x={x!r})")
        if not close(mul(x, one), x):
            raise ValueError(f"{semiring.name}: one is not an identity for mul (x={x!r})")
        if not close(mul(x, zero), zero):
            raise ValueError(f"{semiring.name}: zero does not annihilate mul (x={x!r})")

    for x in samples:
        for y in samples:
            if not close(add(x, y), add(y, x)):
                raise ValueError(f"{semiring.name}: add is not commutative ({x!r}, {y!r})")
            for z in samples:
                if not close(add(add(x, y), z), add(x, add(y, z))):
                    raise ValueError(f"{semiring.name}: add is not associative ({x!r},{y!r},{z!r})")
                if not close(mul(mul(x, y), z), mul(x, mul(y, z))):
                    raise ValueError(f"{semiring.name}: mul is not associative ({x!r},{y!r},{z!r})")
                if not close(mul(x, add(y, z)), add(mul(x, y), mul(x, z))):
                    raise ValueError(
                        f"{semiring.name}: mul does not distribute over add ({x!r},{y!r},{z!r})"
                    )

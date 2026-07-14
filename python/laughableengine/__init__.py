"""Small, exact polynomial rings backed by the native laughableengine core."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Union

from ._laughableengine import (  # type: ignore[import-not-found]
    ColonClosureResult,
    ColonClosureStep,
    ColonClosureTransition,
    CotangentClassProof,
    CotangentH1InputError,
    CotangentH1Presentation,
    CotangentH1ResourceLimit,
    CycleAuditResult,
    CycleScreenResult,
    DivisionResult,
    ExactMatrix,
    FullH1ResourceLimit,
    H1ActionResult,
    H1ScreenResult,
    Ideal,
    InverseSystemDiscoveryRecord,
    InverseSystemResourceLimit,
    InverseSystemResult,
    MatrixSpaceRankResult,
    Polynomial,
    Ring,
    SparseExactMatrix,
    _make_gf_ring,
    _make_qq_ring,
)

__version__ = "0.3.0"

__all__ = [
    "ColonClosureResult",
    "ColonClosureStep",
    "ColonClosureTransition",
    "CotangentClassProof",
    "CotangentH1InputError",
    "CotangentH1Presentation",
    "CotangentH1ResourceLimit",
    "CycleAuditResult",
    "CycleScreenResult",
    "DivisionResult",
    "ExactMatrix",
    "FullH1ResourceLimit",
    "GF",
    "H1ActionResult",
    "H1ScreenResult",
    "Ideal",
    "InverseSystemDiscoveryRecord",
    "InverseSystemResourceLimit",
    "InverseSystemResult",
    "MatrixSpaceRankResult",
    "Polynomial",
    "QQ",
    "Ring",
    "SparseExactMatrix",
    "__version__",
    "elementary_symmetric",
]


def _variable_names(variables: Union[str, Iterable[str]]) -> list[str]:
    if isinstance(variables, str):
        names = [name for name in re.split(r"[\s,]+", variables.strip()) if name]
    else:
        names = list(variables)

    if not names:
        raise ValueError("a polynomial ring needs at least one variable")
    if not all(isinstance(name, str) for name in names):
        raise TypeError("polynomial variable names must be strings")
    return names


def _order_name(order: str) -> str:
    if not isinstance(order, str):
        raise TypeError("the monomial order must be a string")
    normalized = order.lower().replace("_", "").replace("-", "")
    aliases = {
        "lex": "lex",
        "lexicographic": "lex",
        "grevlex": "grevlex",
        "degrevlex": "grevlex",
        "gradedreverselexicographic": "grevlex",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError("order must be 'lex' or 'grevlex'") from error


def QQ(
    variables: Union[str, Iterable[str]],
    order: str = "grevlex",
) -> Ring:
    """Construct ``QQ[variables]`` with lex or grevlex ordering."""

    return _make_qq_ring(_variable_names(variables), _order_name(order))


def GF(
    modulus: int,
    variables: Union[str, Iterable[str]],
    order: str = "grevlex",
) -> Ring:
    """Construct ``GF(modulus)[variables]`` for a prime modulus below 2^31."""

    if not isinstance(modulus, int):
        raise TypeError("the finite-field modulus must be an integer")
    if modulus < 2 or modulus >= 2**31:
        raise ValueError("the finite-field prime modulus must satisfy 2 <= p < 2^31")
    return _make_gf_ring(
        modulus, _variable_names(variables), _order_name(order)
    )


def elementary_symmetric(
    polynomials: Iterable[Polynomial], degree: int
) -> Polynomial:
    """Return the elementary symmetric polynomial of the requested degree."""

    values = tuple(polynomials)
    if not values:
        raise ValueError("elementary_symmetric needs at least one polynomial")
    if not all(isinstance(value, Polynomial) for value in values):
        raise TypeError("elementary_symmetric inputs must be polynomials")
    if not isinstance(degree, int):
        raise TypeError("the elementary-symmetric degree must be an integer")
    if degree < 0:
        raise ValueError("the elementary-symmetric degree must be nonnegative")

    ring = values[0].ring
    if any(value.ring is not ring for value in values[1:]):
        raise ValueError(
            "elementary_symmetric inputs must belong to one exact ring context"
        )
    if degree > len(values):
        return ring.zero()

    coefficients = [ring.zero() for _ in range(degree + 1)]
    coefficients[0] = ring.one()
    for value_index, value in enumerate(values, start=1):
        for index in range(min(value_index, degree), 0, -1):
            coefficients[index] = (
                coefficients[index] + value * coefficients[index - 1]
            )
    return coefficients[degree]

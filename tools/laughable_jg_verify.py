#!/usr/bin/env python3
"""Independent verifier for a raw (J, g) cotangent-H1 certificate.

This file intentionally uses only the Python standard library.  In particular,
it neither imports nor invokes laughableengine, Sage, Singular, or Macaulay2.

The accepted JSON object has exactly these keys (plus the fixed schema tag):

    {
      "schema": "laughable-jg-v1",
      "field": {"kind": "GF", "modulus": "5"},
      "variables": ["x"],
      "order": "grevlex",
      "ideal_generators": [
        [{"coefficient": "1", "exponents": [5]}]
      ],
      "g": [{"coefficient": "1", "exponents": [5]}]
    }

For QQ, a coefficient is instead a canonical object such as
{"numerator":"-2","denominator":"3"}.  Terms must be nonzero, unique,
and sorted from greatest to least in the declared monomial order.  There are
no fields for discovery-side bases, remainders, ranks, or Boolean claims.
"""

import argparse
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import json
import math
import re
import sys
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple, Union


Coefficient = Union[int, Fraction]
Monomial = Tuple[int, ...]
Polynomial = Dict[Monomial, Coefficient]


class SchemaError(ValueError):
    """The input is not a canonical laughable-jg-v1 document."""


class ResourceLimit(RuntimeError):
    """A verifier-controlled resource budget was exhausted."""


class InternalVerificationError(RuntimeError):
    """An invariant of the independent implementation failed."""


@dataclass(frozen=True)
class Limits:
    max_input_bytes: int = 2_000_000
    max_variables: int = 6
    max_generators: int = 128
    max_square_generators: int = 10_000
    max_input_terms: int = 50_000
    max_polynomial_terms: int = 200_000
    max_exponent: int = 65_535
    max_coefficient_bits: int = 16_384
    max_buchberger_pairs: int = 200_000
    max_reduction_steps: int = 2_000_000
    max_arithmetic_operations: int = 20_000_000
    max_standard_monomials: int = 100_000
    max_matrix_entries: int = 5_000_000
    max_rank_operations: int = 20_000_000


class Meter:
    def __init__(self, limits: Limits) -> None:
        self.limits = limits
        self.input_terms = 0
        self.buchberger_pairs = 0
        self.reduction_steps = 0
        self.arithmetic_operations = 0
        self.rank_operations = 0

    def add_input_terms(self, count: int) -> None:
        self.input_terms += count
        if self.input_terms > self.limits.max_input_terms:
            raise ResourceLimit("input term limit exceeded")

    def add_pair(self) -> None:
        self.buchberger_pairs += 1
        if self.buchberger_pairs > self.limits.max_buchberger_pairs:
            raise ResourceLimit("Buchberger pair limit exceeded")

    def add_reduction_step(self) -> None:
        self.reduction_steps += 1
        if self.reduction_steps > self.limits.max_reduction_steps:
            raise ResourceLimit("polynomial reduction step limit exceeded")

    def add_arithmetic(self, count: int = 1) -> None:
        if count < 0:
            raise InternalVerificationError("negative arithmetic operation count")
        self.arithmetic_operations += count
        if self.arithmetic_operations > self.limits.max_arithmetic_operations:
            raise ResourceLimit("polynomial arithmetic operation limit exceeded")

    def add_rank_operations(self, count: int = 1) -> None:
        if count < 0:
            raise InternalVerificationError("negative rank operation count")
        self.rank_operations += count
        if self.rank_operations > self.limits.max_rank_operations:
            raise ResourceLimit("matrix rank operation limit exceeded")

    def check_polynomial_size(self, count: int) -> None:
        if count > self.limits.max_polynomial_terms:
            raise ResourceLimit("live polynomial term limit exceeded")

    def check_monomial(self, monomial: Monomial) -> None:
        if any(exponent < 0 or exponent > self.limits.max_exponent
               for exponent in monomial):
            raise ResourceLimit("generated monomial exponent limit exceeded")

    def check_coefficient(self, coefficient: Coefficient) -> None:
        if isinstance(coefficient, Fraction):
            bits = max(
                abs(coefficient.numerator).bit_length(),
                coefficient.denominator.bit_length(),
            )
            if bits > self.limits.max_coefficient_bits:
                raise ResourceLimit("rational coefficient bit limit exceeded")


@dataclass(frozen=True)
class Field:
    kind: str
    modulus: Optional[int] = None

    def zero(self) -> Coefficient:
        return Fraction(0) if self.kind == "QQ" else 0

    def one(self) -> Coefficient:
        return Fraction(1) if self.kind == "QQ" else 1

    def from_int(self, value: int) -> Coefficient:
        if self.kind == "QQ":
            return Fraction(value)
        assert self.modulus is not None
        return value % self.modulus

    def normalize(self, value: Coefficient) -> Coefficient:
        if self.kind == "QQ":
            if not isinstance(value, Fraction):
                return Fraction(value)
            return value
        assert self.modulus is not None
        return int(value) % self.modulus

    def is_zero(self, value: Coefficient) -> bool:
        return self.normalize(value) == 0

    def add(self, left: Coefficient, right: Coefficient) -> Coefficient:
        return self.normalize(left + right)

    def subtract(self, left: Coefficient, right: Coefficient) -> Coefficient:
        return self.normalize(left - right)

    def negate(self, value: Coefficient) -> Coefficient:
        return self.normalize(-value)

    def multiply(self, left: Coefficient, right: Coefficient) -> Coefficient:
        return self.normalize(left * right)

    def inverse(self, value: Coefficient) -> Coefficient:
        value = self.normalize(value)
        if self.is_zero(value):
            raise InternalVerificationError("attempted inversion of zero")
        if self.kind == "QQ":
            assert isinstance(value, Fraction)
            return 1 / value
        assert self.modulus is not None
        return pow(int(value), self.modulus - 2, self.modulus)

    def divide(self, numerator: Coefficient, denominator: Coefficient) -> Coefficient:
        return self.multiply(numerator, self.inverse(denominator))


@dataclass(frozen=True)
class MonomialOrder:
    name: str
    variable_count: int

    def key(self, monomial: Monomial) -> Tuple[int, ...]:
        if len(monomial) != self.variable_count:
            raise InternalVerificationError("monomial has the wrong arity")
        if self.name == "lex":
            return monomial
        return (sum(monomial),) + tuple(-value for value in reversed(monomial))


@dataclass(frozen=True)
class Certificate:
    field: Field
    variables: Tuple[str, ...]
    order: MonomialOrder
    ideal_generators: Tuple[Polynomial, ...]
    g: Polynomial


_SIGNED_INTEGER = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")
_UNSIGNED_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _strict_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError("duplicate JSON object key: " + key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise SchemaError("non-finite JSON numeric constant is forbidden: " + value)


def _reject_float(value: str) -> None:
    raise SchemaError("JSON floating-point numbers are forbidden: " + value)


def _parse_json_integer(value: str) -> int:
    # Exponents are the only JSON numbers in the schema and are tiny.  Bound
    # conversion before constructing a Python bigint from hostile input.
    digits = value[1:] if value.startswith("-") else value
    if len(digits) > 20:
        raise ResourceLimit("JSON integer digit limit exceeded")
    return int(value)


def _expect_exact_keys(value: Any, expected: Iterable[str], path: str) -> Dict[str, Any]:
    if type(value) is not dict:
        raise SchemaError(path + " must be an object")
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        details: List[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise SchemaError(path + " has wrong keys (" + "; ".join(details) + ")")
    return value


def _parse_signed_integer(value: Any, path: str, bit_limit: int) -> int:
    if type(value) is not str or _SIGNED_INTEGER.fullmatch(value) is None:
        raise SchemaError(path + " must be a canonical signed decimal string")
    negative = value.startswith("-")
    digits = value[1:] if negative else value
    result = _bounded_decimal_integer(digits, path, bit_limit)
    if negative:
        result = -result
    if abs(result).bit_length() > bit_limit:
        raise ResourceLimit(path + " exceeds the coefficient bit limit")
    return result


def _parse_unsigned_integer(value: Any, path: str, bit_limit: int) -> int:
    if type(value) is not str or _UNSIGNED_INTEGER.fullmatch(value) is None:
        raise SchemaError(path + " must be a canonical unsigned decimal string")
    result = _bounded_decimal_integer(value, path, bit_limit)
    if result.bit_length() > bit_limit:
        raise ResourceLimit(path + " exceeds the coefficient bit limit")
    return result


def _bounded_decimal_integer(digits: str, path: str, bit_limit: int) -> int:
    # Avoid Python's version-dependent whole-string bigint digit limit while
    # still bounding work before conversion.
    decimal_limit = (bit_limit * 30_103) // 100_000 + 2
    if len(digits) > decimal_limit:
        raise ResourceLimit(path + " exceeds the coefficient bit limit")
    result = 0
    first = len(digits) % 9
    position = 0
    if first:
        result = int(digits[:first])
        position = first
    while position < len(digits):
        result = result * 1_000_000_000 + int(digits[position:position + 9])
        position += 9
    return result


def _is_prime(value: int) -> bool:
    """Deterministic Miller-Rabin for the accepted 64-bit modulus range."""
    if value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    # This set is deterministic for every n < 2^64.
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        residue = pow(base, exponent, value)
        if residue in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            residue = (residue * residue) % value
            if residue == value - 1:
                break
        else:
            return False
    return True


def _parse_field(value: Any, limits: Limits) -> Field:
    if type(value) is not dict:
        raise SchemaError("field must be an object")
    kind = value.get("kind")
    if kind == "QQ":
        _expect_exact_keys(value, ("kind",), "field")
        return Field("QQ")
    if kind == "GF":
        _expect_exact_keys(value, ("kind", "modulus"), "field")
        modulus = _parse_unsigned_integer(
            value["modulus"], "field.modulus", limits.max_coefficient_bits
        )
        if modulus >= 2 ** 64:
            raise SchemaError("field.modulus must be smaller than 2^64")
        if not _is_prime(modulus):
            raise SchemaError("field.modulus must be prime")
        return Field("GF", modulus)
    raise SchemaError("field.kind must be QQ or GF")


def _parse_coefficient(value: Any, field: Field, path: str, limits: Limits) -> Coefficient:
    if field.kind == "GF":
        coefficient = _parse_unsigned_integer(value, path, limits.max_coefficient_bits)
        assert field.modulus is not None
        if coefficient >= field.modulus:
            raise SchemaError(path + " must be in the canonical range 0..p-1")
        if coefficient == 0:
            raise SchemaError(path + " is zero; zero terms must be omitted")
        return coefficient

    coefficient_object = _expect_exact_keys(
        value, ("numerator", "denominator"), path
    )
    numerator = _parse_signed_integer(
        coefficient_object["numerator"], path + ".numerator", limits.max_coefficient_bits
    )
    denominator = _parse_unsigned_integer(
        coefficient_object["denominator"],
        path + ".denominator",
        limits.max_coefficient_bits,
    )
    if denominator == 0:
        raise SchemaError(path + ".denominator must be positive")
    if numerator == 0:
        raise SchemaError(path + " is zero; zero terms must be omitted")
    if math.gcd(abs(numerator), denominator) != 1:
        raise SchemaError(path + " must be a reduced rational number")
    return Fraction(numerator, denominator)


def _parse_polynomial(
    value: Any,
    field: Field,
    order: MonomialOrder,
    meter: Meter,
    path: str,
) -> Polynomial:
    if type(value) is not list:
        raise SchemaError(path + " must be a list of terms")
    meter.add_input_terms(len(value))
    if len(value) > meter.limits.max_polynomial_terms:
        raise ResourceLimit(path + " exceeds the per-polynomial term limit")
    result: Polynomial = {}
    previous_key: Optional[Tuple[int, ...]] = None
    for index, raw_term in enumerate(value):
        term_path = "%s[%d]" % (path, index)
        term = _expect_exact_keys(raw_term, ("coefficient", "exponents"), term_path)
        raw_exponents = term["exponents"]
        if type(raw_exponents) is not list or len(raw_exponents) != order.variable_count:
            raise SchemaError(term_path + ".exponents has the wrong length")
        exponents: List[int] = []
        for variable, exponent in enumerate(raw_exponents):
            if type(exponent) is not int or isinstance(exponent, bool):
                raise SchemaError(
                    "%s.exponents[%d] must be a JSON integer" % (term_path, variable)
                )
            if exponent < 0 or exponent > meter.limits.max_exponent:
                raise ResourceLimit(term_path + " exponent limit exceeded")
            exponents.append(exponent)
        monomial = tuple(exponents)
        key = order.key(monomial)
        if previous_key is not None and not previous_key > key:
            raise SchemaError(path + " terms must be unique and strictly descending")
        previous_key = key
        coefficient = _parse_coefficient(
            term["coefficient"], field, term_path + ".coefficient", meter.limits
        )
        result[monomial] = coefficient
    return result


def parse_certificate(document: Any, limits: Limits) -> Certificate:
    top = _expect_exact_keys(
        document,
        ("schema", "field", "variables", "order", "ideal_generators", "g"),
        "certificate",
    )
    if top["schema"] != "laughable-jg-v1":
        raise SchemaError("unsupported certificate schema")
    field = _parse_field(top["field"], limits)

    raw_variables = top["variables"]
    if type(raw_variables) is not list:
        raise SchemaError("variables must be a list")
    if not 1 <= len(raw_variables) <= limits.max_variables:
        raise SchemaError("variables must contain between 1 and %d names" % limits.max_variables)
    variables: List[str] = []
    for index, name in enumerate(raw_variables):
        if type(name) is not str or _VARIABLE.fullmatch(name) is None:
            raise SchemaError("variables[%d] is not a valid ASCII identifier" % index)
        if name in variables:
            raise SchemaError("variable names must be unique")
        variables.append(name)

    if top["order"] not in ("lex", "grevlex"):
        raise SchemaError("order must be lex or grevlex")
    order = MonomialOrder(top["order"], len(variables))
    meter = Meter(limits)

    raw_generators = top["ideal_generators"]
    if type(raw_generators) is not list:
        raise SchemaError("ideal_generators must be a list")
    if len(raw_generators) > limits.max_generators:
        raise ResourceLimit("ideal generator limit exceeded")
    generators = tuple(
        _parse_polynomial(value, field, order, meter, "ideal_generators[%d]" % index)
        for index, value in enumerate(raw_generators)
    )
    g = _parse_polynomial(top["g"], field, order, meter, "g")
    return Certificate(field, tuple(variables), order, generators, g)


def _store_term(
    polynomial: Polynomial,
    monomial: Monomial,
    coefficient: Coefficient,
    field: Field,
    meter: Meter,
) -> None:
    meter.check_monomial(monomial)
    coefficient = field.normalize(coefficient)
    meter.check_coefficient(coefficient)
    if field.is_zero(coefficient):
        polynomial.pop(monomial, None)
    else:
        polynomial[monomial] = coefficient
    meter.check_polynomial_size(len(polynomial))


def _monomial_divides(divisor: Monomial, value: Monomial) -> bool:
    return all(left <= right for left, right in zip(divisor, value))


def _monomial_add(left: Monomial, right: Monomial, meter: Meter) -> Monomial:
    result = tuple(a + b for a, b in zip(left, right))
    meter.check_monomial(result)
    return result


def _monomial_subtract(value: Monomial, divisor: Monomial) -> Monomial:
    if not _monomial_divides(divisor, value):
        raise InternalVerificationError("inexact monomial division")
    return tuple(a - b for a, b in zip(value, divisor))


def _monomial_lcm(left: Monomial, right: Monomial) -> Monomial:
    return tuple(max(a, b) for a, b in zip(left, right))


def _leading_monomial(polynomial: Polynomial, order: MonomialOrder) -> Monomial:
    if not polynomial:
        raise InternalVerificationError("zero polynomial has no leading monomial")
    return max(polynomial, key=order.key)


def _poly_subtract(
    left: Polynomial, right: Polynomial, field: Field, meter: Meter
) -> Polynomial:
    meter.add_arithmetic(len(left) + len(right))
    result = dict(left)
    for monomial, coefficient in right.items():
        _store_term(
            result,
            monomial,
            field.subtract(result.get(monomial, field.zero()), coefficient),
            field,
            meter,
        )
    return result


def _poly_multiply_by_term(
    polynomial: Polynomial,
    coefficient: Coefficient,
    monomial: Monomial,
    field: Field,
    meter: Meter,
) -> Polynomial:
    if field.is_zero(coefficient) or not polynomial:
        return {}
    meter.add_arithmetic(len(polynomial))
    result: Polynomial = {}
    for source_monomial, source_coefficient in polynomial.items():
        target = _monomial_add(source_monomial, monomial, meter)
        _store_term(
            result,
            target,
            field.multiply(source_coefficient, coefficient),
            field,
            meter,
        )
    return result


def _poly_multiply(
    left: Polynomial, right: Polynomial, field: Field, meter: Meter
) -> Polynomial:
    if not left or not right:
        return {}
    meter.add_arithmetic(len(left) * len(right))
    result: Polynomial = {}
    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            target = _monomial_add(left_monomial, right_monomial, meter)
            combined = field.add(
                result.get(target, field.zero()),
                field.multiply(left_coefficient, right_coefficient),
            )
            _store_term(result, target, combined, field, meter)
    return result


def _monic(
    polynomial: Polynomial, order: MonomialOrder, field: Field, meter: Meter
) -> Polynomial:
    if not polynomial:
        return {}
    leading = _leading_monomial(polynomial, order)
    inverse = field.inverse(polynomial[leading])
    zero_monomial = (0,) * order.variable_count
    return _poly_multiply_by_term(polynomial, inverse, zero_monomial, field, meter)


def _normal_form(
    polynomial: Polynomial,
    divisors: Sequence[Polynomial],
    order: MonomialOrder,
    field: Field,
    meter: Meter,
) -> Polynomial:
    if any(not divisor for divisor in divisors):
        raise InternalVerificationError("zero divisor in normal-form computation")
    pending = dict(polynomial)
    remainder: Polynomial = {}
    meter.check_polynomial_size(len(pending))
    divisor_leads = [(_leading_monomial(value, order), value) for value in divisors]
    while pending:
        meter.add_reduction_step()
        leading = _leading_monomial(pending, order)
        leading_coefficient = pending[leading]
        selected: Optional[Tuple[Monomial, Polynomial]] = None
        for divisor_leading, divisor in divisor_leads:
            if _monomial_divides(divisor_leading, leading):
                selected = (divisor_leading, divisor)
                break
        if selected is None:
            del pending[leading]
            _store_term(remainder, leading, leading_coefficient, field, meter)
            continue

        divisor_leading, divisor = selected
        quotient_monomial = _monomial_subtract(leading, divisor_leading)
        quotient_coefficient = field.divide(
            leading_coefficient, divisor[divisor_leading]
        )
        meter.add_arithmetic(len(divisor))
        for monomial, coefficient in divisor.items():
            target = _monomial_add(monomial, quotient_monomial, meter)
            updated = field.subtract(
                pending.get(target, field.zero()),
                field.multiply(quotient_coefficient, coefficient),
            )
            _store_term(pending, target, updated, field, meter)
        meter.check_polynomial_size(len(pending) + len(remainder))
    return remainder


def _s_polynomial(
    left: Polynomial,
    right: Polynomial,
    order: MonomialOrder,
    field: Field,
    meter: Meter,
) -> Polynomial:
    left_leading = _leading_monomial(left, order)
    right_leading = _leading_monomial(right, order)
    common = _monomial_lcm(left_leading, right_leading)
    left_factor = _monomial_subtract(common, left_leading)
    right_factor = _monomial_subtract(common, right_leading)
    left_scaled = _poly_multiply_by_term(
        left, field.inverse(left[left_leading]), left_factor, field, meter
    )
    right_scaled = _poly_multiply_by_term(
        right, field.inverse(right[right_leading]), right_factor, field, meter
    )
    return _poly_subtract(left_scaled, right_scaled, field, meter)


def _is_nonzero_constant(polynomial: Polynomial) -> bool:
    return len(polynomial) == 1 and all(value == 0 for value in next(iter(polynomial)))


def _reduce_basis(
    basis: Sequence[Polynomial],
    order: MonomialOrder,
    field: Field,
    meter: Meter,
) -> List[Polynomial]:
    if not basis:
        return []
    leads = [_leading_monomial(value, order) for value in basis]
    keep = [True] * len(basis)
    for index, leading in enumerate(leads):
        for other, other_leading in enumerate(leads):
            if index == other:
                continue
            if _monomial_divides(other_leading, leading) and (
                leading != other_leading or other < index
            ):
                keep[index] = False
                break
    reduced = [_monic(value, order, field, meter)
               for value, retained in zip(basis, keep) if retained]
    for index in range(len(reduced)):
        others = reduced[:index] + reduced[index + 1:]
        remainder = _normal_form(reduced[index], others, order, field, meter)
        if not remainder:
            raise InternalVerificationError("a reduced Groebner basis element vanished")
        reduced[index] = _monic(remainder, order, field, meter)
    reduced.sort(
        key=lambda value: order.key(_leading_monomial(value, order)), reverse=True
    )
    return reduced


def _verify_groebner_basis(
    basis: Sequence[Polynomial],
    generators: Sequence[Polynomial],
    order: MonomialOrder,
    field: Field,
    meter: Meter,
) -> None:
    for generator in generators:
        if _normal_form(generator, basis, order, field, meter):
            raise InternalVerificationError("computed basis lost an input generator")
    for right in range(1, len(basis)):
        for left in range(right):
            meter.add_pair()
            s_value = _s_polynomial(
                basis[left], basis[right], order, field, meter
            )
            if _normal_form(s_value, basis, order, field, meter):
                raise InternalVerificationError("computed basis failed Buchberger's criterion")


def _groebner_basis(
    generators: Sequence[Polynomial],
    order: MonomialOrder,
    field: Field,
    meter: Meter,
) -> List[Polynomial]:
    basis: List[Polynomial] = []
    for generator in generators:
        if not generator:
            continue
        remainder = (
            _normal_form(generator, basis, order, field, meter)
            if basis else dict(generator)
        )
        if not remainder:
            continue
        remainder = _monic(remainder, order, field, meter)
        if _is_nonzero_constant(remainder):
            one = {(0,) * order.variable_count: field.one()}
            _verify_groebner_basis([one], generators, order, field, meter)
            return [one]
        basis.append(remainder)

    pairs: Deque[Tuple[int, int]] = deque()
    for right in range(1, len(basis)):
        for left in range(right):
            pairs.append((left, right))
    while pairs:
        meter.add_pair()
        left, right = pairs.popleft()
        s_value = _s_polynomial(basis[left], basis[right], order, field, meter)
        remainder = _normal_form(s_value, basis, order, field, meter)
        if not remainder:
            continue
        remainder = _monic(remainder, order, field, meter)
        if _is_nonzero_constant(remainder):
            one = {(0,) * order.variable_count: field.one()}
            _verify_groebner_basis([one], generators, order, field, meter)
            return [one]
        new_index = len(basis)
        basis.append(remainder)
        for index in range(new_index):
            pairs.append((index, new_index))

    reduced = _reduce_basis(basis, order, field, meter)
    _verify_groebner_basis(reduced, generators, order, field, meter)
    return reduced


def _derivative(
    polynomial: Polynomial,
    variable: int,
    field: Field,
    meter: Meter,
) -> Polynomial:
    result: Polynomial = {}
    meter.add_arithmetic(len(polynomial))
    for monomial, coefficient in polynomial.items():
        exponent = monomial[variable]
        if exponent == 0:
            continue
        target = list(monomial)
        target[variable] -= 1
        derivative_coefficient = field.multiply(coefficient, field.from_int(exponent))
        if not field.is_zero(derivative_coefficient):
            _store_term(
                result, tuple(target), derivative_coefficient, field, meter
            )
    return result


def _is_unit_basis(basis: Sequence[Polynomial]) -> bool:
    return bool(basis) and _is_nonzero_constant(basis[0])


def _finite_quotient_with_order(
    basis: Sequence[Polynomial], order: MonomialOrder
) -> bool:
    if _is_unit_basis(basis):
        return True
    pure_power_seen = [False] * order.variable_count
    for polynomial in basis:
        leading = _leading_monomial(polynomial, order)
        support = [index for index, exponent in enumerate(leading) if exponent]
        if len(support) == 1:
            pure_power_seen[support[0]] = True
    return all(pure_power_seen)


def _standard_monomials(
    basis: Sequence[Polynomial],
    order: MonomialOrder,
    meter: Meter,
) -> List[Monomial]:
    if not _finite_quotient_with_order(basis, order):
        raise InternalVerificationError("standard monomials requested for infinite quotient")
    if _is_unit_basis(basis):
        return []
    leading_monomials = [_leading_monomial(value, order) for value in basis]
    zero = (0,) * order.variable_count
    frontier: Deque[Monomial] = deque([zero])
    seen = {zero}
    result: List[Monomial] = []
    while frontier:
        monomial = frontier.popleft()
        result.append(monomial)
        if len(result) > meter.limits.max_standard_monomials:
            raise ResourceLimit("standard monomial limit exceeded")
        for variable in range(order.variable_count):
            child = list(monomial)
            child[variable] += 1
            child_tuple = tuple(child)
            meter.check_monomial(child_tuple)
            if child_tuple in seen:
                continue
            if any(_monomial_divides(leading, child_tuple)
                   for leading in leading_monomials):
                continue
            seen.add(child_tuple)
            if len(seen) > meter.limits.max_standard_monomials:
                raise ResourceLimit("standard monomial limit exceeded")
            frontier.append(child_tuple)
    result.sort(key=order.key)
    return result


def _supported_at_origin(
    basis: Sequence[Polynomial],
    standard: Sequence[Monomial],
    order: MonomialOrder,
    field: Field,
    meter: Meter,
) -> bool:
    if not standard:
        return True
    exponent = len(standard)
    for variable in range(order.variable_count):
        monomial = [0] * order.variable_count
        monomial[variable] = exponent
        meter.check_monomial(tuple(monomial))
        power = {tuple(monomial): field.one()}
        if _normal_form(power, basis, order, field, meter):
            return False
    return True


def _square_generators(
    generators: Sequence[Polynomial],
    field: Field,
    meter: Meter,
) -> List[Polynomial]:
    count = len(generators) * (len(generators) + 1) // 2
    if count > meter.limits.max_square_generators:
        raise ResourceLimit("ideal-square generator limit exceeded")
    result: List[Polynomial] = []
    for right in range(len(generators)):
        for left in range(right + 1):
            result.append(_poly_multiply(
                generators[left], generators[right], field, meter
            ))
    return result


def _matrix_rank(
    matrix: List[List[Coefficient]],
    column_count: int,
    field: Field,
    meter: Meter,
) -> int:
    if not matrix or column_count == 0:
        return 0
    if any(len(row) != column_count for row in matrix):
        raise InternalVerificationError("matrix has inconsistent row lengths")
    rank = 0
    row_count = len(matrix)
    for column in range(column_count):
        pivot = None
        for row in range(rank, row_count):
            if not field.is_zero(matrix[row][column]):
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        inverse = field.inverse(matrix[rank][column])
        for target_column in range(column, column_count):
            meter.add_rank_operations()
            value = field.multiply(matrix[rank][target_column], inverse)
            meter.check_coefficient(value)
            matrix[rank][target_column] = value
        for row in range(rank + 1, row_count):
            factor = matrix[row][column]
            if field.is_zero(factor):
                continue
            for target_column in range(column, column_count):
                meter.add_rank_operations(2)
                value = field.subtract(
                    matrix[row][target_column],
                    field.multiply(factor, matrix[rank][target_column]),
                )
                meter.check_coefficient(value)
                matrix[row][target_column] = value
        rank += 1
        if rank == column_count:
            break
    return rank


def _multiplication_rank(
    g: Polynomial,
    domain: Sequence[Monomial],
    codomain: Sequence[Monomial],
    square_basis: Sequence[Polynomial],
    order: MonomialOrder,
    field: Field,
    meter: Meter,
) -> int:
    entries = len(domain) * len(codomain)
    if entries > meter.limits.max_matrix_entries:
        raise ResourceLimit("multiplication matrix entry limit exceeded")
    row_index = {monomial: index for index, monomial in enumerate(codomain)}
    matrix = [
        [field.zero() for _ in range(len(domain))]
        for _ in range(len(codomain))
    ]
    for column, monomial in enumerate(domain):
        product = _poly_multiply_by_term(g, field.one(), monomial, field, meter)
        remainder = _normal_form(product, square_basis, order, field, meter)
        for target, coefficient in remainder.items():
            row = row_index.get(target)
            if row is None:
                raise InternalVerificationError(
                    "normal form lies outside the J^2 standard basis"
                )
            matrix[row][column] = coefficient
    return _matrix_rank(matrix, len(domain), field, meter)


def _blank_report(status: str, reason: str) -> Dict[str, Any]:
    return {
        "cycle_valid": None,
        "derivatives_in_J": None,
        "faithful": False,
        "finite_quotient": None,
        "g_in_J": None,
        "length_P_mod_J2": None,
        "length_Q": None,
        "multiplication_rank": None,
        "reason": reason,
        "status": status,
        "supported_at_origin": None,
        "verified": False,
    }


def verify(certificate: Certificate, limits: Limits) -> Dict[str, Any]:
    meter = Meter(limits)
    field = certificate.field
    order = certificate.order
    generators = list(certificate.ideal_generators)

    ideal_basis = _groebner_basis(generators, order, field, meter)
    square_generators = _square_generators(generators, field, meter)
    square_basis = _groebner_basis(square_generators, order, field, meter)

    g_remainder = _normal_form(certificate.g, ideal_basis, order, field, meter)
    g_in_ideal = not g_remainder
    derivatives = [
        _derivative(certificate.g, variable, field, meter)
        for variable in range(order.variable_count)
    ]
    derivative_memberships = [
        not _normal_form(value, ideal_basis, order, field, meter)
        for value in derivatives
    ]
    derivatives_in_ideal = all(derivative_memberships)
    cycle_valid = g_in_ideal and derivatives_in_ideal

    finite = _finite_quotient_with_order(ideal_basis, order)
    supported = False
    length_q: Optional[int] = None
    length_square: Optional[int] = None
    multiplication_rank: Optional[int] = None
    unit_quotient = _is_unit_basis(ideal_basis)
    if finite:
        standard = _standard_monomials(ideal_basis, order, meter)
        if not _finite_quotient_with_order(square_basis, order):
            raise InternalVerificationError("J is zero-dimensional but J^2 is not")
        square_standard = _standard_monomials(square_basis, order, meter)
        length_q = len(standard)
        length_square = len(square_standard)
        supported = _supported_at_origin(
            ideal_basis, standard, order, field, meter
        )
        if g_in_ideal and not unit_quotient:
            multiplication_rank = _multiplication_rank(
                certificate.g,
                standard,
                square_standard,
                square_basis,
                order,
                field,
                meter,
            )

    injective = (
        multiplication_rank is not None
        and length_q is not None
        and multiplication_rank == length_q
    )
    faithful = (
        finite
        and not unit_quotient
        and supported
        and cycle_valid
        and injective
    )
    reasons: List[str] = []
    if unit_quotient:
        reasons.append("unit quotient")
    elif not finite:
        reasons.append("J is positive-dimensional")
    if finite and not supported:
        reasons.append("J is not supported at the origin")
    if not g_in_ideal:
        reasons.append("g is not in J")
    if not derivatives_in_ideal:
        reasons.append("a derivative of g is not in J")
    if finite and g_in_ideal and not unit_quotient and not injective:
        reasons.append("multiplication by g is not injective")

    return {
        "cycle_valid": cycle_valid,
        "derivatives_in_J": derivatives_in_ideal,
        "faithful": faithful,
        "finite_quotient": finite,
        "g_in_J": g_in_ideal,
        "length_P_mod_J2": length_square,
        "length_Q": length_q,
        "multiplication_rank": multiplication_rank,
        "reason": "verified" if faithful else "; ".join(reasons),
        "status": "verified" if faithful else "not_faithful",
        "supported_at_origin": supported,
        "verified": faithful,
    }


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise SchemaError("command line: " + message)


def _nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="independently verify a raw laughable J,g certificate"
    )
    parser.add_argument("certificate", help="certificate JSON path, or - for stdin")
    parser.add_argument("--max-input-bytes", type=_nonnegative_integer, default=2_000_000)
    parser.add_argument("--max-input-terms", type=_nonnegative_integer, default=50_000)
    parser.add_argument("--max-polynomial-terms", type=_nonnegative_integer, default=200_000)
    parser.add_argument("--max-exponent", type=_nonnegative_integer, default=65_535)
    parser.add_argument("--max-coefficient-bits", type=_nonnegative_integer, default=16_384)
    parser.add_argument("--max-buchberger-pairs", type=_nonnegative_integer, default=200_000)
    parser.add_argument("--max-reduction-steps", type=_nonnegative_integer, default=2_000_000)
    parser.add_argument("--max-arithmetic-operations", type=_nonnegative_integer, default=20_000_000)
    parser.add_argument("--max-standard-monomials", type=_nonnegative_integer, default=100_000)
    parser.add_argument("--max-matrix-entries", type=_nonnegative_integer, default=5_000_000)
    parser.add_argument("--max-rank-operations", type=_nonnegative_integer, default=20_000_000)
    return parser


def _limits_from_args(args: argparse.Namespace) -> Limits:
    return Limits(
        max_input_bytes=args.max_input_bytes,
        max_input_terms=args.max_input_terms,
        max_polynomial_terms=args.max_polynomial_terms,
        max_exponent=args.max_exponent,
        max_coefficient_bits=args.max_coefficient_bits,
        max_buchberger_pairs=args.max_buchberger_pairs,
        max_reduction_steps=args.max_reduction_steps,
        max_arithmetic_operations=args.max_arithmetic_operations,
        max_standard_monomials=args.max_standard_monomials,
        max_matrix_entries=args.max_matrix_entries,
        max_rank_operations=args.max_rank_operations,
    )


def _read_document(path: str, byte_limit: int) -> Any:
    if path == "-":
        data = sys.stdin.buffer.read(byte_limit + 1)
    else:
        with open(path, "rb") as source:
            data = source.read(byte_limit + 1)
    if len(data) > byte_limit:
        raise ResourceLimit("input byte limit exceeded")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SchemaError("certificate is not valid UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_parse_json_integer,
        )
    except json.JSONDecodeError as error:
        raise SchemaError("invalid JSON: " + error.msg) from error


def _emit(report: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = _parser().parse_args(argv)
        limits = _limits_from_args(args)
        document = _read_document(args.certificate, limits.max_input_bytes)
        certificate = parse_certificate(document, limits)
        report = verify(certificate, limits)
        _emit(report)
        return 0 if report["faithful"] else 2
    except ResourceLimit as error:
        _emit(_blank_report("resource_limit", str(error)))
        return 3
    except MemoryError:
        _emit(_blank_report("resource_limit", "memory allocation failed"))
        return 3
    except SchemaError as error:
        _emit(_blank_report("malformed", str(error)))
        return 1
    except OSError as error:
        _emit(_blank_report("io_error", str(error)))
        return 1
    except Exception as error:  # A stable boundary for a standalone checker.
        _emit(_blank_report("internal_error", str(error)))
        return 1


if __name__ == "__main__":
    sys.exit(main())

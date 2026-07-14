"""Independent Sage oracle for laughableengine's mandatory H1 regressions.

This file is intentionally not part of the default CTest run: Sage is a
development oracle, not a runtime dependency.  Run it with

    sage tests/oracles/cotangent_h1_reference.sage
"""

from sage.all import GF, QQ, PolynomialRing, block_matrix, matrix, vector


def coordinates(ideal, basis, polynomial):
    remainder = ideal.reduce(polynomial)
    return vector(
        ideal.ring().base_ring(),
        [remainder.monomial_coefficient(monomial) for monomial in basis],
    )


def cotangent_data(ring, ideal):
    field = ring.base_ring()
    variables = ring.gens()
    square = ideal * ideal
    quotient_basis = list(ideal.normal_basis())
    square_basis = list(square.normal_basis())
    n = len(quotient_basis)
    n2 = len(square_basis)

    reduction = matrix(
        field,
        n,
        n2,
        lambda row, column: coordinates(
            ideal, quotient_basis, square_basis[column]
        )[row],
    )
    derivatives = [
        matrix(
            field,
            n,
            n2,
            lambda row, column, variable=variable: coordinates(
                ideal,
                quotient_basis,
                square_basis[column].derivative(variable),
            )[row],
        )
        for variable in variables
    ]
    h1_kernel = block_matrix(
        [[part] for part in [reduction] + derivatives], subdivide=False
    ).right_kernel()

    multiplications = [
        matrix(
            field,
            n,
            n,
            lambda row, column, variable=variable: coordinates(
                ideal, quotient_basis, variable * quotient_basis[column]
            )[row],
        )
        for variable in variables
    ]
    socle_kernel = block_matrix(
        [[part] for part in multiplications], subdivide=False
    ).right_kernel()

    h1_basis = [
        sum(coeff * monomial for coeff, monomial in zip(row, square_basis))
        for row in h1_kernel.basis()
    ]
    socle_basis = [
        sum(coeff * monomial for coeff, monomial in zip(row, quotient_basis))
        for row in socle_kernel.basis()
    ]
    products = [
        coordinates(square, square_basis, socle * cycle)
        for socle in socle_basis
        for cycle in h1_basis
    ]
    product_span_dimension = matrix(field, products).rank() if products else 0

    return {
        "square": square,
        "quotient_basis": quotient_basis,
        "square_basis": square_basis,
        "length": n,
        "square_length": n2,
        "h1_basis": h1_basis,
        "h1_dimension": h1_kernel.dimension(),
        "socle_basis": socle_basis,
        "socle_dimension": socle_kernel.dimension(),
        "product_span_dimension": product_span_dimension,
    }


def action_rank(data, cycle):
    field = data["square"].ring().base_ring()
    rows = [
        coordinates(
            data["square"], data["square_basis"], socle * cycle
        )
        for socle in data["socle_basis"]
    ]
    return matrix(field, rows).rank() if rows else 0


# Three-variable seed from the kernel specification.
P = PolynomialRing(QQ, names=("x", "y", "z"), order="degrevlex")
x, y, z = P.gens()
f = y**5 + z**5 + y**2 * z**2
J = P.ideal(x**2, x * y, x * z, f, f.derivative(y), f.derivative(z))
seed = cotangent_data(P, J)
assert (seed["length"], seed["square_length"]) == (11, 48)
assert (seed["h1_dimension"], seed["socle_dimension"]) == (19, 3)
# All action products span one line, and at least one product is nonzero, so
# the maximum rank of any individual action is exactly one.
assert seed["product_span_dimension"] == 1
assert max(action_rank(seed, cycle) for cycle in seed["h1_basis"]) == 1


# Known Tjurina near-hit.  The 19/3 dimensions above do not belong to this
# two-variable example; its independently checked dimensions are 15/2.
P = PolynomialRing(QQ, names=("x", "y"), order="degrevlex")
x, y = P.gens()
G = x**3 + y**7 + x * y**5
J = P.ideal(G, G.derivative(x), G.derivative(y))
tjurina = cotangent_data(P, J)
assert (tjurina["length"], tjurina["square_length"]) == (11, 34)
assert (tjurina["h1_dimension"], tjurina["socle_dimension"]) == (15, 2)
assert tjurina["product_span_dimension"] == 1
assert action_rank(tjurina, G) == 1
assert tjurina["square"].quotient(P.ideal(G)) != J

# The exact colon-closure orbit for this near-hit terminates at the unit ideal.
closure_lengths = []
current = J
while True:
    closure_lengths.append(len(current.normal_basis()))
    if current.is_one():
        break
    current = (current * current).quotient(P.ideal(G))
assert closure_lengths == [11, 9, 6, 2, 0]


# A homogeneous characteristic-zero case has identically zero socle action.
P = PolynomialRing(QQ, names=("x", "y"), order="degrevlex")
x, y = P.gens()
homogeneous = cotangent_data(P, P.ideal(x**2, y**2))
assert homogeneous["product_span_dimension"] == 0


# Finite-dimensional does not imply support at the origin.
P = PolynomialRing(QQ, 1, names=("x",), order="degrevlex")
(x,) = P.gens()
not_origin_supported = P.ideal(x**2 - x)
assert len(not_origin_supported.normal_basis()) == 2
assert not not_origin_supported.reduce(x**2).is_zero()


# Colon equality alone is not enough when the representative is not a cycle.
J = P.ideal(x**2)
g = x**2
assert (J * J).quotient(P.ideal(g)) == J
assert not J.reduce(g.derivative(x)).is_zero()


# In characteristic p, [x^p] in (x^p)/(x^(2p)) is faithful.
p = 5
P = PolynomialRing(GF(p), 1, names=("x",), order="degrevlex")
(x,) = P.gens()
J = P.ideal(x**p)
frobenius = cotangent_data(P, J)
assert (frobenius["length"], frobenius["square_length"]) == (p, 2 * p)
assert action_rank(frobenius, x**p) == 1
assert frobenius["square"].quotient(P.ideal(x**p)) == J
nonprimitive = x ** (p + 1)
assert J.reduce(nonprimitive.derivative(x)).is_zero()
nonprimitive_colon = frobenius["square"].quotient(P.ideal(nonprimitive))
assert len(nonprimitive_colon.normal_basis()) == p - 1
assert nonprimitive_colon != J


print("Sage cotangent-H1 reference checks passed")

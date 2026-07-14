"""Compute the E10 cotangent-H1 statement from general algebraic objects."""

from laughableengine import QQ, elementary_symmetric


P = QQ("x1 x2 x3 x4 x5 x6 x7 x8 x9 x10")
x = P.gens()
F = sum((variable**3 for variable in x), P.zero()) + elementary_symmetric(x, 4)
partials = [F.derivative(variable) for variable in x]

# Build the ideal, quotient, conormal module, derivative map, and its kernel.
# None of these objects knows the conclusion that this example will assert.
J = P.local_ideal(generators=partials, maximal_power=4)
R = J.quotient()
conormal = R.conormal_module()
derivative = conormal.derivative_map()
H1 = derivative.kernel()

# Check the representative and its derivatives using the already-built
# quotient, then construct the class.
assert R.remainder(F).is_zero()
assert all(R.remainder(partial).is_zero() for partial in partials)
xi = H1.class_of(F)

# Compute the actual annihilator ideal and its preimage in P.
ann = xi.annihilator()
colon = ann.preimage()

# Materialize the complete kernel, not only its dimension or a modular image.
h1_basis = H1.basis(max_coordinate_entries=20_000_000)
h1_coordinates = H1.kernel_coordinates(max_coordinate_entries=20_000_000)

assert R.length == 176
assert R.square_quotient_length == 2728
assert H1.constraint_matrix.shape[1] == R.square_quotient_length
assert conormal.dimension == 2552
assert H1.dimension == len(h1_basis) == 1873
assert len(h1_coordinates) == 2092  # sparse (basis, coordinate, value) triples

assert ann.dimension == 0
assert ann.generators == []
assert ann == R.zero_ideal()
assert colon == J
assert len(colon.generators()) == 725

print(R)
print(conormal)
print(H1)
print(xi)
print(ann)

"""Complete exact QQ verification of the E10 cotangent-H1 example."""

from laughableengine import QQ, elementary_symmetric


P = QQ("x1 x2 x3 x4 x5 x6 x7 x8 x9 x10")
x = P.gens()
F = sum((variable**3 for variable in x), P.zero()) + elementary_symmetric(x, 4)
H = P.cotangent_h1(
    generators=[F.derivative(variable) for variable in x],
    maximal_power=4,
)
proof = H.verify_class(F)

# Materialize the complete kernel, not only its dimension or a modular image.
h1_basis = H.h1_basis(max_coordinate_entries=20_000_000)
h1_coordinates = H.h1_kernel_coordinates(
    max_coordinate_entries=20_000_000
)

assert H.length_Q == 176
assert H.length_P_mod_J2 == 2728
assert H.conormal_dimension == 2552
assert H.h1_dimension == len(h1_basis) == 1873
assert len(h1_coordinates) == 2092  # sparse (basis, coordinate, value) triples

assert proof.status == "complete"
assert proof.cycle_valid
assert proof.multiplication_rank == 176
assert proof.annihilator_dimension == 0
assert proof.annihilator_generators == []
assert len(proof.colon_generators) == 725
assert proof.faithful
assert proof.colon_equals_ideal

print(H)
print(proof)

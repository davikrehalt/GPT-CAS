import json
from fractions import Fraction

import pytest

from laughableengine import (
    ColonClosureResult,
    CotangentClassProof,
    CotangentH1InputError,
    CotangentH1Presentation,
    CotangentH1ResourceLimit,
    CycleAuditResult,
    CycleScreenResult,
    DivisionResult,
    ExactMatrix,
    FullH1ResourceLimit,
    GF,
    H1ActionResult,
    H1ScreenResult,
    Ideal,
    InverseSystemDiscoveryRecord,
    InverseSystemResourceLimit,
    InverseSystemResult,
    MatrixSpaceRankResult,
    Polynomial,
    QQ,
    Ring,
    SparseExactMatrix,
    __version__,
    elementary_symmetric,
)


def assert_exact_matrix(matrix, shape, scalar_type):
    assert isinstance(matrix, ExactMatrix)
    assert matrix.shape == shape
    assert (matrix.row_count, matrix.column_count) == shape
    rows = matrix.rows
    assert len(rows) == shape[0]
    assert all(len(row) == shape[1] for row in rows)
    assert all(isinstance(value, scalar_type) for row in rows for value in row)
    assert matrix.entries == rows
    assert matrix.to_list() == rows


def test_qq_ring_parsing_and_polynomial_arithmetic():
    ring = QQ("x, y", order="lex")
    x, y = ring.gens()

    assert isinstance(ring, Ring)
    assert isinstance(x, Polynomial)
    assert ring.variables == ["x", "y"]
    assert ring.order == "lex"
    assert ring.characteristic == 0
    assert ring.coefficient_field == "QQ"
    assert str(ring) == "QQ[x, y; order=lex]"
    assert x.ring is ring

    expanded = (x + y) ** 3
    assert expanded == ring("x^3 + 3*x^2*y + 3*x*y^2 + y^3")
    assert str(expanded) == "x^3 + 3*x^2*y + 3*x*y^2 + y^3"
    assert repr(expanded) == str(expanded)
    assert ring(expanded) == expanded

    assert x + 2 == 2 + x == ring("x + 2")
    assert x - 2 == ring("x - 2")
    assert 2 - x == ring("-x + 2")
    assert x * 3 == 3 * x == ring("3*x")
    assert -x == ring("-x")
    assert x**0 == ring.one()

    huge = 10**100
    assert ring(huge) == ring(str(huge))
    assert x + huge == ring(f"x + {huge}")
    assert huge - x == ring(f"{huge} - x")
    assert x * huge == ring(f"{huge}*x")
    assert x / 2 == ring("1/2*x")
    assert x / -2 == ring("-1/2*x")
    assert x / huge == ring(f"1/{huge}*x")
    assert ring("1/2") * x == x / 2

    with pytest.raises(ValueError, match="division by zero"):
        _ = x / 0
    with pytest.raises(TypeError):
        _ = x / (x + 1)


def test_polynomial_inspection_and_derivatives():
    ring = QQ(["x", "y", "z"])
    x, y, z = ring.gens()
    polynomial = y**5 + z**5 + y**2 * z**2

    assert polynomial.degree() == 5
    assert polynomial.term_count() == 3
    assert not polynomial.is_zero()
    assert ring.zero().degree() is None
    assert ring.zero().is_zero()
    assert polynomial.diff("x").is_zero()
    assert polynomial.diff(1) == polynomial.derivative(y)
    assert polynomial.diff(y) == ring("5*y^4 + 2*y*z^2")
    assert polynomial.diff(z) == ring("5*z^4 + 2*y^2*z")
    assert (2 * x).monic() == x

    with pytest.raises(ValueError, match="ring generator"):
        polynomial.diff(x + y)
    with pytest.raises(ValueError, match="nonnegative"):
        x ** -1
    with pytest.raises(OverflowError):
        x ** (10**100)


def test_elementary_symmetric_is_small_explicit_and_ring_safe():
    ring = QQ("a b c d")
    a, b, c, d = ring.gens()

    assert elementary_symmetric([a, b, c, d], 0) == ring.one()
    assert elementary_symmetric([a, b, c, d], 2) == ring(
        "a*b + a*c + a*d + b*c + b*d + c*d"
    )
    assert elementary_symmetric([a, b, c, d], 4) == a * b * c * d
    assert elementary_symmetric([a, b], 3).is_zero()

    with pytest.raises(ValueError, match="at least one polynomial"):
        elementary_symmetric([], 0)
    with pytest.raises(TypeError, match="must be polynomials"):
        elementary_symmetric([a, 1], 1)
    with pytest.raises(TypeError, match="must be an integer"):
        elementary_symmetric([a], 1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        elementary_symmetric([a], -1)
    foreign = QQ("a")
    with pytest.raises(ValueError, match="one exact ring context"):
        elementary_symmetric([a, foreign.gen("a")], 1)


def test_direct_cotangent_h1_presentation_and_faithful_class_over_gf():
    ring = GF(5, "x")
    (x,) = ring.gens()

    h1 = ring.cotangent_h1(generators=[], maximal_power=5)
    assert isinstance(h1, CotangentH1Presentation)
    assert h1.ring is ring
    assert h1.generators == []
    assert h1.maximal_power == 5
    assert h1.quotient_length == h1.length_Q == 5
    assert h1.square_quotient_length == h1.length_P_mod_J2 == 10
    assert h1.conormal_dimension == 5
    assert h1.h1_dimension == 5
    assert "h1_dimension=5" in repr(h1)

    reduction = h1.reduction_matrix
    assert isinstance(reduction, SparseExactMatrix)
    assert reduction.shape == (5, 10)
    assert (reduction.row_count, reduction.column_count, reduction.nnz) == (
        5,
        10,
        5,
    )
    assert reduction.at(0, 0) == 1
    assert reduction[0, 5] == 0
    assert reduction.entries() == [(index, index, 1) for index in range(5)]
    assert "nnz=5" in repr(reduction)
    assert h1.derivative_matrix.shape == (5, 10)
    assert h1.cycle_matrix.shape == (10, 10)
    assert h1.h1_relation_matrix.shape == h1.cycle_matrix.shape

    assert h1.ideal_generators() == [x**5]
    assert h1.quotient_basis() == [ring.one(), x, x**2, x**3, x**4]
    assert h1.square_quotient_basis()[-1] == x**9
    assert h1.h1_kernel_coordinates() == [
        (index, index + 5, 1) for index in range(5)
    ]
    assert h1.h1_basis() == [x**5, x**6, x**7, x**8, x**9]

    proof = h1.verify_class(x**5)
    assert isinstance(proof, CotangentClassProof)
    assert proof.ring is ring
    assert proof.status == "complete"
    assert proof.representative == x**5
    assert proof.in_ideal and proof.cycle and proof.cycle_valid
    assert proof.rank == proof.multiplication_rank == 5
    assert proof.ann == proof.annihilator_dimension == 0
    assert proof.annihilator_basis == proof.annihilator_generators == []
    assert proof.colon_generators == [x**5]
    assert proof.faithful
    assert proof.colon_equals and proof.colon_equals_ideal
    assert proof.conclusive
    assert proof.multiplication_matrix.shape == (10, 5)
    assert proof.multiplication_matrix.nnz == 5
    assert "faithful=True" in repr(proof)

    with pytest.raises(AttributeError):
        h1.h1_dimension = 0
    with pytest.raises(CotangentH1ResourceLimit, match="coordinate entries"):
        h1.quotient_basis(max_coordinate_entries=4)


def test_direct_cotangent_h1_negative_states_sparse_bases_and_types():
    ring = QQ("x")
    (x,) = ring.gens()
    h1 = ring.cotangent_h1(generators=[], maximal_power=2)

    assert h1.h1_dimension == 1
    assert h1.conormal_dimension == 2
    assert h1.h1_kernel_coordinates() == [(0, 3, Fraction(1))]
    assert h1.h1_basis() == [x**3]
    assert h1.conormal_basis() == [x**2, x**3]
    assert h1.quotient_remainder(x**2).is_zero()

    outside = h1.verify_class(x)
    assert outside.status == "not_in_ideal"
    assert not outside.in_ideal and not outside.cycle
    assert outside.rank is None and outside.ann is None
    assert outside.multiplication_matrix is None

    noncycle = h1.verify_class(x**2)
    assert noncycle.status == "not_cycle"
    assert noncycle.in_ideal and not noncycle.cycle

    cycle = h1.verify_class(x**3)
    assert cycle.status == "complete"
    assert cycle.rank == 1 and cycle.ann == 1
    assert cycle.annihilator_generators == [x]
    assert cycle.colon_generators == [x**2, x]
    assert not cycle.faithful and not cycle.colon_equals

    with pytest.raises(CotangentH1ResourceLimit, match="sparse kernel nonzeros"):
        h1.h1_basis(max_coordinate_entries=0)
    with pytest.raises(TypeError):
        ring.cotangent_h1(generators=ring.ideal([x**2]), maximal_power=2)
    with pytest.raises(CotangentH1InputError, match="maximal_power"):
        ring.cotangent_h1(generators=[], maximal_power=0)
    foreign = QQ("x")
    with pytest.raises(ValueError, match="exact ring context"):
        ring.cotangent_h1(
            generators=[foreign.gen("x")], maximal_power=2
        )


def test_ordered_division_and_normal_forms():
    ring = QQ("x y", order="lex")
    x, y = ring.gens()
    polynomial = x**2 * y + x * y**2 + y**2
    divisors = [x * y - 1, y**2 - 1]

    result = ring.divide(polynomial, divisors)
    assert isinstance(result, DivisionResult)
    assert result.quotients == [x + y, ring.one()]
    assert result.remainder == x + y + 1
    assert polynomial == sum(
        (quotient * divisor for quotient, divisor in zip(result.quotients, divisors)),
        result.remainder,
    )
    assert ring.nf(polynomial, divisors) == result.remainder
    assert ring.NF(polynomial, divisors) == result.remainder


def test_reduced_groebner_basis_api():
    ring = QQ(["x", "y"], order="lex")
    x, y = ring.gens()
    generators = [x * y - 1, y**2 - 1]

    assert not ring.is_gb(generators)
    basis = ring.gb(generators)
    assert basis == [x - y, y**2 - 1]
    assert ring.GB(generators) == basis
    assert ring.groebner_basis(generators) == basis
    assert ring.is_groebner_basis(basis)
    assert all(ring.normal_form(generator, basis).is_zero() for generator in generators)
    assert ring.gb([]) == []
    assert ring.is_gb([])


def test_prime_field_ring_and_characteristic_arithmetic():
    ring = GF(5, "x y", order="graded_reverse_lexicographic")
    x, y = ring.gens()

    assert ring.characteristic == 5
    assert ring.coefficient_field == "GF(5)"
    assert ring.order == "grevlex"
    assert ring("7*x - 3*y") == 2 * x + 2 * y
    assert x / 2 == 3 * x
    assert (x + y) ** 5 == x**5 + y**5
    assert (x**5).diff(x).is_zero()


def test_distinct_ring_contexts_are_rejected():
    first = QQ("x y", order="lex")
    second = QQ("x y", order="lex")
    x = first.gen("x")
    other_x = second.gen("x")

    assert x != other_x
    with pytest.raises(ValueError, match="same exact ring context"):
        _ = x + other_x
    with pytest.raises(ValueError, match="exact ring context"):
        first(other_x)
    with pytest.raises(ValueError, match="exact ring context"):
        first.gb([x, other_x])


def test_factory_validation():
    assert __version__ == "0.3.0"
    with pytest.raises(ValueError, match="prime modulus"):
        GF(6, ["x"])
    with pytest.raises(ValueError, match="2 <= p <"):
        GF(-5, ["x"])
    with pytest.raises(ValueError, match="2 <= p <"):
        GF(2**40, ["x"])
    with pytest.raises(ValueError, match="at least one variable"):
        QQ([])
    with pytest.raises(ValueError, match="order"):
        QQ(["x"], order="not-an-order")
    with pytest.raises(TypeError, match="variable names"):
        QQ(["x", 2])


def test_ideal_operations_and_true_batched_normal_forms():
    ring = QQ("x y", order="lex")
    x, y = ring.gens()
    ideal = ring.ideal([x * y - 1, y**2 - 1])

    assert isinstance(ideal, Ideal)
    assert ideal.ring is ring
    assert ideal.generators == ideal.groebner_basis == [x - y, y**2 - 1]
    assert ideal.gb() == ideal.generators
    assert "Ideal([x - y, y^2 - 1]" in repr(ideal)
    assert x - y in ideal
    assert x + y not in ideal
    assert ideal.nf(x**2) == ring.one()
    assert ideal.NF(x**2) == ring.one()
    assert ideal.nfs([x**2, x * y, y**3]) == [ring.one(), ring.one(), y]
    assert ring.nfs([x**2, x * y, y**3], ideal) == [ring.one(), ring.one(), y]
    assert ring.nfs([x**2, x * y, y**3], ideal.generators) == [
        ring.one(),
        ring.one(),
        y,
    ]
    assert ring.NFs([x**2], ideal) == [ring.one()]

    assert not ideal.is_zero()
    assert not ideal.is_unit()
    assert ideal.is_zero_dimensional()
    assert not ideal.supported_at_origin()
    assert ideal.dimension() == ideal.quotient_dimension() == 2
    assert len(ideal.standard_monomials()) == 2
    assert ring.ideal([]).is_zero()
    assert ring.ideal([ring.one()]).is_unit()

    local = ring.ideal([x**3, y**2])
    other = ring.ideal([x, y])
    assert local.supported_at_origin()
    assert local.dimension() == 6
    assert local.square() == local * local
    assert (local + other) == other
    assert local.square().colon(x**3) == local

    elimination_source = ring.ideal([x - y, y**2])
    eliminated_by_name = elimination_source.eliminate(["x"])
    eliminated_by_index = elimination_source.eliminate([0])
    assert eliminated_by_name == eliminated_by_index == ring.ideal([y**2])


def test_finite_field_ideal_surface_and_context_rejection():
    ring = GF(5, "x y")
    x, y = ring.gens()
    ideal = ring.ideal([x**2 - y, y**2])
    assert ideal.dimension() == 4
    assert ideal.supported_at_origin()
    assert ideal.nfs([x**2, y, x**4]) == [y, y, ring.zero()]
    assert ideal.square() == ideal * ideal

    foreign = GF(5, "x y")
    foreign_x = foreign.gen("x")
    foreign_ideal = foreign.ideal([foreign_x])
    with pytest.raises(ValueError, match="exact ring context"):
        ring.ideal([x, foreign_x])
    with pytest.raises(ValueError, match="exact ring context"):
        _ = ideal + foreign_ideal
    with pytest.raises(ValueError, match="exact ring context"):
        _ = foreign_x in ideal
    with pytest.raises(ValueError, match="exact ring context"):
        ring.nfs([x], foreign_ideal)
    assert ideal != foreign.ideal([foreign("x^2-y"), foreign("y^2")])


def test_seed_cycle_audit_evidence():
    ring = QQ("x y z")
    x, y, z = ring.gens()
    seed = y**5 + z**5 + y**2 * z**2
    generators = [
        x**2,
        x * y,
        x * z,
        seed,
        seed.diff(y),
        seed.diff(z),
    ]

    audit = ring.audit_cycle(generators, seed)
    assert isinstance(audit, CycleAuditResult)
    assert audit.status == "complete"
    assert audit.conclusive
    assert audit.length_Q == audit.quotient_length == 11
    assert audit.g_in_J is True
    assert audit.derivatives_in_J is True
    assert audit.cycle_valid is True
    assert audit.primitive is True
    assert all(remainder.is_zero() for remainder in audit.derivative_remainders)
    assert audit.squared_ideal.dimension() == 48
    assert audit.colon_quotient_length == 9
    assert audit.annihilator_length == 2
    assert audit.colon_equals_ideal is False
    assert audit.annihilator_zero is False
    assert_exact_matrix(audit.colon_multiplication_matrix, (48, 11), Fraction)
    assert_exact_matrix(audit.colon_kernel_basis, (2, 11), Fraction)
    assert audit.colon_kernel_coordinates.shape == (2, 11)
    assert len(audit.colon_kernel_lifts) == audit.annihilator_length
    assert audit.kernel_lifts == audit.colon_kernel_lifts
    assert all(
        remainder.is_zero()
        for remainder in audit.colon_lift_product_remainders
    )
    assert audit.lift_product_remainders == audit.colon_lift_product_remainders
    assert all(remainder.is_zero() for remainder in audit.ideal_in_colon_remainders)
    assert any(not remainder.is_zero() for remainder in audit.colon_in_ideal_remainders)
    assert not audit.faithful
    assert "status='complete'" in repr(audit)


def test_characteristic_five_faithful_audit_and_full_h1():
    ring = GF(5, "x")
    (x,) = ring.gens()
    ideal = ring.ideal([x**5])

    audit = ring.audit_cycle(ideal, x**5)
    assert audit.status == "complete"
    assert audit.length_Q == 5
    assert audit.cycle_valid is True
    assert audit.primitive is True
    assert audit.colon_ideal == ideal
    assert audit.colon_equals_ideal is True
    assert audit.annihilator_zero is True
    assert audit.annihilator_length == 0
    assert audit.faithful
    assert_exact_matrix(audit.colon_multiplication_matrix, (10, 5), int)
    assert_exact_matrix(audit.colon_kernel_basis, (0, 5), int)
    assert audit.colon_kernel_basis.rows == []

    result = ring.full_h1_action(ideal)
    assert isinstance(result, H1ActionResult)
    assert result.status == "complete"
    assert (result.length_Q, result.length_P_mod_J2) == (5, 10)
    assert (result.conormal_dimension, result.h1_dimension) == (5, 5)
    assert result.socle_dimension == 1
    assert len(result.h1_basis) == 5
    assert len(result.socle_basis) == 1
    assert_exact_matrix(result.reduction_map, (5, 10), int)
    assert result.reduction_matrix.rows == result.reduction_map.rows
    assert_exact_matrix(result.differential, (5, 10), int)
    assert len(result.variable_multiplication_matrices) == 1
    assert_exact_matrix(result.variable_multiplication_matrices[0], (5, 5), int)
    assert len(result.h1_multiplication_matrices) == 5
    assert all(
        matrix.shape == (5, 5) for matrix in result.h1_multiplication_matrices
    )
    assert len(result.action_matrices) == 5
    assert all(matrix.shape == (5, 1) for matrix in result.action_matrices)
    assert all(
        isinstance(value, int)
        for matrix in result.action_matrices
        for row in matrix.rows
        for value in row
    )
    assert result.common_product_space_dimension == 1
    assert result.maximum_individual_socle_action_rank == 1
    assert isinstance(result.individual_rank, MatrixSpaceRankResult)
    assert result.rank_proof == "proven_full_column_rank"
    assert result.individual_rank.has_full_column_rank_witness
    assert all(isinstance(value, int) for value in result.witness_coefficients)
    assert result.best_h1_polynomial in ideal
    assert result.has_faithful_witness
    assert result.faithful_witness_audit.faithful
    assert result.common_annihilator_diagnostic == ideal

    witness_limited = ring.full_h1_action(
        ideal, max_witness_audit_matrix_entries=0
    )
    assert witness_limited.status == "resource_limit"
    assert not witness_limited.conclusive
    assert witness_limited.faithful_witness_audit.status == "resource_limit"
    assert not witness_limited.faithful_witness_audit.conclusive
    assert not witness_limited.has_faithful_witness

    with pytest.raises(FullH1ResourceLimit, match="resource limit exceeded"):
        ring.full_h1_action(ideal, max_matrix_entries=0)
    with pytest.raises(FullH1ResourceLimit, match="Groebner resource limit"):
        ring.full_h1_action(ideal, max_groebner_basis_polynomials=0)

    limited_ring = QQ("u v")
    u, v = limited_ring.gens()
    limited_ideal = limited_ring.ideal([u, v**2])
    with pytest.raises(
        FullH1ResourceLimit, match="conormal coordinate solve needs 30"
    ):
        limited_ring.full_h1_action(limited_ideal, max_matrix_entries=29)


def test_near_hit_h1_and_colon_closure_lengths():
    ring = QQ("x y")
    x, y = ring.gens()
    potential = x**3 + y**7 + x * y**5
    ideal = ring.ideal([potential, potential.diff(x), potential.diff(y)])

    result = ring.full_h1_action(ideal)
    assert (result.length_Q, result.length_P_mod_J2) == (11, 34)
    assert (result.conormal_dimension, result.h1_dimension) == (23, 15)
    assert result.socle_dimension == 2
    assert_exact_matrix(result.reduction_map, (11, 34), Fraction)
    assert_exact_matrix(result.differential, (22, 34), Fraction)
    assert [matrix.shape for matrix in result.variable_multiplication_matrices] == [
        (11, 11),
        (11, 11),
    ]
    assert len(result.h1_multiplication_matrices) == 15
    assert all(
        matrix.shape == (23, 11)
        for matrix in result.h1_multiplication_matrices
    )
    assert len(result.action_matrices) == 15
    assert all(matrix.shape == (23, 2) for matrix in result.action_matrices)
    assert all(
        isinstance(value, Fraction)
        for matrix in result.action_matrices
        for row in matrix.rows
        for value in row
    )
    assert result.common_product_space_dimension == 1
    assert result.maximum_individual_socle_action_rank == 1
    assert result.rank_proof == "proven_maximum"
    assert not result.individual_rank.has_full_column_rank_witness
    assert result.faithful_witness_audit is None
    assert all(
        isinstance(value, Fraction) for value in result.witness_coefficients
    )

    closure = ring.colon_closure(ideal, potential)
    assert isinstance(closure, ColonClosureResult)
    assert closure.status == "unit_ideal"
    assert closure.conclusive
    assert not closure.faithful_fixed_point_found
    assert closure.lengths == [11, 9, 6, 2, 0]
    assert len(closure.transitions) == 4
    assert all(step.index == index for index, step in enumerate(closure.steps))
    assert all(transition.current_subset_next for transition in closure.transitions)
    assert all(not transition.equal for transition in closure.transitions)

    bounded = ring.colon_closure(ideal, potential, max_steps=2)
    assert bounded.status == "resource_limit"
    assert not bounded.conclusive
    assert bounded.lengths == [11, 9, 6]


def test_audit_states_full_h1_errors_and_high_level_context_checks():
    ring = QQ("x y")
    x, y = ring.gens()
    positive = ring.ideal([x])
    positive_audit = ring.audit_cycle(positive, x)
    assert positive_audit.status == "positive_dimensional"
    assert not positive_audit.finite_quotient
    with pytest.raises(ValueError, match="zero-dimensional"):
        ring.full_h1_action(positive)

    univariate = QQ("x")
    (ux,) = univariate.gens()
    unsupported = univariate.ideal([ux**2 - ux])
    unsupported_audit = univariate.audit_cycle(unsupported, (ux**2 - ux) ** 2)
    assert unsupported_audit.status == "unsupported_at_origin"
    with pytest.raises(ValueError, match="origin"):
        univariate.full_h1_action(unsupported)

    unit = univariate.ideal([univariate.one()])
    assert univariate.audit_cycle(unit, univariate.one()).status == "unit_ideal"
    with pytest.raises(ValueError, match="proper nonzero quotient"):
        univariate.full_h1_action(unit)

    exhausted = univariate.audit_cycle(
        univariate.ideal([ux**2]), ux**2, max_matrix_entries=0
    )
    assert exhausted.status == "resource_limit"
    assert not exhausted.conclusive
    assert exhausted.resource_detail

    foreign = QQ("x")
    foreign_ideal = foreign.ideal([foreign.gen("x") ** 2])
    with pytest.raises(ValueError, match="exact ring context"):
        univariate.audit_cycle(foreign_ideal, ux**2)
    with pytest.raises(ValueError, match="exact ring context"):
        univariate.audit_cycle(univariate.ideal([ux**2]), foreign.gen("x") ** 2)
    with pytest.raises(ValueError, match="exact ring context"):
        univariate.full_h1_action(foreign_ideal)


def test_macaulay_annihilator_metadata_conventions_and_limits():
    operators = QQ("x y")
    x, y = operators.gens()
    dual = QQ("X Y", order="lex")
    X, Y = dual.gens()

    ordinary = operators.macaulay_annihilator(dual_generators=[X**2 + Y])
    assert isinstance(ordinary, InverseSystemResult)
    assert ordinary.convention == "ordinary"
    assert ordinary.dual_generators == [X**2 + Y]
    assert ordinary.maximum_degree == ordinary.maximum_dual_degree == 2
    assert ordinary.action_shape == (6, 10)
    assert (ordinary.action_row_count, ordinary.action_column_count) == (6, 10)
    assert ordinary.action_nonzeros == ordinary.action_nnz == 5
    assert ordinary.action_rank == ordinary.quotient_length == ordinary.length_Q == 3
    assert ordinary.kernel_dimension == 7
    assert ordinary.truncated_kernel_generator_count == 7
    assert ordinary.operator_exponents[:6] == [
        [0, 0],
        [1, 0],
        [0, 1],
        [2, 0],
        [1, 1],
        [0, 2],
    ]
    assert ordinary.annihilator == operators.ideal(
        [x**2 - 2 * y, x * y, y**2]
    )
    assert ordinary.ideal == ordinary.annihilator
    with pytest.raises(AttributeError):
        ordinary.quotient_length = 99

    divided = operators.macaulay_annihilator(
        [X**2 + Y], convention="divided_powers"
    )
    assert divided.convention == "divided_powers"
    assert divided.annihilator == operators.ideal([x**2 - y, x * y, y**2])

    with pytest.raises(ValueError, match="convention"):
        operators.macaulay_annihilator([X], convention="mystery")
    with pytest.raises(ValueError, match="at least one dual generator"):
        operators.macaulay_annihilator([])
    with pytest.raises(InverseSystemResourceLimit, match="resource limit"):
        operators.macaulay_annihilator([X**2 + Y], max_basis_monomials=0)
    with pytest.raises(InverseSystemResourceLimit, match="elimination operations"):
        operators.macaulay_annihilator(
            [X**2 + Y], max_elimination_operations=0
        )
    with pytest.raises(InverseSystemResourceLimit, match="invariant replay"):
        operators.macaulay_annihilator(
            [X**2 + Y],
            verify_invariants=True,
            max_invariant_replay_checks=0,
        )
    with pytest.raises(
        InverseSystemResourceLimit, match="inverse-system Groebner construction"
    ):
        operators.macaulay_annihilator(
            [X**2 + Y], max_groebner_basis_polynomials=0
        )

    wrong_field_dual = GF(5, "X Y")
    with pytest.raises(ValueError, match="same coefficient field"):
        operators.macaulay_annihilator([wrong_field_dual.gen("X")])
    other_dual = QQ("X Y", order="lex")
    with pytest.raises(ValueError, match="one exact dual-ring context"):
        operators.macaulay_annihilator([X, other_dual.gen("X")])


def test_packed_cycle_and_h1_screens_and_failure_states():
    ring = GF(5, "x")
    (x,) = ring.gens()
    ideal = ring.ideal([x**5])

    cycle = ring.screen_cycle(ideal_generators=ideal, g=x**5)
    assert isinstance(cycle, CycleScreenResult)
    assert cycle.status == "complete"
    assert cycle.conclusive
    assert (cycle.length_Q, cycle.length_P_mod_J2) == (5, 10)
    assert cycle.g_in_J and cycle.derivatives_in_J and cycle.cycle_valid
    assert cycle.multiplication_rank == 5
    assert cycle.full_column_rank_candidate
    assert cycle.certified_faithful and cycle.hit
    assert cycle.certification.faithful
    assert cycle.detail is None

    miss = ring.screen_cycle(ideal, x**2)
    assert miss.status == "complete"
    assert not miss.g_in_J
    assert not miss.cycle_valid
    assert not miss.full_column_rank_candidate
    assert not miss.certified_faithful
    assert miss.certification is None

    h1 = ring.screen_full_h1(ideal_generators=[x**5])
    assert isinstance(h1, H1ScreenResult)
    assert h1.status == "complete"
    assert h1.conclusive
    assert (h1.length_Q, h1.length_P_mod_J2) == (5, 10)
    assert (h1.conormal_dimension, h1.h1_dimension, h1.socle_dimension) == (
        5,
        5,
        1,
    )
    assert h1.modulus == 5
    assert isinstance(h1.leading_ideal_fingerprint, int)
    assert h1.leading_ideal_fingerprint != 0
    assert isinstance(h1.leading_ideal_signature, bytes)
    assert h1.leading_ideal_signature
    assert h1.rank_lower_bound == h1.rank_upper_bound == 1
    assert h1.maximum_individual_rank == 1
    assert h1.rank_proof == "proven_full_column_rank"
    assert h1.full_socle_rank_candidate
    assert h1.certified_faithful and h1.hit
    assert h1.witness in ideal
    assert h1.witness_polynomial == h1.witness
    assert all(isinstance(value, int) for value in h1.witness_coefficients)

    limited_cycle = ring.screen_cycle(ideal, x**5, max_matrix_entries=0)
    assert limited_cycle.status == "resource_limit"
    assert not limited_cycle.conclusive
    assert limited_cycle.resource_detail
    limited_h1 = ring.screen_full_h1(ideal, max_matrix_entries=0)
    assert limited_h1.status == "resource_limit"
    assert not limited_h1.conclusive
    assert limited_h1.resource_detail
    groebner_limited_h1 = ring.screen_full_h1(
        ideal, max_groebner_basis_polynomials=0
    )
    assert groebner_limited_h1.status == "resource_limit"
    assert not groebner_limited_h1.conclusive
    assert "Groebner resource limit" in groebner_limited_h1.resource_detail

    plane = GF(5, "u v")
    u, _ = plane.gens()
    positive = plane.ideal([u])
    invalid_cycle = plane.screen_cycle(positive, u)
    assert invalid_cycle.status == "invalid_input"
    assert not invalid_cycle.conclusive
    assert invalid_cycle.detail
    invalid_h1 = plane.screen_full_h1(positive)
    assert invalid_h1.status == "invalid_input"
    assert not invalid_h1.conclusive
    assert invalid_h1.maximum_individual_rank is None

    rationals = QQ("u")
    (u,) = rationals.gens()
    with pytest.raises(ValueError, match=r"only over GF\(p\)"):
        rationals.screen_cycle([u**2], u**2)
    with pytest.raises(ValueError, match=r"only over GF\(p\)"):
        rationals.screen_full_h1([u**2])


def test_parallel_cycle_screen_is_deterministic_and_compact():
    ring = GF(5, "x")
    (x,) = ring.gens()
    ideal = ring.ideal([x**5])
    candidates = [
        x**5 if index % 3 == 0 else ring.zero() if index % 3 == 1 else x**6
        for index in range(48)
    ]

    sequential = ring.screen_cycles_parallel(
        ideal, candidates, workers=1, certify_hits=False
    )
    parallel = ring.screen_cycles_parallel(
        ideal, candidates, workers=4, certify_hits=False
    )
    assert len(sequential) == len(parallel) == len(candidates)
    assert all(isinstance(result, CycleScreenResult) for result in parallel)
    fields = lambda result: (
        result.status,
        result.length_Q,
        result.length_P_mod_J2,
        result.g_in_J,
        result.derivatives_in_J,
        result.cycle_valid,
        result.multiplication_rank,
        result.full_column_rank_candidate,
        result.certified_faithful,
    )
    assert [fields(result) for result in parallel] == [
        fields(result) for result in sequential
    ]
    assert all(result.certification is None for result in parallel)

    with pytest.raises(ValueError, match="exact ring context"):
        ring.screen_cycles_parallel(
            ideal, [GF(5, "x").gen("x")], workers=2
        )


def test_parallel_inverse_system_search_order_retention_and_resources():
    operators = GF(5, "x")
    dual = GF(5, "X")
    (X,) = dual.gens()
    candidates = [X**4, X**3] * 8

    sequential = operators.search_inverse_systems(
        candidates=candidates, workers=1
    )
    parallel = operators.search_inverse_systems(candidates, workers=4)
    assert len(sequential) == len(parallel) == len(candidates)
    assert all(
        isinstance(record, InverseSystemDiscoveryRecord) for record in parallel
    )
    fields = lambda record: (
        record.candidate_index,
        record.status,
        record.maximum_dual_degree,
        record.action_rank,
        record.kernel_dimension,
        record.quotient_length,
        record.h1_dimension,
        record.socle_dimension,
        record.modulus,
        record.leading_ideal_fingerprint,
        record.leading_ideal_signature,
        record.rank_lower_bound,
        record.rank_upper_bound,
        record.rank_proof,
        record.maximum_individual_rank,
        record.full_rank_candidate,
        record.certified_faithful,
        str(record.witness) if record.witness is not None else None,
    )
    assert [fields(record) for record in parallel] == [
        fields(record) for record in sequential
    ]
    assert [record.candidate_index for record in parallel] == list(
        range(len(candidates))
    )
    assert sequential[0].certified_faithful
    assert sequential[0].modulus == 5
    assert sequential[0].leading_ideal_fingerprint != 0
    assert isinstance(sequential[0].leading_ideal_signature, bytes)
    assert sequential[0].leading_ideal_signature
    assert sequential[0].retained_dual_generators == [X**4]
    assert sequential[0].annihilator_basis
    retained_annihilator = operators.ideal(sequential[0].annihilator_basis)
    assert sequential[0].witness in retained_annihilator
    assert not sequential[1].certified_faithful
    assert sequential[1].retained_dual_generators == []
    assert sequential[1].retained_annihilator_basis == []

    retained = operators.search_inverse_systems(
        [[X**3]], workers=2, retain_all=True
    )
    assert retained[0].dual_generators == [X**3]
    assert retained[0].annihilator_basis

    limited = operators.search_inverse_systems(
        [[X**4]], max_basis_monomials=0
    )
    assert limited[0].status == "resource_limit"
    assert not limited[0].conclusive
    assert limited[0].resource_detail

    groebner_limited = operators.search_inverse_systems(
        [[X**4]], max_groebner_basis_polynomials=0
    )
    assert groebner_limited[0].status == "resource_limit"
    assert not groebner_limited[0].conclusive
    assert (
        "inverse-system Groebner construction"
        in groebner_limited[0].resource_detail
    )

    invalid = operators.search_inverse_systems([[]])
    assert invalid[0].status == "invalid_input"
    assert invalid[0].detail

    with pytest.raises(ValueError, match=r"only over GF\(p\)"):
        QQ("x").search_inverse_systems([[QQ("X").gen("X")]])


def test_matrix_space_resource_limits_remain_inconclusive_in_discovery():
    operators = GF(2, "x y")
    dual = GF(2, "X Y")
    X, Y = dual.gens()
    candidate = [Y**3 + X**2 + Y**2 + 1, X**3]
    generated = operators.macaulay_annihilator(
        candidate, convention="divided_powers"
    )

    screen = operators.screen_full_h1(
        generated.annihilator, max_minors=0
    )
    assert screen.status == "resource_limit"
    assert not screen.conclusive
    assert (screen.rank_lower_bound, screen.rank_upper_bound) == (1, 2)
    assert screen.rank_proof == "resource_limit"
    assert screen.maximum_individual_rank is None
    assert screen.resource_detail

    record = operators.search_inverse_systems(
        [candidate],
        convention="divided_powers",
        retain_all=True,
        max_minors=0,
    )[0]
    assert record.status == "resource_limit"
    assert not record.conclusive
    assert (record.rank_lower_bound, record.rank_upper_bound) == (1, 2)
    assert record.rank_proof == "resource_limit"
    assert record.maximum_individual_rank == record.rank_lower_bound == 1
    assert record.modulus == screen.modulus == 2
    assert record.leading_ideal_fingerprint == screen.leading_ideal_fingerprint
    assert record.leading_ideal_signature == screen.leading_ideal_signature
    assert record.resource_detail


def test_two_prime_screen_metadata_has_distinct_moduli_and_exact_signature():
    first = GF(5, "x")
    second = GF(7, "x")
    first_x = first.gen("x")
    second_x = second.gen("x")
    first_screen = first.screen_full_h1([first_x**5], certify_hits=False)
    second_screen = second.screen_full_h1([second_x**5], certify_hits=False)

    assert (first_screen.modulus, second_screen.modulus) == (5, 7)
    assert first_screen.modulus != second_screen.modulus
    assert first_screen.leading_ideal_fingerprint == (
        second_screen.leading_ideal_fingerprint
    )
    assert first_screen.leading_ideal_signature == (
        second_screen.leading_ideal_signature
    )
    assert isinstance(first_screen.leading_ideal_signature, bytes)
    assert b"\x00" in first_screen.leading_ideal_signature


def test_jg_certificate_is_strict_raw_json_for_gf_and_qq():
    finite = GF(5, "x")
    (x,) = finite.gens()
    encoded = finite.make_jg_certificate(ideal_generators=[x**5], g=x**5)
    assert isinstance(encoded, str)
    document = json.loads(encoded)
    assert document == {
        "schema": "laughable-jg-v1",
        "field": {"kind": "GF", "modulus": "5"},
        "variables": ["x"],
        "order": "grevlex",
        "ideal_generators": [
            [{"coefficient": "1", "exponents": [5]}]
        ],
        "g": [{"coefficient": "1", "exponents": [5]}],
    }
    assert not ({"groebner_basis", "rank", "faithful"} & document.keys())

    rationals = QQ("u v", order="lex")
    u, v = rationals.gens()
    rational_document = json.loads(
        rationals.make_jg_certificate([u**2 + v / 2, v**2], u**2 + v / 2)
    )
    half = rational_document["ideal_generators"][0][1]["coefficient"]
    assert half == {"numerator": "1", "denominator": "2"}

    with pytest.raises(ValueError, match="at least one raw ideal generator"):
        finite.make_jg_certificate([], x)
    foreign = GF(5, "x")
    with pytest.raises(ValueError, match="exact ring context"):
        finite.make_jg_certificate([foreign.gen("x")], x)

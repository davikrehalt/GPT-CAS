#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/field.hpp"
#include "laughableengine/h1.hpp"
#include "laughableengine/ideal.hpp"
#include "laughableengine/inverse_system.hpp"
#include "laughableengine/polynomial.hpp"

namespace {

using laughableengine::ApolarityConvention;
using laughableengine::GF;
using laughableengine::Ideal;
using laughableengine::InverseSystemInputError;
using laughableengine::InverseSystemInputIssue;
using laughableengine::InverseSystemLimits;
using laughableengine::InverseSystemResourceKind;
using laughableengine::InverseSystemResourceLimit;
using laughableengine::Order;
using laughableengine::Polynomial;
using laughableengine::QQ;
using laughableengine::RationalField;
using laughableengine::full_h1_action;
using laughableengine::macaulay_annihilator;
using laughableengine::make_ring;

[[noreturn]] void fail(const std::string& message, int line) {
  throw std::runtime_error("line " + std::to_string(line) + ": " + message);
}

#define CHECK(expression)                                                     \
  do {                                                                        \
    if (!(expression)) {                                                      \
      fail("CHECK failed: " #expression, __LINE__);                          \
    }                                                                         \
  } while (false)

template <typename Exception, typename Function>
Exception expect_throw(Function&& function, int line) {
  try {
    std::invoke(std::forward<Function>(function));
  } catch (const Exception& error) {
    return error;
  } catch (const std::exception& error) {
    fail("unexpected exception type: " + std::string(error.what()), line);
  }
  fail("expected exception was not thrown", line);
}

#define EXPECT_THROW(exception, expression)                                  \
  expect_throw<exception>([&] { static_cast<void>(expression); }, __LINE__)

void test_ordinary_inhomogeneous_generator_and_matrix_layout() {
  const auto operators = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  const auto dual = make_ring(QQ(), {"X", "Y"}, Order::Lex);
  const auto x = operators.gen("x");
  const auto y = operators.gen("y");
  const auto X = dual.gen("X");
  const auto Y = dual.gen("Y");

  InverseSystemLimits checked_limits;
  checked_limits.verify_invariants = true;
  const auto result = macaulay_annihilator(
      operators, {X.pow(2) + Y},
      ApolarityConvention::OrdinaryDifferentiation, checked_limits);

  CHECK(result.convention ==
        ApolarityConvention::OrdinaryDifferentiation);
  CHECK(result.maximum_degree == 2);
  CHECK(result.output_exponents.size() == 6);
  CHECK(result.operator_exponents.size() == 10);
  CHECK(result.action_matrix.row_count() == 6);
  CHECK(result.action_matrix.column_count() == 10);
  CHECK(result.action_matrix.nnz() == 5);
  CHECK(result.action_rank == 3);
  CHECK(result.kernel_dimension == 7);
  CHECK(result.truncated_kernel_generator_count == 7);
  CHECK(result.quotient_length == 3);
  CHECK(result.action_matrix.rank() == result.action_rank);

  // Degree first, then descending lexicographic exponent tuple.
  CHECK(result.operator_exponents[0][0] == 0);
  CHECK(result.operator_exponents[0][1] == 0);
  CHECK(result.operator_exponents[1][0] == 1);
  CHECK(result.operator_exponents[1][1] == 0);
  CHECK(result.operator_exponents[2][0] == 0);
  CHECK(result.operator_exponents[2][1] == 1);
  CHECK(result.operator_exponents[3][0] == 2);
  CHECK(result.operator_exponents[3][1] == 0);
  CHECK(result.operator_exponents[4][0] == 1);
  CHECK(result.operator_exponents[4][1] == 1);
  CHECK(result.operator_exponents[5][0] == 0);
  CHECK(result.operator_exponents[5][1] == 2);

  const auto& field = operators.field();
  CHECK(result.action_matrix(3, 0) == field.one());  // 1 . X^2
  CHECK(result.action_matrix(2, 0) == field.one());  // 1 . Y
  CHECK(result.action_matrix(1, 1) == field.from_integer(2));
  CHECK(result.action_matrix(0, 2) == field.one());
  CHECK(result.action_matrix(0, 3) == field.from_integer(2));

  const Ideal<RationalField> expected(
      operators, {x.pow(2) - operators.integer(2) * y,
                  x * y, y.pow(2)});
  CHECK(result.annihilator == expected);
  CHECK(result.annihilator.supported_at_origin());
}

void test_divided_power_convention_is_explicit() {
  const auto operators = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  const auto dual = make_ring(QQ(), {"X", "Y"}, Order::Grevlex);
  const auto x = operators.gen("x");
  const auto y = operators.gen("y");
  const auto X = dual.gen("X");
  const auto Y = dual.gen("Y");

  const auto result = macaulay_annihilator(
      operators, {X.pow(2) + Y}, ApolarityConvention::DividedPowers);
  const Ideal<RationalField> expected(
      operators, {x.pow(2) - y, x * y, y.pow(2)});
  CHECK(result.convention == ApolarityConvention::DividedPowers);
  CHECK(result.action_rank == 3);
  CHECK(result.quotient_length == 3);
  CHECK(result.annihilator == expected);
  CHECK(result.action_matrix(1, 1) == operators.field().one());
  CHECK(result.action_matrix(0, 3) == operators.field().one());
}

void test_homogeneous_result_feeds_full_h1() {
  const auto operators = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  const auto dual = make_ring(QQ(), {"X", "Y"}, Order::Lex);
  const auto x = operators.gen("x");
  const auto y = operators.gen("y");
  const auto X = dual.gen("X");
  const auto Y = dual.gen("Y");

  const auto generated = macaulay_annihilator(operators, {X * Y});
  const Ideal<RationalField> expected(
      operators, {x.pow(2), y.pow(2)});
  CHECK(generated.annihilator == expected);
  CHECK(generated.action_rank == 4);
  CHECK(generated.quotient_length == 4);

  const auto h1 = full_h1_action(generated.annihilator);
  CHECK(h1.length_Q == 4);
  CHECK(h1.individual_rank.exact_maximum == 0);
}

void test_multiple_dual_generators() {
  const auto operators = make_ring(QQ(), {"x", "y"}, Order::Lex);
  const auto dual = make_ring(QQ(), {"X", "Y"}, Order::Grevlex);
  const auto x = operators.gen("x");
  const auto y = operators.gen("y");
  const auto X = dual.gen("X");
  const auto Y = dual.gen("Y");

  const auto result = macaulay_annihilator(operators, {X, Y});
  const Ideal<RationalField> expected(
      operators, {x.pow(2), x * y, y.pow(2)});
  CHECK(result.maximum_degree == 1);
  CHECK(result.action_matrix.row_count() == 6);
  CHECK(result.action_matrix.column_count() == 6);
  CHECK(result.action_rank == 3);
  CHECK(result.kernel_dimension == 3);
  CHECK(result.quotient_length == 3);
  CHECK(result.annihilator == expected);
  CHECK(result.dual_generators ==
        std::vector<Polynomial<RationalField>>({X, Y}));
}

void test_mixed_degrees_and_zero_generator_are_preserved() {
  const auto operators = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  const auto dual = make_ring(QQ(), {"X", "Y"}, Order::Lex);
  const auto x = operators.gen("x");
  const auto y = operators.gen("y");
  const auto X = dual.gen("X");
  const auto Y = dual.gen("Y");
  const auto high = X.pow(3) + Y;

  const auto base = macaulay_annihilator(operators, {high, X});
  const auto with_zero =
      macaulay_annihilator(operators, {dual.zero(), high, X});
  const Ideal<RationalField> expected(
      operators,
      {x * y, y.pow(2), x.pow(3) - operators.integer(6) * y});

  CHECK(base.maximum_degree == 3);
  CHECK(base.action_rank == 4);
  CHECK(base.quotient_length == 4);
  CHECK(base.annihilator == expected);
  CHECK(with_zero.annihilator == expected);
  CHECK(with_zero.action_rank == base.action_rank);
  CHECK(with_zero.action_matrix.row_count() == 30);
  CHECK(with_zero.dual_generators.front().is_zero());
}

void test_positive_characteristic_conventions() {
  const auto operators = make_ring(GF(2), {"x"}, Order::Grevlex);
  const auto dual = make_ring(GF(2), {"X"}, Order::Lex);
  const auto x = operators.gen("x");
  const auto X = dual.gen("X");

  const auto ordinary = macaulay_annihilator(
      operators, {X.pow(2)},
      ApolarityConvention::OrdinaryDifferentiation);
  const auto divided = macaulay_annihilator(
      operators, {X.pow(2)}, ApolarityConvention::DividedPowers);

  CHECK(ordinary.annihilator ==
        decltype(ordinary.annihilator)(operators, {x}));
  CHECK(ordinary.action_rank == 1);
  CHECK(ordinary.quotient_length == 1);
  CHECK(divided.annihilator ==
        decltype(divided.annihilator)(operators, {x.pow(3)}));
  CHECK(divided.action_rank == 3);
  CHECK(divided.quotient_length == 3);
}

void test_input_validation() {
  const auto operators = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  const auto dual_one = make_ring(QQ(), {"X"}, Order::Lex);
  const auto arity_error = EXPECT_THROW(
      InverseSystemInputError,
      macaulay_annihilator(operators, {dual_one.gen("X")}));
  CHECK(arity_error.issue() ==
        InverseSystemInputIssue::VariableCountMismatch);

  const auto gf5_operators = make_ring(GF(5), {"x"}, Order::Grevlex);
  const auto gf7_dual = make_ring(GF(7), {"X"}, Order::Grevlex);
  const auto field_error = EXPECT_THROW(
      InverseSystemInputError,
      macaulay_annihilator(gf5_operators, {gf7_dual.gen("X")}));
  CHECK(field_error.issue() ==
        InverseSystemInputIssue::CoefficientFieldMismatch);

  const auto dual_a = make_ring(QQ(), {"X", "Y"}, Order::Lex);
  const auto dual_b = make_ring(QQ(), {"X", "Y"}, Order::Lex);
  const auto mixed_error = EXPECT_THROW(
      InverseSystemInputError,
      macaulay_annihilator(
          operators, {dual_a.gen("X"), dual_b.gen("Y")}));
  CHECK(mixed_error.issue() == InverseSystemInputIssue::MixedDualRings);

  const std::vector<Polynomial<RationalField>> empty;
  const auto empty_error = EXPECT_THROW(
      InverseSystemInputError,
      macaulay_annihilator(operators, empty));
  CHECK(empty_error.issue() == InverseSystemInputIssue::NoDualGenerators);

  const auto zero_error = EXPECT_THROW(
      InverseSystemInputError,
      macaulay_annihilator(operators, {dual_a.zero()}));
  CHECK(zero_error.issue() == InverseSystemInputIssue::ZeroDualModule);

  const auto large_dual = make_ring(QQ(), {"X"}, Order::Lex);
  const auto large_operators = make_ring(QQ(), {"x"}, Order::Lex);
  const auto enormous = large_dual.monomial(
      large_dual.field().one(), {std::numeric_limits<std::uint16_t>::max()});
  const auto degree_error = EXPECT_THROW(
      InverseSystemInputError,
      macaulay_annihilator(large_operators, {enormous}));
  CHECK(degree_error.issue() == InverseSystemInputIssue::DegreeTooLarge);
}

void test_resource_limits_are_typed_and_preallocation_safe() {
  const auto operators = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  const auto dual = make_ring(QQ(), {"X", "Y"}, Order::Grevlex);
  const auto X = dual.gen("X");
  const auto Y = dual.gen("Y");

  InverseSystemLimits basis_limited;
  basis_limited.max_basis_monomials = 1;
  const auto basis_error = EXPECT_THROW(
      InverseSystemResourceLimit,
      macaulay_annihilator(
          operators, {X.pow(2) + Y},
          ApolarityConvention::OrdinaryDifferentiation, basis_limited));
  CHECK(basis_error.kind() == InverseSystemResourceKind::BasisMonomials);
  CHECK(basis_error.limit() == 1);

  InverseSystemLimits row_limited;
  row_limited.max_action_rows = 0;
  const auto row_error = EXPECT_THROW(
      InverseSystemResourceLimit,
      macaulay_annihilator(
          operators, {X}, ApolarityConvention::OrdinaryDifferentiation,
          row_limited));
  CHECK(row_error.kind() == InverseSystemResourceKind::ActionRows);

  InverseSystemLimits nonzero_limited;
  nonzero_limited.max_action_nonzeros = 0;
  const auto nonzero_error = EXPECT_THROW(
      InverseSystemResourceLimit,
      macaulay_annihilator(
          operators, {X}, ApolarityConvention::OrdinaryDifferentiation,
          nonzero_limited));
  CHECK(nonzero_error.kind() ==
        InverseSystemResourceKind::ActionNonzeros);

  InverseSystemLimits kernel_limited;
  kernel_limited.max_kernel_coordinate_entries = 0;
  const auto kernel_error = EXPECT_THROW(
      InverseSystemResourceLimit,
      macaulay_annihilator(
          operators, {X}, ApolarityConvention::OrdinaryDifferentiation,
          kernel_limited));
  CHECK(kernel_error.kind() ==
        InverseSystemResourceKind::KernelCoordinates);
  CHECK(kernel_error.observed() > 0);

  InverseSystemLimits live_limited;
  live_limited.max_elimination_live_nonzeros = 0;
  const auto live_error = EXPECT_THROW(
      InverseSystemResourceLimit,
      macaulay_annihilator(
          operators, {X}, ApolarityConvention::OrdinaryDifferentiation,
          live_limited));
  CHECK(live_error.kind() ==
        InverseSystemResourceKind::EliminationLiveNonzeros);

  InverseSystemLimits operation_limited;
  operation_limited.max_elimination_operations = 0;
  const auto operation_error = EXPECT_THROW(
      InverseSystemResourceLimit,
      macaulay_annihilator(
          operators, {X}, ApolarityConvention::OrdinaryDifferentiation,
          operation_limited));
  CHECK(operation_error.kind() ==
        InverseSystemResourceKind::EliminationOperations);

  InverseSystemLimits replay_limited;
  replay_limited.verify_invariants = true;
  replay_limited.max_invariant_replay_checks = 0;
  const auto replay_error = EXPECT_THROW(
      InverseSystemResourceLimit,
      macaulay_annihilator(
          operators, {X}, ApolarityConvention::OrdinaryDifferentiation,
          replay_limited));
  CHECK(replay_error.kind() ==
        InverseSystemResourceKind::InvariantReplayChecks);

  InverseSystemLimits groebner_limited;
  groebner_limited.groebner.max_basis_polynomials = 0;
  const auto groebner_error = EXPECT_THROW(
      InverseSystemResourceLimit,
      macaulay_annihilator(
          operators, {X}, ApolarityConvention::OrdinaryDifferentiation,
          groebner_limited));
  CHECK(groebner_error.kind() ==
        InverseSystemResourceKind::GroebnerConstruction);
}

}  // namespace

int main() {
  try {
    test_ordinary_inhomogeneous_generator_and_matrix_layout();
    test_divided_power_convention_is_explicit();
    test_homogeneous_result_feeds_full_h1();
    test_multiple_dual_generators();
    test_mixed_degrees_and_zero_generator_are_preserved();
    test_positive_characteristic_conventions();
    test_input_validation();
    test_resource_limits_are_typed_and_preallocation_safe();
    std::cout << "inverse-system tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

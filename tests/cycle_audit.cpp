#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/cycle_audit.hpp"

namespace {

using laughableengine::ColonClosureLimits;
using laughableengine::ColonClosureStopStatus;
using laughableengine::CycleAuditLimits;
using laughableengine::CycleAuditStatus;
using laughableengine::GF;
using laughableengine::Ideal;
using laughableengine::Order;
using laughableengine::Polynomial;
using laughableengine::PrimeField;
using laughableengine::QQ;
using laughableengine::RationalField;
using laughableengine::audit_cycle;
using laughableengine::colon_closure;
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
void expect_throw(Function&& function, int line) {
  try {
    std::invoke(std::forward<Function>(function));
  } catch (const Exception&) {
    return;
  } catch (const std::exception& error) {
    fail("unexpected exception type: " + std::string(error.what()), line);
  }
  fail("expected exception was not thrown", line);
}

#define EXPECT_THROW(exception, expression)                                   \
  expect_throw<exception>([&] { static_cast<void>(expression); }, __LINE__)

template <typename Field>
void check_zero_remainders(
    const std::vector<Polynomial<Field>>& remainders,
    int line) {
  for (const auto& remainder : remainders) {
    if (!remainder.is_zero()) {
      fail("expected a zero normal-form remainder", line);
    }
  }
}

#define CHECK_ZERO_REMAINDERS(remainders)                                     \
  check_zero_remainders((remainders), __LINE__)

void test_colon_equality_does_not_hide_noncycle() {
  const auto ring = make_ring(QQ(), {"x"}, Order::Lex);
  const auto x = ring.gen("x");
  const Ideal<RationalField> ideal(ring, {x.pow(2)});
  const auto audit = audit_cycle(ideal, x.pow(2));

  CHECK(audit.status() == CycleAuditStatus::Complete);
  CHECK(audit.ideal() == ideal);
  CHECK(audit.conclusive());
  CHECK(audit.finite_quotient());
  CHECK(audit.supported_at_origin());
  CHECK(audit.quotient_length() == std::optional<std::size_t>(2));
  CHECK(audit.polynomial_in_ideal() == std::optional<bool>(true));
  CHECK(audit.derivatives_in_ideal() == std::optional<bool>(false));
  CHECK(audit.cycle_valid() == std::optional<bool>(false));
  CHECK(audit.primitive() == std::optional<bool>(true));
  CHECK(audit.maximal_times_ideal().has_value());
  CHECK(*audit.maximal_times_ideal() ==
        Ideal<RationalField>(ring, {x.pow(3)}));
  CHECK(audit.membership_evidence().has_value());
  CHECK(audit.membership_evidence()->polynomial_remainder().is_zero());
  CHECK(audit.membership_evidence()->derivative_remainders().size() == 1);
  CHECK(audit.membership_evidence()->derivative_remainders()[0] ==
        ring.integer(2) * x);

  CHECK(audit.ideal_square().has_value());
  CHECK(*audit.ideal_square() == Ideal<RationalField>(ring, {x.pow(4)}));
  CHECK(audit.colon_evidence().has_value());
  const auto& colon = *audit.colon_evidence();
  CHECK(colon.multiplication_matrix().row_count() == 4);
  CHECK(colon.multiplication_matrix().column_count() == 2);
  CHECK(colon.annihilator_dimension() == 0);
  CHECK(colon.colon_quotient_length() == 2);
  CHECK(colon.colon_ideal() == ideal);
  CHECK(colon.colon_equals_ideal());
  CHECK(colon.annihilator_zero());
  CHECK(audit.colon_equals_ideal() == std::optional<bool>(true));
  CHECK(audit.annihilator_zero() == std::optional<bool>(true));
  CHECK_ZERO_REMAINDERS(colon.ideal_in_colon_remainders());
  CHECK_ZERO_REMAINDERS(colon.colon_in_ideal_remainders());
  CHECK(!audit.faithful_cycle());
}

void test_characteristic_p_faithful_and_nonprimitive() {
  const auto ring = make_ring(GF(5), {"x"}, Order::Lex);
  const auto x = ring.gen("x");
  const Ideal<PrimeField> ideal(ring, {x.pow(5)});

  const auto faithful = audit_cycle(ideal, x.pow(5));
  CHECK(faithful.status() == CycleAuditStatus::Complete);
  CHECK(faithful.cycle_valid() == std::optional<bool>(true));
  CHECK(faithful.primitive() == std::optional<bool>(true));
  CHECK(faithful.colon_equals_ideal() == std::optional<bool>(true));
  CHECK(faithful.annihilator_zero() == std::optional<bool>(true));
  CHECK(faithful.faithful_cycle());
  CHECK(faithful.colon_evidence()->multiplication_matrix().row_count() == 10);
  CHECK(faithful.colon_evidence()->multiplication_matrix().column_count() == 5);

  const auto nonprimitive = audit_cycle(ideal, x.pow(6));
  CHECK(nonprimitive.status() == CycleAuditStatus::Complete);
  CHECK(nonprimitive.cycle_valid() == std::optional<bool>(true));
  CHECK(nonprimitive.primitive() == std::optional<bool>(false));
  CHECK(nonprimitive.colon_evidence()->annihilator_dimension() == 1);
  CHECK(nonprimitive.colon_evidence()->colon_quotient_length() == 4);
  CHECK(nonprimitive.colon_evidence()->colon_ideal() ==
        Ideal<PrimeField>(ring, {x.pow(4)}));
  CHECK(nonprimitive.colon_equals_ideal() == std::optional<bool>(false));
  CHECK(nonprimitive.annihilator_zero() == std::optional<bool>(false));
  CHECK(!nonprimitive.faithful_cycle());
  CHECK(nonprimitive.colon_evidence()->kernel_lifts().size() == 1);
  CHECK_ZERO_REMAINDERS(
      nonprimitive.colon_evidence()->lift_product_remainders());
}

void test_inapplicable_and_resource_states() {
  const auto ring = make_ring(QQ(), {"x"}, Order::Lex);
  const auto x = ring.gen("x");
  const Ideal<RationalField> ideal(ring, {x.pow(2)});

  const auto zero = audit_cycle(ideal, ring.zero());
  CHECK(zero.status() == CycleAuditStatus::Complete);
  CHECK(zero.cycle_valid() == std::optional<bool>(true));
  CHECK(zero.primitive() == std::optional<bool>(false));
  CHECK(zero.colon_evidence()->annihilator_dimension() == 2);
  CHECK(zero.colon_evidence()->colon_quotient_length() == 0);
  CHECK(zero.colon_evidence()->colon_ideal().is_unit());
  CHECK(!zero.faithful_cycle());

  const auto one = audit_cycle(ideal, ring.one());
  CHECK(one.status() == CycleAuditStatus::PolynomialNotInIdeal);
  CHECK(one.polynomial_in_ideal() == std::optional<bool>(false));
  CHECK(one.cycle_valid() == std::optional<bool>(false));
  CHECK(!one.primitive().has_value());
  CHECK(!one.colon_evidence().has_value());
  CHECK(!one.colon_equals_ideal().has_value());
  CHECK(!one.annihilator_zero().has_value());
  CHECK(!one.faithful_cycle());

  const Ideal<RationalField> unit(ring, {ring.one()});
  const auto unit_audit = audit_cycle(unit, ring.one());
  CHECK(unit_audit.status() == CycleAuditStatus::UnitIdeal);
  CHECK(unit_audit.quotient_length() == std::optional<std::size_t>(0));
  CHECK(unit_audit.colon_equals_ideal() == std::optional<bool>(true));
  CHECK(unit_audit.annihilator_zero() == std::optional<bool>(true));
  CHECK(!unit_audit.faithful_cycle());

  const auto two_variable_ring =
      make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  const auto tx = two_variable_ring.gen("x");
  const Ideal<RationalField> positive(two_variable_ring, {tx.pow(2)});
  const auto positive_audit = audit_cycle(positive, tx.pow(2));
  CHECK(positive_audit.status() == CycleAuditStatus::PositiveDimensional);
  CHECK(!positive_audit.finite_quotient());
  CHECK(!positive_audit.quotient_length().has_value());
  CHECK(!positive_audit.colon_evidence().has_value());
  CHECK(!positive_audit.faithful_cycle());

  const auto defining = x.pow(2) - x;
  const Ideal<RationalField> unsupported(ring, {defining});
  const auto unsupported_audit = audit_cycle(unsupported, defining.pow(2));
  CHECK(unsupported_audit.status() == CycleAuditStatus::UnsupportedAtOrigin);
  CHECK(unsupported_audit.finite_quotient());
  CHECK(!unsupported_audit.supported_at_origin());
  CHECK(unsupported_audit.cycle_valid() == std::optional<bool>(true));
  CHECK(unsupported_audit.colon_evidence().has_value());
  CHECK(!unsupported_audit.faithful_cycle());

  CycleAuditLimits no_matrix;
  no_matrix.max_matrix_entries = 0;
  const auto exhausted = audit_cycle(ideal, x.pow(2), no_matrix);
  CHECK(exhausted.status() == CycleAuditStatus::ResourceLimit);
  CHECK(exhausted.ideal() == ideal);
  CHECK(!exhausted.conclusive());
  CHECK(exhausted.resource_detail().has_value());
  CHECK(!exhausted.colon_evidence().has_value());
  CHECK(!exhausted.colon_equals_ideal().has_value());
  CHECK(!exhausted.faithful_cycle());

  const auto foreign = make_ring(QQ(), {"x"}, Order::Lex);
  EXPECT_THROW(std::invalid_argument,
               audit_cycle(ideal, foreign.gen("x")));
}

void test_seed_audit() {
  const auto ring = make_ring(QQ(), {"x", "y", "z"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto z = ring.gen("z");
  const auto f = y.pow(5) + z.pow(5) + y.pow(2) * z.pow(2);
  const Ideal<RationalField> ideal(
      ring,
      {x.pow(2), x * y, x * z, f, f.derivative("y"), f.derivative("z")});
  const auto audit = audit_cycle(ideal, f);

  CHECK(audit.status() == CycleAuditStatus::Complete);
  CHECK(audit.quotient_length() == std::optional<std::size_t>(11));
  CHECK(audit.cycle_valid() == std::optional<bool>(true));
  CHECK(audit.primitive() == std::optional<bool>(true));
  CHECK(audit.ideal_square()->quotient_dimension() == 48);
  CHECK(audit.colon_evidence().has_value());
  const auto& evidence = *audit.colon_evidence();
  CHECK(evidence.multiplication_matrix().row_count() == 48);
  CHECK(evidence.multiplication_matrix().column_count() == 11);
  CHECK(evidence.annihilator_dimension() == 2);
  CHECK(evidence.colon_quotient_length() == 9);
  const Ideal<RationalField> expected(
      ring,
      {y.pow(4), z.pow(4), y.pow(2) * z, y * z.pow(2), x.pow(2), x * y,
       x * z});
  CHECK(evidence.colon_ideal() == expected);
  CHECK(!evidence.colon_equals_ideal());
  CHECK(!evidence.annihilator_zero());
  CHECK_ZERO_REMAINDERS(evidence.lift_product_remainders());
  CHECK_ZERO_REMAINDERS(evidence.ideal_in_colon_remainders());
  CHECK(!audit.faithful_cycle());
}

void test_colon_closure() {
  const auto ring = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto potential = x.pow(3) + y.pow(7) + x * y.pow(5);
  const Ideal<RationalField> ideal(
      ring,
      {potential, potential.derivative("x"), potential.derivative("y")});

  const auto closure = colon_closure(ideal, potential);
  CHECK(closure.status() == ColonClosureStopStatus::UnitIdeal);
  CHECK(closure.conclusive());
  CHECK(!closure.faithful_fixed_point_found());
  std::vector<std::size_t> lengths;
  for (const auto& step : closure.steps()) {
    CHECK(step.quotient_length().has_value());
    lengths.push_back(*step.quotient_length());
  }
  CHECK(lengths == std::vector<std::size_t>({11, 9, 6, 2, 0}));
  CHECK(closure.transitions().size() == 4);
  for (const auto& transition : closure.transitions()) {
    CHECK(transition.current_subset_next());
    CHECK(!transition.equal());
    CHECK_ZERO_REMAINDERS(transition.current_in_next_remainders());
  }

  ColonClosureLimits two_steps;
  two_steps.max_steps = 2;
  const auto bounded = colon_closure(ideal, potential, two_steps);
  CHECK(bounded.status() == ColonClosureStopStatus::ResourceLimit);
  CHECK(!bounded.conclusive());
  CHECK(!bounded.faithful_fixed_point_found());
  CHECK(bounded.resource_detail().has_value());
  CHECK(bounded.steps().size() == 3);
  CHECK(*bounded.steps()[0].quotient_length() == 11);
  CHECK(*bounded.steps()[1].quotient_length() == 9);
  CHECK(*bounded.steps()[2].quotient_length() == 6);

  const auto finite_ring = make_ring(GF(5), {"x"}, Order::Lex);
  const auto fx = finite_ring.gen("x");
  const Ideal<PrimeField> faithful_ideal(finite_ring, {fx.pow(5)});
  const auto fixed = colon_closure(faithful_ideal, fx.pow(5));
  CHECK(fixed.status() == ColonClosureStopStatus::ProperFixedPoint);
  CHECK(fixed.faithful_fixed_point_found());
  CHECK(fixed.steps().size() == 2);
  CHECK(*fixed.steps()[0].quotient_length() == 5);
  CHECK(*fixed.steps()[1].quotient_length() == 5);
  CHECK(fixed.transitions().size() == 1);
  CHECK(fixed.transitions()[0].equal());

  CHECK(colon_closure(faithful_ideal, finite_ring.one()).status() ==
        ColonClosureStopStatus::InvalidStart);
  const Ideal<PrimeField> finite_unit(finite_ring, {finite_ring.one()});
  CHECK(colon_closure(finite_unit, finite_ring.one()).status() ==
        ColonClosureStopStatus::UnitIdeal);

  ColonClosureLimits no_matrix;
  no_matrix.audit.max_matrix_entries = 0;
  const auto exhausted =
      colon_closure(faithful_ideal, fx.pow(5), no_matrix);
  CHECK(exhausted.status() == ColonClosureStopStatus::ResourceLimit);
  CHECK(!exhausted.conclusive());
  CHECK(exhausted.steps().size() == 1);

  const auto foreign = make_ring(GF(5), {"x"}, Order::Lex);
  EXPECT_THROW(std::invalid_argument,
               colon_closure(faithful_ideal, foreign.gen("x")));
}

}  // namespace

int main() {
  try {
    test_colon_equality_does_not_hide_noncycle();
    test_characteristic_p_faithful_and_nonprimitive();
    test_inapplicable_and_resource_states();
    test_seed_audit();
    test_colon_closure();
    std::cout << "laughableengine: cycle audit tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "laughableengine cycle audit test failure: " << error.what()
              << '\n';
    return 1;
  }
}

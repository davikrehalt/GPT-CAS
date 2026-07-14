#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/laughableengine.hpp"

namespace {

using laughableengine::GF;
using laughableengine::GroebnerLimits;
using laughableengine::GroebnerResourceLimit;
using laughableengine::Order;
using laughableengine::Polynomial;
using laughableengine::QQ;
using laughableengine::RationalField;
using laughableengine::divide;
using laughableengine::groebner_basis;
using laughableengine::is_groebner_basis;
using laughableengine::make_ring;
using laughableengine::normal_form;
using laughableengine::s_polynomial;

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
void check_reconstruction(
    const Polynomial<Field>& dividend,
    const std::vector<Polynomial<Field>>& divisors,
    const laughableengine::DivisionResult<Field>& result) {
  CHECK(divisors.size() == result.quotients.size());
  auto reconstructed = result.remainder;
  for (std::size_t index = 0; index < divisors.size(); ++index) {
    reconstructed = reconstructed + result.quotients[index] * divisors[index];
  }
  CHECK(reconstructed == dividend);

  for (const auto& term : result.remainder.terms()) {
    for (const auto& divisor : divisors) {
      CHECK(!term.monomial.is_divisible_by(
          divisor.leading_term()->monomial));
    }
  }
}

template <typename Field>
void check_reduced_basis(const std::vector<Polynomial<Field>>& basis) {
  CHECK(is_groebner_basis(basis));
  for (std::size_t index = 0; index < basis.size(); ++index) {
    CHECK(!basis[index].is_zero());
    CHECK(basis[index].coefficient_field().is_one(
        basis[index].leading_term()->coefficient));
    for (std::size_t other = 0; other < basis.size(); ++other) {
      if (index == other) {
        continue;
      }
      for (const auto& term : basis[index].terms()) {
        CHECK(!term.monomial.is_divisible_by(
            basis[other].leading_term()->monomial));
      }
    }
  }
}

void test_ordered_division_over_qq() {
  const auto ring = make_ring(QQ(), {"x", "y"}, Order::Lex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto f = x.pow(2) * y + x * y.pow(2) + y.pow(2);
  const auto g1 = x * y - ring.one();
  const auto g2 = y.pow(2) - ring.one();

  const std::vector<Polynomial<RationalField>> divisors{g1, g2};
  const auto result = divide(f, divisors);
  CHECK(result.quotients[0] == x + y);
  CHECK(result.quotients[1] == ring.one());
  CHECK(result.remainder == x + y + ring.one());
  check_reconstruction(f, divisors, result);

  const std::vector<Polynomial<RationalField>> reversed{g2, g1};
  const auto reversed_result = divide(f, reversed);
  CHECK(reversed_result.quotients[0] == x + ring.one());
  CHECK(reversed_result.quotients[1] == x);
  CHECK(reversed_result.remainder == ring.integer(2) * x + ring.one());
  check_reconstruction(f, reversed, reversed_result);

  const std::vector<Polynomial<RationalField>> no_divisors;
  const auto unchanged = divide(f, no_divisors);
  CHECK(unchanged.quotients.empty());
  CHECK(unchanged.remainder == f);
  CHECK(normal_form(ring.zero(), divisors).is_zero());

  EXPECT_THROW(std::invalid_argument, divide(f, {ring.zero()}));
  const auto foreign = make_ring(QQ(), {"x", "y"}, Order::Lex);
  EXPECT_THROW(std::invalid_argument, divide(ring.zero(), {foreign.zero()}));
}

void test_finite_field_division() {
  const auto ring = make_ring(GF(5), {"x"}, Order::Lex);
  const auto x = ring.gen("x");
  const auto f = ring.integer(2) * x.pow(2) + x;
  const auto g = ring.integer(3) * x;
  const auto result = divide(f, {g});
  CHECK(result.quotients.size() == 1);
  CHECK(result.quotients[0] == ring.integer(4) * x + ring.integer(2));
  CHECK(result.remainder.is_zero());
  check_reconstruction(f, std::vector{g}, result);
}

void test_s_polynomial() {
  const auto ring = make_ring(QQ(), {"x", "y"}, Order::Lex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto f = ring.integer(2) * x.pow(2) * y + x;
  const auto g = ring.integer(3) * x * y.pow(2) + y;
  const auto sixth = ring.constant(ring.field().fraction(1, 6));
  CHECK(s_polynomial(f, g) == sixth * x * y);
  EXPECT_THROW(std::invalid_argument, s_polynomial(ring.zero(), g));
}

void test_reduced_groebner_basis() {
  const auto ring = make_ring(QQ(), {"x", "y"}, Order::Lex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto first = x.pow(2) - y;
  const auto second = x * y - ring.one();
  const std::vector<Polynomial<RationalField>> generators{first, second};

  CHECK(!is_groebner_basis(generators));
  const auto basis = groebner_basis(generators);
  const std::vector<Polynomial<RationalField>> expected{
      x - y.pow(2), y.pow(3) - ring.one()};
  CHECK(basis == expected);
  check_reduced_basis(basis);
  for (const auto& generator : generators) {
    CHECK(normal_form(generator, basis).is_zero());
  }
  const auto shuffled = groebner_basis(std::vector<Polynomial<RationalField>>{
      ring.zero(), ring.integer(-3) * second, first,
      ring.integer(2) * first, second});
  CHECK(shuffled == basis);

  const auto copied_ring = ring;
  CHECK(normal_form(copied_ring.gen("x") - copied_ring.gen("y").pow(2), basis)
            .is_zero());
  const auto foreign = make_ring(QQ(), {"x", "y"}, Order::Lex);
  EXPECT_THROW(std::invalid_argument,
               groebner_basis(std::vector{first, foreign.gen("x")}));
}

void test_zero_and_unit_ideals() {
  const auto ring = make_ring(QQ(), {"x"}, Order::Lex);
  const auto x = ring.gen("x");

  const auto unit = groebner_basis({x, x + ring.one()});
  CHECK(unit.size() == 1);
  CHECK(unit.front() == ring.one());

  const std::vector<Polynomial<RationalField>> empty;
  CHECK(groebner_basis(empty).empty());
  CHECK(groebner_basis(std::vector{ring.zero(), ring.zero()}).empty());
  CHECK(is_groebner_basis(empty));
}

void test_characteristic_p_regression() {
  const auto ring = make_ring(GF(5), {"x"}, Order::Lex);
  const auto x = ring.gen("x");
  const auto basis = groebner_basis({x.pow(5)});
  CHECK(basis.size() == 1);
  CHECK(basis.front() == x.pow(5));
  CHECK(x.pow(5).derivative("x").is_zero());
  check_reduced_basis(basis);
}

void test_cotangent_shaped_ideal() {
  const auto ring = make_ring(QQ(), {"x", "y", "z"});
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto z = ring.gen("z");
  const auto seed = y.pow(5) + z.pow(5) + y.pow(2) * z.pow(2);
  const std::vector<Polynomial<RationalField>> generators{
      x.pow(2), x * y, x * z, seed, seed.derivative("y"),
      seed.derivative("z")};

  const auto basis = groebner_basis(generators);
  const auto two_fifths = ring.constant(ring.field().fraction(2, 5));
  const std::vector<Polynomial<RationalField>> expected{
      y.pow(4) + two_fifths * y * z.pow(2),
      y.pow(3) * z,
      y.pow(2) * z.pow(2),
      y * z.pow(3),
      z.pow(4) + two_fifths * y.pow(2) * z,
      x.pow(2),
      x * y,
      x * z};
  CHECK(basis == expected);
  check_reduced_basis(basis);
  for (const auto& generator : generators) {
    CHECK(normal_form(generator, basis).is_zero());
  }
}

void test_user_tjurina_seed() {
  const auto rational_ring = make_ring(QQ(), {"x", "y"});
  const auto x = rational_ring.gen("x");
  const auto y = rational_ring.gen("y");
  const auto potential = x.pow(3) + y.pow(7) + x * y.pow(5);
  const std::vector<Polynomial<RationalField>> generators{
      potential, potential.derivative("x"), potential.derivative("y")};

  const auto basis = groebner_basis(generators);
  const auto minus_twenty_one_fifths =
      rational_ring.constant(rational_ring.field().fraction(-21, 5));
  const std::vector<Polynomial<RationalField>> expected{
      x * y.pow(4) + minus_twenty_one_fifths * x.pow(2) * y,
      y.pow(5) + rational_ring.integer(3) * x.pow(2),
      x.pow(2) * y.pow(2),
      x.pow(3)};
  CHECK(basis == expected);
  check_reduced_basis(basis);
  for (const auto& generator : generators) {
    CHECK(normal_form(generator, basis).is_zero());
  }
  const auto twenty_one_fifths =
      rational_ring.constant(rational_ring.field().fraction(21, 5));
  CHECK(normal_form(x * y.pow(4) + y.pow(5), basis) ==
        twenty_one_fifths * x.pow(2) * y -
            rational_ring.integer(3) * x.pow(2));

  const auto finite_ring = make_ring(GF(101), {"x", "y"});
  const auto fx = finite_ring.gen("x");
  const auto fy = finite_ring.gen("y");
  const auto finite_potential =
      fx.pow(3) + fy.pow(7) + fx * fy.pow(5);
  const auto finite_basis = groebner_basis(
      {finite_potential, finite_potential.derivative("x"),
       finite_potential.derivative("y")});
  const auto finite_expected = std::vector{
      fx * fy.pow(4) + finite_ring.integer(16) * fx.pow(2) * fy,
      fy.pow(5) + finite_ring.integer(3) * fx.pow(2),
      fx.pow(2) * fy.pow(2),
      fx.pow(3)};
  CHECK(finite_basis == finite_expected);
  check_reduced_basis(finite_basis);
  CHECK(normal_form(fx * fy.pow(4) + fy.pow(5), finite_basis) ==
        finite_ring.integer(-16) * fx.pow(2) * fy -
            finite_ring.integer(3) * fx.pow(2));
}

void test_groebner_resource_limits() {
  const auto ring = make_ring(QQ(), {"x", "y"});
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");

  GroebnerLimits basis_limit;
  basis_limit.max_basis_polynomials = 1;
  EXPECT_THROW(
      GroebnerResourceLimit,
      groebner_basis(std::vector{x, y}, basis_limit));

  GroebnerLimits pair_limit;
  pair_limit.max_critical_pairs = 0;
  EXPECT_THROW(
      GroebnerResourceLimit,
      groebner_basis(std::vector{x, y}, pair_limit));

  GroebnerLimits step_limit;
  step_limit.max_reduction_steps = 0;
  EXPECT_THROW(
      GroebnerResourceLimit,
      groebner_basis(std::vector{x, y}, step_limit));

  GroebnerLimits live_limit;
  live_limit.max_live_terms = 0;
  EXPECT_THROW(
      GroebnerResourceLimit,
      groebner_basis(std::vector{x}, live_limit));

  GroebnerLimits term_limit;
  term_limit.max_basis_terms = 0;
  EXPECT_THROW(
      GroebnerResourceLimit,
      groebner_basis(std::vector{x}, term_limit));

  EXPECT_THROW(
      GroebnerResourceLimit,
      is_groebner_basis(std::vector{x, y}, pair_limit));
}

}  // namespace

int main() {
  try {
    test_ordered_division_over_qq();
    test_finite_field_division();
    test_s_polynomial();
    test_reduced_groebner_basis();
    test_zero_and_unit_ideals();
    test_characteristic_p_regression();
    test_cotangent_shaped_ideal();
    test_user_tjurina_seed();
    test_groebner_resource_limits();
    std::cout << "laughableengine: division and Groebner tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "laughableengine test failure: " << error.what() << '\n';
    return 1;
  }
}

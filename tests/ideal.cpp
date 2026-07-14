#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/ideal.hpp"

namespace {

using laughableengine::GF;
using laughableengine::GroebnerLimits;
using laughableengine::GroebnerResourceLimit;
using laughableengine::Ideal;
using laughableengine::Order;
using laughableengine::Polynomial;
using laughableengine::PrimeField;
using laughableengine::QQ;
using laughableengine::RationalField;
using laughableengine::ResourceLimitExceeded;
using laughableengine::ResourceLimitKind;
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

void test_zero_unit_membership_and_equality() {
  const auto ring = make_ring(QQ(), {"x", "y"}, Order::Lex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const Ideal<RationalField> zero(
      ring, std::vector<Polynomial<RationalField>>{});
  CHECK(zero.is_zero());
  CHECK(!zero.is_unit());
  CHECK(zero.contains(ring.zero()));
  CHECK(!zero.contains(x));
  CHECK(!zero.is_zero_dimensional());
  CHECK(!zero.supported_at_origin());
  CHECK(zero.colon(ring.zero()).is_unit());
  CHECK(zero.colon(ring.one()) == zero);
  EXPECT_THROW(std::domain_error, zero.colon(x));

  const Ideal<RationalField> unit(ring, {ring.integer(-7)});
  CHECK(unit.is_unit());
  CHECK(!unit.is_zero());
  CHECK(unit.groebner_basis() ==
        std::vector<Polynomial<RationalField>>{ring.one()});
  CHECK(unit.contains(x + y));
  CHECK(unit.is_zero_dimensional());
  CHECK(unit.quotient_dimension() == 0);
  CHECK(unit.supported_at_origin());

  const Ideal<RationalField> ideal(ring, {x.pow(2) - y, y.pow(2)});
  CHECK(ideal.contains(x.pow(2) - y));
  CHECK(ideal.contains(x.pow(4)));
  CHECK(!ideal.contains(x));
  CHECK(ideal.normal_form(x.pow(2)) == y);
  CHECK((ideal.normal_forms({x.pow(2), y.pow(2), x}) ==
         std::vector<Polynomial<RationalField>>{y, ring.zero(), x}));

  const Ideal<RationalField> same(
      ring,
      {ring.integer(3) * (x.pow(2) - y), y.pow(2), x.pow(4)});
  CHECK(ideal == same);
  CHECK(ideal.equals(same));
  CHECK(ideal.is_subset_of(same));
  CHECK(same.is_subset_of(ideal));

  const auto foreign_ring = make_ring(QQ(), {"x", "y"}, Order::Lex);
  const Ideal<RationalField> foreign_zero(
      foreign_ring, std::vector<Polynomial<RationalField>>{});
  CHECK(!(zero == foreign_zero));
  EXPECT_THROW(std::invalid_argument, zero.equals(foreign_zero));
  EXPECT_THROW(std::invalid_argument, zero.is_subset_of(foreign_zero));
  EXPECT_THROW(std::invalid_argument, ideal.sum(foreign_zero));
  EXPECT_THROW(std::invalid_argument, ideal.contains(foreign_ring.gen("x")));
}

void test_sum_product_and_square() {
  const auto ring = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const Ideal<RationalField> left(ring, {x.pow(2), y});
  const Ideal<RationalField> right(ring, {x, y.pow(2)});

  const Ideal<RationalField> expected_sum(ring, {x, y});
  CHECK(left.sum(right) == expected_sum);
  CHECK(left + right == expected_sum);

  const Ideal<RationalField> expected_product(
      ring, {x.pow(3), x * y, y.pow(3)});
  CHECK(left.product(right) == expected_product);
  CHECK(left * right == expected_product);

  const Ideal<RationalField> expected_square(
      ring, {x.pow(4), x.pow(2) * y, y.pow(2)});
  CHECK(left.square() == expected_square);

  const Ideal<RationalField> unit(ring, {ring.one()});
  const Ideal<RationalField> zero(
      ring, std::vector<Polynomial<RationalField>>{});
  CHECK(left.product(unit) == left);
  CHECK(left.product(zero) == zero);
  CHECK(zero.square() == zero);
}

void test_rational_and_finite_field_colons() {
  const auto ring = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const Ideal<RationalField> ideal(
      ring, {x.pow(2), x * y, y.pow(3)});
  const Ideal<RationalField> expected(ring, {x, y});
  const auto colon = ideal.colon(x);
  CHECK(colon == expected);
  CHECK(colon.quotient_dimension() == 1);
  for (const auto& generator : colon.generators()) {
    CHECK(ideal.contains(generator * x));
  }
  CHECK(ideal.colon(ring.one()) == ideal);
  CHECK(ideal.principal_colon(x) == expected);
  CHECK(ideal.colon(ring.zero()).is_unit());
  CHECK(ideal.colon(x.pow(2)).is_unit());

  const Ideal<RationalField> positive_dimensional(ring, {x.pow(2)});
  CHECK(positive_dimensional.colon(ring.one()) == positive_dimensional);
  CHECK(positive_dimensional.colon(x.pow(2)).is_unit());
  EXPECT_THROW(std::domain_error, positive_dimensional.colon(x));

  const auto finite_ring = make_ring(GF(5), {"x", "y"}, Order::Lex);
  const auto finite_x = finite_ring.gen("x");
  const auto finite_y = finite_ring.gen("y");
  const Ideal<PrimeField> finite_ideal(
      finite_ring, {finite_x.pow(2), finite_y.pow(2)});
  const Ideal<PrimeField> finite_expected(
      finite_ring, {finite_x, finite_y.pow(2)});
  CHECK(finite_ideal.colon(finite_x) == finite_expected);
  CHECK(finite_ideal.colon(finite_ring.integer(2)) == finite_ideal);

  const auto foreign = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  EXPECT_THROW(std::invalid_argument, ideal.colon(foreign.gen("x")));
}

void test_support_at_origin_uses_actual_nilpotence() {
  const auto ring = make_ring(QQ(), {"x"}, Order::Lex);
  const auto x = ring.gen("x");
  const Ideal<RationalField> origin(ring, {x.pow(2)});
  CHECK(origin.is_zero_dimensional());
  CHECK(origin.quotient_dimension() == 2);
  CHECK(origin.supported_at_origin());

  const Ideal<RationalField> two_points(ring, {x.pow(2) - x});
  CHECK(two_points.is_zero_dimensional());
  CHECK(two_points.quotient_dimension() == 2);
  CHECK(!two_points.supported_at_origin());
  CHECK(!two_points.contains(x.pow(2)));

  // Generic zero-dimensional colon must also work away from the local case.
  CHECK(two_points.colon(x + ring.one()) == two_points);
  const Ideal<RationalField> second_point(ring, {x - ring.one()});
  CHECK(two_points.colon(x) == second_point);
}

void test_elimination_and_validation() {
  const auto ring = make_ring(QQ(), {"x", "y", "z"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto z = ring.gen("z");

  const Ideal<RationalField> first(ring, {x * y - ring.one(), x - z});
  const Ideal<RationalField> first_expected(ring, {y * z - ring.one()});
  const auto eliminate_x = first.eliminate(0);
  CHECK(eliminate_x == first_expected);
  CHECK(eliminate_x.contains(y * z - ring.one()));

  const Ideal<RationalField> second(
      ring, {x - y, y - z, z.pow(2) - ring.one()});
  const Ideal<RationalField> second_expected(
      ring, {z.pow(2) - ring.one()});
  CHECK(second.eliminate({0, 1}) == second_expected);
  CHECK(second.eliminate({1, 0}) == second_expected);
  const Ideal<RationalField> eliminate_middle_expected(
      ring, {x - z, z.pow(2) - ring.one()});
  CHECK(second.eliminate({1}) == eliminate_middle_expected);
  const Ideal<RationalField> eliminate_outer_expected(
      ring, {y.pow(2) - ring.one()});
  CHECK(second.eliminate({2, 0}) == eliminate_outer_expected);

  CHECK(second.eliminate({0, 1, 2}).is_zero());
  const Ideal<RationalField> unit(ring, {ring.one()});
  CHECK(unit.eliminate({0, 1, 2}).is_unit());

  EXPECT_THROW(std::invalid_argument,
               first.eliminate(std::vector<std::size_t>{}));
  EXPECT_THROW(std::invalid_argument, first.eliminate({0, 1, 2, 2}));
  EXPECT_THROW(std::invalid_argument, first.eliminate({0, 0}));
  EXPECT_THROW(std::out_of_range, first.eliminate({3}));

  const auto foreign = make_ring(QQ(), {"x", "y", "z"}, Order::Grevlex);
  EXPECT_THROW(std::invalid_argument,
               eliminate_x.contains(foreign.gen("y") * foreign.gen("z") -
                                    foreign.one()));

  const auto finite_ring =
      make_ring(GF(5), {"x", "y"}, Order::Grevlex);
  const auto finite_x = finite_ring.gen("x");
  const auto finite_y = finite_ring.gen("y");
  const Ideal<PrimeField> finite_ideal(
      finite_ring,
      {finite_x - finite_y, finite_x.pow(2) + finite_ring.one()});
  const Ideal<PrimeField> finite_expected(
      finite_ring, {finite_y.pow(2) + finite_ring.one()});
  CHECK(finite_ideal.eliminate(0) == finite_expected);
}

void test_sage_seed_colon() {
  const auto ring = make_ring(QQ(), {"x", "y", "z"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto z = ring.gen("z");
  const auto f = y.pow(5) + z.pow(5) + y.pow(2) * z.pow(2);
  const Ideal<RationalField> ideal(
      ring,
      {x.pow(2), x * y, x * z, f, f.derivative("y"), f.derivative("z")});
  CHECK(ideal.quotient_dimension() == 11);

  const auto square = ideal.square();
  CHECK(square.quotient_dimension() == 48);
  const auto colon = square.colon(f);
  CHECK(colon.quotient_dimension() == 9);
  const Ideal<RationalField> expected(
      ring,
      {y.pow(4), z.pow(4), y.pow(2) * z, y * z.pow(2), x.pow(2), x * y,
       x * z});
  CHECK(colon == expected);
  CHECK(ideal.is_subset_of(colon));
  CHECK(!(ideal == colon));
  for (const auto& generator : colon.generators()) {
    CHECK(square.contains(generator * f));
  }
}

void test_colon_dense_matrix_guard() {
  const auto ring = make_ring(GF(5), {"x"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const Ideal<PrimeField> ideal(ring, {x.pow(2237)});
  try {
    static_cast<void>(ideal.colon(x));
  } catch (const ResourceLimitExceeded& error) {
    CHECK(error.kind() == ResourceLimitKind::MatrixEntries);
    CHECK(error.limit() == 5'000'000);
    CHECK(error.observed() == 2237U * 2237U);
    return;
  }
  fail("expected principal-colon matrix guard", __LINE__);
}

void test_ideal_operations_preflight_groebner_limits() {
  const auto ring = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const Ideal<RationalField> ideal(ring, {x.pow(2), y.pow(2)});

  GroebnerLimits generator_limit;
  generator_limit.max_basis_polynomials = 2;
  EXPECT_THROW(GroebnerResourceLimit, ideal.square(generator_limit));

  GroebnerLimits term_limit;
  term_limit.max_basis_terms = 1;
  EXPECT_THROW(GroebnerResourceLimit, ideal.sum(ideal, term_limit));
  EXPECT_THROW(GroebnerResourceLimit, ideal.product(ideal, term_limit));
}

}  // namespace

int main() {
  try {
    test_zero_unit_membership_and_equality();
    test_sum_product_and_square();
    test_rational_and_finite_field_colons();
    test_support_at_origin_uses_actual_nilpotence();
    test_elimination_and_validation();
    test_sage_seed_colon();
    test_colon_dense_matrix_guard();
    test_ideal_operations_preflight_groebner_limits();
    std::cout << "laughableengine: ideal tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "laughableengine ideal test failure: " << error.what() << '\n';
    return 1;
  }
}

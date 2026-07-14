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
#include "laughableengine/polynomial.hpp"
#include "laughableengine/truncated_monomial_space.hpp"

namespace {

using laughableengine::GF;
using laughableengine::Monomial;
using laughableengine::MonomialIndexRange;
using laughableengine::PolynomialRing;
using laughableengine::TruncatedMonomialSpace;
using laughableengine::TruncatedMonomialSpaceLimits;
using laughableengine::TruncatedMonomialSpaceResourceLimit;
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

std::vector<std::uint16_t> active_exponents(
    const Monomial& monomial,
    std::size_t variable_count) {
  return std::vector<std::uint16_t>(
      monomial.exponents().begin(),
      monomial.exponents().begin() +
          static_cast<std::ptrdiff_t>(variable_count));
}

void test_shape_and_degree_blocks() {
  const TruncatedMonomialSpace space(3, 4);
  CHECK(space.variable_count() == 3);
  CHECK(space.cutoff() == 4);
  CHECK(space.size() == 20);
  CHECK(space.degree_ranges().size() == 4);
  CHECK((space.degree_range(0) == MonomialIndexRange{0, 1}));
  CHECK((space.degree_range(1) == MonomialIndexRange{1, 4}));
  CHECK((space.degree_range(2) == MonomialIndexRange{4, 10}));
  CHECK((space.degree_range(3) == MonomialIndexRange{10, 20}));
  CHECK(space.monomials_of_degree(2).size() == 6);
  EXPECT_THROW(std::out_of_range, space.degree_range(4));
  EXPECT_THROW(std::out_of_range, space.monomial(space.size()));

  const std::vector<std::vector<std::uint16_t>> expected_degree_two{
      {2, 0, 0}, {1, 1, 0}, {1, 0, 1},
      {0, 2, 0}, {0, 1, 1}, {0, 0, 2}};
  std::vector<std::vector<std::uint16_t>> actual_degree_two;
  for (const auto& monomial : space.monomials_of_degree(2)) {
    actual_degree_two.push_back(active_exponents(monomial, 3));
  }
  CHECK(actual_degree_two == expected_degree_two);
}

void test_index_lookup_is_an_exact_inverse() {
  const TruncatedMonomialSpace space(4, 6);
  for (std::size_t index = 0; index < space.size(); ++index) {
    CHECK(space.find_index(space.monomial(index)) == index);
    CHECK(space.index_of(space.monomial(index)) == index);
  }

  const auto four_variables = make_ring(GF(101), {"w", "x", "y", "z"});
  const auto degree_six = four_variables.gen(0).pow(6);
  CHECK(!space.find_index(degree_six.leading_term()->monomial).has_value());
  EXPECT_THROW(
      std::out_of_range,
      space.index_of(degree_six.leading_term()->monomial));

  const TruncatedMonomialSpace three_variable_space(3, 6);
  CHECK(!three_variable_space
             .find_index(four_variables.gen(3).leading_term()->monomial)
             .has_value());
}

void test_truncated_products() {
  const auto ring = make_ring(GF(7), {"x", "y"});
  const auto x = ring.gen(0);
  const auto y = ring.gen(1);
  const TruncatedMonomialSpace space(2, 4);

  const auto one_index = space.index_of(ring.one().leading_term()->monomial);
  const auto x_index = space.index_of(x.leading_term()->monomial);
  const auto y_index = space.index_of(y.leading_term()->monomial);
  const auto x_squared_index =
      space.index_of(x.pow(2).leading_term()->monomial);
  const auto xy_index = space.index_of((x * y).leading_term()->monomial);

  CHECK(space.product_index(one_index, x_index) == x_index);
  CHECK(space.product_index(x_index, y_index) == xy_index);
  CHECK(space.product_index(y_index, x_index) == xy_index);
  CHECK(space.product_index(x_squared_index, x_index) ==
        space.index_of(x.pow(3).leading_term()->monomial));
  CHECK(!space.product_index(x_squared_index, xy_index).has_value());
  EXPECT_THROW(std::out_of_range, space.product_index(space.size(), 0));
}

void test_variable_steps_and_derivatives() {
  const auto ring = make_ring(GF(7), {"x", "y"});
  const auto x = ring.gen(0);
  const auto y = ring.gen(1);
  const TruncatedMonomialSpace space(2, 4);

  const auto x_index = space.index_of(x.leading_term()->monomial);
  const auto x_squared_index =
      space.index_of(x.pow(2).leading_term()->monomial);
  const auto x_cubed_index =
      space.index_of(x.pow(3).leading_term()->monomial);
  const auto x_squared_y_index =
      space.index_of((x.pow(2) * y).leading_term()->monomial);
  const auto xy_index = space.index_of((x * y).leading_term()->monomial);

  CHECK(space.multiply_by_variable(x_squared_index, 0) == x_cubed_index);
  CHECK(space.multiply_by_variable(x_squared_index, 1) ==
        x_squared_y_index);
  CHECK(!space.multiply_by_variable(x_cubed_index, 1).has_value());

  const auto derivative_x = space.differentiate(x_squared_y_index, 0);
  CHECK(derivative_x.has_value());
  CHECK(derivative_x->first == xy_index);
  CHECK(derivative_x->second == 2);
  const auto derivative_y = space.differentiate(x_squared_y_index, 1);
  CHECK(derivative_y.has_value());
  CHECK(derivative_y->first == x_squared_index);
  CHECK(derivative_y->second == 1);
  CHECK(!space.differentiate(x_index, 1).has_value());

  EXPECT_THROW(std::out_of_range, space.multiply_by_variable(x_index, 2));
  EXPECT_THROW(std::out_of_range, space.differentiate(x_index, 2));
  EXPECT_THROW(std::out_of_range, space.multiply_by_variable(space.size(), 0));
  EXPECT_THROW(std::out_of_range, space.differentiate(space.size(), 0));
}

void test_explicit_polynomial_conversion() {
  const auto ring = make_ring(GF(11), {"x", "y", "z"});
  const auto x = ring.gen(0);
  const auto y = ring.gen(1);
  const auto z = ring.gen(2);
  const TruncatedMonomialSpace space(3, 5);
  const auto source = x.pow(2) * y * z;
  const auto index = space.index_of(source.leading_term()->monomial);
  CHECK(space.as_polynomial(ring, index) == source);

  const auto wrong_ring = make_ring(GF(11), {"x", "y"});
  EXPECT_THROW(std::invalid_argument, space.as_polynomial(wrong_ring, index));
}

void test_input_and_resource_guards() {
  EXPECT_THROW(std::invalid_argument, TruncatedMonomialSpace(0, 3));
  EXPECT_THROW(
      std::invalid_argument,
      TruncatedMonomialSpace(Monomial::maximum_variables + 1, 3));
  EXPECT_THROW(std::invalid_argument, TruncatedMonomialSpace(3, 0));
  EXPECT_THROW(
      std::invalid_argument,
      TruncatedMonomialSpace(
          1,
          static_cast<std::size_t>(
              std::numeric_limits<std::uint16_t>::max()) +
              2));

  try {
    static_cast<void>(TruncatedMonomialSpace(
        3, 4, TruncatedMonomialSpaceLimits{19}));
    fail("expected resource limit was not thrown", __LINE__);
  } catch (const TruncatedMonomialSpaceResourceLimit& error) {
    CHECK(error.limit() == 19);
    CHECK(error.required() == 20);
  }

  // This shape is rejected before either allocation or enumeration.
  EXPECT_THROW(
      std::overflow_error,
      TruncatedMonomialSpace(
          Monomial::maximum_variables,
          static_cast<std::size_t>(
              std::numeric_limits<std::uint16_t>::max()) +
              1));

  const TruncatedMonomialSpace all_variables(
      Monomial::maximum_variables, 2,
      TruncatedMonomialSpaceLimits{Monomial::maximum_variables + 1});
  CHECK(all_variables.size() == Monomial::maximum_variables + 1);
}

}  // namespace

int main() {
  try {
    test_shape_and_degree_blocks();
    test_index_lookup_is_an_exact_inverse();
    test_truncated_products();
    test_variable_steps_and_derivatives();
    test_explicit_polynomial_conversion();
    test_input_and_resource_guards();
    std::cout << "laughableengine: truncated monomial space tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "laughableengine truncated monomial space test failure: "
              << error.what() << '\n';
    return 1;
  }
}

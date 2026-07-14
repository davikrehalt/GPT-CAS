#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/quotient_context.hpp"

namespace {

using laughableengine::DenseMatrix;
using laughableengine::GF;
using laughableengine::Ideal;
using laughableengine::InfiniteQuotientError;
using laughableengine::Order;
using laughableengine::Polynomial;
using laughableengine::PolynomialRing;
using laughableengine::PrimeField;
using laughableengine::QQ;
using laughableengine::QuotientContext;
using laughableengine::RationalField;
using laughableengine::make_ring;
using laughableengine::multiply;

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
std::vector<typename Field::Element> unit_vector(
    const Field& field,
    std::size_t dimension,
    std::size_t index) {
  std::vector<typename Field::Element> result(dimension, field.zero());
  result[index] = field.one();
  return result;
}

template <typename Field>
Polynomial<Field> random_polynomial(
    const PolynomialRing<Field>& ring,
    std::mt19937& generator) {
  std::uniform_int_distribution<long> coefficient(-4, 4);
  std::uniform_int_distribution<unsigned> exponent(0, 5);
  auto result = ring.zero();
  const auto x = ring.gen(0);
  const auto y = ring.gen(1);
  for (std::size_t term = 0; term < 7; ++term) {
    result = result + ring.integer(coefficient(generator)) *
                          x.pow(exponent(generator)) *
                          y.pow(exponent(generator));
  }
  return result;
}

template <typename Field>
void check_matrix_columns(
    const QuotientContext<Field>& quotient,
    const Polynomial<Field>& multiplier,
    const DenseMatrix<Field>& matrix) {
  CHECK(matrix.row_count() == quotient.dimension());
  CHECK(matrix.column_count() == quotient.dimension());
  for (std::size_t column = 0; column < quotient.dimension(); ++column) {
    const auto expected = quotient.coordinates(
        multiplier * quotient.polynomial_basis()[column]);
    CHECK(matrix.multiply_column(
              unit_vector(quotient.field(), quotient.dimension(), column)) ==
          expected);
  }
}

void test_rational_nonmonomial_quotient() {
  const auto ring = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const Ideal<RationalField> ideal(ring, {x.pow(2) - y, y.pow(2)});
  const QuotientContext<RationalField> quotient(ideal);

  CHECK(quotient.ideal() == ideal);
  CHECK(quotient.ring().zero().same_ring(ring.zero()));
  CHECK(quotient.dimension() == 4);
  CHECK(quotient.standard_monomials().size() == 4);
  CHECK(quotient.polynomial_basis().size() == 4);
  CHECK(!quotient.is_zero_quotient());
  CHECK(quotient.supported_at_origin());
  CHECK(quotient.variable_nilpotency_index(0) == 4);
  CHECK(quotient.variable_nilpotency_index(1) == 2);
  CHECK(quotient.variable_nilpotency_index("x") == 4);
  CHECK(quotient.variable_nilpotency_index("y") == 2);

  CHECK(quotient.coordinates(x.pow(2)) == quotient.coordinates(y));
  CHECK(quotient.remainder(x.pow(2) - y).is_zero());
  const auto polynomial = x.pow(5) + ring.integer(3) * x * y + y;
  const auto coordinates = quotient.coordinates(polynomial);
  CHECK(quotient.representative(coordinates) ==
        quotient.remainder(polynomial));
  CHECK(quotient.coordinates(quotient.representative(coordinates)) ==
        coordinates);

  const std::vector<Polynomial<RationalField>> batch_input{
      x.pow(2), y, polynomial, x.pow(2) - y};
  const auto batch = quotient.reduce_batch(batch_input);
  CHECK(batch.remainders.size() == batch_input.size());
  CHECK(batch.coordinates.size() == batch_input.size());
  CHECK(quotient.remainders(batch_input) == batch.remainders);
  CHECK(quotient.coordinates(batch_input) == batch.coordinates);
  for (std::size_t index = 0; index < batch_input.size(); ++index) {
    CHECK(batch.remainders[index] == ideal.normal_form(batch_input[index]));
    CHECK(quotient.representative(batch.coordinates[index]) ==
          batch.remainders[index]);
  }

  const std::vector<Polynomial<RationalField>> empty_batch;
  const auto empty_result = quotient.reduce_batch(empty_batch);
  CHECK(empty_result.remainders.empty());
  CHECK(empty_result.coordinates.empty());

  const auto multiply_x = quotient.dense_variable_multiplication_matrix(0);
  const auto multiply_y = quotient.variable_multiplication_matrix("y");
  CHECK(multiply(multiply_x, multiply_x) == multiply_y);
  CHECK(multiply(multiply_x, multiply_y) ==
        multiply(multiply_y, multiply_x));
  CHECK(multiply(multiply_y, multiply_y) ==
        DenseMatrix<RationalField>(quotient.field(), 4, 4));
  check_matrix_columns(quotient, x, multiply_x);
  check_matrix_columns(quotient, y, multiply_y);

  CHECK(quotient.sparse_variable_multiplication_matrix(0).to_dense() ==
        multiply_x);
  CHECK(quotient.sparse_variable_multiplication_matrix("y").to_dense() ==
        multiply_y);
  CHECK(quotient.sparse_multiplication_matrix(polynomial).to_dense() ==
        quotient.dense_multiplication_matrix(polynomial));
  check_matrix_columns(
      quotient, polynomial,
      quotient.dense_multiplication_matrix(polynomial));

  EXPECT_THROW(
      std::out_of_range,
      quotient.variable_nilpotency_index(quotient.ring().variable_count()));
  EXPECT_THROW(
      std::invalid_argument,
      quotient.variable_nilpotency_index("not_a_variable"));
}

void test_finite_field_quotient_and_field_tags() {
  const auto ring = make_ring(GF(5), {"x", "y"});
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const Ideal<PrimeField> ideal(ring, {x.pow(2) - y, y.pow(2)});
  const QuotientContext<PrimeField> quotient(ideal);

  CHECK(quotient.dimension() == 4);
  CHECK(quotient.supported_at_origin());
  CHECK(quotient.variable_nilpotency_indices().size() == 2);
  CHECK(quotient.variable_nilpotency_index(0) == 4);
  CHECK(quotient.variable_nilpotency_index(1) == 2);
  CHECK(quotient.coordinates(x.pow(2)) == quotient.coordinates(y));

  const auto multiply_x = quotient.dense_variable_multiplication_matrix("x");
  const auto multiply_y = quotient.dense_variable_multiplication_matrix("y");
  CHECK(multiply(multiply_x, multiply_x) == multiply_y);
  CHECK(multiply(multiply_x, multiply_y) ==
        multiply(multiply_y, multiply_x));
  CHECK(quotient.sparse_variable_multiplication_matrix("x").to_dense() ==
        multiply_x);

  const auto foreign_ring = make_ring(GF(5), {"x", "y"});
  EXPECT_THROW(
      std::invalid_argument,
      quotient.coordinates(foreign_ring.gen("x")));

  const auto foreign_field = GF(7);
  std::vector<PrimeField::Element> foreign_coordinates(
      quotient.dimension(), foreign_field.zero());
  EXPECT_THROW(
      std::invalid_argument,
      quotient.representative(foreign_coordinates));
}

void test_origin_support_unit_and_positive_dimension() {
  const auto univariate = make_ring(QQ(), {"x"});
  const auto x = univariate.gen("x");
  const Ideal<RationalField> nonlocal_ideal(
      univariate, {x.pow(2) - x});
  const QuotientContext<RationalField> nonlocal(nonlocal_ideal);
  CHECK(nonlocal.dimension() == 2);
  CHECK(!nonlocal.supported_at_origin());
  CHECK(!nonlocal.variable_nilpotency_index(0).has_value());
  const auto multiply_x = nonlocal.variable_multiplication_matrix(0);
  CHECK(multiply(multiply_x, multiply_x) == multiply_x);

  const Ideal<RationalField> local_ideal(univariate, {x.pow(2)});
  const QuotientContext<RationalField> local(local_ideal);
  CHECK(local.supported_at_origin());
  CHECK(local.variable_nilpotency_index(0) == 2);

  const auto bivariate = make_ring(QQ(), {"x", "y"});
  const auto bx = bivariate.gen("x");
  const auto by = bivariate.gen("y");
  const Ideal<RationalField> unit_ideal(bivariate, {bivariate.one()});
  const QuotientContext<RationalField> unit(unit_ideal);
  CHECK(unit.is_zero_quotient());
  CHECK(unit.dimension() == 0);
  CHECK(unit.standard_monomials().empty());
  CHECK(unit.polynomial_basis().empty());
  CHECK(unit.supported_at_origin());
  CHECK(unit.variable_nilpotency_indices().size() == 2);
  CHECK(unit.variable_nilpotency_index(0) == 0);
  CHECK(unit.variable_nilpotency_index(1) == 0);
  CHECK(unit.remainder(bivariate.one()).is_zero());
  CHECK(unit.coordinates(bx + by).empty());
  const std::vector<RationalField::Element> empty_coordinates;
  CHECK(unit.representative(empty_coordinates).is_zero());

  const auto dense_zero = unit.dense_variable_multiplication_matrix(0);
  const auto sparse_zero = unit.sparse_variable_multiplication_matrix(1);
  CHECK(dense_zero.row_count() == 0);
  CHECK(dense_zero.column_count() == 0);
  CHECK(sparse_zero.row_count() == 0);
  CHECK(sparse_zero.column_count() == 0);
  CHECK(sparse_zero.to_dense() == dense_zero);

  const std::vector<Polynomial<RationalField>> unit_batch{
      bivariate.one(), bx, by};
  const auto reduced_unit_batch = unit.reduce_batch(unit_batch);
  CHECK(reduced_unit_batch.remainders.size() == 3);
  CHECK(reduced_unit_batch.coordinates.size() == 3);
  for (std::size_t index = 0; index < unit_batch.size(); ++index) {
    CHECK(reduced_unit_batch.remainders[index].is_zero());
    CHECK(reduced_unit_batch.coordinates[index].empty());
  }

  const Ideal<RationalField> positive_dimensional(bivariate, {bx});
  EXPECT_THROW(
      InfiniteQuotientError,
      QuotientContext<RationalField>(positive_dimensional));

  const Ideal<RationalField> mixed_support(
      bivariate, {bx.pow(2), by.pow(2) - by});
  const QuotientContext<RationalField> mixed(mixed_support);
  CHECK(!mixed.supported_at_origin());
  CHECK(mixed.variable_nilpotency_index(0) == 2);
  CHECK(!mixed.variable_nilpotency_index(1).has_value());
}

QuotientContext<RationalField> make_escaping_quotient_context() {
  const auto ring = make_ring(QQ(), {"u", "v"});
  const auto u = ring.gen("u");
  const auto v = ring.gen("v");
  return QuotientContext<RationalField>(
      Ideal<RationalField>(ring, {u.pow(2) - v, v.pow(2)}));
}

void test_context_lifetime_copy_and_move() {
  auto escaped = make_escaping_quotient_context();
  auto copied = escaped;
  auto moved = std::move(copied);
  const auto u = moved.ring().gen("u");
  const auto v = moved.ring().gen("v");
  CHECK(moved.dimension() == 4);
  CHECK(moved.coordinates(u.pow(2)) == moved.coordinates(v));
  CHECK(moved.variable_nilpotency_index("u") == 4);
  CHECK(moved.sparse_multiplication_matrix(u).to_dense() ==
        moved.dense_multiplication_matrix(u));
}

template <typename Field>
void run_random_product_coordinate_checks(
    const PolynomialRing<Field>& ring,
    const QuotientContext<Field>& quotient,
    std::uint32_t seed) {
  std::mt19937 generator(seed);
  for (std::size_t trial = 0; trial < 30; ++trial) {
    const auto left = random_polynomial(ring, generator);
    const auto right = random_polynomial(ring, generator);
    const auto product = left * right;
    const auto left_coordinates = quotient.coordinates(left);
    const auto right_coordinates = quotient.coordinates(right);
    const auto product_coordinates = quotient.coordinates(product);

    const auto left_dense = quotient.dense_multiplication_matrix(left);
    const auto right_dense = quotient.dense_multiplication_matrix(right);
    CHECK(left_dense.multiply_column(right_coordinates) == product_coordinates);
    CHECK(right_dense.multiply_column(left_coordinates) == product_coordinates);
    CHECK(quotient.sparse_multiplication_matrix(left).multiply_column(
              right_coordinates) == product_coordinates);
    CHECK(multiply(left_dense, right_dense) ==
          quotient.dense_multiplication_matrix(product));

    const std::vector<Polynomial<Field>> batch_input{left, right, product};
    const auto batch = quotient.reduce_batch(batch_input);
    CHECK(batch.coordinates[0] == left_coordinates);
    CHECK(batch.coordinates[1] == right_coordinates);
    CHECK(batch.coordinates[2] == product_coordinates);
    for (std::size_t index = 0; index < batch_input.size(); ++index) {
      CHECK(batch.remainders[index] == quotient.remainder(batch_input[index]));
      CHECK(quotient.representative(batch.coordinates[index]) ==
            batch.remainders[index]);
    }

    for (std::size_t variable = 0; variable < ring.variable_count();
         ++variable) {
      const auto leibniz = left.derivative(variable) * right +
                           left * right.derivative(variable);
      CHECK(quotient.coordinates(product.derivative(variable)) ==
            quotient.coordinates(leibniz));
    }
  }
}

void test_random_product_coordinate_differentials() {
  const auto rational_ring = make_ring(QQ(), {"x", "y"});
  const auto x = rational_ring.gen("x");
  const auto y = rational_ring.gen("y");
  const QuotientContext<RationalField> rational_quotient(
      Ideal<RationalField>(
          rational_ring, {x.pow(2) - y, y.pow(2)}));
  run_random_product_coordinate_checks(
      rational_ring, rational_quotient, 0x51A9U);

  const auto finite_ring = make_ring(GF(101), {"x", "y"});
  const auto fx = finite_ring.gen("x");
  const auto fy = finite_ring.gen("y");
  const QuotientContext<PrimeField> finite_quotient(
      Ideal<PrimeField>(finite_ring, {fx.pow(2) - fy, fy.pow(2)}));
  run_random_product_coordinate_checks(
      finite_ring, finite_quotient, 0xC0FFEEU);
}

}  // namespace

int main() {
  try {
    test_rational_nonmonomial_quotient();
    test_finite_field_quotient_and_field_tags();
    test_origin_support_unit_and_positive_dimension();
    test_context_lifetime_copy_and_move();
    test_random_product_coordinate_differentials();
    std::cout << "laughableengine: quotient context tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "laughableengine quotient context test failure: "
              << error.what() << '\n';
    return 1;
  }
}

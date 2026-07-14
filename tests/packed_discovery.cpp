#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "laughableengine/compiled_quotient.hpp"
#include "laughableengine/ideal.hpp"
#include "laughableengine/packed_polynomial.hpp"
#include "laughableengine/packed_prime.hpp"

namespace {

using laughableengine::CompiledPrimeQuotientLimits;
using laughableengine::CompiledPrimeQuotientPlan;
using laughableengine::DenseMatrix;
using laughableengine::GF;
using laughableengine::Ideal;
using laughableengine::LinearSolveStatus;
using laughableengine::Order;
using laughableengine::PackedDiscoveryResourceLimit;
using laughableengine::PackedMonomial;
using laughableengine::PackedPrimeArithmetic;
using laughableengine::PackedPrimeMatrix;
using laughableengine::PackedPrimePolynomial;
using laughableengine::PackedPrimeSparseMatrix;
using laughableengine::PackedPrimeTerm;
using laughableengine::Polynomial;
using laughableengine::PolynomialRing;
using laughableengine::PrimeField;
using laughableengine::QuotientContext;

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

std::vector<std::vector<std::uint32_t>> exact_vectors_to_raw(
    const std::vector<std::vector<PrimeField::Element>>& vectors) {
  std::vector<std::vector<std::uint32_t>> result;
  result.reserve(vectors.size());
  for (const auto& vector : vectors) {
    std::vector<std::uint32_t> raw;
    raw.reserve(vector.size());
    for (const auto entry : vector) {
      raw.push_back(entry.value());
    }
    result.push_back(std::move(raw));
  }
  return result;
}

void test_packed_arithmetic() {
  for (const auto modulus :
       {2U, 5U, 101U, PrimeField::maximum_modulus}) {
    const auto field = GF(modulus);
    const PackedPrimeArithmetic packed(modulus);
    for (std::uint32_t left = 0; left < std::min(modulus, 64U); ++left) {
      for (std::uint32_t right = 0; right < std::min(modulus, 64U); ++right) {
        const auto exact_left = field.from_unsigned(left);
        const auto exact_right = field.from_unsigned(right);
        CHECK(packed.add(left, right) ==
              field.add(exact_left, exact_right).value());
        CHECK(packed.subtract(left, right) ==
              field.subtract(exact_left, exact_right).value());
        CHECK(packed.multiply(left, right) ==
              field.multiply(exact_left, exact_right).value());
      }
      if (left != 0) {
        CHECK(packed.inverse(left) == field.inverse(field.from_unsigned(left)).value());
      }
    }
  }
}

void test_packed_matrix_differential() {
  const auto field = GF(101);
  std::uint32_t state = 0x9e3779b9U;
  auto next = [&state]() {
    state = state * 1664525U + 1013904223U;
    return state;
  };
  for (std::size_t rows = 0; rows <= 6; ++rows) {
    for (std::size_t columns = 0; columns <= 6; ++columns) {
      for (std::size_t sample = 0; sample < 20; ++sample) {
        std::vector<PrimeField::Element> exact_entries;
        std::vector<std::uint32_t> packed_entries;
        exact_entries.reserve(rows * columns);
        packed_entries.reserve(rows * columns);
        for (std::size_t index = 0; index < rows * columns; ++index) {
          const auto raw = next() % field.modulus();
          exact_entries.push_back(field.from_unsigned(raw));
          packed_entries.push_back(raw);
        }
        const DenseMatrix<PrimeField> exact(
            field, rows, columns, exact_entries);
        const PackedPrimeMatrix packed(
            field.modulus(), rows, columns, packed_entries);
        CHECK(packed.to_exact() == exact);
        CHECK(packed.rank() == exact.rank());
        CHECK(packed.right_kernel_basis() ==
              exact_vectors_to_raw(exact.right_kernel_basis()));
        CHECK(packed.rank_and_kernel().rank +
                  packed.rank_and_kernel().kernel_basis.size() ==
              columns);

        std::vector<PrimeField::Element> exact_vector;
        std::vector<std::uint32_t> raw_vector;
        for (std::size_t column = 0; column < columns; ++column) {
          const auto raw = next() % field.modulus();
          exact_vector.push_back(field.from_unsigned(raw));
          raw_vector.push_back(raw);
        }
        const auto exact_rhs = exact.multiply_column(exact_vector);
        std::vector<std::uint32_t> raw_rhs;
        for (const auto entry : exact_rhs) {
          raw_rhs.push_back(entry.value());
        }
        CHECK(packed.multiply_column(raw_vector) == raw_rhs);
        const auto exact_solution = exact.solve(exact_rhs);
        const auto packed_solution = packed.solve(raw_rhs);
        CHECK(packed_solution.status == exact_solution.status);
        if (exact_solution.particular_solution.has_value()) {
          std::vector<std::uint32_t> expected;
          for (const auto entry : *exact_solution.particular_solution) {
            expected.push_back(entry.value());
          }
          CHECK(packed_solution.particular_solution == expected);
        }
        CHECK(packed_solution.homogeneous_basis ==
              exact_vectors_to_raw(exact_solution.homogeneous_basis));
      }
    }
  }
}

void test_packed_sparse_matrix() {
  const std::vector<PackedPrimeSparseMatrix::Triplet> triplets{
      {0, 0, 1}, {0, 0, 6}, {1, 0, 2},
      {3, 0, 1}, {0, 2, 3}, {2, 4, 4}};
  const PackedPrimeSparseMatrix sparse(7, 4, 5, triplets);
  CHECK(sparse.nnz() == 4);
  CHECK(sparse.multiply_column(std::vector<std::uint32_t>{2, 0, 3, 0, 5}) ==
        (std::vector<std::uint32_t>{2, 4, 6, 2}));
  CHECK(sparse.to_dense().multiply_column(
            std::vector<std::uint32_t>{2, 0, 3, 0, 5}) ==
        sparse.multiply_column(
            std::vector<std::uint32_t>{2, 0, 3, 0, 5}));

  const PackedPrimeMatrix right(
      7, 5, 3,
      std::vector<std::uint32_t>{
          2, 1, 6, 0, 3, 4, 3, 5, 2, 1, 0, 6, 5, 4, 3});
  CHECK(multiply(sparse, right) == multiply(sparse.to_dense(), right));
}

void test_packed_polynomials() {
  const PolynomialRing<PrimeField> ring(
      GF(101), {"x", "y"}, Order::Grevlex);
  const auto x = ring.gen(0);
  const auto y = ring.gen(1);
  const auto left = x * x + ring.integer(3) * x * y + ring.integer(5);
  const auto right = y * y + ring.integer(7) * x - ring.integer(5);
  const auto packed_left = PackedPrimePolynomial::from_exact(left);
  const auto packed_right = PackedPrimePolynomial::from_exact(right);
  CHECK(packed_left.to_exact(ring) == left);
  CHECK(packed_right.to_exact(ring) == right);
  CHECK((packed_left + packed_right).to_exact(ring) == left + right);
  CHECK((packed_left * packed_right).to_exact(ring) == left * right);

  const auto high = ring.monomial(ring.field().one(), {256, 0});
  EXPECT_THROW(
      std::overflow_error, PackedPrimePolynomial::from_exact(high));
  EXPECT_THROW(
      std::invalid_argument,
      PackedMonomial::from_word(
          PackedMonomial::from_exponents(
              std::vector<std::uint16_t>{2, 1}, 2).word() ^
          (std::uint64_t{1} << 48U)));

  const auto hidden_second_variable = PackedMonomial::from_exponents(
      std::vector<std::uint16_t>{0, 1}, 2);
  EXPECT_THROW(
      std::invalid_argument,
      PackedPrimePolynomial(
          101, 1, Order::Grevlex,
          std::vector<PackedPrimeTerm>{{hidden_second_variable, 1}}));
  const auto packed_one = PackedMonomial::from_exponents(
      std::vector<std::uint16_t>{0}, 1);
  EXPECT_THROW(
      std::invalid_argument,
      hidden_second_variable.multiplied_by(packed_one, 1));
  EXPECT_THROW(
      std::invalid_argument,
      hidden_second_variable.is_divisible_by(packed_one, 1));
  EXPECT_THROW(
      std::invalid_argument,
      PackedMonomial::compare(
          hidden_second_variable, packed_one, Order::Grevlex, 1));

  const PolynomialRing<PrimeField> seven_variable_ring(
      GF(101), {"x1", "x2", "x3", "x4", "x5", "x6", "x7"},
      Order::Grevlex);
  CHECK(seven_variable_ring.variable_count() == 7);
  EXPECT_THROW(
      std::invalid_argument,
      PackedPrimePolynomial::from_exact(seven_variable_ring.gen("x7")));
}

void test_compiled_quotient_batch_and_fingerprints() {
  const PolynomialRing<PrimeField> ring(
      GF(101), {"x", "y"}, Order::Grevlex);
  const auto x = ring.gen(0);
  const auto y = ring.gen(1);
  const Ideal<PrimeField> ideal(ring, {x * x - y, y * y});
  const CompiledPrimeQuotientPlan plan(ideal);
  const QuotientContext<PrimeField> reference(ideal);
  CHECK(plan.dimension() == reference.dimension());
  CHECK(plan.variable_actions().size() == 2);
  CHECK(CompiledPrimeQuotientPlan::scalar_normal_form_calls_per_batch == 0);

  std::vector<Polynomial<PrimeField>> polynomials;
  for (std::size_t exponent = 0; exponent < 18; ++exponent) {
    polynomials.push_back(
        (x + y).pow(exponent) + ring.integer(static_cast<long>(exponent)));
  }
  std::vector<PackedPrimePolynomial> packed;
  for (const auto& polynomial : polynomials) {
    packed.push_back(PackedPrimePolynomial::from_exact(polynomial));
  }
  const auto actual = plan.reduce_coordinates_batch(packed);
  const auto expected = reference.coordinates(polynomials);
  CHECK(actual.row_count() == expected.size());
  for (std::size_t row = 0; row < expected.size(); ++row) {
    std::vector<std::uint32_t> raw;
    for (const auto entry : expected[row]) {
      raw.push_back(entry.value());
    }
    CHECK(std::vector<std::uint32_t>(
              actual.row(row).begin(), actual.row(row).end()) == raw);
  }

  const auto multiplier = PackedPrimePolynomial::from_exact(x + y);
  CHECK(plan.multiplication_matrix(multiplier).to_exact() ==
        reference.dense_multiplication_matrix(x + y));

  const CompiledPrimeQuotientPlan permuted(
      Ideal<PrimeField>(ring, {y * y, x * x - y, x * x - y}));
  CHECK(plan.full_signature() == permuted.full_signature());
  CHECK(plan.full_fingerprint() == permuted.full_fingerprint());
  const CompiledPrimeQuotientPlan changed(
      Ideal<PrimeField>(ring, {x * x - ring.integer(2) * y, y * y}));
  CHECK(plan.leading_signature() == changed.leading_signature());
  CHECK(plan.leading_fingerprint() == changed.leading_fingerprint());
  CHECK(plan.full_signature() != changed.full_signature());
  CHECK(plan.full_fingerprint() != changed.full_fingerprint());

  CompiledPrimeQuotientLimits limits;
  limits.max_batch_entries = 1;
  EXPECT_THROW(
      PackedDiscoveryResourceLimit,
      plan.reduce_coordinates_batch(packed, limits));
  limits.max_batch_entries = 1'000;
  limits.max_variable_action_applications = 0;
  EXPECT_THROW(
      PackedDiscoveryResourceLimit,
      plan.reduce_coordinates_batch(packed, limits));

  std::size_t shared_action_count = 0;
  CompiledPrimeQuotientLimits shared_limits;
  shared_limits.max_variable_action_applications = 1;
  EXPECT_THROW(
      PackedDiscoveryResourceLimit,
      plan.multiply_by_monomial(
          PackedMonomial::from_exponents(
              std::vector<std::uint16_t>{2, 0}, 2),
          std::vector<std::uint32_t>(plan.dimension(), 1),
          shared_action_count, shared_limits));

  const PolynomialRing<PrimeField> univariate(
      GF(101), {"t"}, Order::Grevlex);
  const auto t = univariate.gen(0);
  const CompiledPrimeQuotientPlan unit_plan(
      Ideal<PrimeField>(univariate, {univariate.one()}));
  CHECK(unit_plan.dimension() == 0);
  const std::vector<Polynomial<PrimeField>> unit_inputs{t, t + univariate.one()};
  const auto unit_remainders =
      unit_plan.reduce_coordinates_batch(unit_inputs);
  CHECK(unit_remainders.row_count() == unit_inputs.size());
  CHECK(unit_remainders.column_count() == 0);
  CHECK(unit_remainders.row(0).empty());

  const CompiledPrimeQuotientPlan one_variable_plan(
      Ideal<PrimeField>(univariate, {t * t}));
  const auto inactive_lane = PackedMonomial::from_exponents(
      std::vector<std::uint16_t>{0, 1}, 2);
  EXPECT_THROW(
      std::invalid_argument,
      one_variable_plan.multiply_by_monomial(
          inactive_lane,
          std::vector<std::uint32_t>(one_variable_plan.dimension(), 1)));
}

}  // namespace

int main() {
  try {
    test_packed_arithmetic();
    test_packed_matrix_differential();
    test_packed_sparse_matrix();
    test_packed_polynomials();
    test_compiled_quotient_batch_and_fingerprints();
    std::cout << "packed discovery tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "packed discovery test failure: " << error.what() << '\n';
    return 1;
  }
}

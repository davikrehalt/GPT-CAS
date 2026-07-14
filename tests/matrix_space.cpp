#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/field.hpp"
#include "laughableengine/matrix.hpp"
#include "laughableengine/matrix_space.hpp"

namespace {

using laughableengine::DenseMatrix;
using laughableengine::GF;
using laughableengine::MatrixSpaceRankProof;
using laughableengine::MatrixSpaceSearchLimits;
using laughableengine::PrimeField;
using laughableengine::QQ;
using laughableengine::RationalField;
using laughableengine::analyze_matrix_space;

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
DenseMatrix<Field> integer_matrix(
    Field field,
    std::size_t rows,
    std::size_t columns,
    std::initializer_list<long> values) {
  std::vector<typename Field::Element> entries;
  entries.reserve(values.size());
  for (const auto value : values) {
    entries.push_back(field.from_integer(value));
  }
  return DenseMatrix<Field>(
      std::move(field), rows, columns, std::move(entries));
}

template <typename Field>
std::size_t witness_rank(
    const Field& field,
    const std::vector<DenseMatrix<Field>>& basis,
    const std::vector<typename Field::Element>& coefficients) {
  DenseMatrix<Field> combined(
      field, basis.front().row_count(), basis.front().column_count());
  for (std::size_t row = 0; row < combined.row_count(); ++row) {
    for (std::size_t column = 0; column < combined.column_count(); ++column) {
      auto value = field.zero();
      for (std::size_t parameter = 0; parameter < basis.size(); ++parameter) {
        value = field.add(
            value,
            field.multiply(coefficients[parameter],
                           basis[parameter](row, column)));
      }
      combined.set(row, column, std::move(value));
    }
  }
  return combined.rank();
}

template <typename Field>
std::vector<DenseMatrix<Field>> diagonal_pitfall(const Field& field) {
  return {
      integer_matrix(field, 3, 3, {1, 0, 0, 0, 0, 0, 0, 0, 1}),
      integer_matrix(field, 3, 3, {0, 0, 0, 0, 1, 0, 0, 0, 1}),
  };
}

void test_rational_symbolic_full_rank_witness() {
  const auto field = QQ();
  const auto basis = diagonal_pitfall(field);
  const auto result = analyze_matrix_space(field, 3, 3, basis);
  CHECK(result.lower_bound == 3);
  CHECK(result.upper_bound == 3);
  CHECK(result.generic_maximum == 3);
  CHECK(result.exact_maximum == 3);
  CHECK(result.proof == MatrixSpaceRankProof::ProvenFullColumnRank);
  CHECK(result.has_full_column_rank_witness());
  CHECK(result.witness_coefficients.has_value());
  CHECK(witness_rank(field, basis, *result.witness_coefficients) == 3);
  CHECK(result.minors_tested == 1);
  CHECK(result.determinant_products_tested == 6);
}

void test_small_finite_field_symbolic_pitfall_is_exhausted() {
  const auto field = GF(2);
  const auto basis = diagonal_pitfall(field);
  const auto result = analyze_matrix_space(field, 3, 3, basis);
  CHECK(result.generic_maximum == 3);
  CHECK(result.exact_maximum == 2);
  CHECK(result.lower_bound == 2);
  CHECK(result.upper_bound == 2);
  CHECK(result.proof == MatrixSpaceRankProof::ProvenMaximum);
  CHECK(!result.has_full_column_rank_witness());
  CHECK(result.finite_field_evaluations == 4);
  CHECK(result.witness_coefficients.has_value());
  CHECK(witness_rank(field, basis, *result.witness_coefficients) == 2);
}

void test_large_enough_finite_field_symbolic_witness() {
  const auto field = GF(5);
  const auto basis = diagonal_pitfall(field);
  const auto result = analyze_matrix_space(field, 3, 3, basis);
  CHECK(result.generic_maximum == 3);
  CHECK(result.exact_maximum == 3);
  CHECK(result.proof == MatrixSpaceRankProof::ProvenFullColumnRank);
  CHECK(result.finite_field_evaluations == 0);
  CHECK(result.witness_coefficients.has_value());
  CHECK(witness_rank(field, basis, *result.witness_coefficients) == 3);
}

void test_common_image_bound_proves_rank_one() {
  const auto field = QQ();
  const std::vector<DenseMatrix<RationalField>> basis{
      integer_matrix(field, 4, 3, {1, 2, 3, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0}),
      integer_matrix(field, 4, 3, {2, -1, 4, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0}),
  };
  const auto result = analyze_matrix_space(field, 4, 3, basis);
  CHECK(result.lower_bound == 1);
  CHECK(result.upper_bound == 1);
  CHECK(result.generic_maximum == 1);
  CHECK(result.exact_maximum == 1);
  CHECK(result.proof == MatrixSpaceRankProof::ProvenMaximum);
  CHECK(result.minors_tested == 0);
}

void test_zero_space_and_empty_domain() {
  const auto field = QQ();
  const std::vector<DenseMatrix<RationalField>> no_parameters;
  const auto zero = analyze_matrix_space(field, 5, 3, no_parameters);
  CHECK(zero.exact_maximum == 0);
  CHECK(zero.proof == MatrixSpaceRankProof::ProvenMaximum);
  CHECK(zero.witness_coefficients.has_value());
  CHECK(zero.witness_coefficients->empty());

  const auto empty_domain =
      analyze_matrix_space(field, 5, 0, no_parameters);
  CHECK(empty_domain.exact_maximum == 0);
  CHECK(empty_domain.has_full_column_rank_witness());
  CHECK(empty_domain.proof == MatrixSpaceRankProof::ProvenFullColumnRank);
}

void test_generic_only_and_resource_limit_statuses() {
  const auto field = GF(2);
  const auto basis = diagonal_pitfall(field);
  MatrixSpaceSearchLimits no_exhaustion;
  no_exhaustion.max_finite_field_evaluations = 3;
  const auto generic =
      analyze_matrix_space(field, 3, 3, basis, no_exhaustion);
  CHECK(generic.generic_maximum == 3);
  CHECK(!generic.exact_maximum.has_value());
  CHECK(generic.lower_bound == 2);
  CHECK(generic.upper_bound == 3);
  CHECK(generic.proof == MatrixSpaceRankProof::GenericOnly);

  MatrixSpaceSearchLimits no_minors;
  no_minors.max_minors = 0;
  const auto limited = analyze_matrix_space(field, 3, 3, basis, no_minors);
  CHECK(limited.proof == MatrixSpaceRankProof::ResourceLimit);
  CHECK(!limited.generic_maximum.has_value());
  CHECK(!limited.exact_maximum.has_value());
  CHECK(limited.lower_bound == 2);
  CHECK(limited.upper_bound == 3);

  MatrixSpaceSearchLimits no_determinant_products;
  no_determinant_products.max_determinant_products = 0;
  const auto determinant_limited =
      analyze_matrix_space(field, 3, 3, basis, no_determinant_products);
  CHECK(determinant_limited.proof == MatrixSpaceRankProof::ResourceLimit);
  CHECK(determinant_limited.minors_tested == 1);
  CHECK(determinant_limited.determinant_products_tested == 0);
}

void test_validation() {
  const auto field = GF(5);
  const std::vector<DenseMatrix<PrimeField>> wrong_shape{
      integer_matrix(field, 2, 2, {1, 0, 0, 1})};
  EXPECT_THROW(
      std::invalid_argument,
      analyze_matrix_space(field, 3, 3, wrong_shape));

  const auto foreign = GF(7);
  const std::vector<DenseMatrix<PrimeField>> wrong_field{
      integer_matrix(foreign, 2, 2, {1, 0, 0, 1})};
  EXPECT_THROW(
      std::invalid_argument,
      analyze_matrix_space(field, 2, 2, wrong_field));
}

void test_randomized_gf5_against_exhaustive_search() {
  const auto field = GF(5);
  std::uint32_t state = 0x6d2b79f5U;
  const auto next = [&state]() {
    state = state * 1664525U + 1013904223U;
    return state;
  };

  for (std::size_t sample = 0; sample < 80; ++sample) {
    const std::size_t rows = 1 + next() % 3U;
    const std::size_t columns = 1 + next() % 3U;
    const std::size_t parameters = 1 + next() % 3U;
    std::vector<DenseMatrix<PrimeField>> basis;
    basis.reserve(parameters);
    for (std::size_t parameter = 0; parameter < parameters; ++parameter) {
      std::vector<PrimeField::Element> entries;
      entries.reserve(rows * columns);
      for (std::size_t index = 0; index < rows * columns; ++index) {
        entries.push_back(field.from_unsigned(next() % 5U));
      }
      basis.emplace_back(field, rows, columns, std::move(entries));
    }

    std::size_t exhaustive_maximum = 0;
    std::size_t evaluations = 1;
    for (std::size_t parameter = 0; parameter < parameters; ++parameter) {
      evaluations *= 5;
    }
    std::vector<std::uint32_t> digits(parameters, 0);
    for (std::size_t evaluation = 0; evaluation < evaluations; ++evaluation) {
      std::vector<PrimeField::Element> coefficients;
      coefficients.reserve(parameters);
      for (const auto digit : digits) {
        coefficients.push_back(field.from_unsigned(digit));
      }
      exhaustive_maximum = std::max(
          exhaustive_maximum, witness_rank(field, basis, coefficients));
      for (std::size_t position = 0; position < parameters; ++position) {
        ++digits[position];
        if (digits[position] < 5) {
          break;
        }
        digits[position] = 0;
      }
    }

    const auto result =
        analyze_matrix_space(field, rows, columns, basis);
    CHECK(result.exact_maximum == exhaustive_maximum);
    CHECK(result.lower_bound == exhaustive_maximum);
    CHECK(result.upper_bound == exhaustive_maximum);
    CHECK(result.witness_coefficients.has_value());
    CHECK(witness_rank(field, basis, *result.witness_coefficients) ==
          exhaustive_maximum);
  }
}

}  // namespace

int main() {
  try {
    test_rational_symbolic_full_rank_witness();
    test_small_finite_field_symbolic_pitfall_is_exhausted();
    test_large_enough_finite_field_symbolic_witness();
    test_common_image_bound_proves_rank_one();
    test_zero_space_and_empty_domain();
    test_generic_only_and_resource_limit_statuses();
    test_validation();
    test_randomized_gf5_against_exhaustive_search();
    std::cout << "matrix-space tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "matrix-space test failure: " << error.what() << '\n';
    return 1;
  }
}

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "laughableengine/field.hpp"
#include "laughableengine/matrix.hpp"
#include "laughableengine/sparse_matrix.hpp"

namespace {

using laughableengine::DenseMatrix;
using laughableengine::GF;
using laughableengine::LinearSolveStatus;
using laughableengine::PrimeField;
using laughableengine::QQ;
using laughableengine::RationalField;
using laughableengine::SparseMatrix;
using laughableengine::SparseEliminationLimits;
using laughableengine::SparseEliminationResourceKind;
using laughableengine::SparseEliminationResourceLimit;

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

#define EXPECT_THROW(exception, expression)                                   \
  expect_throw<exception>([&] { static_cast<void>(expression); }, __LINE__)

template <typename Field>
SparseMatrix<Field> integer_sparse_matrix(
    Field field,
    std::size_t rows,
    std::size_t columns,
    std::initializer_list<std::tuple<std::size_t, std::size_t, long>> values) {
  std::vector<typename SparseMatrix<Field>::Triplet> triplets;
  triplets.reserve(values.size());
  for (const auto& [row, column, value] : values) {
    triplets.push_back(typename SparseMatrix<Field>::Triplet{
        row, column, field.from_integer(value)});
  }
  return SparseMatrix<Field>(
      std::move(field), rows, columns, std::move(triplets));
}

template <typename Field>
DenseMatrix<Field> integer_dense_matrix(
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
std::vector<typename Field::Element> integer_vector(
    const Field& field,
    std::initializer_list<long> values) {
  std::vector<typename Field::Element> result;
  result.reserve(values.size());
  for (const auto value : values) {
    result.push_back(field.from_integer(value));
  }
  return result;
}

template <typename Element>
void check_vectors_equal(
    const std::vector<Element>& actual,
    const std::vector<Element>& expected,
    int line) {
  if (actual != expected) {
    fail("vectors differ", line);
  }
}

#define CHECK_VECTOR(actual, expected)                                        \
  check_vectors_equal((actual), (expected), __LINE__)

template <typename Field>
void check_kernel(const SparseMatrix<Field>& matrix, int line) {
  const auto basis = matrix.right_kernel_basis();
  if (basis.size() != matrix.column_count() - matrix.rank()) {
    fail("rank-nullity failed", line);
  }
  for (const auto& vector : basis) {
    if (vector.size() != matrix.column_count()) {
      fail("kernel vector has the wrong dimension", line);
    }
    for (const auto& entry : matrix.multiply_column(vector)) {
      if (!matrix.field().is_zero(entry)) {
        fail("reported kernel vector has nonzero image", line);
      }
    }
  }
}

#define CHECK_KERNEL(matrix) check_kernel((matrix), __LINE__)

void test_canonical_triplets_lookup_and_transpose() {
  const auto field = QQ();
  const auto matrix = integer_sparse_matrix(
      field,
      4,
      5,
      {{3, 4, 7},
       {0, 1, 2},
       {2, 3, 4},
       {1, 0, 1},
       {0, 1, 3},
       {3, 2, -2},
       {2, 3, -4},
       {2, 2, 0}});

  CHECK(!SparseMatrix<RationalField>::elimination_uses_dense_fallback);
  CHECK(matrix.row_count() == 4);
  CHECK(matrix.column_count() == 5);
  CHECK(matrix.nnz() == 4);
  CHECK(matrix.nonzero_count() == 4);
  CHECK(matrix(0, 1) == field.from_integer(5));
  CHECK(matrix(1, 0) == field.one());
  CHECK(matrix(2, 3) == field.zero());
  CHECK(matrix(3, 2) == field.from_integer(-2));
  CHECK(matrix(3, 4) == field.from_integer(7));
  CHECK(matrix(0, 4) == field.zero());

  CHECK(matrix.to_dense() == integer_dense_matrix(
      field,
      4,
      5,
      {0, 5, 0, 0, 0,
       1, 0, 0, 0, 0,
       0, 0, 0, 0, 0,
       0, 0, -2, 0, 7}));
  CHECK_VECTOR(
      matrix.multiply_column(integer_vector(field, {1, 2, 3, 4, 5})),
      integer_vector(field, {10, 1, 0, 29}));

  const auto transpose = matrix.transpose();
  CHECK(transpose.row_count() == 5);
  CHECK(transpose.column_count() == 4);
  CHECK(transpose.nonzero_count() == 4);
  CHECK(transpose(1, 0) == field.from_integer(5));
  CHECK(transpose(4, 3) == field.from_integer(7));
  CHECK(transpose.transpose() == matrix);

  const auto reordered = integer_sparse_matrix(
      field,
      4,
      5,
      {{0, 1, 5}, {3, 2, -2}, {1, 0, 1}, {3, 4, 7}});
  CHECK(reordered == matrix);

  EXPECT_THROW(std::out_of_range, matrix.at(4, 0));
  EXPECT_THROW(std::out_of_range, matrix.at(0, 5));
  EXPECT_THROW(
      std::invalid_argument,
      matrix.multiply_column(integer_vector(field, {1, 2})));
}

void test_sparse_linear_algebra_and_solves() {
  const auto field = QQ();
  const auto matrix = integer_sparse_matrix(
      field,
      3,
      4,
      {{0, 0, 1}, {0, 1, 2}, {0, 2, 3}, {0, 3, 4},
       {1, 0, 2}, {1, 1, 4}, {1, 2, 6}, {1, 3, 8},
       {2, 1, 1}, {2, 2, 1}, {2, 3, 1}});
  CHECK(matrix.rank() == 2);
  CHECK_KERNEL(matrix);
  CHECK(matrix.right_kernel_basis() ==
        matrix.to_dense().right_kernel_basis());
  const auto sparse_kernel = matrix.right_kernel_basis_sparse();
  CHECK(sparse_kernel.size() == matrix.right_kernel_basis().size());
  for (std::size_t index = 0; index < sparse_kernel.size(); ++index) {
    auto expanded = std::vector<RationalField::Element>(
        matrix.column_count(), field.zero());
    for (const auto& entry : sparse_kernel[index]) {
      CHECK(entry.column < expanded.size());
      CHECK(!field.is_zero(entry.value));
      expanded[entry.column] = entry.value;
    }
    CHECK(expanded == matrix.right_kernel_basis()[index]);
  }
  const auto sparse_rank_kernel = matrix.rank_and_sparse_kernel();
  CHECK(sparse_rank_kernel.rank == matrix.rank());
  CHECK(sparse_rank_kernel.kernel_basis == sparse_kernel);
  CHECK(matrix.column_space_basis() ==
        matrix.to_dense().column_space_basis());

  const auto underdetermined =
      matrix.solve(integer_vector(field, {3, 6, 2}));
  CHECK(underdetermined.status == LinearSolveStatus::Underdetermined);
  CHECK(underdetermined.particular_solution.has_value());
  CHECK_VECTOR(
      *underdetermined.particular_solution,
      integer_vector(field, {-1, 2, 0, 0}));
  CHECK_VECTOR(
      matrix.multiply_column(*underdetermined.particular_solution),
      integer_vector(field, {3, 6, 2}));
  for (const auto& vector : underdetermined.homogeneous_basis) {
    for (const auto& entry : matrix.multiply_column(vector)) {
      CHECK(field.is_zero(entry));
    }
  }

  const auto inconsistent =
      matrix.solve(integer_vector(field, {3, 7, 2}));
  CHECK(inconsistent.status == LinearSolveStatus::Inconsistent);
  CHECK(!inconsistent.particular_solution.has_value());

  const auto diagonal = integer_sparse_matrix(
      field, 3, 3, {{0, 0, 2}, {1, 1, -3}, {2, 2, 5}});
  const auto unique = diagonal.solve(integer_vector(field, {4, 6, 15}));
  CHECK(unique.status == LinearSolveStatus::Unique);
  CHECK(unique.particular_solution.has_value());
  CHECK_VECTOR(
      *unique.particular_solution, integer_vector(field, {2, -2, 3}));
  CHECK(unique.homogeneous_basis.empty());
}

void test_empty_shapes() {
  const auto field = QQ();
  const std::vector<SparseMatrix<RationalField>::Triplet> no_triplets;
  const std::vector<RationalField::Element> empty;

  const SparseMatrix<RationalField> no_equations(
      field, 0, 3, no_triplets);
  CHECK(no_equations.nonzero_count() == 0);
  CHECK(no_equations.rank() == 0);
  CHECK(no_equations.right_kernel_basis().size() == 3);
  CHECK(no_equations.column_space_basis().empty());
  CHECK(no_equations.multiply_column(
            integer_vector(field, {1, 2, 3})).empty());
  const auto free_solution = no_equations.solve(empty);
  CHECK(free_solution.status == LinearSolveStatus::Underdetermined);
  CHECK(free_solution.particular_solution.has_value());
  CHECK(free_solution.particular_solution->size() == 3);
  CHECK(free_solution.homogeneous_basis.size() == 3);

  const SparseMatrix<RationalField> no_unknowns(
      field, 3, 0, no_triplets);
  CHECK(no_unknowns.rank() == 0);
  CHECK(no_unknowns.right_kernel_basis().empty());
  CHECK(no_unknowns.column_space_basis().empty());
  const auto zero_image = no_unknowns.multiply_column(empty);
  CHECK(zero_image.size() == 3);
  for (const auto& entry : zero_image) {
    CHECK(field.is_zero(entry));
  }
  const auto unique = no_unknowns.solve(integer_vector(field, {0, 0, 0}));
  CHECK(unique.status == LinearSolveStatus::Unique);
  CHECK(unique.particular_solution.has_value());
  CHECK(unique.particular_solution->empty());
  const auto inconsistent =
      no_unknowns.solve(integer_vector(field, {0, 1, 0}));
  CHECK(inconsistent.status == LinearSolveStatus::Inconsistent);

  const SparseMatrix<RationalField> empty_matrix(
      field, 0, 0, no_triplets);
  CHECK(empty_matrix.rank() == 0);
  CHECK(empty_matrix.transpose() == empty_matrix);
  const auto empty_solution = empty_matrix.solve(empty);
  CHECK(empty_solution.status == LinearSolveStatus::Unique);
  CHECK(empty_solution.particular_solution.has_value());
  CHECK(empty_solution.particular_solution->empty());
}

void test_validation_and_finite_field_cancellation() {
  const auto field = GF(5);
  const auto cancelled = integer_sparse_matrix(
      field,
      2,
      2,
      {{0, 0, 2}, {0, 0, 3}, {1, 1, 4}, {1, 1, 1}});
  CHECK(cancelled.nonzero_count() == 0);
  CHECK(cancelled.rank() == 0);

  const auto foreign = GF(7);
  std::vector<SparseMatrix<PrimeField>::Triplet> foreign_triplets{
      {0, 0, foreign.one()}};
  EXPECT_THROW(
      std::invalid_argument,
      SparseMatrix<PrimeField>(field, 1, 1, foreign_triplets));

  const auto identity = integer_sparse_matrix(
      field, 2, 2, {{0, 0, 1}, {1, 1, 1}});
  const std::vector<PrimeField::Element> foreign_vector{foreign.one(),
                                                        foreign.zero()};
  EXPECT_THROW(
      std::invalid_argument, identity.multiply_column(foreign_vector));
  EXPECT_THROW(std::invalid_argument, identity.solve(foreign_vector));

  std::vector<SparseMatrix<PrimeField>::Triplet> outside{
      {2, 0, field.one()}};
  EXPECT_THROW(
      std::out_of_range,
      SparseMatrix<PrimeField>(field, 2, 2, outside));
  EXPECT_THROW(
      std::length_error,
      SparseMatrix<PrimeField>(
          field,
          std::numeric_limits<std::size_t>::max(),
          0,
          std::vector<SparseMatrix<PrimeField>::Triplet>{}));
}

void test_sparse_elimination_resource_limits_precede_dense_outputs() {
  const auto field = GF(7);
  const auto matrix = integer_sparse_matrix(
      field, 2, 3, {{0, 0, 1}, {0, 2, 1}, {1, 1, 1}});

  SparseEliminationLimits live_limits;
  live_limits.max_live_nonzeros = 0;
  EXPECT_THROW(SparseEliminationResourceLimit, matrix.rank(live_limits));

  SparseEliminationLimits operation_limits;
  operation_limits.max_arithmetic_operations = 0;
  const auto operation_error = EXPECT_THROW(
      SparseEliminationResourceLimit, matrix.rank(operation_limits));
  CHECK(operation_error.kind() ==
        SparseEliminationResourceKind::ArithmeticOperations);

  SparseEliminationLimits kernel_limits;
  kernel_limits.max_kernel_coordinate_entries = 0;
  const auto kernel_error = EXPECT_THROW(
      SparseEliminationResourceLimit,
      matrix.rank_and_kernel(kernel_limits));
  CHECK(kernel_error.kind() ==
        SparseEliminationResourceKind::KernelCoordinates);
  CHECK(kernel_error.observed() == 3);

  SparseEliminationLimits sparse_kernel_limits;
  sparse_kernel_limits.max_kernel_nonzeros = 0;
  const auto sparse_kernel_error = EXPECT_THROW(
      SparseEliminationResourceLimit,
      matrix.rank_and_sparse_kernel(sparse_kernel_limits));
  CHECK(sparse_kernel_error.kind() ==
        SparseEliminationResourceKind::KernelNonzeros);

  SparseEliminationLimits output_limits;
  output_limits.max_output_coordinate_entries = 0;
  const auto output_error = EXPECT_THROW(
      SparseEliminationResourceLimit,
      matrix.column_space_basis(output_limits));
  CHECK(output_error.kind() ==
        SparseEliminationResourceKind::OutputCoordinates);
  CHECK(output_error.observed() == 4);
}

void test_randomized_sparse_dense_differential() {
  const auto field = GF(7);
  std::uint32_t state = 0x51a09e37U;
  auto next = [&state]() {
    state = state * 1664525U + 1013904223U;
    return state;
  };

  for (std::size_t rows = 0; rows <= 5; ++rows) {
    for (std::size_t columns = 0; columns <= 5; ++columns) {
      for (std::size_t sample = 0; sample < 15; ++sample) {
        std::vector<SparseMatrix<PrimeField>::Triplet> triplets;
        std::vector<PrimeField::Element> accumulated(
            rows * columns, field.zero());
        const auto add_triplet = [&](std::size_t row,
                                     std::size_t column,
                                     PrimeField::Element value) {
          triplets.push_back({row, column, value});
          auto& entry = accumulated[row * columns + column];
          entry = field.add(entry, value);
        };

        if (rows != 0 && columns != 0) {
          const auto count = rows * columns + sample % 5;
          for (std::size_t index = 0; index < count; ++index) {
            const auto row = next() % rows;
            const auto column = next() % columns;
            const auto coefficient =
                static_cast<long>(next() % 13U) - 6L;
            const auto value = field.from_integer(coefficient);
            add_triplet(row, column, value);
            if ((index % 5) == 0) {
              add_triplet(row, column, field.negate(value));
            }
          }
        }

        const SparseMatrix<PrimeField> sparse(
            field, rows, columns, std::move(triplets));
        const DenseMatrix<PrimeField> dense(
            field, rows, columns, std::move(accumulated));
        CHECK(sparse.to_dense() == dense);
        CHECK(sparse.nonzero_count() == [&] {
          std::size_t count = 0;
          for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
              if (!field.is_zero(dense(row, column))) {
                ++count;
              }
            }
          }
          return count;
        }());
        CHECK(sparse.transpose().to_dense() == dense.transpose());
        CHECK(sparse.rank() == dense.rank());
        CHECK(sparse.right_kernel_basis() == dense.right_kernel_basis());
        CHECK(sparse.column_space_basis() == dense.column_space_basis());
        CHECK_KERNEL(sparse);

        std::vector<PrimeField::Element> vector;
        vector.reserve(columns);
        for (std::size_t column = 0; column < columns; ++column) {
          vector.push_back(field.from_unsigned(next() % 7U));
        }
        CHECK_VECTOR(
            sparse.multiply_column(vector), dense.multiply_column(vector));

        const auto right_hand_side = dense.multiply_column(vector);
        const auto sparse_solution = sparse.solve(right_hand_side);
        const auto dense_solution = dense.solve(right_hand_side);
        CHECK(sparse_solution.status == dense_solution.status);
        CHECK(sparse_solution.particular_solution ==
              dense_solution.particular_solution);
        CHECK(sparse_solution.homogeneous_basis ==
              dense_solution.homogeneous_basis);
      }
    }
  }
}

void test_large_native_sparse_elimination() {
  const auto field = GF(7);
  constexpr std::size_t dimension = 20'000;

  std::vector<SparseMatrix<PrimeField>::Triplet> triplets;
  triplets.reserve(dimension);
  std::vector<PrimeField::Element> right_hand_side;
  right_hand_side.reserve(dimension);
  for (std::size_t index = 0; index < dimension; ++index) {
    triplets.push_back(
        {index, index, field.one()});
    right_hand_side.push_back(field.from_unsigned(index % field.modulus()));
  }

  const SparseMatrix<PrimeField> identity(
      field, dimension, dimension, std::move(triplets));
  CHECK(identity.nonzero_count() == dimension);
  CHECK(identity.rank() == dimension);
  CHECK(identity.right_kernel_basis().empty());

  const auto solution = identity.solve(right_hand_side);
  CHECK(solution.status == LinearSolveStatus::Unique);
  CHECK(solution.particular_solution.has_value());
  CHECK_VECTOR(*solution.particular_solution, right_hand_side);
  CHECK(solution.homogeneous_basis.empty());
}

}  // namespace

int main() {
  try {
    test_canonical_triplets_lookup_and_transpose();
    test_sparse_linear_algebra_and_solves();
    test_empty_shapes();
    test_validation_and_finite_field_cancellation();
    test_sparse_elimination_resource_limits_precede_dense_outputs();
    test_randomized_sparse_dense_differential();
    test_large_native_sparse_elimination();
    std::cout << "sparse matrix tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "sparse matrix test failure: " << error.what() << '\n';
    return 1;
  }
}

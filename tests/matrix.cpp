#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/field.hpp"
#include "laughableengine/matrix.hpp"

namespace {

using laughableengine::DenseMatrix;
using laughableengine::GF;
using laughableengine::LinearSolveStatus;
using laughableengine::PrimeField;
using laughableengine::QQ;
using laughableengine::RationalField;

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
void check_zero_vector(
    const Field& field,
    const std::vector<typename Field::Element>& vector,
    int line) {
  for (const auto& entry : vector) {
    if (!field.is_zero(entry)) {
      fail("vector is not zero", line);
    }
  }
}

#define CHECK_ZERO_VECTOR(field, vector)                                      \
  check_zero_vector((field), (vector), __LINE__)

template <typename Field>
void check_kernel(const DenseMatrix<Field>& matrix, int line) {
  const auto basis = matrix.right_kernel_basis();
  if (basis.size() != matrix.column_count() - matrix.rank()) {
    fail("rank-nullity failed", line);
  }
  for (const auto& vector : basis) {
    if (vector.size() != matrix.column_count()) {
      fail("kernel vector has the wrong dimension", line);
    }
    const auto image = matrix.multiply_column(vector);
    for (const auto& entry : image) {
      if (!matrix.field().is_zero(entry)) {
        fail("reported kernel vector has nonzero image", line);
      }
    }
  }
}

#define CHECK_KERNEL(matrix) check_kernel((matrix), __LINE__)

void test_rational_rref_kernel_image_and_transpose() {
  const auto field = QQ();
  const auto matrix = integer_matrix(
      field,
      3,
      4,
      {1, 2, 3, 4,
       2, 4, 6, 8,
       0, 1, 1, 1});

  const auto reduction = matrix.rref();
  CHECK(reduction.pivot_columns == std::vector<std::size_t>({0, 1}));
  CHECK(reduction.reduced == integer_matrix(
      field,
      3,
      4,
      {1, 0, 1, 2,
       0, 1, 1, 1,
       0, 0, 0, 0}));
  CHECK(matrix.rank() == 2);
  CHECK(matrix(1, 0) == field.from_integer(2));

  const auto kernel = matrix.right_kernel_basis();
  CHECK(kernel.size() == 2);
  CHECK_VECTOR(kernel[0], integer_vector(field, {-1, -1, 1, 0}));
  CHECK_VECTOR(kernel[1], integer_vector(field, {-2, -1, 0, 1}));
  CHECK_KERNEL(matrix);

  const auto image = matrix.column_space_basis();
  CHECK(image.size() == 2);
  CHECK_VECTOR(image[0], integer_vector(field, {1, 2, 0}));
  CHECK_VECTOR(image[1], integer_vector(field, {2, 4, 1}));

  const auto transpose = matrix.transpose();
  CHECK(transpose.row_count() == 4);
  CHECK(transpose.column_count() == 3);
  CHECK(transpose(3, 1) == field.from_integer(8));
  CHECK(transpose.transpose() == matrix);

  std::vector<RationalField::Element> fractional_entries;
  fractional_entries.push_back(field.fraction(1, 2));
  fractional_entries.push_back(field.fraction(1, 3));
  const DenseMatrix<RationalField> fractional(
      field, 1, 2, std::move(fractional_entries));
  const auto fractional_kernel = fractional.right_kernel_basis();
  CHECK(fractional_kernel.size() == 1);
  CHECK(fractional_kernel[0][0] == field.fraction(-2, 3));
  CHECK(fractional_kernel[0][1] == field.one());
  CHECK_KERNEL(fractional);
}

void test_rational_solves() {
  const auto field = QQ();
  const auto invertible = integer_matrix(field, 2, 2, {2, 1, 1, -1});
  const auto unique = invertible.solve(integer_vector(field, {5, 1}));
  CHECK(unique.status == LinearSolveStatus::Unique);
  CHECK(unique.particular_solution.has_value());
  CHECK(unique.homogeneous_basis.empty());
  CHECK_VECTOR(
      *unique.particular_solution, integer_vector(field, {2, 1}));
  CHECK_VECTOR(
      invertible.multiply_column(*unique.particular_solution),
      integer_vector(field, {5, 1}));

  const auto rectangular = integer_matrix(
      field,
      3,
      4,
      {1, 2, 3, 4,
       2, 4, 6, 8,
       0, 1, 1, 1});
  const auto underdetermined =
      rectangular.solve(integer_vector(field, {3, 6, 2}));
  CHECK(underdetermined.status == LinearSolveStatus::Underdetermined);
  CHECK(underdetermined.particular_solution.has_value());
  CHECK_VECTOR(
      *underdetermined.particular_solution,
      integer_vector(field, {-1, 2, 0, 0}));
  CHECK_VECTOR(
      rectangular.multiply_column(*underdetermined.particular_solution),
      integer_vector(field, {3, 6, 2}));
  CHECK(underdetermined.homogeneous_basis.size() == 2);
  for (const auto& vector : underdetermined.homogeneous_basis) {
    CHECK_ZERO_VECTOR(field, rectangular.multiply_column(vector));
  }

  const auto inconsistent =
      rectangular.solve(integer_vector(field, {3, 7, 2}));
  CHECK(inconsistent.status == LinearSolveStatus::Inconsistent);
  CHECK(!inconsistent.particular_solution.has_value());
  CHECK(inconsistent.homogeneous_basis.empty());

  EXPECT_THROW(
      std::invalid_argument,
      rectangular.solve(integer_vector(field, {1, 2})));
}

void test_finite_field_arithmetic_and_validation() {
  const auto field = GF(5);
  const auto rank_one = integer_matrix(
      field,
      2,
      3,
      {1, 2, 3,
       2, 4, 1});
  CHECK(rank_one.rank() == 1);
  const auto reduction = rank_one.rref();
  CHECK(reduction.pivot_columns == std::vector<std::size_t>({0}));
  CHECK(reduction.reduced == integer_matrix(
      field,
      2,
      3,
      {1, 2, 3,
       0, 0, 0}));
  const auto kernel = rank_one.right_kernel_basis();
  CHECK(kernel.size() == 2);
  CHECK_VECTOR(kernel[0], integer_vector(field, {3, 1, 0}));
  CHECK_VECTOR(kernel[1], integer_vector(field, {2, 0, 1}));
  CHECK_KERNEL(rank_one);

  const auto invertible = integer_matrix(field, 2, 2, {1, 2, 3, 4});
  const auto solution = invertible.solve(integer_vector(field, {0, 1}));
  CHECK(solution.status == LinearSolveStatus::Unique);
  CHECK(solution.particular_solution.has_value());
  CHECK_VECTOR(
      *solution.particular_solution, integer_vector(field, {1, 2}));

  const auto foreign = GF(7);
  std::vector<PrimeField::Element> foreign_entry{foreign.one()};
  EXPECT_THROW(
      std::invalid_argument,
      DenseMatrix<PrimeField>(field, 1, 1, foreign_entry));
  EXPECT_THROW(
      std::invalid_argument,
      invertible.multiply_column(foreign_entry));
  EXPECT_THROW(std::invalid_argument, invertible.solve(foreign_entry));

  auto mutable_matrix = DenseMatrix<PrimeField>(field, 1, 1);
  EXPECT_THROW(
      std::invalid_argument, mutable_matrix.set(0, 0, foreign.one()));
}

void test_empty_shapes() {
  const auto field = QQ();
  const DenseMatrix<RationalField> no_equations(field, 0, 3);
  CHECK(no_equations.rank() == 0);
  CHECK(no_equations.rref().reduced.row_count() == 0);
  CHECK(no_equations.rref().reduced.column_count() == 3);
  CHECK(no_equations.column_space_basis().empty());
  CHECK_KERNEL(no_equations);
  CHECK(no_equations.right_kernel_basis().size() == 3);

  const std::vector<RationalField::Element> empty;
  const auto free_solution = no_equations.solve(empty);
  CHECK(free_solution.status == LinearSolveStatus::Underdetermined);
  CHECK(free_solution.particular_solution.has_value());
  CHECK(free_solution.particular_solution->size() == 3);
  CHECK_ZERO_VECTOR(field, *free_solution.particular_solution);
  CHECK(free_solution.homogeneous_basis.size() == 3);

  const auto transpose = no_equations.transpose();
  CHECK(transpose.row_count() == 3);
  CHECK(transpose.column_count() == 0);
  CHECK(transpose.transpose() == no_equations);

  const DenseMatrix<RationalField> no_unknowns(field, 3, 0);
  CHECK(no_unknowns.rank() == 0);
  CHECK(no_unknowns.right_kernel_basis().empty());
  CHECK(no_unknowns.column_space_basis().empty());
  CHECK_ZERO_VECTOR(field, no_unknowns.multiply_column(empty));

  const auto empty_unique =
      no_unknowns.solve(integer_vector(field, {0, 0, 0}));
  CHECK(empty_unique.status == LinearSolveStatus::Unique);
  CHECK(empty_unique.particular_solution.has_value());
  CHECK(empty_unique.particular_solution->empty());
  CHECK(empty_unique.homogeneous_basis.empty());

  const auto empty_inconsistent =
      no_unknowns.solve(integer_vector(field, {0, 1, 0}));
  CHECK(empty_inconsistent.status == LinearSolveStatus::Inconsistent);
  CHECK(!empty_inconsistent.particular_solution.has_value());

  const DenseMatrix<RationalField> empty_matrix(field, 0, 0);
  CHECK(empty_matrix.rank() == 0);
  CHECK(empty_matrix.right_kernel_basis().empty());
  const auto empty_system = empty_matrix.solve(empty);
  CHECK(empty_system.status == LinearSolveStatus::Unique);
  CHECK(empty_system.particular_solution.has_value());
  CHECK(empty_system.particular_solution->empty());
}

void test_bounds_and_shape_errors() {
  const auto field = QQ();
  auto matrix = DenseMatrix<RationalField>(field, 2, 3);
  CHECK(field.is_zero(matrix(1, 2)));
  matrix.set(1, 2, field.one());
  CHECK(matrix(1, 2) == field.one());

  EXPECT_THROW(std::out_of_range, matrix.at(2, 0));
  EXPECT_THROW(std::out_of_range, matrix.at(0, 3));
  EXPECT_THROW(std::out_of_range, matrix.set(9, 9, field.zero()));
  EXPECT_THROW(
      std::invalid_argument,
      DenseMatrix<RationalField>(
          field, 2, 2, integer_vector(field, {1, 2, 3})));
  EXPECT_THROW(
      std::length_error,
      DenseMatrix<RationalField>(
          field, std::numeric_limits<std::size_t>::max(), 2));
  EXPECT_THROW(
      std::invalid_argument,
      matrix.multiply_column(integer_vector(field, {1, 2})));

  const DenseMatrix<RationalField> no_rows(field, 0, 2);
  EXPECT_THROW(std::out_of_range, no_rows.at(0, 0));
}

void test_small_finite_field_properties() {
  const auto field = GF(5);
  std::uint32_t state = 0x8f31a2c7U;
  auto next = [&state]() {
    state = state * 1664525U + 1013904223U;
    return state;
  };

  for (std::size_t rows = 0; rows <= 4; ++rows) {
    for (std::size_t columns = 0; columns <= 5; ++columns) {
      for (std::size_t sample = 0; sample < 20; ++sample) {
        std::vector<PrimeField::Element> entries;
        entries.reserve(rows * columns);
        for (std::size_t index = 0; index < rows * columns; ++index) {
          entries.push_back(field.from_unsigned(next() % 5U));
        }
        const DenseMatrix<PrimeField> matrix(
            field, rows, columns, std::move(entries));
        CHECK_KERNEL(matrix);
        CHECK(matrix.column_space_basis().size() == matrix.rank());

        const auto reduction = matrix.rref();
        CHECK(reduction.reduced.rref().reduced == reduction.reduced);
        for (std::size_t pivot_row = 0;
             pivot_row < reduction.pivot_columns.size(); ++pivot_row) {
          const auto pivot_column = reduction.pivot_columns[pivot_row];
          for (std::size_t row = 0; row < rows; ++row) {
            const auto expected =
                row == pivot_row ? field.one() : field.zero();
            CHECK(reduction.reduced(row, pivot_column) == expected);
          }
        }

        std::vector<PrimeField::Element> witness;
        witness.reserve(columns);
        for (std::size_t column = 0; column < columns; ++column) {
          witness.push_back(field.from_unsigned(next() % 5U));
        }
        const auto right_hand_side = matrix.multiply_column(witness);
        const auto solution = matrix.solve(right_hand_side);
        CHECK(solution.status != LinearSolveStatus::Inconsistent);
        CHECK(solution.particular_solution.has_value());
        CHECK_VECTOR(
            matrix.multiply_column(*solution.particular_solution),
            right_hand_side);
        for (const auto& vector : solution.homogeneous_basis) {
          CHECK_ZERO_VECTOR(field, matrix.multiply_column(vector));
        }
      }
    }
  }
}

}  // namespace

int main() {
  try {
    test_rational_rref_kernel_image_and_transpose();
    test_rational_solves();
    test_finite_field_arithmetic_and_validation();
    test_empty_shapes();
    test_bounds_and_shape_errors();
    test_small_finite_field_properties();
    std::cout << "matrix tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "matrix test failure: " << error.what() << '\n';
    return 1;
  }
}

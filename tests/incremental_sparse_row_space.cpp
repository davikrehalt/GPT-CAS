#include <cstddef>
#include <functional>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/field.hpp"
#include "laughableengine/incremental_sparse_row_space.hpp"
#include "laughableengine/matrix.hpp"

namespace {

using laughableengine::GF;
using laughableengine::IncrementalSparseRowSpace;
using laughableengine::IncrementalSparseRowSpaceOptions;
using laughableengine::IncrementalSparseRowSpaceResourceKind;
using laughableengine::IncrementalSparseRowSpaceResourceLimit;
using laughableengine::DenseMatrix;
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
using Space = IncrementalSparseRowSpace<Field>;

template <typename Field>
typename Space<Field>::SparseVector sparse_vector(
    const Field& field,
    std::initializer_list<std::pair<std::size_t, long>> entries) {
  typename Space<Field>::SparseVector result;
  result.reserve(entries.size());
  for (const auto& [column, value] : entries) {
    result.push_back({column, field.from_integer(value)});
  }
  return result;
}

template <typename SparseVector>
void check_sparse_vector(
    const SparseVector& actual,
    const SparseVector& expected,
    int line) {
  if (actual != expected) {
    fail("sparse vectors differ", line);
  }
}

#define CHECK_SPARSE(field, actual, expected)                                 \
  do {                                                                        \
    static_cast<void>(field);                                                 \
    check_sparse_vector((actual), (expected), __LINE__);                      \
  } while (false)

void test_stable_basis_and_exact_coordinates() {
  const auto field = QQ();
  IncrementalSparseRowSpaceOptions options;
  options.track_insertion_coordinates = true;
  Space<RationalField> space(field, 5, options);

  // Duplicate columns are combined, zeros are discarded, and storage is
  // sorted: this accepted input is 2e_0 + 3e_2.
  const auto first = space.insert(sparse_vector(
      field, {{2, 1}, {0, 2}, {2, 2}, {4, 0}}));
  CHECK(first.inserted);
  CHECK(first.insertion_index == std::optional<std::size_t>(0));
  CHECK(first.pivot_column == std::optional<std::size_t>(0));
  CHECK(!first.dependence_coordinates.has_value());
  CHECK_SPARSE(
      field, space.basis_vector(0), sparse_vector(field, {{0, 2}, {2, 3}}));
  CHECK(space.pivot_column(0) == 0);

  const auto second = space.insert(
      sparse_vector(field, {{1, 1}, {2, 1}}));
  CHECK(second.inserted);
  CHECK(second.insertion_index == std::optional<std::size_t>(1));
  CHECK(second.pivot_column == std::optional<std::size_t>(1));

  // 2*basis(0) - 3*basis(1).
  const auto dependent = space.insert(
      sparse_vector(field, {{0, 4}, {1, -3}, {2, 3}}));
  CHECK(!dependent.inserted);
  CHECK(!dependent.insertion_index.has_value());
  CHECK(!dependent.pivot_column.has_value());
  CHECK(dependent.dependence_coordinates.has_value());
  CHECK_SPARSE(
      field,
      *dependent.dependence_coordinates,
      sparse_vector(field, {{0, 2}, {1, -3}}));

  const auto reduced = space.normal_form_with_coordinates(
      sparse_vector(field, {{4, 5}, {2, 3}, {1, -3}, {0, 4}}));
  CHECK_SPARSE(
      field, reduced.normal_form, sparse_vector(field, {{4, 5}}));
  CHECK_SPARSE(
      field,
      reduced.insertion_coordinates,
      sparse_vector(field, {{0, 2}, {1, -3}}));

  CHECK(space.rank() == 2);
  CHECK(space.pivot_columns() == std::vector<std::size_t>({0, 1}));
  CHECK(space.nonpivot_columns() == std::vector<std::size_t>({2, 3, 4}));
  CHECK(space.basis_nonzero_count() == 4);
  CHECK(space.pivot_row_nonzero_count() == 4);
  CHECK(space.coordinate_nonzero_count() == 2);
  CHECK(space.live_nonzero_count() == 10);
}

void test_pivots_are_deterministic_not_insertion_sorted() {
  const auto field = GF(7);
  Space<PrimeField> space(field, 6);

  const auto late = space.insert(sparse_vector(field, {{5, 1}}));
  CHECK(late.inserted && late.insertion_index == 0 && late.pivot_column == 5);

  // e_2 + e_5 reduces by the already-stored e_5 row and gets pivot 2.
  const auto early = space.insert(sparse_vector(field, {{2, 1}, {5, 1}}));
  CHECK(early.inserted && early.insertion_index == 1 && early.pivot_column == 2);
  CHECK(space.pivot_columns() == std::vector<std::size_t>({2, 5}));
  CHECK(space.nonpivot_columns() ==
        std::vector<std::size_t>({0, 1, 3, 4}));
  CHECK_SPARSE(
      field, space.basis_vector(1), sparse_vector(field, {{2, 1}, {5, 1}}));
  CHECK_SPARSE(
      field, space.normal_form(sparse_vector(field, {{2, 3}, {5, 3}})),
      sparse_vector(field, {}));
}

void test_readonly_reductions_do_not_change_lifetime_meter() {
  const auto field = QQ();
  IncrementalSparseRowSpaceOptions options;
  options.track_insertion_coordinates = true;
  Space<RationalField> space(field, 3, options);
  static_cast<void>(space.insert(sparse_vector(field, {{0, 2}, {1, 1}})));

  const auto before = space.arithmetic_operation_count();
  const Space<RationalField>& frozen = space;
  const auto reduced = frozen.normal_form_with_coordinates_readonly(
      sparse_vector(field, {{0, 4}, {1, 2}, {2, 7}}));
  CHECK_SPARSE(
      field, reduced.normal_form, sparse_vector(field, {{2, 7}}));
  CHECK_SPARSE(
      field, reduced.insertion_coordinates, sparse_vector(field, {{0, 2}}));
  CHECK(space.arithmetic_operation_count() == before);
  CHECK_SPARSE(
      field,
      frozen.normal_form_readonly(sparse_vector(field, {{0, 4}, {1, 2}})),
      sparse_vector(field, {}));
  CHECK(space.arithmetic_operation_count() == before);
}

void test_validation_and_tracking_is_explicit() {
  const auto field = GF(7);
  Space<PrimeField> space(field, 2);
  static_cast<void>(space.insert(sparse_vector(field, {{0, 1}})));

  const auto dependent = space.insert(sparse_vector(field, {{0, 3}}));
  CHECK(!dependent.inserted);
  CHECK(!dependent.dependence_coordinates.has_value());
  EXPECT_THROW(
      std::logic_error,
      space.normal_form_with_coordinates(sparse_vector(field, {{0, 1}})));
  EXPECT_THROW(
      std::logic_error,
      static_cast<const Space<PrimeField>&>(space)
          .normal_form_with_coordinates_readonly(
              sparse_vector(field, {{0, 1}})));

  EXPECT_THROW(
      std::out_of_range,
      space.normal_form(sparse_vector(field, {{2, 1}})));
  EXPECT_THROW(std::out_of_range, space.basis_vector(1));
  CHECK(space.rank() == 1);

  const auto foreign = GF(5);
  Space<PrimeField>::SparseVector wrong_field{{1, foreign.one()}};
  EXPECT_THROW(std::invalid_argument, space.normal_form(wrong_field));

  Space<PrimeField> zero_dimensional(field, 0);
  const auto zero = zero_dimensional.insert({});
  CHECK(!zero.inserted);
  CHECK(zero_dimensional.rank() == 0);
  EXPECT_THROW(
      std::out_of_range,
      zero_dimensional.insert(sparse_vector(field, {{0, 1}})));
}

std::vector<PrimeField::Element> dense_coordinates(
    const PrimeField& field,
    std::size_t dimension,
    const Space<PrimeField>::SparseVector& sparse) {
  std::vector<PrimeField::Element> result(dimension, field.zero());
  for (const auto& entry : sparse) {
    result[entry.column] = field.add(result[entry.column], entry.value);
  }
  return result;
}

std::size_t dense_row_rank(
    const PrimeField& field,
    std::size_t dimension,
    const std::vector<Space<PrimeField>::SparseVector>& rows) {
  std::vector<PrimeField::Element> entries;
  entries.reserve(rows.size() * dimension);
  for (const auto& row : rows) {
    auto dense = dense_coordinates(field, dimension, row);
    entries.insert(entries.end(), dense.begin(), dense.end());
  }
  return DenseMatrix<PrimeField>(
             field, rows.size(), dimension, std::move(entries))
      .rank();
}

void test_randomized_dense_rank_and_coordinate_differential() {
  const auto field = GF(7);
  constexpr std::size_t dimension = 8;
  IncrementalSparseRowSpaceOptions options;
  options.track_insertion_coordinates = true;
  Space<PrimeField> space(field, dimension, options);
  std::vector<Space<PrimeField>::SparseVector> attempted_rows;
  std::mt19937 random(0x51A7U);

  for (std::size_t trial = 0; trial < 30; ++trial) {
    Space<PrimeField>::SparseVector candidate;
    for (std::size_t column = 0; column < dimension; ++column) {
      if ((random() % 3U) == 0U) {
        candidate.push_back(
            {column, field.from_unsigned(random() % field.modulus())});
      }
    }
    if (!candidate.empty() && (random() % 2U) == 0U) {
      const auto copied = candidate[random() % candidate.size()];
      candidate.push_back(copied);
    }
    std::shuffle(candidate.begin(), candidate.end(), random);

    const auto decomposition =
        static_cast<const Space<PrimeField>&>(space)
            .normal_form_with_coordinates_readonly(candidate);
    auto reconstructed =
        dense_coordinates(field, dimension, decomposition.normal_form);
    for (const auto& coordinate : decomposition.insertion_coordinates) {
      for (const auto& basis_entry : space.basis_vector(coordinate.column)) {
        reconstructed[basis_entry.column] = field.add(
            reconstructed[basis_entry.column],
            field.multiply(coordinate.value, basis_entry.value));
      }
    }
    CHECK(reconstructed == dense_coordinates(field, dimension, candidate));

    const auto old_rank = space.rank();
    attempted_rows.push_back(candidate);
    const auto expected_rank =
        dense_row_rank(field, dimension, attempted_rows);
    const auto inserted = space.insert(std::move(candidate));
    CHECK(inserted.inserted == (expected_rank != old_rank));
    CHECK(space.rank() == expected_rank);
    if (inserted.inserted) {
      CHECK(inserted.insertion_index == old_rank);
    }
  }
}

void test_resource_guards_leave_the_space_unchanged() {
  const auto field = GF(7);

  IncrementalSparseRowSpaceOptions vector_options;
  vector_options.limits.max_basis_vectors = 0;
  Space<PrimeField> no_vectors(field, 2, vector_options);
  const auto vector_error = EXPECT_THROW(
      IncrementalSparseRowSpaceResourceLimit,
      no_vectors.insert(sparse_vector(field, {{0, 1}})));
  CHECK(vector_error.kind() ==
        IncrementalSparseRowSpaceResourceKind::BasisVectors);
  CHECK(vector_error.limit() == 0);
  CHECK(vector_error.observed() == 1);
  CHECK(no_vectors.rank() == 0);
  CHECK(no_vectors.arithmetic_operation_count() == 0);

  IncrementalSparseRowSpaceOptions live_options;
  live_options.limits.max_live_nonzeros = 1;
  Space<PrimeField> little_memory(field, 3, live_options);
  const auto live_error = EXPECT_THROW(
      IncrementalSparseRowSpaceResourceLimit,
      little_memory.insert(sparse_vector(field, {{0, 1}, {1, 1}})));
  CHECK(live_error.kind() ==
        IncrementalSparseRowSpaceResourceKind::LiveNonzeros);
  CHECK(live_error.limit() == 1);
  CHECK(live_error.observed() == 2);
  CHECK(little_memory.rank() == 0);

  IncrementalSparseRowSpaceOptions operation_options;
  operation_options.limits.max_arithmetic_operations = 0;
  Space<PrimeField> no_arithmetic(field, 2, operation_options);
  const auto operation_error = EXPECT_THROW(
      IncrementalSparseRowSpaceResourceLimit,
      no_arithmetic.insert(sparse_vector(field, {{0, 2}})));
  CHECK(operation_error.kind() ==
        IncrementalSparseRowSpaceResourceKind::ArithmeticOperations);
  CHECK(operation_error.limit() == 0);
  CHECK(operation_error.observed() == 1);
  CHECK(no_arithmetic.rank() == 0);
  CHECK(no_arithmetic.arithmetic_operation_count() == 0);

  // Read-only calls use the arithmetic budget per call and never consume the
  // lifetime meter, including when the per-call guard fires.
  IncrementalSparseRowSpaceOptions readonly_options;
  readonly_options.limits.max_arithmetic_operations = 2;
  Space<PrimeField> readonly_space(field, 2, readonly_options);
  static_cast<void>(readonly_space.insert(sparse_vector(field, {{0, 1}})));
  const auto before = readonly_space.arithmetic_operation_count();
  const auto readonly_error = EXPECT_THROW(
      IncrementalSparseRowSpaceResourceLimit,
      static_cast<const Space<PrimeField>&>(readonly_space)
          .normal_form_readonly(sparse_vector(field, {{0, 1}, {0, 1}})));
  CHECK(readonly_error.kind() ==
        IncrementalSparseRowSpaceResourceKind::ArithmeticOperations);
  CHECK(readonly_space.arithmetic_operation_count() == before);
}

}  // namespace

int main() {
  try {
    test_stable_basis_and_exact_coordinates();
    test_pivots_are_deterministic_not_insertion_sorted();
    test_readonly_reductions_do_not_change_lifetime_meter();
    test_validation_and_tracking_is_explicit();
    test_randomized_dense_rank_and_coordinate_differential();
    test_resource_guards_leave_the_space_unchanged();
    std::cout << "incremental sparse row-space tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "incremental sparse row-space test failure: "
              << error.what() << '\n';
    return 1;
  }
}

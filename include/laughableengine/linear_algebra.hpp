#pragma once

#include <cstddef>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "laughableengine/matrix.hpp"

namespace laughableengine {

namespace linear_algebra_detail {

inline std::size_t checked_sum(
    std::size_t left,
    std::size_t right,
    const char* message) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::length_error(message);
  }
  return left + right;
}

template <typename Field>
void require_same_field(
    const Field& left,
    const Field& right,
    const char* operation) {
  if (!(left == right)) {
    throw std::invalid_argument(operation);
  }
}

}  // namespace linear_algebra_detail

// Build a matrix whose supplied vectors are columns. row_count is explicit so
// that an empty column list still denotes a well-defined row_count-by-zero
// matrix.
template <typename Field>
[[nodiscard]] DenseMatrix<Field> matrix_from_columns(
    Field field,
    std::size_t row_count,
    std::span<const std::vector<typename Field::Element>> columns) {
  using Element = typename Field::Element;

  for (const auto& column : columns) {
    if (column.size() != row_count) {
      throw std::invalid_argument(
          "matrix column length does not match the requested row count");
    }
  }

  if (columns.size() != 0 &&
      row_count > std::numeric_limits<std::size_t>::max() / columns.size()) {
    throw std::length_error("matrix dimensions overflow size_t");
  }
  std::vector<Element> entries;
  entries.reserve(row_count * columns.size());
  for (std::size_t row = 0; row < row_count; ++row) {
    for (const auto& column : columns) {
      entries.push_back(column[row]);
    }
  }
  return DenseMatrix<Field>(
      std::move(field), row_count, columns.size(), std::move(entries));
}

template <typename Field>
[[nodiscard]] DenseMatrix<Field> matrix_from_columns(
    Field field,
    std::size_t row_count,
    const std::vector<std::vector<typename Field::Element>>& columns) {
  return matrix_from_columns(
      std::move(field), row_count,
      std::span<const std::vector<typename Field::Element>>(
          columns.data(), columns.size()));
}

template <typename Field>
[[nodiscard]] DenseMatrix<Field> matrix_from_columns(
    Field field,
    std::size_t row_count,
    std::initializer_list<std::vector<typename Field::Element>> columns) {
  return matrix_from_columns(
      std::move(field), row_count,
      std::span<const std::vector<typename Field::Element>>(
          columns.begin(), columns.size()));
}

// Matrix multiplication with the column-vector convention:
//
//             left : k^m -> k^n, right : k^r -> k^m.
template <typename Field>
[[nodiscard]] DenseMatrix<Field> multiply(
    const DenseMatrix<Field>& left,
    const DenseMatrix<Field>& right) {
  linear_algebra_detail::require_same_field(
      left.field(), right.field(),
      "matrix multiplication requires one coefficient field");
  if (left.column_count() != right.row_count()) {
    throw std::invalid_argument(
        "matrix multiplication has incompatible dimensions");
  }

  Field field = left.field();
  const auto rows = left.row_count();
  const auto inner = left.column_count();
  const auto columns = right.column_count();
  if (columns != 0 &&
      rows > std::numeric_limits<std::size_t>::max() / columns) {
    throw std::length_error("matrix product dimensions overflow size_t");
  }
  std::vector<typename Field::Element> entries(
      rows * columns, field.zero());
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t index = 0; index < inner; ++index) {
      const auto left_entry = left(row, index);
      if (field.is_zero(left_entry)) {
        continue;
      }
      for (std::size_t column = 0; column < columns; ++column) {
        auto& entry = entries[row * columns + column];
        entry = field.add(
            entry,
            field.multiply(left_entry, right(index, column)));
      }
    }
  }
  return DenseMatrix<Field>(
      std::move(field), rows, columns, std::move(entries));
}

// The explicit field and column count preserve the shape of an empty stack.
template <typename Field>
[[nodiscard]] DenseMatrix<Field> vertical_stack(
    Field field,
    std::size_t column_count,
    std::span<const DenseMatrix<Field>> blocks) {
  std::size_t row_count = 0;
  for (const auto& block : blocks) {
    linear_algebra_detail::require_same_field(
        field, block.field(),
        "vertical stacking requires one coefficient field");
    if (block.column_count() != column_count) {
      throw std::invalid_argument(
          "vertical stack blocks must have equal column counts");
    }
    row_count = linear_algebra_detail::checked_sum(
        row_count, block.row_count(),
        "vertical stack row count overflows size_t");
  }

  if (column_count != 0 &&
      row_count > std::numeric_limits<std::size_t>::max() / column_count) {
    throw std::length_error("vertical stack dimensions overflow size_t");
  }
  std::vector<typename Field::Element> entries;
  entries.reserve(row_count * column_count);
  for (const auto& block : blocks) {
    for (std::size_t row = 0; row < block.row_count(); ++row) {
      for (std::size_t column = 0; column < column_count; ++column) {
        entries.push_back(block(row, column));
      }
    }
  }
  return DenseMatrix<Field>(
      std::move(field), row_count, column_count, std::move(entries));
}

template <typename Field>
[[nodiscard]] DenseMatrix<Field> vertical_stack(
    Field field,
    std::size_t column_count,
    const std::vector<DenseMatrix<Field>>& blocks) {
  return vertical_stack(
      std::move(field), column_count,
      std::span<const DenseMatrix<Field>>(blocks.data(), blocks.size()));
}

template <typename Field>
[[nodiscard]] DenseMatrix<Field> vertical_stack(
    std::span<const DenseMatrix<Field>> blocks) {
  if (blocks.empty()) {
    throw std::invalid_argument(
        "an empty vertical stack requires an explicit field and column count");
  }
  return vertical_stack(
      blocks.front().field(), blocks.front().column_count(), blocks);
}

template <typename Field>
[[nodiscard]] DenseMatrix<Field> vertical_stack(
    const std::vector<DenseMatrix<Field>>& blocks) {
  return vertical_stack(
      std::span<const DenseMatrix<Field>>(blocks.data(), blocks.size()));
}

template <typename Field>
[[nodiscard]] DenseMatrix<Field> vertical_stack(
    std::initializer_list<DenseMatrix<Field>> blocks) {
  return vertical_stack(
      std::span<const DenseMatrix<Field>>(blocks.begin(), blocks.size()));
}

// The explicit field and row count preserve the shape of an empty stack.
template <typename Field>
[[nodiscard]] DenseMatrix<Field> horizontal_stack(
    Field field,
    std::size_t row_count,
    std::span<const DenseMatrix<Field>> blocks) {
  std::size_t column_count = 0;
  for (const auto& block : blocks) {
    linear_algebra_detail::require_same_field(
        field, block.field(),
        "horizontal stacking requires one coefficient field");
    if (block.row_count() != row_count) {
      throw std::invalid_argument(
          "horizontal stack blocks must have equal row counts");
    }
    column_count = linear_algebra_detail::checked_sum(
        column_count, block.column_count(),
        "horizontal stack column count overflows size_t");
  }

  if (column_count != 0 &&
      row_count > std::numeric_limits<std::size_t>::max() / column_count) {
    throw std::length_error("horizontal stack dimensions overflow size_t");
  }
  std::vector<typename Field::Element> entries;
  entries.reserve(row_count * column_count);
  for (std::size_t row = 0; row < row_count; ++row) {
    for (const auto& block : blocks) {
      for (std::size_t column = 0; column < block.column_count(); ++column) {
        entries.push_back(block(row, column));
      }
    }
  }
  return DenseMatrix<Field>(
      std::move(field), row_count, column_count, std::move(entries));
}

template <typename Field>
[[nodiscard]] DenseMatrix<Field> horizontal_stack(
    Field field,
    std::size_t row_count,
    const std::vector<DenseMatrix<Field>>& blocks) {
  return horizontal_stack(
      std::move(field), row_count,
      std::span<const DenseMatrix<Field>>(blocks.data(), blocks.size()));
}

template <typename Field>
[[nodiscard]] DenseMatrix<Field> horizontal_stack(
    std::span<const DenseMatrix<Field>> blocks) {
  if (blocks.empty()) {
    throw std::invalid_argument(
        "an empty horizontal stack requires an explicit field and row count");
  }
  return horizontal_stack(
      blocks.front().field(), blocks.front().row_count(), blocks);
}

template <typename Field>
[[nodiscard]] DenseMatrix<Field> horizontal_stack(
    const std::vector<DenseMatrix<Field>>& blocks) {
  return horizontal_stack(
      std::span<const DenseMatrix<Field>>(blocks.data(), blocks.size()));
}

template <typename Field>
[[nodiscard]] DenseMatrix<Field> horizontal_stack(
    std::initializer_list<DenseMatrix<Field>> blocks) {
  return horizontal_stack(
      std::span<const DenseMatrix<Field>>(blocks.begin(), blocks.size()));
}

// An immutable linear subspace of k^ambient_dimension. Its basis is stored as
// independent columns in a canonical reduced form, so equal subspaces over the
// same field have equal basis matrices. In particular, the zero subspace is an
// ambient_dimension-by-zero matrix rather than a dimensionless empty list.
template <typename Field>
class LinearSubspace {
 public:
  using Element = typename Field::Element;
  using Vector = std::vector<Element>;

  [[nodiscard]] static LinearSubspace zero(
      Field field,
      std::size_t ambient_dimension) {
    return LinearSubspace(
        DenseMatrix<Field>(std::move(field), ambient_dimension, 0));
  }

  [[nodiscard]] static LinearSubspace whole(
      Field field,
      std::size_t ambient_dimension) {
    DenseMatrix<Field> identity(
        field, ambient_dimension, ambient_dimension);
    for (std::size_t index = 0; index < ambient_dimension; ++index) {
      identity.set(index, index, field.one());
    }
    return LinearSubspace(std::move(identity));
  }

  [[nodiscard]] static LinearSubspace span(
      Field field,
      std::size_t ambient_dimension,
      std::span<const Vector> generators) {
    return LinearSubspace(canonical_column_basis(matrix_from_columns(
        std::move(field), ambient_dimension, generators)));
  }

  [[nodiscard]] static LinearSubspace span(
      Field field,
      std::size_t ambient_dimension,
      const std::vector<Vector>& generators) {
    return span(
        std::move(field), ambient_dimension,
        std::span<const Vector>(generators.data(), generators.size()));
  }

  [[nodiscard]] static LinearSubspace span(
      Field field,
      std::size_t ambient_dimension,
      std::initializer_list<Vector> generators) {
    return span(
        std::move(field), ambient_dimension,
        std::span<const Vector>(generators.begin(), generators.size()));
  }

  [[nodiscard]] static LinearSubspace from_independent_columns(
      DenseMatrix<Field> basis) {
    if (basis.rank() != basis.column_count()) {
      throw std::invalid_argument(
          "a subspace basis must have independent columns");
    }
    return LinearSubspace(canonical_column_basis(std::move(basis)));
  }

  [[nodiscard]] static LinearSubspace kernel(
      const DenseMatrix<Field>& matrix) {
    const auto generators = matrix.right_kernel_basis();
    return span(matrix.field(), matrix.column_count(), generators);
  }

  [[nodiscard]] static LinearSubspace image(
      const DenseMatrix<Field>& matrix) {
    const auto generators = matrix.column_space_basis();
    return span(matrix.field(), matrix.row_count(), generators);
  }

  [[nodiscard]] static LinearSubspace intersection(
      const LinearSubspace& left,
      const LinearSubspace& right) {
    linear_algebra_detail::require_same_field(
        left.field(), right.field(),
        "subspace intersection requires one coefficient field");
    if (left.ambient_dimension() != right.ambient_dimension()) {
      throw std::invalid_argument(
          "subspace intersection requires equal ambient dimensions");
    }

    std::vector<Vector> coupled_columns;
    coupled_columns.reserve(left.dimension() + right.dimension());
    for (const auto& column : left.basis_vectors()) {
      coupled_columns.push_back(column);
    }
    for (auto column : right.basis_vectors()) {
      for (auto& entry : column) {
        entry = left.field().negate(entry);
      }
      coupled_columns.push_back(std::move(column));
    }

    const auto coupled = matrix_from_columns(
        left.field(), left.ambient_dimension(), coupled_columns);
    const auto relations = coupled.right_kernel_basis();
    std::vector<Vector> intersection_generators;
    intersection_generators.reserve(relations.size());
    for (const auto& relation : relations) {
      const std::span<const Element> left_coordinates(
          relation.data(), left.dimension());
      intersection_generators.push_back(
          left.from_coordinates(left_coordinates));
    }
    return span(
        left.field(), left.ambient_dimension(), intersection_generators);
  }

  [[nodiscard]] const Field& field() const noexcept { return basis_.field(); }
  [[nodiscard]] std::size_t ambient_dimension() const noexcept {
    return basis_.row_count();
  }
  [[nodiscard]] std::size_t dimension() const noexcept {
    return basis_.column_count();
  }
  [[nodiscard]] const DenseMatrix<Field>& basis_matrix() const noexcept {
    return basis_;
  }

  [[nodiscard]] std::vector<Vector> basis_vectors() const {
    std::vector<Vector> result;
    result.reserve(dimension());
    for (std::size_t column = 0; column < dimension(); ++column) {
      Vector vector;
      vector.reserve(ambient_dimension());
      for (std::size_t row = 0; row < ambient_dimension(); ++row) {
        vector.push_back(basis_(row, column));
      }
      result.push_back(std::move(vector));
    }
    return result;
  }

  [[nodiscard]] bool contains(std::span<const Element> vector) const {
    if (vector.size() != ambient_dimension()) {
      throw std::invalid_argument(
          "subspace membership vector has the wrong ambient dimension");
    }
    return basis_.solve(vector).status != LinearSolveStatus::Inconsistent;
  }

  [[nodiscard]] bool contains(const Vector& vector) const {
    return contains(std::span<const Element>(vector.data(), vector.size()));
  }

  [[nodiscard]] Vector coordinates(
      std::span<const Element> vector) const {
    if (vector.size() != ambient_dimension()) {
      throw std::invalid_argument(
          "subspace coordinate vector has the wrong ambient dimension");
    }
    auto solution = basis_.solve(vector);
    if (solution.status == LinearSolveStatus::Inconsistent) {
      throw std::domain_error("vector does not belong to the subspace");
    }
    if (solution.status != LinearSolveStatus::Unique ||
        !solution.particular_solution.has_value()) {
      throw std::logic_error(
          "internal error: subspace basis coordinates are not unique");
    }
    return std::move(*solution.particular_solution);
  }

  [[nodiscard]] Vector coordinates(const Vector& vector) const {
    return coordinates(
        std::span<const Element>(vector.data(), vector.size()));
  }

  // Lift a coordinate vector to the ambient vector space. This operation is
  // deliberately polynomial-free; quotient layers can separately turn the
  // resulting ambient coordinates into polynomial representatives.
  [[nodiscard]] Vector from_coordinates(
      std::span<const Element> coordinates) const {
    if (coordinates.size() != dimension()) {
      throw std::invalid_argument(
          "subspace coordinate vector has the wrong dimension");
    }
    return basis_.multiply_column(coordinates);
  }

  [[nodiscard]] Vector from_coordinates(const Vector& coordinates) const {
    return from_coordinates(
        std::span<const Element>(coordinates.data(), coordinates.size()));
  }

  friend bool operator==(
      const LinearSubspace&,
      const LinearSubspace&) = default;

 private:
  explicit LinearSubspace(DenseMatrix<Field> canonical_basis)
      : basis_(std::move(canonical_basis)) {}

  [[nodiscard]] static DenseMatrix<Field> canonical_column_basis(
      DenseMatrix<Field> generators) {
    const auto ambient_dimension = generators.row_count();
    const auto row_reduction = generators.transpose().rref();
    std::vector<Vector> canonical_columns;
    canonical_columns.reserve(row_reduction.pivot_columns.size());
    for (std::size_t row = 0;
         row < row_reduction.pivot_columns.size(); ++row) {
      Vector column;
      column.reserve(ambient_dimension);
      for (std::size_t ambient = 0; ambient < ambient_dimension; ++ambient) {
        column.push_back(row_reduction.reduced(row, ambient));
      }
      canonical_columns.push_back(std::move(column));
    }
    return matrix_from_columns(
        generators.field(), ambient_dimension, canonical_columns);
  }

  DenseMatrix<Field> basis_;
};

}  // namespace laughableengine

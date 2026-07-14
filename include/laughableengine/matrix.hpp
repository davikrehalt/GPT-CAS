#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "laughableengine/field.hpp"

namespace laughableengine {

template <typename Field>
class DenseMatrix;

template <typename Field>
struct RrefResult;

enum class LinearSolveStatus { Inconsistent, Unique, Underdetermined };

template <typename Field>
struct LinearSolveResult;

// A small exact dense matrix. Entries are stored row-major, while every
// linear-algebra operation uses the conventional column-vector action
//
//                         A : k^columns -> k^rows.
//
// In particular, every vector returned by right_kernel_basis() has length
// column_count() and satisfies A*v = 0.
template <typename Field>
class DenseMatrix {
 public:
  using Element = typename Field::Element;

  DenseMatrix(Field field, std::size_t rows, std::size_t columns)
      : field_(std::move(field)),
        rows_(rows),
        columns_(columns),
        entries_(checked_entry_count(rows, columns), field_.zero()) {}

  DenseMatrix(
      Field field,
      std::size_t rows,
      std::size_t columns,
      std::vector<Element> row_major_entries)
      : field_(std::move(field)),
        rows_(rows),
        columns_(columns),
        entries_(std::move(row_major_entries)) {
    const auto expected = checked_entry_count(rows_, columns_);
    if (entries_.size() != expected) {
      throw std::invalid_argument(
          "dense matrix entry count does not match its shape");
    }
    for (auto& entry : entries_) {
      entry = field_.canonical(entry);
    }
  }

  [[nodiscard]] const Field& field() const noexcept { return field_; }
  [[nodiscard]] std::size_t row_count() const noexcept { return rows_; }
  [[nodiscard]] std::size_t column_count() const noexcept { return columns_; }

  [[nodiscard]] const Element& at(
      std::size_t row,
      std::size_t column) const {
    return entries_[checked_index(row, column)];
  }

  [[nodiscard]] const Element& operator()(
      std::size_t row,
      std::size_t column) const {
    return at(row, column);
  }

  void set(std::size_t row, std::size_t column, Element value) {
    entries_[checked_index(row, column)] =
        field_.canonical(std::move(value));
  }

  [[nodiscard]] DenseMatrix transpose() const {
    DenseMatrix result(field_, columns_, rows_);
    for (std::size_t row = 0; row < rows_; ++row) {
      for (std::size_t column = 0; column < columns_; ++column) {
        result.entries_[column * rows_ + row] =
            entries_[row * columns_ + column];
      }
    }
    return result;
  }

  // Return A*v. The input is interpreted as a column vector and must have
  // exactly column_count() entries.
  [[nodiscard]] std::vector<Element> multiply_column(
      std::span<const Element> vector) const {
    if (vector.size() != columns_) {
      throw std::invalid_argument(
          "matrix-vector multiplication has incompatible dimensions");
    }

    std::vector<Element> canonical;
    canonical.reserve(vector.size());
    for (const auto& entry : vector) {
      canonical.push_back(field_.canonical(entry));
    }

    std::vector<Element> result(rows_, field_.zero());
    for (std::size_t row = 0; row < rows_; ++row) {
      auto value = field_.zero();
      for (std::size_t column = 0; column < columns_; ++column) {
        value = field_.add(
            value,
            field_.multiply(
                entries_[row * columns_ + column], canonical[column]));
      }
      result[row] = std::move(value);
    }
    return result;
  }

  [[nodiscard]] RrefResult<Field> rref() const;
  [[nodiscard]] std::size_t rank() const;

  // Return a basis of ker(A). Each basis element is a column vector encoded
  // as a vector of length column_count(). The basis is ordered by increasing
  // free-column index.
  [[nodiscard]] std::vector<std::vector<Element>>
  right_kernel_basis() const;

  // Return original columns of A whose indices are pivot columns. They form
  // a basis of im(A) in k^row_count().
  [[nodiscard]] std::vector<std::vector<Element>>
  column_space_basis() const;

  // Solve A*x=b. For a consistent system, particular_solution is present and
  // every solution is
  //
  //     particular_solution + sum(c_i * homogeneous_basis[i]).
  //
  // The homogeneous basis is empty exactly when the solution is unique.
  [[nodiscard]] LinearSolveResult<Field> solve(
      std::span<const Element> right_hand_side) const;

  friend bool operator==(const DenseMatrix&, const DenseMatrix&) = default;

 private:
  [[nodiscard]] static std::size_t checked_entry_count(
      std::size_t rows,
      std::size_t columns) {
    if (columns != 0 &&
        rows > std::numeric_limits<std::size_t>::max() / columns) {
      throw std::length_error("dense matrix dimensions overflow size_t");
    }
    return rows * columns;
  }

  [[nodiscard]] std::size_t checked_index(
      std::size_t row,
      std::size_t column) const {
    if (row >= rows_ || column >= columns_) {
      throw std::out_of_range("dense matrix index is out of range");
    }
    return row * columns_ + column;
  }

  [[nodiscard]] RrefResult<Field> row_reduce(
      std::size_t pivot_column_limit) const;

  Field field_;
  std::size_t rows_;
  std::size_t columns_;
  std::vector<Element> entries_;
};

template <typename Field>
struct RrefResult {
  DenseMatrix<Field> reduced;
  std::vector<std::size_t> pivot_columns;
};

template <typename Field>
struct LinearSolveResult {
  using Element = typename Field::Element;

  LinearSolveStatus status;
  std::optional<std::vector<Element>> particular_solution;
  std::vector<std::vector<Element>> homogeneous_basis;
};

template <typename Field>
RrefResult<Field> DenseMatrix<Field>::row_reduce(
    std::size_t pivot_column_limit) const {
  if (pivot_column_limit > columns_) {
    throw std::logic_error("RREF pivot limit exceeds the matrix width");
  }

  auto reduced = *this;
  std::vector<std::size_t> pivot_columns;
  pivot_columns.reserve(std::min(rows_, pivot_column_limit));

  std::size_t pivot_row = 0;
  for (std::size_t column = 0;
       column < pivot_column_limit && pivot_row < rows_;
       ++column) {
    std::size_t selected = pivot_row;
    while (selected < rows_ &&
           field_.is_zero(
               reduced.entries_[selected * columns_ + column])) {
      ++selected;
    }
    if (selected == rows_) {
      continue;
    }

    if (selected != pivot_row) {
      for (std::size_t entry_column = 0; entry_column < columns_;
           ++entry_column) {
        std::swap(
            reduced.entries_[pivot_row * columns_ + entry_column],
            reduced.entries_[selected * columns_ + entry_column]);
      }
    }

    const auto inverse = field_.inverse(
        reduced.entries_[pivot_row * columns_ + column]);
    for (std::size_t entry_column = 0; entry_column < columns_;
         ++entry_column) {
      auto& entry =
          reduced.entries_[pivot_row * columns_ + entry_column];
      entry = field_.multiply(entry, inverse);
    }

    for (std::size_t row = 0; row < rows_; ++row) {
      if (row == pivot_row) {
        continue;
      }
      const auto factor = reduced.entries_[row * columns_ + column];
      if (field_.is_zero(factor)) {
        continue;
      }
      for (std::size_t entry_column = 0; entry_column < columns_;
           ++entry_column) {
        auto& entry = reduced.entries_[row * columns_ + entry_column];
        entry = field_.subtract(
            entry,
            field_.multiply(
                factor,
                reduced.entries_[pivot_row * columns_ + entry_column]));
      }
    }

    pivot_columns.push_back(column);
    ++pivot_row;
  }

  return RrefResult<Field>{
      std::move(reduced), std::move(pivot_columns)};
}

template <typename Field>
RrefResult<Field> DenseMatrix<Field>::rref() const {
  return row_reduce(columns_);
}

template <typename Field>
std::size_t DenseMatrix<Field>::rank() const {
  return rref().pivot_columns.size();
}

template <typename Field>
std::vector<std::vector<typename DenseMatrix<Field>::Element>>
DenseMatrix<Field>::right_kernel_basis() const {
  auto reduction = rref();
  std::vector<bool> is_pivot(columns_, false);
  for (const auto column : reduction.pivot_columns) {
    is_pivot[column] = true;
  }

  std::vector<std::vector<Element>> basis;
  basis.reserve(columns_ - reduction.pivot_columns.size());
  for (std::size_t free_column = 0; free_column < columns_;
       ++free_column) {
    if (is_pivot[free_column]) {
      continue;
    }

    std::vector<Element> vector(columns_, field_.zero());
    vector[free_column] = field_.one();
    for (std::size_t pivot_row = 0;
         pivot_row < reduction.pivot_columns.size(); ++pivot_row) {
      vector[reduction.pivot_columns[pivot_row]] = field_.negate(
          reduction.reduced.entries_[pivot_row * columns_ + free_column]);
    }
    basis.push_back(std::move(vector));
  }
  return basis;
}

template <typename Field>
std::vector<std::vector<typename DenseMatrix<Field>::Element>>
DenseMatrix<Field>::column_space_basis() const {
  const auto pivot_columns = rref().pivot_columns;
  std::vector<std::vector<Element>> basis;
  basis.reserve(pivot_columns.size());
  for (const auto column : pivot_columns) {
    std::vector<Element> vector;
    vector.reserve(rows_);
    for (std::size_t row = 0; row < rows_; ++row) {
      vector.push_back(entries_[row * columns_ + column]);
    }
    basis.push_back(std::move(vector));
  }
  return basis;
}

template <typename Field>
LinearSolveResult<Field> DenseMatrix<Field>::solve(
    std::span<const Element> right_hand_side) const {
  if (right_hand_side.size() != rows_) {
    throw std::invalid_argument(
        "linear solve right-hand side has incompatible dimensions");
  }
  if (columns_ == std::numeric_limits<std::size_t>::max()) {
    throw std::length_error("augmented matrix width overflows size_t");
  }

  std::vector<Element> augmented_entries;
  augmented_entries.reserve(checked_entry_count(rows_, columns_ + 1));
  for (std::size_t row = 0; row < rows_; ++row) {
    for (std::size_t column = 0; column < columns_; ++column) {
      augmented_entries.push_back(entries_[row * columns_ + column]);
    }
    augmented_entries.push_back(field_.canonical(right_hand_side[row]));
  }
  DenseMatrix augmented(
      field_, rows_, columns_ + 1, std::move(augmented_entries));
  auto reduction = augmented.row_reduce(columns_);

  for (std::size_t row = 0; row < rows_; ++row) {
    bool zero_left_hand_side = true;
    for (std::size_t column = 0; column < columns_; ++column) {
      if (!field_.is_zero(reduction.reduced(row, column))) {
        zero_left_hand_side = false;
        break;
      }
    }
    if (zero_left_hand_side &&
        !field_.is_zero(reduction.reduced(row, columns_))) {
      return LinearSolveResult<Field>{
          LinearSolveStatus::Inconsistent, std::nullopt, {}};
    }
  }

  std::vector<Element> particular(columns_, field_.zero());
  for (std::size_t pivot_row = 0;
       pivot_row < reduction.pivot_columns.size(); ++pivot_row) {
    particular[reduction.pivot_columns[pivot_row]] =
        reduction.reduced(pivot_row, columns_);
  }

  std::vector<bool> is_pivot(columns_, false);
  for (const auto column : reduction.pivot_columns) {
    is_pivot[column] = true;
  }
  std::vector<std::vector<Element>> homogeneous_basis;
  homogeneous_basis.reserve(columns_ - reduction.pivot_columns.size());
  for (std::size_t free_column = 0; free_column < columns_;
       ++free_column) {
    if (is_pivot[free_column]) {
      continue;
    }
    std::vector<Element> vector(columns_, field_.zero());
    vector[free_column] = field_.one();
    for (std::size_t pivot_row = 0;
         pivot_row < reduction.pivot_columns.size(); ++pivot_row) {
      vector[reduction.pivot_columns[pivot_row]] = field_.negate(
          reduction.reduced(pivot_row, free_column));
    }
    homogeneous_basis.push_back(std::move(vector));
  }

  const auto status = homogeneous_basis.empty()
                          ? LinearSolveStatus::Unique
                          : LinearSolveStatus::Underdetermined;
  return LinearSolveResult<Field>{
      status,
      std::optional<std::vector<Element>>(std::move(particular)),
      std::move(homogeneous_basis)};
}

}  // namespace laughableengine

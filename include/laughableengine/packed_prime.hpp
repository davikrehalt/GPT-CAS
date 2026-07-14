#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "laughableengine/field.hpp"
#include "laughableengine/matrix.hpp"

namespace laughableengine {

// Raw finite-field arithmetic for discovery kernels.  The modulus is stored
// once per container instead of once per coefficient.  PrimeField remains the
// checked public/reference representation used at API boundaries.
class PackedPrimeArithmetic {
 public:
  explicit PackedPrimeArithmetic(std::uint32_t modulus)
      : modulus_(PrimeField(modulus).modulus()) {}

  [[nodiscard]] std::uint32_t modulus() const noexcept { return modulus_; }
  [[nodiscard]] constexpr std::uint32_t zero() const noexcept { return 0; }
  [[nodiscard]] constexpr std::uint32_t one() const noexcept { return 1; }

  [[nodiscard]] std::uint32_t canonical(std::uint32_t value) const noexcept {
    return value < modulus_ ? value : value % modulus_;
  }

  [[nodiscard]] std::uint32_t from_integer(std::int64_t value) const noexcept {
    const auto modulus = static_cast<std::int64_t>(modulus_);
    auto result = value % modulus;
    if (result < 0) {
      result += modulus;
    }
    return static_cast<std::uint32_t>(result);
  }

  [[nodiscard]] std::uint32_t add(
      std::uint32_t left,
      std::uint32_t right) const noexcept {
    const auto sum = static_cast<std::uint64_t>(left) + right;
    return static_cast<std::uint32_t>(
        sum >= modulus_ ? sum - modulus_ : sum);
  }

  [[nodiscard]] std::uint32_t subtract(
      std::uint32_t left,
      std::uint32_t right) const noexcept {
    return left >= right ? left - right : modulus_ - (right - left);
  }

  [[nodiscard]] std::uint32_t negate(std::uint32_t value) const noexcept {
    return value == 0 ? 0 : modulus_ - value;
  }

  [[nodiscard]] std::uint32_t multiply(
      std::uint32_t left,
      std::uint32_t right) const noexcept {
    return static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(left) * right) % modulus_);
  }

  [[nodiscard]] std::uint32_t power(
      std::uint32_t base,
      std::uint64_t exponent) const noexcept {
    auto result = one();
    while (exponent != 0) {
      if ((exponent & 1U) != 0) {
        result = multiply(result, base);
      }
      exponent >>= 1U;
      if (exponent != 0) {
        base = multiply(base, base);
      }
    }
    return result;
  }

  [[nodiscard]] std::uint32_t inverse(std::uint32_t value) const {
    if (value == 0 || value >= modulus_) {
      throw std::domain_error(
          value == 0 ? "zero has no multiplicative inverse"
                     : "packed finite-field residue is not canonical");
    }
    return power(value, static_cast<std::uint64_t>(modulus_) - 2U);
  }

 private:
  std::uint32_t modulus_;
};

struct PackedPrimeRref;
struct PackedPrimeRankKernel;
struct PackedPrimeSolveResult;
class PackedPrimeSparseMatrix;

// A contiguous row-major GF(p) matrix for discovery.  Entries are uint32_t
// residues and arithmetic dispatches through one matrix-owned modulus.
class PackedPrimeMatrix {
 public:
  PackedPrimeMatrix(
      std::uint32_t modulus,
      std::size_t rows,
      std::size_t columns)
      : arithmetic_(modulus),
        rows_(rows),
        columns_(columns),
        entries_(checked_entries(rows, columns), 0) {}

  PackedPrimeMatrix(
      std::uint32_t modulus,
      std::size_t rows,
      std::size_t columns,
      std::vector<std::uint32_t> entries)
      : arithmetic_(modulus),
        rows_(rows),
        columns_(columns),
        entries_(std::move(entries)) {
    if (entries_.size() != checked_entries(rows_, columns_)) {
      throw std::invalid_argument(
          "packed matrix entry count does not match its shape");
    }
    for (auto& entry : entries_) {
      entry = arithmetic_.canonical(entry);
    }
  }

  [[nodiscard]] static PackedPrimeMatrix from_exact(
      const DenseMatrix<PrimeField>& matrix) {
    std::vector<std::uint32_t> entries;
    entries.reserve(checked_entries(
        matrix.row_count(), matrix.column_count()));
    for (std::size_t row = 0; row < matrix.row_count(); ++row) {
      for (std::size_t column = 0; column < matrix.column_count(); ++column) {
        entries.push_back(matrix(row, column).value());
      }
    }
    return PackedPrimeMatrix(
        matrix.field().modulus(), matrix.row_count(), matrix.column_count(),
        std::move(entries));
  }

  [[nodiscard]] std::uint32_t modulus() const noexcept {
    return arithmetic_.modulus();
  }
  [[nodiscard]] std::size_t row_count() const noexcept { return rows_; }
  [[nodiscard]] std::size_t column_count() const noexcept { return columns_; }
  [[nodiscard]] const std::vector<std::uint32_t>& entries() const noexcept {
    return entries_;
  }

  [[nodiscard]] std::uint32_t operator()(
      std::size_t row,
      std::size_t column) const {
    return entries_[checked_index(row, column)];
  }

  void set(std::size_t row, std::size_t column, std::uint32_t value) {
    entries_[checked_index(row, column)] = arithmetic_.canonical(value);
  }

  [[nodiscard]] std::vector<std::uint32_t> multiply_column(
      std::span<const std::uint32_t> vector) const {
    if (vector.size() != columns_) {
      throw std::invalid_argument(
          "packed matrix-vector multiplication has incompatible dimensions");
    }
    std::vector<std::uint32_t> result(rows_, 0);
    for (std::size_t row = 0; row < rows_; ++row) {
      auto value = std::uint32_t{0};
      for (std::size_t column = 0; column < columns_; ++column) {
        const auto input = arithmetic_.canonical(vector[column]);
        value = arithmetic_.add(
            value,
            arithmetic_.multiply(
                entries_[row * columns_ + column], input));
      }
      result[row] = value;
    }
    return result;
  }

  [[nodiscard]] PackedPrimeMatrix transpose() const {
    PackedPrimeMatrix result(modulus(), columns_, rows_);
    for (std::size_t row = 0; row < rows_; ++row) {
      for (std::size_t column = 0; column < columns_; ++column) {
        result.entries_[column * rows_ + row] =
            entries_[row * columns_ + column];
      }
    }
    return result;
  }

  [[nodiscard]] DenseMatrix<PrimeField> to_exact() const {
    const PrimeField field(modulus());
    std::vector<PrimeField::Element> entries;
    entries.reserve(entries_.size());
    for (const auto entry : entries_) {
      entries.push_back(field.from_unsigned(entry));
    }
    return DenseMatrix<PrimeField>(
        field, rows_, columns_, std::move(entries));
  }

  [[nodiscard]] PackedPrimeRref rref() const;
  [[nodiscard]] PackedPrimeRankKernel rank_and_kernel(
      bool compute_kernel = true) const;
  [[nodiscard]] std::size_t rank() const;
  [[nodiscard]] std::vector<std::vector<std::uint32_t>>
  right_kernel_basis() const;
  [[nodiscard]] PackedPrimeSolveResult solve(
      std::span<const std::uint32_t> right_hand_side) const;

  friend bool operator==(
      const PackedPrimeMatrix& left,
      const PackedPrimeMatrix& right) noexcept {
    return left.modulus() == right.modulus() && left.rows_ == right.rows_ &&
           left.columns_ == right.columns_ && left.entries_ == right.entries_;
  }

  friend PackedPrimeMatrix multiply(
      const PackedPrimeSparseMatrix& left,
      const PackedPrimeMatrix& right);

 private:
  [[nodiscard]] static std::size_t checked_entries(
      std::size_t rows,
      std::size_t columns) {
    if (columns != 0 &&
        rows > std::numeric_limits<std::size_t>::max() / columns) {
      throw std::length_error("packed matrix dimensions overflow size_t");
    }
    return rows * columns;
  }

  [[nodiscard]] std::size_t checked_index(
      std::size_t row,
      std::size_t column) const {
    if (row >= rows_ || column >= columns_) {
      throw std::out_of_range("packed matrix index is out of range");
    }
    return row * columns_ + column;
  }

  [[nodiscard]] PackedPrimeRref row_reduce(
      std::size_t pivot_column_limit) const;

  PackedPrimeArithmetic arithmetic_;
  std::size_t rows_;
  std::size_t columns_;
  std::vector<std::uint32_t> entries_;
};

// Deterministic compressed-column raw-residue matrix.  It is optimized for
// repeated action on coordinate columns and for sparse-times-dense assembly.
class PackedPrimeSparseMatrix {
 public:
  struct Triplet {
    std::size_t row;
    std::size_t column;
    std::uint32_t value;
  };

  PackedPrimeSparseMatrix(
      std::uint32_t modulus,
      std::size_t rows,
      std::size_t columns,
      std::vector<Triplet> triplets)
      : arithmetic_(modulus), rows_(rows), columns_(columns) {
    if (columns_ == std::numeric_limits<std::size_t>::max()) {
      throw std::length_error(
          "packed sparse column-offset count overflows size_t");
    }
    for (auto& triplet : triplets) {
      if (triplet.row >= rows_ || triplet.column >= columns_) {
        throw std::out_of_range(
            "packed sparse triplet is outside its shape");
      }
      triplet.value = arithmetic_.canonical(triplet.value);
    }
    std::sort(triplets.begin(), triplets.end(), [](const auto& left,
                                                    const auto& right) {
      if (left.column != right.column) {
        return left.column < right.column;
      }
      return left.row < right.row;
    });
    column_offsets_.reserve(columns_ + 1);
    row_indices_.reserve(triplets.size());
    values_.reserve(triplets.size());
    std::size_t index = 0;
    for (std::size_t column = 0; column < columns_; ++column) {
      column_offsets_.push_back(values_.size());
      while (index < triplets.size() &&
             triplets[index].column == column) {
        const auto row = triplets[index].row;
        auto value = std::uint32_t{0};
        do {
          value = arithmetic_.add(value, triplets[index].value);
          ++index;
        } while (index < triplets.size() &&
                 triplets[index].column == column &&
                 triplets[index].row == row);
        if (value != 0) {
          row_indices_.push_back(row);
          values_.push_back(value);
        }
      }
    }
    column_offsets_.push_back(values_.size());
  }

  template <typename ExactSparseMatrix>
  [[nodiscard]] static PackedPrimeSparseMatrix from_exact(
      const ExactSparseMatrix& matrix) {
    std::vector<Triplet> triplets;
    triplets.reserve(matrix.nnz());
    for (std::size_t column = 0; column < matrix.column_count(); ++column) {
      for (std::size_t row = 0; row < matrix.row_count(); ++row) {
        const auto value = matrix(row, column);
        if (!matrix.field().is_zero(value)) {
          triplets.push_back(Triplet{row, column, value.value()});
        }
      }
    }
    return PackedPrimeSparseMatrix(
        matrix.field().modulus(), matrix.row_count(), matrix.column_count(),
        std::move(triplets));
  }

  [[nodiscard]] std::uint32_t modulus() const noexcept {
    return arithmetic_.modulus();
  }
  [[nodiscard]] std::size_t row_count() const noexcept { return rows_; }
  [[nodiscard]] std::size_t column_count() const noexcept { return columns_; }
  [[nodiscard]] std::size_t nnz() const noexcept { return values_.size(); }

  [[nodiscard]] std::vector<std::uint32_t> multiply_column(
      std::span<const std::uint32_t> vector) const {
    if (vector.size() != columns_) {
      throw std::invalid_argument(
          "packed sparse matrix-vector multiplication has incompatible dimensions");
    }
    std::vector<std::uint32_t> result(rows_, 0);
    for (std::size_t column = 0; column < columns_; ++column) {
      const auto factor = arithmetic_.canonical(vector[column]);
      if (factor == 0) {
        continue;
      }
      for (std::size_t index = column_offsets_[column];
           index < column_offsets_[column + 1]; ++index) {
        const auto row = row_indices_[index];
        result[row] = arithmetic_.add(
            result[row], arithmetic_.multiply(values_[index], factor));
      }
    }
    return result;
  }

  [[nodiscard]] PackedPrimeMatrix to_dense() const {
    PackedPrimeMatrix result(modulus(), rows_, columns_);
    for (std::size_t column = 0; column < columns_; ++column) {
      for (std::size_t index = column_offsets_[column];
           index < column_offsets_[column + 1]; ++index) {
        result.set(row_indices_[index], column, values_[index]);
      }
    }
    return result;
  }

  friend PackedPrimeMatrix multiply(
      const PackedPrimeSparseMatrix& left,
      const PackedPrimeMatrix& right) {
    if (left.modulus() != right.modulus() ||
        left.column_count() != right.row_count()) {
      throw std::invalid_argument(
          "packed sparse-dense multiplication has incompatible operands");
    }
    PackedPrimeMatrix result(
        left.modulus(), left.row_count(), right.column_count());
    for (std::size_t inner = 0; inner < left.column_count(); ++inner) {
      for (std::size_t index = left.column_offsets_[inner];
           index < left.column_offsets_[inner + 1]; ++index) {
        const auto row = left.row_indices_[index];
        const auto factor = left.values_[index];
        for (std::size_t column = 0; column < right.column_count(); ++column) {
          const auto right_index = inner * right.columns_ + column;
          const auto result_index = row * result.columns_ + column;
          const auto product = left.arithmetic_.multiply(
              factor, right.entries_[right_index]);
          result.entries_[result_index] = left.arithmetic_.add(
              result.entries_[result_index], product);
        }
      }
    }
    return result;
  }

 private:
  PackedPrimeArithmetic arithmetic_;
  std::size_t rows_;
  std::size_t columns_;
  std::vector<std::size_t> column_offsets_;
  std::vector<std::size_t> row_indices_;
  std::vector<std::uint32_t> values_;
};

struct PackedPrimeRref {
  PackedPrimeMatrix reduced;
  std::vector<std::size_t> pivot_columns;
};

struct PackedPrimeRankKernel {
  std::size_t rank;
  std::vector<std::size_t> pivot_columns;
  std::vector<std::vector<std::uint32_t>> kernel_basis;
};

struct PackedPrimeSolveResult {
  LinearSolveStatus status;
  std::optional<std::vector<std::uint32_t>> particular_solution;
  std::vector<std::vector<std::uint32_t>> homogeneous_basis;
};

inline PackedPrimeRref PackedPrimeMatrix::row_reduce(
    std::size_t pivot_column_limit) const {
  if (pivot_column_limit > columns_) {
    throw std::logic_error("packed RREF pivot limit exceeds matrix width");
  }
  auto reduced = *this;
  std::vector<std::size_t> pivots;
  pivots.reserve(std::min(rows_, pivot_column_limit));
  std::size_t pivot_row = 0;
  for (std::size_t column = 0;
       column < pivot_column_limit && pivot_row < rows_; ++column) {
    std::size_t selected = pivot_row;
    while (selected < rows_ &&
           reduced.entries_[selected * columns_ + column] == 0) {
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

    const auto inverse = reduced.arithmetic_.inverse(
        reduced.entries_[pivot_row * columns_ + column]);
    for (std::size_t entry_column = 0; entry_column < columns_;
         ++entry_column) {
      auto& entry =
          reduced.entries_[pivot_row * columns_ + entry_column];
      entry = reduced.arithmetic_.multiply(entry, inverse);
    }
    for (std::size_t row = 0; row < rows_; ++row) {
      if (row == pivot_row) {
        continue;
      }
      const auto factor = reduced.entries_[row * columns_ + column];
      if (factor == 0) {
        continue;
      }
      for (std::size_t entry_column = 0; entry_column < columns_;
           ++entry_column) {
        auto& entry = reduced.entries_[row * columns_ + entry_column];
        entry = reduced.arithmetic_.subtract(
            entry,
            reduced.arithmetic_.multiply(
                factor,
                reduced.entries_[pivot_row * columns_ + entry_column]));
      }
    }
    pivots.push_back(column);
    ++pivot_row;
  }
  return PackedPrimeRref{std::move(reduced), std::move(pivots)};
}

inline PackedPrimeRref PackedPrimeMatrix::rref() const {
  return row_reduce(columns_);
}

inline PackedPrimeRankKernel PackedPrimeMatrix::rank_and_kernel(
    bool compute_kernel) const {
  auto reduction = rref();
  std::vector<std::vector<std::uint32_t>> kernel;
  if (compute_kernel) {
    std::vector<bool> is_pivot(columns_, false);
    for (const auto pivot : reduction.pivot_columns) {
      is_pivot[pivot] = true;
    }
    kernel.reserve(columns_ - reduction.pivot_columns.size());
    for (std::size_t free_column = 0; free_column < columns_; ++free_column) {
      if (is_pivot[free_column]) {
        continue;
      }
      std::vector<std::uint32_t> vector(columns_, 0);
      vector[free_column] = 1;
      for (std::size_t pivot_row = 0;
           pivot_row < reduction.pivot_columns.size(); ++pivot_row) {
        vector[reduction.pivot_columns[pivot_row]] = arithmetic_.negate(
            reduction.reduced(pivot_row, free_column));
      }
      kernel.push_back(std::move(vector));
    }
  }
  const auto rank = reduction.pivot_columns.size();
  return PackedPrimeRankKernel{
      rank, std::move(reduction.pivot_columns), std::move(kernel)};
}

inline std::size_t PackedPrimeMatrix::rank() const {
  return rank_and_kernel(false).rank;
}

inline std::vector<std::vector<std::uint32_t>>
PackedPrimeMatrix::right_kernel_basis() const {
  return rank_and_kernel(true).kernel_basis;
}

inline PackedPrimeSolveResult PackedPrimeMatrix::solve(
    std::span<const std::uint32_t> right_hand_side) const {
  if (right_hand_side.size() != rows_) {
    throw std::invalid_argument(
        "packed linear solve right-hand side has incompatible dimensions");
  }
  if (columns_ == std::numeric_limits<std::size_t>::max()) {
    throw std::length_error("packed augmented matrix width overflows size_t");
  }
  std::vector<std::uint32_t> augmented;
  augmented.reserve(checked_entries(rows_, columns_ + 1));
  for (std::size_t row = 0; row < rows_; ++row) {
    const auto first = entries_.begin() +
                       static_cast<std::ptrdiff_t>(row * columns_);
    augmented.insert(
        augmented.end(), first,
        first + static_cast<std::ptrdiff_t>(columns_));
    augmented.push_back(arithmetic_.canonical(right_hand_side[row]));
  }
  PackedPrimeMatrix system(
      modulus(), rows_, columns_ + 1, std::move(augmented));
  auto reduction = system.row_reduce(columns_);
  for (std::size_t row = 0; row < rows_; ++row) {
    bool left_zero = true;
    for (std::size_t column = 0; column < columns_; ++column) {
      if (reduction.reduced(row, column) != 0) {
        left_zero = false;
        break;
      }
    }
    if (left_zero && reduction.reduced(row, columns_) != 0) {
      return PackedPrimeSolveResult{
          LinearSolveStatus::Inconsistent, std::nullopt, {}};
    }
  }

  std::vector<std::uint32_t> particular(columns_, 0);
  for (std::size_t pivot_row = 0;
       pivot_row < reduction.pivot_columns.size(); ++pivot_row) {
    particular[reduction.pivot_columns[pivot_row]] =
        reduction.reduced(pivot_row, columns_);
  }
  std::vector<bool> is_pivot(columns_, false);
  for (const auto pivot : reduction.pivot_columns) {
    is_pivot[pivot] = true;
  }
  std::vector<std::vector<std::uint32_t>> homogeneous;
  homogeneous.reserve(columns_ - reduction.pivot_columns.size());
  for (std::size_t free_column = 0; free_column < columns_; ++free_column) {
    if (is_pivot[free_column]) {
      continue;
    }
    std::vector<std::uint32_t> vector(columns_, 0);
    vector[free_column] = 1;
    for (std::size_t pivot_row = 0;
         pivot_row < reduction.pivot_columns.size(); ++pivot_row) {
      vector[reduction.pivot_columns[pivot_row]] = arithmetic_.negate(
          reduction.reduced(pivot_row, free_column));
    }
    homogeneous.push_back(std::move(vector));
  }
  return PackedPrimeSolveResult{
      homogeneous.empty() ? LinearSolveStatus::Unique
                          : LinearSolveStatus::Underdetermined,
      std::move(particular), std::move(homogeneous)};
}

inline PackedPrimeMatrix multiply(
    const PackedPrimeMatrix& left,
    const PackedPrimeMatrix& right) {
  if (left.modulus() != right.modulus() ||
      left.column_count() != right.row_count()) {
    throw std::invalid_argument(
        "packed matrix multiplication has incompatible operands");
  }
  PackedPrimeArithmetic arithmetic(left.modulus());
  PackedPrimeMatrix result(
      left.modulus(), left.row_count(), right.column_count());
  for (std::size_t row = 0; row < left.row_count(); ++row) {
    for (std::size_t inner = 0; inner < left.column_count(); ++inner) {
      const auto factor = left(row, inner);
      if (factor == 0) {
        continue;
      }
      for (std::size_t column = 0; column < right.column_count(); ++column) {
        const auto product = arithmetic.multiply(factor, right(inner, column));
        result.set(
            row, column, arithmetic.add(result(row, column), product));
      }
    }
  }
  return result;
}

inline PackedPrimeMatrix packed_matrix_from_columns(
    std::uint32_t modulus,
    std::size_t row_count,
    std::span<const std::vector<std::uint32_t>> columns) {
  PackedPrimeMatrix result(modulus, row_count, columns.size());
  for (std::size_t column = 0; column < columns.size(); ++column) {
    if (columns[column].size() != row_count) {
      throw std::invalid_argument(
          "packed matrix column has the wrong dimension");
    }
    for (std::size_t row = 0; row < row_count; ++row) {
      result.set(row, column, columns[column][row]);
    }
  }
  return result;
}

inline PackedPrimeMatrix packed_vertical_stack(
    std::uint32_t modulus,
    std::size_t column_count,
    std::span<const PackedPrimeMatrix> matrices) {
  std::size_t rows = 0;
  for (const auto& matrix : matrices) {
    if (matrix.modulus() != modulus || matrix.column_count() != column_count) {
      throw std::invalid_argument(
          "packed vertical stack has incompatible matrices");
    }
    if (matrix.row_count() >
        std::numeric_limits<std::size_t>::max() - rows) {
      throw std::length_error("packed vertical stack row count overflows");
    }
    rows += matrix.row_count();
  }
  PackedPrimeMatrix result(modulus, rows, column_count);
  std::size_t offset = 0;
  for (const auto& matrix : matrices) {
    for (std::size_t row = 0; row < matrix.row_count(); ++row) {
      for (std::size_t column = 0; column < column_count; ++column) {
        result.set(offset + row, column, matrix(row, column));
      }
    }
    offset += matrix.row_count();
  }
  return result;
}

}  // namespace laughableengine

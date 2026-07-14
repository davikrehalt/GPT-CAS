#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "laughableengine/matrix.hpp"

namespace laughableengine {

enum class SparseEliminationResourceKind {
  LiveNonzeros,
  ArithmeticOperations,
  KernelCoordinates,
  KernelNonzeros,
  OutputCoordinates,
};

class SparseEliminationResourceLimit : public std::runtime_error {
 public:
  SparseEliminationResourceLimit(
      SparseEliminationResourceKind kind,
      std::size_t limit,
      std::size_t observed)
      : std::runtime_error(make_message(kind, limit, observed)),
        kind_(kind),
        limit_(limit),
        observed_(observed) {}

  [[nodiscard]] SparseEliminationResourceKind kind() const noexcept {
    return kind_;
  }
  [[nodiscard]] std::size_t limit() const noexcept { return limit_; }
  [[nodiscard]] std::size_t observed() const noexcept { return observed_; }

 private:
  [[nodiscard]] static const char* resource_name(
      SparseEliminationResourceKind kind) noexcept {
    switch (kind) {
      case SparseEliminationResourceKind::LiveNonzeros:
        return "live sparse coefficients";
      case SparseEliminationResourceKind::ArithmeticOperations:
        return "sparse elimination arithmetic operations";
      case SparseEliminationResourceKind::KernelCoordinates:
        return "dense kernel coordinates";
      case SparseEliminationResourceKind::KernelNonzeros:
        return "sparse kernel nonzeros";
      case SparseEliminationResourceKind::OutputCoordinates:
        return "dense sparse-matrix output coordinates";
    }
    return "sparse elimination resource";
  }

  [[nodiscard]] static std::string make_message(
      SparseEliminationResourceKind kind,
      std::size_t limit,
      std::size_t observed) {
    return std::string("sparse elimination resource limit exceeded for ") +
           resource_name(kind) + ": limit " + std::to_string(limit) +
           ", attempted " + std::to_string(observed);
  }

  SparseEliminationResourceKind kind_;
  std::size_t limit_;
  std::size_t observed_;
};

struct SparseEliminationLimits {
  std::optional<std::size_t> max_live_nonzeros;
  std::optional<std::size_t> max_arithmetic_operations;
  std::optional<std::size_t> max_kernel_coordinate_entries;
  std::optional<std::size_t> max_kernel_nonzeros;
  std::optional<std::size_t> max_output_coordinate_entries;
};

// An immutable exact sparse matrix in deterministic compressed-row form.
// As with DenseMatrix, it acts on column vectors:
//
//                         A : k^columns -> k^rows.
//
// Construction is the only normalization point. Triplets are checked,
// duplicate positions are added in the coefficient field, and exact zeros
// are discarded.
template <typename Field>
class SparseMatrix {
 public:
  using Element = typename Field::Element;

  struct Triplet {
    std::size_t row;
    std::size_t column;
    Element value;

    friend bool operator==(const Triplet&, const Triplet&) = default;
  };

  struct RankKernelResult {
    std::size_t rank;
    std::vector<std::vector<Element>> kernel_basis;
  };

  struct SparseKernelEntry {
    std::size_t column;
    Element value;

    friend bool operator==(const SparseKernelEntry&, const SparseKernelEntry&) =
        default;
  };

  using SparseKernelVector = std::vector<SparseKernelEntry>;

  struct SparseRankKernelResult {
    std::size_t rank;
    std::vector<SparseKernelVector> kernel_basis;
  };

  static constexpr bool elimination_uses_dense_fallback = false;

  SparseMatrix(
      Field field,
      std::size_t rows,
      std::size_t columns,
      std::vector<Triplet> triplets)
      : field_(std::move(field)), rows_(rows), columns_(columns) {
    if (rows_ == std::numeric_limits<std::size_t>::max()) {
      throw std::length_error("sparse matrix row-offset count overflows size_t");
    }

    for (auto& triplet : triplets) {
      if (triplet.row >= rows_ || triplet.column >= columns_) {
        throw std::out_of_range("sparse matrix triplet is outside its shape");
      }
      triplet.value = field_.canonical(std::move(triplet.value));
    }
    std::sort(triplets.begin(), triplets.end(), [](const auto& left,
                                                    const auto& right) {
      if (left.row != right.row) {
        return left.row < right.row;
      }
      return left.column < right.column;
    });

    column_indices_.reserve(triplets.size());
    values_.reserve(triplets.size());
    row_offsets_.reserve(rows_ + 1);
    std::size_t index = 0;
    for (std::size_t row = 0; row < rows_; ++row) {
      row_offsets_.push_back(values_.size());
      while (index < triplets.size() && triplets[index].row == row) {
        const auto column = triplets[index].column;
        auto value = field_.zero();
        do {
          value = field_.add(value, triplets[index].value);
          ++index;
        } while (index < triplets.size() && triplets[index].row == row &&
                 triplets[index].column == column);
        if (!field_.is_zero(value)) {
          column_indices_.push_back(column);
          values_.push_back(std::move(value));
        }
      }
    }
    row_offsets_.push_back(values_.size());
  }

  [[nodiscard]] const Field& field() const noexcept { return field_; }
  [[nodiscard]] std::size_t row_count() const noexcept { return rows_; }
  [[nodiscard]] std::size_t column_count() const noexcept { return columns_; }
  [[nodiscard]] std::size_t nnz() const noexcept { return values_.size(); }
  [[nodiscard]] std::size_t nonzero_count() const noexcept {
    return nnz();
  }

  // Explicit stored-entry export.  This is O(nnz), never a dense shape scan.
  [[nodiscard]] std::vector<Triplet> triplets() const {
    std::vector<Triplet> result;
    result.reserve(values_.size());
    for (std::size_t row = 0; row < rows_; ++row) {
      for (std::size_t index = row_offsets_[row];
           index < row_offsets_[row + 1]; ++index) {
        result.push_back(
            Triplet{row, column_indices_[index], values_[index]});
      }
    }
    return result;
  }

  [[nodiscard]] Element at(std::size_t row, std::size_t column) const {
    if (row >= rows_ || column >= columns_) {
      throw std::out_of_range("sparse matrix index is out of range");
    }
    const auto first = row_offsets_[row];
    const auto last = row_offsets_[row + 1];
    const auto iterator = std::lower_bound(
        column_indices_.begin() + static_cast<std::ptrdiff_t>(first),
        column_indices_.begin() + static_cast<std::ptrdiff_t>(last),
        column);
    if (iterator ==
            column_indices_.begin() + static_cast<std::ptrdiff_t>(last) ||
        *iterator != column) {
      return field_.zero();
    }
    return values_[static_cast<std::size_t>(iterator - column_indices_.begin())];
  }

  [[nodiscard]] Element operator()(
      std::size_t row,
      std::size_t column) const {
    return at(row, column);
  }

  [[nodiscard]] SparseMatrix transpose() const {
    std::vector<Triplet> triplets;
    triplets.reserve(values_.size());
    for (std::size_t row = 0; row < rows_; ++row) {
      for (std::size_t index = row_offsets_[row];
           index < row_offsets_[row + 1]; ++index) {
        triplets.push_back(
            Triplet{column_indices_[index], row, values_[index]});
      }
    }
    return SparseMatrix(field_, columns_, rows_, std::move(triplets));
  }

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
      for (std::size_t index = row_offsets_[row];
           index < row_offsets_[row + 1]; ++index) {
        value = field_.add(
            value,
            field_.multiply(
                values_[index], canonical[column_indices_[index]]));
      }
      result[row] = std::move(value);
    }
    return result;
  }

  [[nodiscard]] DenseMatrix<Field> to_dense() const {
    DenseMatrix<Field> result(field_, rows_, columns_);
    for (std::size_t row = 0; row < rows_; ++row) {
      for (std::size_t index = row_offsets_[row];
           index < row_offsets_[row + 1]; ++index) {
        result.set(row, column_indices_[index], values_[index]);
      }
    }
    return result;
  }

  [[nodiscard]] std::size_t rank(
      const SparseEliminationLimits& limits = {}) const {
    return sparse_rref(columns_, std::nullopt, limits).pivot_columns.size();
  }

  [[nodiscard]] RankKernelResult rank_and_kernel(
      const SparseEliminationLimits& limits = {}) const {
    const auto reduction = sparse_rref(columns_, std::nullopt, limits);
    return RankKernelResult{
        reduction.pivot_columns.size(),
        kernel_from_reduction(reduction, limits)};
  }

  [[nodiscard]] SparseRankKernelResult rank_and_sparse_kernel(
      const SparseEliminationLimits& limits = {}) const {
    const auto reduction = sparse_rref(columns_, std::nullopt, limits);
    return SparseRankKernelResult{
        reduction.pivot_columns.size(),
        sparse_kernel_from_reduction(reduction, limits)};
  }

  [[nodiscard]] std::vector<std::vector<Element>>
  right_kernel_basis(const SparseEliminationLimits& limits = {}) const {
    const auto reduction = sparse_rref(columns_, std::nullopt, limits);
    return kernel_from_reduction(reduction, limits);
  }

  [[nodiscard]] std::vector<SparseKernelVector>
  right_kernel_basis_sparse(
      const SparseEliminationLimits& limits = {}) const {
    const auto reduction = sparse_rref(columns_, std::nullopt, limits);
    return sparse_kernel_from_reduction(reduction, limits);
  }

  [[nodiscard]] std::vector<std::vector<Element>>
  column_space_basis(const SparseEliminationLimits& limits = {}) const {
    const auto reduction = sparse_rref(columns_, std::nullopt, limits);
    EliminationMeter(limits).check_output_coordinates(
        rows_, reduction.pivot_columns.size());
    std::vector<std::vector<Element>> basis;
    basis.reserve(reduction.pivot_columns.size());
    for (std::size_t index = 0; index < reduction.pivot_columns.size();
         ++index) {
      basis.emplace_back(rows_, field_.zero());
    }

    std::unordered_map<std::size_t, std::size_t> pivot_indices;
    pivot_indices.reserve(reduction.pivot_columns.size());
    for (std::size_t index = 0; index < reduction.pivot_columns.size();
         ++index) {
      pivot_indices.emplace(reduction.pivot_columns[index], index);
    }
    for (std::size_t row = 0; row < rows_; ++row) {
      for (std::size_t index = row_offsets_[row];
           index < row_offsets_[row + 1]; ++index) {
        const auto pivot = pivot_indices.find(column_indices_[index]);
        if (pivot != pivot_indices.end()) {
          basis[pivot->second][row] = values_[index];
        }
      }
    }
    return basis;
  }

  [[nodiscard]] LinearSolveResult<Field> solve(
      std::span<const Element> right_hand_side,
      const SparseEliminationLimits& limits = {}) const {
    if (right_hand_side.size() != rows_) {
      throw std::invalid_argument(
          "linear solve right-hand side has incompatible dimensions");
    }
    if (columns_ == std::numeric_limits<std::size_t>::max()) {
      throw std::length_error("augmented matrix width overflows size_t");
    }

    std::vector<Element> canonical_right_hand_side;
    canonical_right_hand_side.reserve(right_hand_side.size());
    for (const auto& entry : right_hand_side) {
      canonical_right_hand_side.push_back(field_.canonical(entry));
    }
    const auto reduction = sparse_rref(
        columns_, std::span<const Element>(canonical_right_hand_side), limits);

    for (const auto& row : reduction.residual_rows) {
      if (const auto* value = coefficient_at(row, columns_);
          value != nullptr && !field_.is_zero(*value)) {
        return LinearSolveResult<Field>{
            LinearSolveStatus::Inconsistent, std::nullopt, {}};
      }
    }

    std::vector<Element> particular(columns_, field_.zero());
    for (std::size_t pivot = 0; pivot < reduction.pivot_columns.size();
         ++pivot) {
      if (const auto* value =
              coefficient_at(reduction.pivot_rows[pivot], columns_);
          value != nullptr) {
        particular[reduction.pivot_columns[pivot]] = *value;
      }
    }

    auto homogeneous_basis = kernel_from_reduction(reduction, limits);
    const auto status = homogeneous_basis.empty()
                            ? LinearSolveStatus::Unique
                            : LinearSolveStatus::Underdetermined;
    return LinearSolveResult<Field>{
        status,
        std::optional<std::vector<Element>>(std::move(particular)),
        std::move(homogeneous_basis)};
  }

  friend bool operator==(const SparseMatrix&, const SparseMatrix&) = default;

 private:
  struct SparseEntry {
    std::size_t column;
    Element value;
  };

  using SparseRow = std::vector<SparseEntry>;

  struct SparseRrefResult {
    // Pivot rows and pivot columns are both sorted by increasing pivot column.
    // residual_rows contains nonzero rows with no eligible pivot; it is only
    // nonempty for an augmented solve whose left-hand side has vanished.
    std::vector<SparseRow> pivot_rows;
    std::vector<std::size_t> pivot_columns;
    std::vector<SparseRow> residual_rows;
  };

  class EliminationMeter {
   public:
    explicit EliminationMeter(const SparseEliminationLimits& limits)
        : limits_(limits) {}

    [[nodiscard]] static std::size_t saturated_add(
        std::size_t left,
        std::size_t right) noexcept {
      return left > std::numeric_limits<std::size_t>::max() - right
                 ? std::numeric_limits<std::size_t>::max()
                 : left + right;
    }

    [[nodiscard]] static std::size_t saturated_product(
        std::size_t left,
        std::size_t right) noexcept {
      return right != 0 &&
                     left > std::numeric_limits<std::size_t>::max() / right
                 ? std::numeric_limits<std::size_t>::max()
                 : left * right;
    }

    void add_operations(std::size_t count) {
      operations_ = saturated_add(operations_, count);
      check(
          SparseEliminationResourceKind::ArithmeticOperations, operations_,
          limits_.max_arithmetic_operations);
    }

    void check_live(std::size_t observed) const {
      check(
          SparseEliminationResourceKind::LiveNonzeros, observed,
          limits_.max_live_nonzeros);
    }

    void check_kernel_coordinates(
        std::size_t width,
        std::size_t vector_count) const {
      const auto observed = saturated_product(width, vector_count);
      check(
          SparseEliminationResourceKind::KernelCoordinates, observed,
          limits_.max_kernel_coordinate_entries);
    }

    void check_kernel_nonzeros(std::size_t observed) const {
      check(
          SparseEliminationResourceKind::KernelNonzeros, observed,
          limits_.max_kernel_nonzeros);
    }

    void check_output_coordinates(
        std::size_t width,
        std::size_t vector_count) const {
      const auto observed = saturated_product(width, vector_count);
      check(
          SparseEliminationResourceKind::OutputCoordinates, observed,
          limits_.max_output_coordinate_entries);
    }

   private:
    static void check(
        SparseEliminationResourceKind kind,
        std::size_t observed,
        const std::optional<std::size_t>& limit) {
      if (limit.has_value() && observed > *limit) {
        throw SparseEliminationResourceLimit(
            kind, *limit, observed);
      }
    }

    const SparseEliminationLimits& limits_;
    std::size_t operations_ = 0;
  };

  [[nodiscard]] static const Element* coefficient_at(
      const SparseRow& row,
      std::size_t column) {
    const auto iterator = std::lower_bound(
        row.begin(), row.end(), column,
        [](const SparseEntry& entry, std::size_t candidate) {
          return entry.column < candidate;
        });
    return iterator != row.end() && iterator->column == column
               ? &iterator->value
               : nullptr;
  }

  [[nodiscard]] SparseRow subtract_multiple(
      const SparseRow& target,
      const SparseRow& pivot,
      const Element& factor,
      EliminationMeter& meter,
      std::size_t retained_nonzeros) const {
    if (target.size() >
        std::numeric_limits<std::size_t>::max() - pivot.size()) {
      throw std::length_error("sparse row workspace overflows size_t");
    }
    const auto maximum_result_size = target.size() + pivot.size();
    meter.add_operations(maximum_result_size);
    meter.check_live(EliminationMeter::saturated_add(
        retained_nonzeros,
        EliminationMeter::saturated_add(target.size(), maximum_result_size)));
    SparseRow result;
    result.reserve(maximum_result_size);

    std::size_t target_index = 0;
    std::size_t pivot_index = 0;
    while (target_index < target.size() || pivot_index < pivot.size()) {
      if (pivot_index == pivot.size() ||
          (target_index < target.size() &&
           target[target_index].column < pivot[pivot_index].column)) {
        result.push_back(target[target_index]);
        ++target_index;
        continue;
      }
      if (target_index == target.size() ||
          pivot[pivot_index].column < target[target_index].column) {
        auto value = field_.negate(
            field_.multiply(factor, pivot[pivot_index].value));
        if (!field_.is_zero(value)) {
          result.push_back(
              SparseEntry{pivot[pivot_index].column, std::move(value)});
        }
        ++pivot_index;
        continue;
      }

      auto value = field_.subtract(
          target[target_index].value,
          field_.multiply(factor, pivot[pivot_index].value));
      if (!field_.is_zero(value)) {
        result.push_back(
            SparseEntry{target[target_index].column, std::move(value)});
      }
      ++target_index;
      ++pivot_index;
    }
    return result;
  }

  void normalize_pivot_row(
      SparseRow& row,
      std::size_t pivot_column,
      EliminationMeter& meter) const {
    const auto* pivot = coefficient_at(row, pivot_column);
    if (pivot == nullptr || field_.is_zero(*pivot)) {
      throw std::logic_error("sparse elimination selected a zero pivot");
    }
    const auto inverse = field_.inverse(*pivot);
    meter.add_operations(row.size());
    for (auto& entry : row) {
      entry.value = field_.multiply(entry.value, inverse);
    }
  }

  [[nodiscard]] SparseRow input_row(
      std::size_t row,
      const std::optional<std::span<const Element>>& right_hand_side) const {
    SparseRow result;
    const auto extra = right_hand_side.has_value() ? std::size_t{1}
                                                    : std::size_t{0};
    const auto stored_size = row_offsets_[row + 1] - row_offsets_[row];
    if (stored_size > std::numeric_limits<std::size_t>::max() - extra) {
      throw std::length_error("sparse row workspace overflows size_t");
    }
    result.reserve(stored_size + extra);
    for (std::size_t index = row_offsets_[row];
         index < row_offsets_[row + 1]; ++index) {
      result.push_back(SparseEntry{column_indices_[index], values_[index]});
    }
    if (right_hand_side.has_value()) {
      const auto& value = (*right_hand_side)[row];
      if (!field_.is_zero(value)) {
        result.push_back(SparseEntry{columns_, value});
      }
    }
    return result;
  }

  [[nodiscard]] SparseRrefResult sparse_rref(
      std::size_t pivot_column_limit,
      std::optional<std::span<const Element>> right_hand_side =
          std::nullopt,
      const SparseEliminationLimits& limits = {}) const {
    if (pivot_column_limit > columns_) {
      throw std::logic_error("sparse RREF pivot limit exceeds matrix width");
    }
    if (right_hand_side.has_value() && right_hand_side->size() != rows_) {
      throw std::logic_error(
          "sparse RREF right-hand side has incompatible dimensions");
    }

    std::vector<SparseRow> pivot_rows;
    std::vector<std::size_t> pivot_columns;
    std::vector<SparseRow> residual_rows;
    pivot_rows.reserve(std::min(rows_, pivot_column_limit));
    pivot_columns.reserve(std::min(rows_, pivot_column_limit));

    // Pivot lookup reduces a new row without scanning absent columns. The
    // incidence table contains only non-pivot entries of established pivot
    // rows, so a new pivot is cleared from exactly the rows that contain it.
    std::unordered_map<std::size_t, std::size_t> pivot_lookup;
    std::unordered_map<std::size_t, std::unordered_set<std::size_t>> incidence;
    pivot_lookup.reserve(std::min(rows_, pivot_column_limit));
    EliminationMeter meter(limits);
    std::size_t retained_nonzeros = 0;

    const auto remove_from_incidence =
        [&](std::size_t row_index, const SparseRow& row_value) {
          const auto own_pivot = pivot_columns[row_index];
          for (const auto& entry : row_value) {
            if (entry.column >= pivot_column_limit ||
                entry.column == own_pivot) {
              continue;
            }
            const auto column = incidence.find(entry.column);
            if (column == incidence.end()) {
              throw std::logic_error(
                  "sparse elimination incidence table lost a row");
            }
            column->second.erase(row_index);
            if (column->second.empty()) {
              incidence.erase(column);
            }
          }
        };
    const auto add_to_incidence =
        [&](std::size_t row_index, const SparseRow& row_value) {
          const auto own_pivot = pivot_columns[row_index];
          for (const auto& entry : row_value) {
            if (entry.column < pivot_column_limit &&
                entry.column != own_pivot) {
              incidence[entry.column].insert(row_index);
            }
          }
        };

    for (std::size_t source_row = 0; source_row < rows_; ++source_row) {
      auto row = input_row(source_row, right_hand_side);
      meter.check_live(EliminationMeter::saturated_add(
          retained_nonzeros, row.size()));

      // Established pivot rows are maintained in reduced form. Eliminating
      // one existing pivot therefore cannot reintroduce another pivot column.
      while (true) {
        std::optional<std::pair<std::size_t, Element>> reduction;
        for (const auto& entry : row) {
          if (entry.column >= pivot_column_limit) {
            break;
          }
          const auto pivot = pivot_lookup.find(entry.column);
          if (pivot != pivot_lookup.end()) {
            reduction.emplace(pivot->second, entry.value);
            break;
          }
        }
        if (!reduction.has_value()) {
          break;
        }
        row = subtract_multiple(
            row, pivot_rows[reduction->first], reduction->second, meter,
            retained_nonzeros);
      }

      if (row.empty() || row.front().column >= pivot_column_limit) {
        if (!row.empty()) {
          retained_nonzeros = EliminationMeter::saturated_add(
              retained_nonzeros, row.size());
          meter.check_live(retained_nonzeros);
          residual_rows.push_back(std::move(row));
        }
        continue;
      }

      const auto pivot_column = row.front().column;
      normalize_pivot_row(row, pivot_column, meter);

      std::vector<std::size_t> affected_rows;
      if (const auto affected = incidence.find(pivot_column);
          affected != incidence.end()) {
        affected_rows.assign(affected->second.begin(), affected->second.end());
        std::sort(affected_rows.begin(), affected_rows.end());
      }
      for (const auto affected_row : affected_rows) {
        const auto* factor =
            coefficient_at(pivot_rows[affected_row], pivot_column);
        if (factor == nullptr) {
          throw std::logic_error(
              "sparse elimination incidence table contains a stale row");
        }
        const auto factor_copy = *factor;
        const auto previous_size = pivot_rows[affected_row].size();
        remove_from_incidence(affected_row, pivot_rows[affected_row]);
        pivot_rows[affected_row] = subtract_multiple(
            pivot_rows[affected_row], row, factor_copy, meter,
            retained_nonzeros);
        retained_nonzeros -= previous_size;
        retained_nonzeros = EliminationMeter::saturated_add(
            retained_nonzeros, pivot_rows[affected_row].size());
        meter.check_live(retained_nonzeros);
        add_to_incidence(affected_row, pivot_rows[affected_row]);
      }

      const auto pivot_index = pivot_rows.size();
      retained_nonzeros = EliminationMeter::saturated_add(
          retained_nonzeros, row.size());
      meter.check_live(retained_nonzeros);
      pivot_columns.push_back(pivot_column);
      pivot_rows.push_back(std::move(row));
      pivot_lookup.emplace(pivot_column, pivot_index);
      add_to_incidence(pivot_index, pivot_rows.back());
    }

    std::vector<std::size_t> order;
    order.reserve(pivot_columns.size());
    for (std::size_t index = 0; index < pivot_columns.size(); ++index) {
      order.push_back(index);
    }
    std::sort(order.begin(), order.end(), [&](std::size_t left,
                                              std::size_t right) {
      return pivot_columns[left] < pivot_columns[right];
    });

    SparseRrefResult result;
    result.pivot_rows.reserve(pivot_rows.size());
    result.pivot_columns.reserve(pivot_columns.size());
    result.residual_rows = std::move(residual_rows);
    for (const auto index : order) {
      result.pivot_columns.push_back(pivot_columns[index]);
      result.pivot_rows.push_back(std::move(pivot_rows[index]));
    }
    return result;
  }

  [[nodiscard]] std::vector<std::vector<Element>> kernel_from_reduction(
      const SparseRrefResult& reduction,
      const SparseEliminationLimits& limits) const {
    EliminationMeter(limits).check_kernel_coordinates(
        columns_, columns_ - reduction.pivot_columns.size());
    std::vector<std::vector<Element>> basis;
    basis.reserve(columns_ - reduction.pivot_columns.size());

    std::size_t next_pivot = 0;
    for (std::size_t free_column = 0; free_column < columns_; ++free_column) {
      if (next_pivot < reduction.pivot_columns.size() &&
          reduction.pivot_columns[next_pivot] == free_column) {
        ++next_pivot;
        continue;
      }

      std::vector<Element> vector(columns_, field_.zero());
      vector[free_column] = field_.one();
      for (std::size_t pivot = 0; pivot < reduction.pivot_columns.size();
           ++pivot) {
        if (const auto* value =
                coefficient_at(reduction.pivot_rows[pivot], free_column);
            value != nullptr) {
          vector[reduction.pivot_columns[pivot]] = field_.negate(*value);
        }
      }
      basis.push_back(std::move(vector));
    }
    return basis;
  }

  [[nodiscard]] std::vector<SparseKernelVector>
  sparse_kernel_from_reduction(
      const SparseRrefResult& reduction,
      const SparseEliminationLimits& limits) const {
    EliminationMeter meter(limits);
    std::vector<SparseKernelVector> basis;
    basis.reserve(columns_ - reduction.pivot_columns.size());
    std::size_t nonzeros = 0;

    std::size_t next_pivot = 0;
    for (std::size_t free_column = 0; free_column < columns_; ++free_column) {
      if (next_pivot < reduction.pivot_columns.size() &&
          reduction.pivot_columns[next_pivot] == free_column) {
        ++next_pivot;
        continue;
      }

      SparseKernelVector vector;
      vector.reserve(reduction.pivot_columns.size() + 1);
      for (std::size_t pivot = 0; pivot < reduction.pivot_columns.size();
           ++pivot) {
        if (const auto* value =
                coefficient_at(reduction.pivot_rows[pivot], free_column);
            value != nullptr) {
          vector.push_back(SparseKernelEntry{
              reduction.pivot_columns[pivot], field_.negate(*value)});
        }
      }
      vector.push_back(SparseKernelEntry{free_column, field_.one()});
      std::sort(vector.begin(), vector.end(), [](const auto& left,
                                                 const auto& right) {
        return left.column < right.column;
      });
      if (vector.size() > std::numeric_limits<std::size_t>::max() - nonzeros) {
        throw std::length_error("sparse kernel nonzero count overflows size_t");
      }
      nonzeros += vector.size();
      meter.check_kernel_nonzeros(nonzeros);
      basis.push_back(std::move(vector));
    }
    return basis;
  }

  Field field_;
  std::size_t rows_;
  std::size_t columns_;
  std::vector<std::size_t> row_offsets_;
  std::vector<std::size_t> column_indices_;
  std::vector<Element> values_;
};

}  // namespace laughableengine

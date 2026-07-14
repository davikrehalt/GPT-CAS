#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace laughableengine {

enum class IncrementalSparseRowSpaceResourceKind {
  LiveNonzeros,
  ArithmeticOperations,
  BasisVectors,
};

class IncrementalSparseRowSpaceResourceLimit : public std::runtime_error {
 public:
  IncrementalSparseRowSpaceResourceLimit(
      IncrementalSparseRowSpaceResourceKind kind,
      std::size_t limit,
      std::size_t observed)
      : std::runtime_error(make_message(kind, limit, observed)),
        kind_(kind),
        limit_(limit),
        observed_(observed) {}

  [[nodiscard]] IncrementalSparseRowSpaceResourceKind kind() const noexcept {
    return kind_;
  }
  [[nodiscard]] std::size_t limit() const noexcept { return limit_; }
  [[nodiscard]] std::size_t observed() const noexcept { return observed_; }

 private:
  [[nodiscard]] static const char* resource_name(
      IncrementalSparseRowSpaceResourceKind kind) noexcept {
    switch (kind) {
      case IncrementalSparseRowSpaceResourceKind::LiveNonzeros:
        return "live incremental-row-space coefficients";
      case IncrementalSparseRowSpaceResourceKind::ArithmeticOperations:
        return "incremental-row-space arithmetic operations";
      case IncrementalSparseRowSpaceResourceKind::BasisVectors:
        return "incremental-row-space basis vectors";
    }
    return "incremental-row-space resource";
  }

  [[nodiscard]] static std::string make_message(
      IncrementalSparseRowSpaceResourceKind kind,
      std::size_t limit,
      std::size_t observed) {
    return std::string("incremental sparse row-space resource limit exceeded for ") +
           resource_name(kind) + ": limit " + std::to_string(limit) +
           ", attempted " + std::to_string(observed);
  }

  IncrementalSparseRowSpaceResourceKind kind_;
  std::size_t limit_;
  std::size_t observed_;
};

struct IncrementalSparseRowSpaceLimits {
  // Includes stored input-basis vectors, normalized pivot rows, optional
  // coordinate rows, and the temporary rows used by the current operation.
  std::optional<std::size_t> max_live_nonzeros;

  // This is a lifetime total for one row-space object.  A failed operation
  // does not consume the budget.
  std::optional<std::size_t> max_arithmetic_operations;

  // Counts accepted input vectors, equivalently the current rank.
  std::optional<std::size_t> max_basis_vectors;
};

struct IncrementalSparseRowSpaceOptions {
  IncrementalSparseRowSpaceLimits limits;

  // Coordinate tracking records each normalized pivot row in the accepted
  // input basis.  Leave it disabled for large rank-only/normal-form jobs.
  bool track_insertion_coordinates = false;
};

// A deterministic, exact, incrementally constructed row space in k^dimension.
//
// Sparse vectors are sorted and canonical after every public operation.  The
// first surviving column is the pivot, so pivot choice does not depend on hash
// iteration order.  Accepted vectors retain stable insertion indices.  When
// coordinate tracking is enabled, normal_form_with_coordinates(v) returns
//
//       v = sum_i coordinates[i] * basis_vector(i) + normal_form.
//
// The object is deliberately mutable only through insertion and metered
// reduction.  All exposed stored vectors are const views.
template <typename Field>
class IncrementalSparseRowSpace {
 public:
  using Element = typename Field::Element;

  struct Entry {
    std::size_t column;
    Element value;

    friend bool operator==(const Entry&, const Entry&) = default;
  };

  using SparseVector = std::vector<Entry>;

  struct NormalFormWithCoordinates {
    SparseVector normal_form;
    // Columns are stable accepted-vector insertion indices.
    SparseVector insertion_coordinates;
  };

  struct InsertResult {
    bool inserted = false;
    // Populated exactly when inserted is true.
    std::optional<std::size_t> insertion_index;
    std::optional<std::size_t> pivot_column;
    // Populated for a dependent input exactly when coordinate tracking is on.
    // The input equals this combination of the accepted input basis.
    std::optional<SparseVector> dependence_coordinates;
  };

  IncrementalSparseRowSpace(
      Field field,
      std::size_t dimension,
      IncrementalSparseRowSpaceOptions options = {})
      : field_(std::move(field)),
        dimension_(dimension),
        options_(std::move(options)),
        row_for_pivot_(dimension_, no_row) {}

  [[nodiscard]] const Field& field() const noexcept { return field_; }
  [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
  [[nodiscard]] std::size_t rank() const noexcept {
    return basis_vectors_.size();
  }
  [[nodiscard]] bool tracks_insertion_coordinates() const noexcept {
    return options_.track_insertion_coordinates;
  }
  [[nodiscard]] const IncrementalSparseRowSpaceLimits& limits() const noexcept {
    return options_.limits;
  }

  [[nodiscard]] std::size_t basis_nonzero_count() const noexcept {
    return basis_nonzeros_;
  }
  [[nodiscard]] std::size_t pivot_row_nonzero_count() const noexcept {
    return pivot_row_nonzeros_;
  }
  [[nodiscard]] std::size_t coordinate_nonzero_count() const noexcept {
    return coordinate_nonzeros_;
  }
  [[nodiscard]] std::size_t live_nonzero_count() const noexcept {
    return persistent_nonzero_count();
  }
  [[nodiscard]] std::size_t arithmetic_operation_count() const noexcept {
    return arithmetic_operations_;
  }

  [[nodiscard]] const SparseVector& basis_vector(
      std::size_t insertion_index) const {
    check_basis_index(insertion_index);
    return basis_vectors_[insertion_index];
  }

  [[nodiscard]] std::size_t pivot_column(
      std::size_t insertion_index) const {
    check_basis_index(insertion_index);
    return pivot_columns_by_insertion_[insertion_index];
  }

  // This is a const audit view.  Reduction users normally need only
  // normal_form(), not the implementation row.
  [[nodiscard]] const SparseVector& normalized_pivot_row(
      std::size_t insertion_index) const {
    check_basis_index(insertion_index);
    return pivot_rows_[insertion_index];
  }

  [[nodiscard]] std::vector<std::size_t> pivot_columns() const {
    std::vector<std::size_t> result;
    result.reserve(rank());
    for (std::size_t column = 0; column < dimension_; ++column) {
      if (row_for_pivot_[column] != no_row) {
        result.push_back(column);
      }
    }
    return result;
  }

  [[nodiscard]] std::vector<std::size_t> nonpivot_columns() const {
    std::vector<std::size_t> result;
    result.reserve(dimension_ - rank());
    for (std::size_t column = 0; column < dimension_; ++column) {
      if (row_for_pivot_[column] == no_row) {
        result.push_back(column);
      }
    }
    return result;
  }

  // The argument is taken by value intentionally: callers may move a work row
  // in, while lvalue callers get the expected safe copy.
  [[nodiscard]] SparseVector normal_form(SparseVector vector) {
    Meter meter(*this, arithmetic_operations_);
    vector = canonicalize(std::move(vector), meter);
    auto reduction = reduce_canonical(
        std::move(vector), false, meter, 0);
    arithmetic_operations_ = meter.operations();
    return std::move(reduction.normal_form);
  }

  [[nodiscard]] NormalFormWithCoordinates normal_form_with_coordinates(
      SparseVector vector) {
    require_coordinate_tracking();
    Meter meter(*this, arithmetic_operations_);
    vector = canonicalize(std::move(vector), meter);
    auto reduction = reduce_canonical(
        std::move(vector), true, meter, 0);
    arithmetic_operations_ = meter.operations();
    return NormalFormWithCoordinates{
        std::move(reduction.normal_form),
        std::move(reduction.coordinates)};
  }

  // Read-only reductions are safe to run concurrently on a frozen row space.
  // Their arithmetic limit is applied per call, starting from zero, and they
  // do not consume the lifetime counter used by insert()/normal_form().
  [[nodiscard]] SparseVector normal_form_readonly(
      SparseVector vector) const {
    Meter meter(*this, 0);
    vector = canonicalize(std::move(vector), meter);
    auto reduction = reduce_canonical(
        std::move(vector), false, meter, 0);
    return std::move(reduction.normal_form);
  }

  [[nodiscard]] NormalFormWithCoordinates
  normal_form_with_coordinates_readonly(SparseVector vector) const {
    require_coordinate_tracking();
    Meter meter(*this, 0);
    vector = canonicalize(std::move(vector), meter);
    auto reduction = reduce_canonical(
        std::move(vector), true, meter, 0);
    return NormalFormWithCoordinates{
        std::move(reduction.normal_form),
        std::move(reduction.coordinates)};
  }

  [[nodiscard]] InsertResult insert(SparseVector vector) {
    Meter meter(*this, arithmetic_operations_);
    auto original = canonicalize(std::move(vector), meter);
    meter.check_live_extra(saturated_product(original.size(), 2));
    auto reduction = reduce_canonical(
        SparseVector(original), options_.track_insertion_coordinates,
        meter, original.size());

    if (reduction.normal_form.empty()) {
      arithmetic_operations_ = meter.operations();
      return InsertResult{
          false,
          std::nullopt,
          std::nullopt,
          options_.track_insertion_coordinates
              ? std::optional<SparseVector>(std::move(reduction.coordinates))
              : std::nullopt};
    }

    const auto insertion_index = rank();
    if (insertion_index == std::numeric_limits<std::size_t>::max()) {
      throw std::length_error(
          "incremental sparse row-space rank overflows size_t");
    }
    meter.check_basis_vectors(insertion_index + 1);

    const auto pivot = reduction.normal_form.front().column;
    if (row_for_pivot_[pivot] != no_row) {
      throw std::logic_error(
          "incremental sparse row-space reduction left an existing pivot");
    }

    meter.add_operations(1);
    const auto inverse = field_.inverse(reduction.normal_form.front().value);
    auto normalized_row = scale(
        reduction.normal_form, inverse, meter,
        original.size() + reduction.coordinates.size());

    SparseVector pivot_coordinates;
    if (options_.track_insertion_coordinates) {
      pivot_coordinates.reserve(reduction.coordinates.size() + 1);
      meter.add_operations(saturated_product(reduction.coordinates.size(), 2));
      for (const auto& entry : reduction.coordinates) {
        auto value = field_.negate(field_.multiply(inverse, entry.value));
        if (!field_.is_zero(value)) {
          pivot_coordinates.push_back(Entry{entry.column, std::move(value)});
        }
      }
      pivot_coordinates.push_back(Entry{insertion_index, inverse});
    }

    const auto added_nonzeros = saturated_add(
        original.size(),
        saturated_add(normalized_row.size(), pivot_coordinates.size()));
    meter.check_live_extra(saturated_add(
        added_nonzeros,
        saturated_add(
            reduction.normal_form.size(), reduction.coordinates.size())));
    const auto new_live_nonzeros =
        saturated_add(persistent_nonzero_count(), added_nonzeros);
    meter.check_live_total(new_live_nonzeros);

    // Reserve every outer vector before committing any logical mutation.
    // Moving the already-built sparse rows is then noexcept.
    basis_vectors_.reserve(insertion_index + 1);
    pivot_rows_.reserve(insertion_index + 1);
    pivot_coordinates_.reserve(insertion_index + 1);
    pivot_columns_by_insertion_.reserve(insertion_index + 1);

    basis_nonzeros_ = saturated_add(basis_nonzeros_, original.size());
    pivot_row_nonzeros_ =
        saturated_add(pivot_row_nonzeros_, normalized_row.size());
    coordinate_nonzeros_ =
        saturated_add(coordinate_nonzeros_, pivot_coordinates.size());
    basis_vectors_.push_back(std::move(original));
    pivot_rows_.push_back(std::move(normalized_row));
    pivot_coordinates_.push_back(std::move(pivot_coordinates));
    pivot_columns_by_insertion_.push_back(pivot);
    row_for_pivot_[pivot] = insertion_index;
    arithmetic_operations_ = meter.operations();

    return InsertResult{
        true, insertion_index, pivot, std::nullopt};
  }

 private:
  static constexpr std::size_t no_row =
      std::numeric_limits<std::size_t>::max();

  struct ReductionData {
    SparseVector normal_form;
    SparseVector coordinates;
  };

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

  class Meter {
   public:
    Meter(
        const IncrementalSparseRowSpace& space,
        std::size_t initial_operations)
        : space_(space), operations_(initial_operations) {}

    void add_operations(std::size_t count) {
      operations_ = saturated_add(operations_, count);
      check(
          IncrementalSparseRowSpaceResourceKind::ArithmeticOperations,
          operations_, space_.options_.limits.max_arithmetic_operations);
    }

    void check_live_extra(std::size_t extra) const {
      check_live_total(saturated_add(space_.persistent_nonzero_count(), extra));
    }

    void check_live_total(std::size_t observed) const {
      check(
          IncrementalSparseRowSpaceResourceKind::LiveNonzeros,
          observed, space_.options_.limits.max_live_nonzeros);
    }

    void check_basis_vectors(std::size_t observed) const {
      check(
          IncrementalSparseRowSpaceResourceKind::BasisVectors,
          observed, space_.options_.limits.max_basis_vectors);
    }

    [[nodiscard]] std::size_t operations() const noexcept {
      return operations_;
    }

   private:
    static void check(
        IncrementalSparseRowSpaceResourceKind kind,
        std::size_t observed,
        const std::optional<std::size_t>& limit) {
      if (limit.has_value() && observed > *limit) {
        throw IncrementalSparseRowSpaceResourceLimit(
            kind, *limit, observed);
      }
    }

    const IncrementalSparseRowSpace& space_;
    std::size_t operations_;
  };

  [[nodiscard]] std::size_t persistent_nonzero_count() const noexcept {
    return saturated_add(
        basis_nonzeros_,
        saturated_add(pivot_row_nonzeros_, coordinate_nonzeros_));
  }

  void check_basis_index(std::size_t insertion_index) const {
    if (insertion_index >= rank()) {
      throw std::out_of_range(
          "incremental sparse row-space basis index is out of range");
    }
  }

  void require_coordinate_tracking() const {
    if (!options_.track_insertion_coordinates) {
      throw std::logic_error(
          "incremental sparse row-space coordinate tracking is disabled");
    }
  }

  [[nodiscard]] SparseVector canonicalize(
      SparseVector vector,
      Meter& meter) const {
    meter.check_live_extra(vector.size());
    for (auto& entry : vector) {
      if (entry.column >= dimension_) {
        throw std::out_of_range(
            "sparse row-space entry is outside the ambient dimension");
      }
      entry.value = field_.canonical(std::move(entry.value));
    }
    std::sort(vector.begin(), vector.end(), [](const Entry& left,
                                                const Entry& right) {
      return left.column < right.column;
    });

    std::size_t read = 0;
    std::size_t write = 0;
    while (read < vector.size()) {
      const auto column = vector[read].column;
      auto value = std::move(vector[read].value);
      ++read;
      while (read < vector.size() && vector[read].column == column) {
        meter.add_operations(1);
        value = field_.add(value, vector[read].value);
        ++read;
      }
      if (!field_.is_zero(value)) {
        vector[write] = Entry{column, std::move(value)};
        ++write;
      }
    }
    vector.erase(
        vector.begin() + static_cast<std::ptrdiff_t>(write), vector.end());
    meter.check_live_extra(vector.size());
    return vector;
  }

  [[nodiscard]] ReductionData reduce_canonical(
      SparseVector vector,
      bool track_coordinates,
      Meter& meter,
      std::size_t extra_live) const {
    SparseVector coordinates;
    std::size_t scan = 0;
    while (scan < vector.size()) {
      const auto pivot = vector[scan].column;
      const auto row_index = row_for_pivot_[pivot];
      if (row_index == no_row) {
        ++scan;
        continue;
      }

      const auto factor = vector[scan].value;
      vector = add_scaled(
          vector, pivot_rows_[row_index], field_.negate(factor), meter,
          saturated_add(extra_live, coordinates.size()));
      if (track_coordinates) {
        coordinates = add_scaled(
            coordinates, pivot_coordinates_[row_index], factor, meter,
            saturated_add(extra_live, vector.size()));
      }
      // The normalized pivot cancels exactly.  Entries before scan are
      // untouched and no new entry can be introduced before this pivot.
    }
    meter.check_live_extra(saturated_add(
        extra_live, saturated_add(vector.size(), coordinates.size())));
    return ReductionData{std::move(vector), std::move(coordinates)};
  }

  [[nodiscard]] SparseVector add_scaled(
      const SparseVector& target,
      const SparseVector& source,
      const Element& factor,
      Meter& meter,
      std::size_t other_live) const {
    if (field_.is_zero(factor) || source.empty()) {
      return target;
    }
    const auto maximum_size = saturated_add(target.size(), source.size());
    meter.add_operations(saturated_product(source.size(), 2));
    meter.check_live_extra(saturated_add(
        other_live, saturated_add(target.size(), maximum_size)));

    SparseVector result;
    result.reserve(maximum_size);
    std::size_t target_index = 0;
    std::size_t source_index = 0;
    while (target_index < target.size() || source_index < source.size()) {
      if (source_index == source.size() ||
          (target_index < target.size() &&
           target[target_index].column < source[source_index].column)) {
        result.push_back(target[target_index]);
        ++target_index;
        continue;
      }

      const auto column = source[source_index].column;
      auto scaled = field_.multiply(factor, source[source_index].value);
      if (target_index == target.size() ||
          column < target[target_index].column) {
        if (!field_.is_zero(scaled)) {
          result.push_back(Entry{column, std::move(scaled)});
        }
        ++source_index;
        continue;
      }

      auto value = field_.add(target[target_index].value, scaled);
      if (!field_.is_zero(value)) {
        result.push_back(Entry{column, std::move(value)});
      }
      ++target_index;
      ++source_index;
    }
    return result;
  }

  [[nodiscard]] SparseVector scale(
      const SparseVector& vector,
      const Element& factor,
      Meter& meter,
      std::size_t other_live) const {
    meter.add_operations(vector.size());
    meter.check_live_extra(saturated_add(
        other_live, saturated_product(vector.size(), 2)));
    SparseVector result;
    result.reserve(vector.size());
    for (const auto& entry : vector) {
      auto value = field_.multiply(factor, entry.value);
      if (!field_.is_zero(value)) {
        result.push_back(Entry{entry.column, std::move(value)});
      }
    }
    return result;
  }

  Field field_;
  std::size_t dimension_;
  IncrementalSparseRowSpaceOptions options_;

  // All four vectors below are indexed by stable accepted-vector insertion
  // index.  row_for_pivot_ maps ambient columns back to that index.
  std::vector<SparseVector> basis_vectors_;
  std::vector<SparseVector> pivot_rows_;
  std::vector<SparseVector> pivot_coordinates_;
  std::vector<std::size_t> pivot_columns_by_insertion_;
  std::vector<std::size_t> row_for_pivot_;

  std::size_t basis_nonzeros_ = 0;
  std::size_t pivot_row_nonzeros_ = 0;
  std::size_t coordinate_nonzeros_ = 0;
  std::size_t arithmetic_operations_ = 0;
};

}  // namespace laughableengine

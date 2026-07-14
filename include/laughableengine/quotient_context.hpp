#pragma once

#include <cstddef>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "laughableengine/ideal.hpp"
#include "laughableengine/linear_algebra.hpp"
#include "laughableengine/sparse_matrix.hpp"

namespace laughableengine {

template <typename Field>
struct QuotientReductionBatch {
  using Element = typename Field::Element;
  using PolynomialType = Polynomial<Field>;

  std::vector<PolynomialType> remainders;
  std::vector<std::vector<Element>> coordinates;
};

namespace quotient_context_detail {

struct MonomialHash {
  [[nodiscard]] std::size_t operator()(const Monomial& monomial) const noexcept {
    std::size_t value = static_cast<std::size_t>(1469598103934665603ULL);
    for (const auto exponent : monomial.exponents()) {
      value ^= exponent;
      value *= static_cast<std::size_t>(1099511628211ULL);
    }
    return value;
  }
};

}  // namespace quotient_context_detail

// A reusable exact context for a finite quotient P/I. The ideal, reducer,
// standard-monomial basis, and coordinate indexing remain tied to the exact
// polynomial-ring context that created them. Matrices use the convention that
// column j is the coordinate vector of the image of basis element j.
template <typename Field>
class QuotientContext {
 private:
  struct Initialization {
    Ideal<Field> ideal;
    StandardMonomialBasis<Field> standard_basis;
  };

 public:
  using Element = typename Field::Element;
  using PolynomialType = Polynomial<Field>;
  using Ring = PolynomialRing<Field>;
  using CoordinateVector = std::vector<Element>;
  using BatchResult = QuotientReductionBatch<Field>;

  explicit QuotientContext(
      Ideal<Field> ideal,
      const StandardMonomialLimits& limits = {})
      : QuotientContext(initialize(std::move(ideal), limits)) {}

  [[nodiscard]] const Ideal<Field>& ideal() const noexcept { return ideal_; }
  [[nodiscard]] const Ring& ring() const noexcept { return ideal_.ring(); }
  [[nodiscard]] const Field& field() const noexcept { return ring().field(); }

  [[nodiscard]] std::size_t dimension() const noexcept {
    return standard_basis_.size();
  }
  [[nodiscard]] bool is_zero_quotient() const noexcept {
    return dimension() == 0;
  }

  [[nodiscard]] const StandardMonomialBasis<Field>& standard_basis() const
      noexcept {
    return standard_basis_;
  }
  [[nodiscard]] const std::vector<Monomial>& standard_monomials() const
      noexcept {
    return standard_basis_.monomials();
  }
  [[nodiscard]] const std::vector<PolynomialType>& polynomial_basis() const
      noexcept {
    return polynomial_basis_;
  }

  [[nodiscard]] PolynomialType remainder(
      const PolynomialType& polynomial,
      const ReductionLimits& limits = {}) const {
    if (!use_compiled_tables(limits)) {
      return ideal_.normal_form(polynomial, limits);
    }
    CoordinateCache cache;
    return representative(coordinates_from_tables(polynomial, cache));
  }

  [[nodiscard]] std::vector<PolynomialType> remainders(
      std::span<const PolynomialType> polynomials,
      const ReductionLimits& limits = {}) const {
    return reduce_batch(polynomials, limits).remainders;
  }

  [[nodiscard]] std::vector<PolynomialType> remainders(
      const std::vector<PolynomialType>& polynomials,
      const ReductionLimits& limits = {}) const {
    return remainders(
        std::span<const PolynomialType>(polynomials.data(), polynomials.size()),
        limits);
  }

  [[nodiscard]] std::vector<PolynomialType> remainders(
      std::initializer_list<PolynomialType> polynomials,
      const ReductionLimits& limits = {}) const {
    return remainders(
        std::span<const PolynomialType>(polynomials.begin(),
                                        polynomials.size()),
        limits);
  }

  // One compiled batch reduction produces both outputs. Coordinates are read
  // directly from the returned normal forms; they are not reduced a second
  // time through the scalar API.
  [[nodiscard]] BatchResult reduce_batch(
      std::span<const PolynomialType> polynomials,
      const ReductionLimits& limits = {}) const {
    if (!use_compiled_tables(limits)) {
      auto reduced = ideal_.normal_forms(polynomials, limits);
      std::vector<CoordinateVector> coordinates;
      coordinates.reserve(reduced.size());
      for (const auto& remainder : reduced) {
        coordinates.push_back(coordinates_of_remainder(remainder));
      }
      return BatchResult{std::move(reduced), std::move(coordinates)};
    }

    CoordinateCache cache;
    std::vector<PolynomialType> reduced;
    std::vector<CoordinateVector> coordinates;
    reduced.reserve(polynomials.size());
    coordinates.reserve(polynomials.size());
    for (const auto& polynomial : polynomials) {
      auto coordinate_vector = coordinates_from_tables(polynomial, cache);
      reduced.push_back(representative(coordinate_vector));
      coordinates.push_back(std::move(coordinate_vector));
    }
    return BatchResult{std::move(reduced), std::move(coordinates)};
  }

  [[nodiscard]] BatchResult reduce_batch(
      const std::vector<PolynomialType>& polynomials,
      const ReductionLimits& limits = {}) const {
    return reduce_batch(
        std::span<const PolynomialType>(polynomials.data(), polynomials.size()),
        limits);
  }

  [[nodiscard]] BatchResult reduce_batch(
      std::initializer_list<PolynomialType> polynomials,
      const ReductionLimits& limits = {}) const {
    return reduce_batch(
        std::span<const PolynomialType>(polynomials.begin(),
                                        polynomials.size()),
        limits);
  }

  [[nodiscard]] CoordinateVector coordinates(
      const PolynomialType& polynomial,
      const ReductionLimits& limits = {}) const {
    if (!use_compiled_tables(limits)) {
      const auto reduced = ideal_.normal_form(polynomial, limits);
      return coordinates_of_remainder(reduced);
    }
    CoordinateCache cache;
    return coordinates_from_tables(polynomial, cache);
  }

  [[nodiscard]] std::vector<CoordinateVector> coordinates(
      std::span<const PolynomialType> polynomials,
      const ReductionLimits& limits = {}) const {
    return reduce_batch(polynomials, limits).coordinates;
  }

  [[nodiscard]] std::vector<CoordinateVector> coordinates(
      const std::vector<PolynomialType>& polynomials,
      const ReductionLimits& limits = {}) const {
    return coordinates(
        std::span<const PolynomialType>(polynomials.data(), polynomials.size()),
        limits);
  }

  [[nodiscard]] std::vector<CoordinateVector> coordinates(
      std::initializer_list<PolynomialType> polynomials,
      const ReductionLimits& limits = {}) const {
    return coordinates(
        std::span<const PolynomialType>(polynomials.begin(),
                                        polynomials.size()),
        limits);
  }

  [[nodiscard]] PolynomialType representative(
      std::span<const Element> coordinates) const {
    return standard_basis_.from_coordinates(coordinates);
  }

  [[nodiscard]] PolynomialType representative(
      const CoordinateVector& coordinates) const {
    return representative(
        std::span<const Element>(coordinates.data(), coordinates.size()));
  }

  [[nodiscard]] DenseMatrix<Field> dense_multiplication_matrix(
      const PolynomialType& polynomial,
      const ReductionLimits& limits = {}) const {
    const auto columns = multiplication_columns(polynomial, limits);
    return matrix_from_columns(field(), dimension(), columns);
  }

  [[nodiscard]] DenseMatrix<Field> multiplication_matrix(
      const PolynomialType& polynomial,
      const ReductionLimits& limits = {}) const {
    return dense_multiplication_matrix(polynomial, limits);
  }

  [[nodiscard]] SparseMatrix<Field> sparse_multiplication_matrix(
      const PolynomialType& polynomial,
      const ReductionLimits& limits = {}) const {
    std::vector<typename SparseMatrix<Field>::Triplet> triplets;
    const auto reduced_polynomial = remainder(polynomial, limits);
    if (!reduced_polynomial.is_zero()) {
      std::vector<PolynomialType> products;
      products.reserve(dimension());
      for (const auto& basis_element : polynomial_basis_) {
        products.push_back(basis_element * reduced_polynomial);
      }
      const auto remainders = this->remainders(products, limits);
      for (std::size_t column = 0; column < remainders.size(); ++column) {
        for (const auto& term : remainders[column].terms()) {
          const auto iterator = coordinate_indices_.find(term.monomial);
          if (iterator == coordinate_indices_.end()) {
            throw std::logic_error(
                "internal error: quotient normal form is outside its basis");
          }
          if (!field().is_zero(term.coefficient)) {
            triplets.push_back(typename SparseMatrix<Field>::Triplet{
                iterator->second, column, term.coefficient});
          }
        }
      }
    }
    return SparseMatrix<Field>(
        field(), dimension(), dimension(), std::move(triplets));
  }

  [[nodiscard]] DenseMatrix<Field> dense_variable_multiplication_matrix(
      std::size_t variable,
      const ReductionLimits& limits = {}) const {
    if (variable >= ring().variable_count()) {
      throw std::out_of_range("quotient variable index is out of range");
    }
    if (!use_compiled_tables(limits)) {
      return dense_multiplication_matrix(ring().gen(variable), limits);
    }
    return variable_multiplication_tables_[variable].to_dense();
  }

  [[nodiscard]] DenseMatrix<Field> dense_variable_multiplication_matrix(
      std::string_view variable,
      const ReductionLimits& limits = {}) const {
    const auto& names = ring().variable_names();
    const auto iterator = std::find(names.begin(), names.end(), variable);
    if (iterator == names.end()) {
      throw std::invalid_argument(
          "unknown polynomial variable: " + std::string(variable));
    }
    return dense_variable_multiplication_matrix(
        static_cast<std::size_t>(iterator - names.begin()), limits);
  }

  [[nodiscard]] DenseMatrix<Field> variable_multiplication_matrix(
      std::size_t variable,
      const ReductionLimits& limits = {}) const {
    return dense_variable_multiplication_matrix(variable, limits);
  }

  [[nodiscard]] DenseMatrix<Field> variable_multiplication_matrix(
      std::string_view variable,
      const ReductionLimits& limits = {}) const {
    return dense_variable_multiplication_matrix(variable, limits);
  }

  [[nodiscard]] SparseMatrix<Field> sparse_variable_multiplication_matrix(
      std::size_t variable,
      const ReductionLimits& limits = {}) const {
    if (variable >= ring().variable_count()) {
      throw std::out_of_range("quotient variable index is out of range");
    }
    if (!use_compiled_tables(limits)) {
      return sparse_multiplication_matrix(ring().gen(variable), limits);
    }
    return variable_multiplication_tables_[variable];
  }

  [[nodiscard]] SparseMatrix<Field> sparse_variable_multiplication_matrix(
      std::string_view variable,
      const ReductionLimits& limits = {}) const {
    const auto& names = ring().variable_names();
    const auto iterator = std::find(names.begin(), names.end(), variable);
    if (iterator == names.end()) {
      throw std::invalid_argument(
          "unknown polynomial variable: " + std::string(variable));
    }
    return sparse_variable_multiplication_matrix(
        static_cast<std::size_t>(iterator - names.begin()), limits);
  }

  // These are the compiled border-rewrite tables. Column j of table i is the
  // normal form of x_i times the j-th standard monomial. They are built once
  // per quotient and are then reused by every unbounded batch operation.
  [[nodiscard]] const std::vector<SparseMatrix<Field>>&
  variable_multiplication_tables() const noexcept {
    return variable_multiplication_tables_;
  }

  [[nodiscard]] bool supported_at_origin() const noexcept {
    return supported_at_origin_;
  }

  // For a nonzero quotient, the value is the least positive n with x_i^n=0.
  // A missing value means x_i is not nilpotent. In the unit quotient 1=0, so
  // the exact convention is nilpotency index zero for every variable.
  [[nodiscard]] const std::vector<std::optional<std::size_t>>&
  variable_nilpotency_indices() const noexcept {
    return nilpotency_indices_;
  }

  [[nodiscard]] std::optional<std::size_t> variable_nilpotency_index(
      std::size_t variable) const {
    if (variable >= nilpotency_indices_.size()) {
      throw std::out_of_range("quotient variable index is out of range");
    }
    return nilpotency_indices_[variable];
  }

  [[nodiscard]] std::optional<std::size_t> variable_nilpotency_index(
      std::string_view variable) const {
    const auto& names = ring().variable_names();
    for (std::size_t index = 0; index < names.size(); ++index) {
      if (names[index] == variable) {
        return nilpotency_indices_[index];
      }
    }
    throw std::invalid_argument(
        "unknown polynomial variable: " + std::string(variable));
  }

 private:
  explicit QuotientContext(Initialization initialization)
      : ideal_(std::move(initialization.ideal)),
        standard_basis_(std::move(initialization.standard_basis)),
        polynomial_basis_(standard_basis_.polynomials()) {
    coordinate_indices_.reserve(standard_basis_.size());
    for (std::size_t index = 0; index < standard_basis_.size(); ++index) {
      coordinate_indices_.emplace(
          standard_basis_.monomials()[index], index);
    }
    if (dimension() != 0) {
      const auto iterator = coordinate_indices_.find(Monomial{});
      if (iterator == coordinate_indices_.end()) {
        throw std::logic_error(
            "internal error: a nonzero quotient basis omits one");
      }
      one_index_ = iterator->second;
    }
    if constexpr (std::is_same_v<Field, PrimeField>) {
      variable_multiplication_tables_ =
          build_variable_multiplication_tables();
    }
    nilpotency_indices_ = compute_nilpotency_indices();
    supported_at_origin_ = true;
    for (const auto index : nilpotency_indices_) {
      if (!index.has_value()) {
        supported_at_origin_ = false;
        break;
      }
    }
  }

  [[nodiscard]] static Initialization initialize(
      Ideal<Field> ideal,
      const StandardMonomialLimits& limits) {
    if (!ideal.is_zero_dimensional()) {
      throw InfiniteQuotientError();
    }
    auto standard_basis = ideal.standard_monomials(limits);
    return Initialization{
        std::move(ideal), std::move(standard_basis)};
  }

  [[nodiscard]] CoordinateVector coordinates_of_remainder(
      const PolynomialType& remainder) const {
    CoordinateVector coordinates(dimension(), field().zero());
    for (const auto& term : remainder.terms()) {
      const auto iterator = coordinate_indices_.find(term.monomial);
      if (iterator == coordinate_indices_.end()) {
        throw std::logic_error(
            "internal error: quotient normal form is outside its basis");
      }
      coordinates[iterator->second] = term.coefficient;
    }
    return coordinates;
  }

  using CoordinateCache = std::unordered_map<
      Monomial,
      CoordinateVector,
      quotient_context_detail::MonomialHash>;

  [[nodiscard]] static bool has_explicit_limits(
      const ReductionLimits& limits) noexcept {
    return limits.max_steps.has_value() ||
           limits.max_live_terms.has_value() ||
           limits.max_batch_steps.has_value();
  }

  [[nodiscard]] static bool use_compiled_tables(
      const ReductionLimits& limits) noexcept {
    if constexpr (std::is_same_v<Field, PrimeField>) {
      return !has_explicit_limits(limits);
    }
    return false;
  }

  void require_polynomial(const PolynomialType& polynomial) const {
    if (!ring().zero().same_ring(polynomial)) {
      throw std::invalid_argument(
          "quotient reduction requires its exact polynomial-ring context");
    }
  }

  [[nodiscard]] CoordinateVector monomial_coordinates_from_tables(
      const Monomial& monomial,
      CoordinateCache& cache) const {
    CoordinateVector zero(dimension(), field().zero());
    if (dimension() == 0) {
      return zero;
    }
    const auto standard = coordinate_indices_.find(monomial);
    if (standard != coordinate_indices_.end()) {
      zero[standard->second] = field().one();
      return zero;
    }
    const auto cached = cache.find(monomial);
    if (cached != cache.end()) {
      return cached->second;
    }

    CoordinateVector result(dimension(), field().zero());
    result[one_index_] = field().one();
    for (std::size_t variable = 0; variable < ring().variable_count();
         ++variable) {
      for (std::uint16_t exponent = 0;
           exponent < monomial.exponent(variable); ++exponent) {
        result = variable_multiplication_tables_[variable].multiply_column(
            result);
      }
    }
    cache.emplace(monomial, result);
    return result;
  }

  [[nodiscard]] CoordinateVector coordinates_from_tables(
      const PolynomialType& polynomial,
      CoordinateCache& cache) const {
    require_polynomial(polynomial);
    CoordinateVector result(dimension(), field().zero());
    for (const auto& term : polynomial.terms()) {
      const auto monomial_coordinates =
          monomial_coordinates_from_tables(term.monomial, cache);
      for (std::size_t row = 0; row < dimension(); ++row) {
        result[row] = field().add(
            result[row],
            field().multiply(term.coefficient, monomial_coordinates[row]));
      }
    }
    return result;
  }

  [[nodiscard]] std::vector<SparseMatrix<Field>>
  build_variable_multiplication_tables() const {
    using Triplet = typename SparseMatrix<Field>::Triplet;
    struct BorderEntry {
      std::size_t variable;
      std::size_t column;
      PolynomialType polynomial;
    };

    std::vector<std::vector<Triplet>> triplets(ring().variable_count());
    std::vector<BorderEntry> border;
    for (std::size_t variable = 0; variable < ring().variable_count();
         ++variable) {
      const auto variable_monomial =
          ring().gen(variable).leading_term()->monomial;
      for (std::size_t column = 0; column < dimension(); ++column) {
        const auto child = standard_monomials()[column].multiplied_by(
            variable_monomial, ring().variable_count());
        const auto standard = coordinate_indices_.find(child);
        if (standard != coordinate_indices_.end()) {
          triplets[variable].push_back(
              Triplet{standard->second, column, field().one()});
        } else {
          border.push_back(BorderEntry{
              variable, column,
              ring().zero().term_like(field().one(), child)});
        }
      }
    }

    std::vector<PolynomialType> border_polynomials;
    border_polynomials.reserve(border.size());
    for (const auto& entry : border) {
      border_polynomials.push_back(entry.polynomial);
    }
    const auto border_remainders = ideal_.normal_forms(border_polynomials);
    for (std::size_t index = 0; index < border.size(); ++index) {
      for (const auto& term : border_remainders[index].terms()) {
        const auto coordinate = coordinate_indices_.find(term.monomial);
        if (coordinate == coordinate_indices_.end()) {
          throw std::logic_error(
              "internal error: a border rewrite is outside the quotient basis");
        }
        triplets[border[index].variable].push_back(Triplet{
            coordinate->second, border[index].column, term.coefficient});
      }
    }

    std::vector<SparseMatrix<Field>> result;
    result.reserve(ring().variable_count());
    for (auto& entries : triplets) {
      result.emplace_back(
          field(), dimension(), dimension(), std::move(entries));
    }
    return result;
  }

  [[nodiscard]] std::vector<CoordinateVector> multiplication_columns(
      const PolynomialType& polynomial,
      const ReductionLimits& limits) const {
    const auto reduced_polynomial = remainder(polynomial, limits);
    std::vector<CoordinateVector> columns;
    columns.reserve(dimension());
    if (reduced_polynomial.is_zero()) {
      for (std::size_t column = 0; column < dimension(); ++column) {
        columns.emplace_back(dimension(), field().zero());
      }
      return columns;
    }

    std::vector<PolynomialType> products;
    products.reserve(dimension());
    for (const auto& basis_element : polynomial_basis_) {
      products.push_back(basis_element * reduced_polynomial);
    }
    return reduce_batch(products, limits).coordinates;
  }

  [[nodiscard]] std::vector<std::optional<std::size_t>>
  compute_nilpotency_indices() const {
    std::vector<std::optional<std::size_t>> result;
    result.reserve(ring().variable_count());
    if (dimension() == 0) {
      result.assign(ring().variable_count(), std::size_t{0});
      return result;
    }

    for (std::size_t variable = 0; variable < ring().variable_count();
         ++variable) {
      auto power = ring().one();
      const auto generator = ring().gen(variable);
      std::optional<std::size_t> nilpotency_index;
      for (std::size_t exponent = 1; exponent <= dimension(); ++exponent) {
        power = ideal_.normal_form(power * generator);
        if (power.is_zero()) {
          nilpotency_index = exponent;
          break;
        }
      }
      result.push_back(nilpotency_index);
    }
    return result;
  }

  Ideal<Field> ideal_;
  StandardMonomialBasis<Field> standard_basis_;
  std::vector<PolynomialType> polynomial_basis_;
  std::unordered_map<
      Monomial,
      std::size_t,
      quotient_context_detail::MonomialHash>
      coordinate_indices_;
  std::size_t one_index_ = 0;
  std::vector<SparseMatrix<Field>> variable_multiplication_tables_;
  std::vector<std::optional<std::size_t>> nilpotency_indices_;
  bool supported_at_origin_ = false;
};

}  // namespace laughableengine

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/incremental_sparse_row_space.hpp"
#include "laughableengine/polynomial.hpp"
#include "laughableengine/sparse_matrix.hpp"
#include "laughableengine/truncated_monomial_space.hpp"

namespace laughableengine {

// This file implements one deliberately narrow presentation:
//
//   J = (generators) + m^N,       Q = P/J.
//
// Since m^(2N) is contained in J^2, every calculation is exact in the finite
// vector space P/m^(2N).  No Groebner basis of J^2 is constructed.

enum class CotangentH1InputIssue {
  EmptyGeneratorRingMismatch,
  GeneratorRingMismatch,
  GeneratorHasConstantTerm,
  InvalidMaximalPower,
};

class CotangentH1InputError : public std::domain_error {
 public:
  explicit CotangentH1InputError(CotangentH1InputIssue issue)
      : std::domain_error(message(issue)), issue_(issue) {}

  [[nodiscard]] CotangentH1InputIssue issue() const noexcept { return issue_; }

 private:
  [[nodiscard]] static const char* message(
      CotangentH1InputIssue issue) noexcept {
    switch (issue) {
      case CotangentH1InputIssue::EmptyGeneratorRingMismatch:
        return "internal cotangent-H1 ring probe failed";
      case CotangentH1InputIssue::GeneratorRingMismatch:
        return "every cotangent-H1 generator must belong to the supplied ring";
      case CotangentH1InputIssue::GeneratorHasConstantTerm:
        return "cotangent-H1 generators must vanish at the origin";
      case CotangentH1InputIssue::InvalidMaximalPower:
        return "maximal_power must be positive and its double must fit the exact degree limit";
    }
    return "invalid cotangent-H1 input";
  }

  CotangentH1InputIssue issue_;
};

class CotangentH1ResourceLimit : public std::runtime_error {
 public:
  explicit CotangentH1ResourceLimit(std::string detail)
      : std::runtime_error(
            "cotangent-H1 resource limit exceeded: " + std::move(detail)) {}
};

struct CotangentH1Options {
  TruncatedMonomialSpaceLimits monomial_space_limits{};
  IncrementalSparseRowSpaceLimits row_space_limits{};
  SparseEliminationLimits matrix_elimination_limits{};

  // Counts seeds and closure products offered to either incremental row
  // space. nullopt disables this guard.
  std::optional<std::size_t> max_generated_rows = 2'000'000;

  // Counts triplets before duplicate coalescing. nullopt disables this guard.
  std::optional<std::size_t> max_matrix_triplets = 20'000'000;

  // d(J^2) subset J is a mathematical invariant and also catches a missing
  // cross term in the truncated square construction.
  bool verify_square_derivatives = true;
};

template <typename Field>
struct CotangentH1Spec {
  PolynomialRing<Field> ring;
  std::vector<Polynomial<Field>> generators;
  std::size_t maximal_power;
};

enum class CotangentClassStatus {
  Complete,
  NotInIdeal,
  NotCycle,
};

template <typename Field>
struct CotangentClassProof {
  using Element = typename Field::Element;
  using PolynomialType = Polynomial<Field>;
  using SparseEntry = typename IncrementalSparseRowSpace<Field>::Entry;

  CotangentClassStatus status;
  PolynomialType representative;
  bool in_ideal;
  std::vector<PolynomialType> derivative_remainders;
  bool cycle_valid;

  // Coordinates in the standard monomial basis of P/J^2.  The image is in
  // J/J^2 when status is Complete.
  std::optional<std::vector<SparseEntry>> class_coordinates;
  std::optional<SparseMatrix<Field>> multiplication_matrix;
  std::optional<std::size_t> multiplication_rank;
  std::optional<std::size_t> annihilator_dimension;

  // Each dense vector is in the documented standard-monomial basis of Q.
  std::vector<std::vector<Element>> annihilator_coordinates;
  // Polynomial lifts whose residue classes generate Ann_Q([g]).
  std::vector<PolynomialType> annihilator_basis;
  // Generators in P for the full preimage J^2:g.  This is J's displayed
  // generator set followed by the annihilator lifts above.
  std::vector<PolynomialType> colon_generators;

  bool faithful = false;
  bool colon_equals_ideal = false;

  [[nodiscard]] bool conclusive() const noexcept { return true; }
};

namespace cotangent_h1_detail {

template <typename Field>
using RowSpace = IncrementalSparseRowSpace<Field>;

template <typename Field>
using SparseVector = typename RowSpace<Field>::SparseVector;

inline constexpr std::size_t no_coordinate =
    std::numeric_limits<std::size_t>::max();

inline void check_increment(
    std::size_t& counter,
    const std::optional<std::size_t>& limit,
    const char* description) {
  if (counter == std::numeric_limits<std::size_t>::max()) {
    throw CotangentH1ResourceLimit(
        std::string(description) + " count overflows size_t");
  }
  ++counter;
  if (limit.has_value() && counter > *limit) {
    throw CotangentH1ResourceLimit(
        std::string(description) + " count " + std::to_string(counter) +
        " exceeds configured limit " + std::to_string(*limit));
  }
}

inline void check_triplets(
    std::size_t current,
    std::size_t additional,
    const CotangentH1Options& options) {
  if (additional > std::numeric_limits<std::size_t>::max() - current) {
    throw CotangentH1ResourceLimit("matrix triplet count overflows size_t");
  }
  const auto total = current + additional;
  if (options.max_matrix_triplets.has_value() &&
      total > *options.max_matrix_triplets) {
    throw CotangentH1ResourceLimit(
        "matrix construction needs more than " +
        std::to_string(*options.max_matrix_triplets) + " triplets");
  }
}

template <typename Field>
[[nodiscard]] SparseVector<Field> polynomial_row(
    const Polynomial<Field>& polynomial,
    const TruncatedMonomialSpace& monomials) {
  SparseVector<Field> result;
  result.reserve(polynomial.term_count());
  for (const auto& term : polynomial.terms()) {
    if (const auto index = monomials.find_index(term.monomial);
        index.has_value()) {
      result.push_back({*index, term.coefficient});
    }
  }
  return result;
}

template <typename Field>
[[nodiscard]] Polynomial<Field> polynomial_from_ambient_row(
    const PolynomialRing<Field>& ring,
    const TruncatedMonomialSpace& monomials,
    std::span<const typename RowSpace<Field>::Entry> row) {
  using TermSpec = typename PolynomialRing<Field>::TermSpec;
  std::vector<TermSpec> terms;
  terms.reserve(row.size());
  for (const auto& entry : row) {
    const auto& monomial = monomials.monomial(entry.column);
    std::vector<std::uint16_t> exponents(
        monomial.exponents().begin(),
        monomial.exponents().begin() +
            static_cast<std::ptrdiff_t>(ring.variable_count()));
    terms.push_back(TermSpec{entry.value, std::move(exponents)});
  }
  return ring.from_terms(std::move(terms));
}

template <typename Field>
[[nodiscard]] SparseVector<Field> multiply_rows(
    const Field& field,
    const TruncatedMonomialSpace& monomials,
    std::span<const typename RowSpace<Field>::Entry> left,
    std::span<const typename RowSpace<Field>::Entry> right) {
  SparseVector<Field> result;
  if (right.size() != 0 &&
      left.size() > std::numeric_limits<std::size_t>::max() / right.size()) {
    throw CotangentH1ResourceLimit("sparse polynomial product overflows size_t");
  }
  result.reserve(left.size() * right.size());
  for (const auto& left_entry : left) {
    for (const auto& right_entry : right) {
      if (const auto product = monomials.product_index(
              left_entry.column, right_entry.column);
          product.has_value()) {
        result.push_back({
            *product,
            field.multiply(left_entry.value, right_entry.value)});
      }
    }
  }
  return result;
}

template <typename Field>
[[nodiscard]] SparseVector<Field> canonicalize_row(
    const Field& field,
    SparseVector<Field> row) {
  std::sort(row.begin(), row.end(), [](const auto& left, const auto& right) {
    return left.column < right.column;
  });
  SparseVector<Field> result;
  result.reserve(row.size());
  for (auto& entry : row) {
    if (!result.empty() && result.back().column == entry.column) {
      result.back().value = field.add(result.back().value, entry.value);
      if (field.is_zero(result.back().value)) {
        result.pop_back();
      }
    } else if (!field.is_zero(entry.value)) {
      result.push_back(std::move(entry));
    }
  }
  return result;
}

template <typename Field>
[[nodiscard]] SparseVector<Field> multiply_row_by_monomial(
    const TruncatedMonomialSpace& monomials,
    std::span<const typename RowSpace<Field>::Entry> row,
    std::size_t monomial) {
  SparseVector<Field> result;
  result.reserve(row.size());
  for (const auto& entry : row) {
    if (const auto product = monomials.product_index(entry.column, monomial);
        product.has_value()) {
      result.push_back({*product, entry.value});
    }
  }
  return result;
}

template <typename Field>
[[nodiscard]] SparseVector<Field> multiply_row_by_variable(
    const TruncatedMonomialSpace& monomials,
    std::span<const typename RowSpace<Field>::Entry> row,
    std::size_t variable) {
  SparseVector<Field> result;
  result.reserve(row.size());
  for (const auto& entry : row) {
    if (const auto product =
            monomials.multiply_by_variable(entry.column, variable);
        product.has_value()) {
      result.push_back({*product, entry.value});
    }
  }
  return result;
}

template <typename Field>
[[nodiscard]] SparseVector<Field> derivative_row(
    const Field& field,
    const TruncatedMonomialSpace& source,
    const TruncatedMonomialSpace& target,
    std::span<const typename RowSpace<Field>::Entry> row,
    std::size_t variable) {
  SparseVector<Field> result;
  result.reserve(row.size());
  for (const auto& entry : row) {
    const auto derivative = source.differentiate(entry.column, variable);
    if (!derivative.has_value()) {
      continue;
    }
    const auto& source_monomial = source.monomial(derivative->first);
    const auto target_index = target.find_index(source_monomial);
    if (!target_index.has_value()) {
      continue;
    }
    result.push_back({
        *target_index,
        field.multiply(
            entry.value, field.from_unsigned(derivative->second))});
  }
  return result;
}

template <typename Field>
[[nodiscard]] std::size_t minimum_degree(
    const TruncatedMonomialSpace& monomials,
    std::span<const typename RowSpace<Field>::Entry> row) {
  if (row.empty()) {
    throw std::logic_error(
        "internal error: minimum degree requested for the zero row");
  }
  auto result = static_cast<std::size_t>(
      monomials.monomial(row.front().column).total_degree());
  for (const auto& entry : row) {
    result = std::min(
        result,
        static_cast<std::size_t>(
            monomials.monomial(entry.column).total_degree()));
  }
  return result;
}

template <typename Field>
[[nodiscard]] std::vector<std::size_t> coordinate_map(
    const RowSpace<Field>& relations) {
  std::vector<std::size_t> result(relations.dimension(), no_coordinate);
  const auto free_columns = relations.nonpivot_columns();
  for (std::size_t coordinate = 0; coordinate < free_columns.size();
       ++coordinate) {
    result[free_columns[coordinate]] = coordinate;
  }
  return result;
}

template <typename Field>
[[nodiscard]] SparseVector<Field> quotient_coordinates_of_normal_form(
    std::span<const typename RowSpace<Field>::Entry> normal_form,
    std::span<const std::size_t> ambient_to_quotient) {
  SparseVector<Field> result;
  result.reserve(normal_form.size());
  for (const auto& entry : normal_form) {
    const auto coordinate = ambient_to_quotient[entry.column];
    if (coordinate == no_coordinate) {
      throw std::logic_error(
          "internal error: row-space normal form retained a pivot column");
    }
    result.push_back({coordinate, entry.value});
  }
  return result;
}

template <typename Field>
[[nodiscard]] SparseVector<Field> normal_form_coordinates(
    const RowSpace<Field>& relations,
    std::span<const std::size_t> ambient_to_quotient,
    SparseVector<Field> ambient_row) {
  auto normal = relations.normal_form_readonly(std::move(ambient_row));
  return quotient_coordinates_of_normal_form<Field>(
      normal, ambient_to_quotient);
}

template <typename Field>
void append_column(
    std::vector<typename SparseMatrix<Field>::Triplet>& triplets,
    std::size_t row_offset,
    std::size_t column,
    std::span<const typename RowSpace<Field>::Entry> entries,
    const CotangentH1Options& options) {
  check_triplets(triplets.size(), entries.size(), options);
  for (const auto& entry : entries) {
    triplets.push_back({row_offset + entry.column, column, entry.value});
  }
}

template <typename Field>
[[nodiscard]] Polynomial<Field> quotient_polynomial_from_dense(
    const PolynomialRing<Field>& ring,
    const TruncatedMonomialSpace& monomials,
    std::span<const std::size_t> basis_ambient_indices,
    std::span<const typename Field::Element> coordinates) {
  if (basis_ambient_indices.size() != coordinates.size()) {
    throw std::logic_error(
        "internal error: quotient coordinate vector has the wrong length");
  }
  using TermSpec = typename PolynomialRing<Field>::TermSpec;
  std::vector<TermSpec> terms;
  for (std::size_t index = 0; index < coordinates.size(); ++index) {
    if (ring.field().is_zero(coordinates[index])) {
      continue;
    }
    const auto& monomial = monomials.monomial(basis_ambient_indices[index]);
    std::vector<std::uint16_t> exponents(
        monomial.exponents().begin(),
        monomial.exponents().begin() +
            static_cast<std::ptrdiff_t>(ring.variable_count()));
    terms.push_back(TermSpec{coordinates[index], std::move(exponents)});
  }
  return ring.from_terms(std::move(terms));
}

template <typename Field>
[[nodiscard]] Polynomial<Field> quotient_polynomial_from_sparse(
    const PolynomialRing<Field>& ring,
    const TruncatedMonomialSpace& monomials,
    std::span<const std::size_t> basis_ambient_indices,
    std::span<const typename SparseMatrix<Field>::SparseKernelEntry>
        coordinates) {
  using TermSpec = typename PolynomialRing<Field>::TermSpec;
  std::vector<TermSpec> terms;
  terms.reserve(coordinates.size());
  for (const auto& entry : coordinates) {
    if (entry.column >= basis_ambient_indices.size()) {
      throw std::logic_error(
          "internal error: sparse quotient coordinate is out of range");
    }
    if (ring.field().is_zero(entry.value)) {
      throw std::logic_error(
          "internal error: sparse quotient coordinate stores zero");
    }
    const auto& monomial =
        monomials.monomial(basis_ambient_indices[entry.column]);
    std::vector<std::uint16_t> exponents(
        monomial.exponents().begin(),
        monomial.exponents().begin() +
            static_cast<std::ptrdiff_t>(ring.variable_count()));
    terms.push_back(TermSpec{entry.value, std::move(exponents)});
  }
  return ring.from_terms(std::move(terms));
}

}  // namespace cotangent_h1_detail

template <typename Field>
class CotangentH1Presentation {
 public:
  using Element = typename Field::Element;
  using PolynomialType = Polynomial<Field>;
  using RowSpace = IncrementalSparseRowSpace<Field>;
  using SparseVector = typename RowSpace::SparseVector;

  [[nodiscard]] const PolynomialRing<Field>& ring() const noexcept {
    return ring_;
  }
  [[nodiscard]] std::span<const PolynomialType> generators() const noexcept {
    return generators_;
  }
  [[nodiscard]] std::size_t maximal_power() const noexcept {
    return maximal_power_;
  }
  [[nodiscard]] std::size_t length_Q() const noexcept {
    return quotient_basis_indices_.size();
  }
  [[nodiscard]] std::size_t length_P_mod_J2() const noexcept {
    return square_quotient_basis_indices_.size();
  }
  [[nodiscard]] std::size_t conormal_dimension() const noexcept {
    return conormal_dimension_;
  }
  [[nodiscard]] std::size_t h1_dimension() const noexcept {
    return h1_dimension_;
  }

  [[nodiscard]] const SparseMatrix<Field>& reduction_matrix() const noexcept {
    return reduction_matrix_;
  }
  [[nodiscard]] const SparseMatrix<Field>& derivative_matrix() const noexcept {
    return derivative_matrix_;
  }
  [[nodiscard]] const SparseMatrix<Field>& cycle_matrix() const noexcept {
    return cycle_matrix_;
  }
  [[nodiscard]] const SparseMatrix<Field>& h1_relation_matrix() const noexcept {
    return cycle_matrix_;
  }

  [[nodiscard]] std::vector<PolynomialType> ideal_generators() const {
    auto result = generators_;
    const auto degree_n = square_monomials_.degree_range(maximal_power_);
    result.reserve(result.size() + degree_n.size());
    for (std::size_t index = degree_n.first; index < degree_n.past_last;
         ++index) {
      result.push_back(square_monomials_.as_polynomial(ring_, index));
    }
    return result;
  }

  [[nodiscard]] std::vector<PolynomialType> quotient_basis() const {
    std::vector<PolynomialType> result;
    result.reserve(length_Q());
    for (const auto index : quotient_basis_indices_) {
      result.push_back(quotient_monomials_.as_polynomial(ring_, index));
    }
    return result;
  }

  [[nodiscard]] std::vector<PolynomialType> square_quotient_basis() const {
    std::vector<PolynomialType> result;
    result.reserve(length_P_mod_J2());
    for (const auto index : square_quotient_basis_indices_) {
      result.push_back(square_monomials_.as_polynomial(ring_, index));
    }
    return result;
  }

  // Explicit bases are opt-in because E10 can have thousands of vectors.
  // The sparse matrix itself is the primary, complete kernel presentation.
  [[nodiscard]] std::vector<PolynomialType> h1_basis(
      const SparseEliminationLimits& limits) const {
    const auto kernel = h1_kernel_coordinates(limits);
    std::vector<PolynomialType> result;
    result.reserve(kernel.size());
    for (const auto& coordinates : kernel) {
      result.push_back(cotangent_h1_detail::quotient_polynomial_from_sparse(
          ring_, square_monomials_, square_quotient_basis_indices_,
          coordinates));
    }
    return result;
  }

  [[nodiscard]] std::vector<typename SparseMatrix<Field>::SparseKernelVector>
  h1_kernel_coordinates(
      const SparseEliminationLimits& limits) const {
    return cycle_matrix_.right_kernel_basis_sparse(limits);
  }

  [[nodiscard]] std::vector<PolynomialType> conormal_basis(
      const SparseEliminationLimits& limits) const {
    const auto kernel = reduction_matrix_.right_kernel_basis_sparse(limits);
    std::vector<PolynomialType> result;
    result.reserve(kernel.size());
    for (const auto& coordinates : kernel) {
      result.push_back(cotangent_h1_detail::quotient_polynomial_from_sparse(
          ring_, square_monomials_, square_quotient_basis_indices_,
          coordinates));
    }
    return result;
  }

  [[nodiscard]] PolynomialType quotient_remainder(
      const PolynomialType& polynomial) const {
    require_same_ring(polynomial);
    auto normal = quotient_relations_.normal_form_readonly(
        cotangent_h1_detail::polynomial_row(polynomial, quotient_monomials_));
    return cotangent_h1_detail::polynomial_from_ambient_row(
        ring_, quotient_monomials_, normal);
  }

  [[nodiscard]] CotangentClassProof<Field> verify_class(
      const PolynomialType& representative) const {
    require_same_ring(representative);
    const auto representative_remainder = quotient_remainder(representative);
    const bool in_ideal = representative_remainder.is_zero();

    std::vector<PolynomialType> derivative_remainders;
    derivative_remainders.reserve(ring_.variable_count());
    bool derivatives_zero = true;
    for (std::size_t variable = 0; variable < ring_.variable_count();
         ++variable) {
      auto remainder = quotient_remainder(representative.derivative(variable));
      derivatives_zero = derivatives_zero && remainder.is_zero();
      derivative_remainders.push_back(std::move(remainder));
    }

    if (!in_ideal || !derivatives_zero) {
      return CotangentClassProof<Field>{
          in_ideal ? CotangentClassStatus::NotCycle
                   : CotangentClassStatus::NotInIdeal,
          representative,
          in_ideal,
          std::move(derivative_remainders),
          false,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          {},
          {},
          {},
          false,
          false};
    }

    auto representative_ambient =
        cotangent_h1_detail::polynomial_row(representative, square_monomials_);
    auto representative_normal = square_relations_.normal_form_readonly(
        representative_ambient);
    auto class_coordinates =
        cotangent_h1_detail::quotient_coordinates_of_normal_form<Field>(
            representative_normal, square_ambient_to_quotient_);

    std::vector<typename SparseMatrix<Field>::Triplet> multiplication_triplets;
    for (std::size_t column = 0; column < length_Q(); ++column) {
      const auto ambient_monomial = quotient_basis_indices_[column];
      auto product = cotangent_h1_detail::multiply_row_by_monomial<Field>(
          square_monomials_, representative_ambient, ambient_monomial);
      auto product_normal =
          square_relations_.normal_form_readonly(std::move(product));

      // Independently confirm that this P/J^2 column lies in J/J^2.
      SparseVector low_part;
      low_part.reserve(product_normal.size());
      for (const auto& entry : product_normal) {
        if (const auto low_index = quotient_monomials_.find_index(
                square_monomials_.monomial(entry.column));
            low_index.has_value()) {
          low_part.push_back({*low_index, entry.value});
        }
      }
      if (!quotient_relations_.normal_form_readonly(std::move(low_part)).empty()) {
        throw std::logic_error(
            "internal error: multiplication column is outside J/J^2");
      }

      auto coordinates =
          cotangent_h1_detail::quotient_coordinates_of_normal_form<Field>(
              product_normal, square_ambient_to_quotient_);
      cotangent_h1_detail::append_column<Field>(
          multiplication_triplets, 0, column, coordinates, options_);
    }

    SparseMatrix<Field> multiplication(
        ring_.field(), length_P_mod_J2(), length_Q(),
        std::move(multiplication_triplets));
    auto rank_kernel = multiplication.rank_and_kernel(
        options_.matrix_elimination_limits);
    if (rank_kernel.rank + rank_kernel.kernel_basis.size() != length_Q()) {
      throw std::logic_error(
          "internal error: class multiplication violates rank-nullity");
    }

    std::vector<PolynomialType> annihilator_basis;
    annihilator_basis.reserve(rank_kernel.kernel_basis.size());
    for (const auto& coordinates : rank_kernel.kernel_basis) {
      annihilator_basis.push_back(
          cotangent_h1_detail::quotient_polynomial_from_dense(
              ring_, quotient_monomials_, quotient_basis_indices_,
              coordinates));
    }
    const bool faithful = rank_kernel.rank == length_Q();
    auto colon_generators = ideal_generators();
    colon_generators.insert(
        colon_generators.end(), annihilator_basis.begin(),
        annihilator_basis.end());

    return CotangentClassProof<Field>{
        CotangentClassStatus::Complete,
        representative,
        true,
        std::move(derivative_remainders),
        true,
        std::optional<SparseVector>(std::move(class_coordinates)),
        std::optional<SparseMatrix<Field>>(std::move(multiplication)),
        rank_kernel.rank,
        rank_kernel.kernel_basis.size(),
        std::move(rank_kernel.kernel_basis),
        std::move(annihilator_basis),
        std::move(colon_generators),
        faithful,
        faithful};
  }

 private:
  CotangentH1Presentation(
      PolynomialRing<Field> ring,
      std::vector<PolynomialType> generators,
      std::size_t maximal_power,
      CotangentH1Options options,
      TruncatedMonomialSpace quotient_monomials,
      RowSpace quotient_relations,
      std::vector<std::size_t> quotient_basis_indices,
      std::vector<std::size_t> quotient_ambient_to_coordinates,
      TruncatedMonomialSpace square_monomials,
      RowSpace square_relations,
      std::vector<std::size_t> square_quotient_basis_indices,
      std::vector<std::size_t> square_ambient_to_quotient,
      SparseMatrix<Field> reduction_matrix,
      SparseMatrix<Field> derivative_matrix,
      SparseMatrix<Field> cycle_matrix,
      std::size_t conormal_dimension,
      std::size_t h1_dimension)
      : ring_(std::move(ring)),
        generators_(std::move(generators)),
        maximal_power_(maximal_power),
        options_(std::move(options)),
        quotient_monomials_(std::move(quotient_monomials)),
        quotient_relations_(std::move(quotient_relations)),
        quotient_basis_indices_(std::move(quotient_basis_indices)),
        quotient_ambient_to_coordinates_(
            std::move(quotient_ambient_to_coordinates)),
        square_monomials_(std::move(square_monomials)),
        square_relations_(std::move(square_relations)),
        square_quotient_basis_indices_(
            std::move(square_quotient_basis_indices)),
        square_ambient_to_quotient_(
            std::move(square_ambient_to_quotient)),
        reduction_matrix_(std::move(reduction_matrix)),
        derivative_matrix_(std::move(derivative_matrix)),
        cycle_matrix_(std::move(cycle_matrix)),
        conormal_dimension_(conormal_dimension),
        h1_dimension_(h1_dimension) {}

  void require_same_ring(const PolynomialType& polynomial) const {
    if (!ring_.zero().same_ring(polynomial)) {
      throw std::invalid_argument(
          "cotangent-H1 polynomials must belong to the presentation ring");
    }
  }

  PolynomialRing<Field> ring_;
  std::vector<PolynomialType> generators_;
  std::size_t maximal_power_;
  CotangentH1Options options_;

  TruncatedMonomialSpace quotient_monomials_;
  RowSpace quotient_relations_;
  std::vector<std::size_t> quotient_basis_indices_;
  std::vector<std::size_t> quotient_ambient_to_coordinates_;

  TruncatedMonomialSpace square_monomials_;
  RowSpace square_relations_;
  std::vector<std::size_t> square_quotient_basis_indices_;
  std::vector<std::size_t> square_ambient_to_quotient_;

  SparseMatrix<Field> reduction_matrix_;
  SparseMatrix<Field> derivative_matrix_;
  SparseMatrix<Field> cycle_matrix_;
  std::size_t conormal_dimension_;
  std::size_t h1_dimension_;

  template <typename OtherField>
  friend CotangentH1Presentation<OtherField> cotangent_h1(
      CotangentH1Spec<OtherField>, const CotangentH1Options&);
};

template <typename Field>
[[nodiscard]] CotangentH1Presentation<Field> cotangent_h1(
    CotangentH1Spec<Field> spec,
    const CotangentH1Options& options = {}) {
  using RowSpace = cotangent_h1_detail::RowSpace<Field>;
  using SparseVector = cotangent_h1_detail::SparseVector<Field>;

  if (spec.maximal_power == 0 ||
      spec.maximal_power >
          static_cast<std::size_t>(
              std::numeric_limits<std::uint16_t>::max()) / 2) {
    throw CotangentH1InputError(
        CotangentH1InputIssue::InvalidMaximalPower);
  }
  const auto ring_probe = spec.ring.zero();
  for (const auto& generator : spec.generators) {
    if (!ring_probe.same_ring(generator)) {
      throw CotangentH1InputError(
          CotangentH1InputIssue::GeneratorRingMismatch);
    }
    if (std::any_of(
            generator.terms().begin(), generator.terms().end(),
            [](const auto& term) {
              return term.monomial.total_degree() == 0;
            })) {
      throw CotangentH1InputError(
          CotangentH1InputIssue::GeneratorHasConstantTerm);
    }
  }

  const auto variable_count = spec.ring.variable_count();
  const auto square_cutoff = spec.maximal_power * 2;
  TruncatedMonomialSpace quotient_monomials(
      variable_count, spec.maximal_power, options.monomial_space_limits);
  TruncatedMonomialSpace square_monomials(
      variable_count, square_cutoff, options.monomial_space_limits);

  IncrementalSparseRowSpaceOptions row_options;
  row_options.limits = options.row_space_limits;
  row_options.track_insertion_coordinates = false;
  RowSpace quotient_relations(
      spec.ring.field(), quotient_monomials.size(), row_options);

  std::size_t generated_rows = 0;
  std::vector<SparseVector> generator_rows;
  generator_rows.reserve(spec.generators.size());
  for (const auto& generator : spec.generators) {
    auto row = cotangent_h1_detail::polynomial_row(
        generator, quotient_monomials);
    if (row.empty()) {
      continue;
    }
    generator_rows.push_back(row);
    const auto minimum = cotangent_h1_detail::minimum_degree<Field>(
        quotient_monomials, row);
    for (std::size_t degree = 0;
         degree + minimum < spec.maximal_power; ++degree) {
      const auto multipliers = quotient_monomials.degree_range(degree);
      for (std::size_t monomial = multipliers.first;
           monomial < multipliers.past_last; ++monomial) {
        cotangent_h1_detail::check_increment(
            generated_rows, options.max_generated_rows, "generated row");
        static_cast<void>(quotient_relations.insert(
            cotangent_h1_detail::multiply_row_by_monomial<Field>(
                quotient_monomials, row, monomial)));
      }
    }
  }

  // The degree ordering is independent of the cutoff, so a row in P/m^N
  // uses the same ambient indices in P/m^(2N).
  RowSpace square_relations(
      spec.ring.field(), square_monomials.size(), row_options);

  for (std::size_t left = 0; left < generator_rows.size(); ++left) {
    for (std::size_t right = left; right < generator_rows.size(); ++right) {
      const auto product = cotangent_h1_detail::canonicalize_row<Field>(
          spec.ring.field(),
          cotangent_h1_detail::multiply_rows<Field>(
              spec.ring.field(), square_monomials,
              generator_rows[left], generator_rows[right]));
      if (product.empty()) {
        continue;
      }
      const auto minimum = cotangent_h1_detail::minimum_degree<Field>(
          square_monomials, product);
      for (std::size_t degree = 0; degree + minimum < square_cutoff;
           ++degree) {
        const auto multipliers = square_monomials.degree_range(degree);
        for (std::size_t monomial = multipliers.first;
             monomial < multipliers.past_last; ++monomial) {
          cotangent_h1_detail::check_increment(
              generated_rows, options.max_generated_rows, "generated row");
          static_cast<void>(square_relations.insert(
              cotangent_h1_detail::multiply_row_by_monomial<Field>(
                  square_monomials, product, monomial)));
        }
      }
    }
  }

  for (const auto& generator : generator_rows) {
    const auto minimum = cotangent_h1_detail::minimum_degree<Field>(
        square_monomials, generator);
    for (std::size_t degree = spec.maximal_power;
         degree + minimum < square_cutoff; ++degree) {
      const auto multipliers = square_monomials.degree_range(degree);
      for (std::size_t monomial = multipliers.first;
           monomial < multipliers.past_last; ++monomial) {
        cotangent_h1_detail::check_increment(
            generated_rows, options.max_generated_rows, "generated row");
        static_cast<void>(square_relations.insert(
            cotangent_h1_detail::multiply_row_by_monomial<Field>(
                square_monomials, generator, monomial)));
      }
    }
  }

  const auto quotient_basis_indices = quotient_relations.nonpivot_columns();
  const auto quotient_ambient_to_coordinates =
      cotangent_h1_detail::coordinate_map(quotient_relations);
  const auto square_quotient_basis_indices =
      square_relations.nonpivot_columns();
  const auto square_ambient_to_quotient =
      cotangent_h1_detail::coordinate_map(square_relations);
  const auto length = quotient_basis_indices.size();
  const auto square_length = square_quotient_basis_indices.size();

  if (options.verify_square_derivatives) {
    for (std::size_t basis = 0; basis < square_relations.rank(); ++basis) {
      const auto& relation = square_relations.normalized_pivot_row(basis);
      for (std::size_t variable = 0; variable < variable_count; ++variable) {
        auto derivative = cotangent_h1_detail::derivative_row<Field>(
            spec.ring.field(), square_monomials, quotient_monomials,
            relation, variable);
        if (!quotient_relations
                 .normal_form_readonly(std::move(derivative))
                 .empty()) {
          throw std::logic_error(
              "internal error: a stored J^2 relation has derivative outside J");
        }
      }
    }
  }

  std::vector<typename SparseMatrix<Field>::Triplet> reduction_triplets;
  std::vector<typename SparseMatrix<Field>::Triplet> derivative_triplets;
  std::vector<typename SparseMatrix<Field>::Triplet> cycle_triplets;

  for (std::size_t column = 0; column < square_length; ++column) {
    const auto ambient = square_quotient_basis_indices[column];
    const auto& monomial = square_monomials.monomial(ambient);

    if (const auto low = quotient_monomials.find_index(monomial);
        low.has_value()) {
      SparseVector unit{{*low, spec.ring.field().one()}};
      auto coordinates = cotangent_h1_detail::normal_form_coordinates<Field>(
          quotient_relations, quotient_ambient_to_coordinates,
          std::move(unit));
      cotangent_h1_detail::append_column<Field>(
          reduction_triplets, 0, column, coordinates, options);
      cotangent_h1_detail::append_column<Field>(
          cycle_triplets, 0, column, coordinates, options);
    }

    for (std::size_t variable = 0; variable < variable_count; ++variable) {
      const auto derivative = square_monomials.differentiate(ambient, variable);
      if (!derivative.has_value()) {
        continue;
      }
      const auto low = quotient_monomials.find_index(
          square_monomials.monomial(derivative->first));
      if (!low.has_value()) {
        continue;
      }
      SparseVector unit{{
          *low, spec.ring.field().from_unsigned(derivative->second)}};
      auto coordinates = cotangent_h1_detail::normal_form_coordinates<Field>(
          quotient_relations, quotient_ambient_to_coordinates,
          std::move(unit));
      cotangent_h1_detail::append_column<Field>(
          derivative_triplets, variable * length, column, coordinates,
          options);
      cotangent_h1_detail::append_column<Field>(
          cycle_triplets, length + variable * length, column, coordinates,
          options);
    }
  }

  SparseMatrix<Field> reduction_matrix(
      spec.ring.field(), length, square_length,
      std::move(reduction_triplets));
  SparseMatrix<Field> derivative_matrix(
      spec.ring.field(), variable_count * length, square_length,
      std::move(derivative_triplets));
  SparseMatrix<Field> cycle_matrix(
      spec.ring.field(), (variable_count + 1) * length, square_length,
      std::move(cycle_triplets));

  const auto reduction_rank =
      reduction_matrix.rank(options.matrix_elimination_limits);
  if (reduction_rank != length) {
    throw std::logic_error(
        "internal error: P/J^2 to P/J is not surjective");
  }
  const auto cycle_rank = cycle_matrix.rank(options.matrix_elimination_limits);
  if (cycle_rank > square_length) {
    throw std::logic_error("internal error: cotangent cycle rank is invalid");
  }
  const auto conormal_dimension = square_length - reduction_rank;
  const auto h1_dimension = square_length - cycle_rank;

  return CotangentH1Presentation<Field>(
      std::move(spec.ring), std::move(spec.generators), spec.maximal_power,
      options, std::move(quotient_monomials),
      std::move(quotient_relations), quotient_basis_indices,
      quotient_ambient_to_coordinates, std::move(square_monomials),
      std::move(square_relations), square_quotient_basis_indices,
      square_ambient_to_quotient, std::move(reduction_matrix),
      std::move(derivative_matrix), std::move(cycle_matrix),
      conormal_dimension, h1_dimension);
}

}  // namespace laughableengine

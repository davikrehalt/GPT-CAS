#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/groebner.hpp"
#include "laughableengine/matrix.hpp"
#include "laughableengine/polynomial.hpp"
#include "laughableengine/quotient.hpp"

namespace laughableengine {

// An ideal always carries an explicit polynomial-ring context and stores its
// deterministic reduced monic Groebner basis.  In particular, the empty basis
// retains the ring needed to distinguish zero ideals in different rings.
template <typename Field>
class Ideal {
 public:
  using Element = typename Field::Element;
  using PolynomialType = Polynomial<Field>;
  using Ring = PolynomialRing<Field>;

  Ideal(const Ring& ring, std::vector<PolynomialType> generators)
      : Ideal(ring, std::move(generators), GroebnerLimits{}) {}

  Ideal(
      const Ring& ring,
      std::vector<PolynomialType> generators,
      const GroebnerLimits& limits)
      : Ideal(
            ring, canonicalize(ring, std::move(generators), limits),
            CanonicalTag{}) {}

  Ideal(const Ring& ring,
        std::initializer_list<PolynomialType> generators)
      : Ideal(ring, std::vector<PolynomialType>(generators)) {}

  Ideal(
      const Ring& ring,
      std::initializer_list<PolynomialType> generators,
      const GroebnerLimits& limits)
      : Ideal(ring, std::vector<PolynomialType>(generators), limits) {}

  [[nodiscard]] const Ring& ring() const noexcept { return ring_; }

  // The canonical basis is also a valid deterministic generating set.
  [[nodiscard]] const std::vector<PolynomialType>& generators() const noexcept {
    return basis_;
  }
  [[nodiscard]] const std::vector<PolynomialType>& groebner_basis() const
      noexcept {
    return basis_;
  }
  [[nodiscard]] const CompiledReducer<Field>& reducer() const noexcept {
    return reducer_;
  }

  [[nodiscard]] PolynomialType normal_form(
      const PolynomialType& polynomial,
      const ReductionLimits& limits = {}) const {
    return reducer_.normal_form(polynomial, limits);
  }

  [[nodiscard]] std::vector<PolynomialType> normal_forms(
      std::span<const PolynomialType> polynomials,
      const ReductionLimits& limits = {}) const {
    return reducer_.normal_forms(polynomials, limits);
  }

  [[nodiscard]] std::vector<PolynomialType> normal_forms(
      const std::vector<PolynomialType>& polynomials,
      const ReductionLimits& limits = {}) const {
    return reducer_.normal_forms(polynomials, limits);
  }

  [[nodiscard]] std::vector<PolynomialType> normal_forms(
      std::initializer_list<PolynomialType> polynomials,
      const ReductionLimits& limits = {}) const {
    return reducer_.normal_forms(polynomials, limits);
  }

  [[nodiscard]] bool contains(
      const PolynomialType& polynomial,
      const ReductionLimits& limits = {}) const {
    return normal_form(polynomial, limits).is_zero();
  }

  [[nodiscard]] bool is_subset_of(const Ideal& other) const {
    require_same_ring(other);
    for (const auto& generator : basis_) {
      if (!other.contains(generator)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool equals(const Ideal& other) const {
    require_same_ring(other);
    return basis_ == other.basis_;
  }

  [[nodiscard]] bool is_zero() const noexcept { return basis_.empty(); }

  [[nodiscard]] bool is_unit() const noexcept {
    return basis_.size() == 1 &&
           basis_.front().leading_term()->monomial.total_degree() == 0;
  }

  [[nodiscard]] bool is_zero_dimensional() const {
    return reducer_.is_zero_dimensional();
  }

  [[nodiscard]] StandardMonomialBasis<Field> standard_monomials(
      const StandardMonomialLimits& limits = {}) const {
    return reducer_.standard_monomials(limits);
  }

  [[nodiscard]] std::size_t quotient_dimension(
      const StandardMonomialLimits& limits = {}) const {
    return reducer_.quotient_dimension(limits);
  }

  // A finite quotient is supported at the origin exactly when every variable
  // is nilpotent.  On an algebra of length n, nilpotence is equivalent to
  // x_i^n=0, so this checks actual ideal membership rather than merely the
  // pure powers in the leading ideal.
  [[nodiscard]] bool supported_at_origin(
      const StandardMonomialLimits& limits = {}) const {
    if (!is_zero_dimensional()) {
      return false;
    }
    const auto length = quotient_dimension(limits);
    if (length == 0) {
      return true;
    }
    for (std::size_t variable = 0; variable < ring_.variable_count();
         ++variable) {
      auto power = ring_.one();
      const auto generator = ring_.gen(variable);
      for (std::size_t exponent = 0;
           exponent < length && !power.is_zero(); ++exponent) {
        power = normal_form(power * generator);
      }
      if (!power.is_zero()) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] Ideal sum(
      const Ideal& other,
      const GroebnerLimits& limits = {}) const {
    require_same_ring(other);
    const auto generator_count = checked_sum_size(
        basis_.size(), other.basis_.size(),
        "ideal-sum generator count overflows size_t");
    check_groebner_input_count(generator_count, limits);
    check_groebner_input_terms(
        saturated_add(basis_term_count(), other.basis_term_count()), limits);
    std::vector<PolynomialType> generators = basis_;
    generators.reserve(generator_count);
    generators.insert(
        generators.end(), other.basis_.begin(), other.basis_.end());
    return Ideal(ring_, std::move(generators), limits);
  }

  [[nodiscard]] Ideal product(
      const Ideal& other,
      const GroebnerLimits& limits = {}) const {
    require_same_ring(other);
    std::vector<PolynomialType> generators;
    const auto generator_count = checked_product_size(
        basis_.size(), other.basis_.size(),
        "ideal product generator count overflows size_t");
    check_groebner_input_count(generator_count, limits);
    std::size_t raw_term_bound = 0;
    for (const auto& left : basis_) {
      for (const auto& right : other.basis_) {
        raw_term_bound = saturated_add(
            raw_term_bound,
            saturated_product(left.term_count(), right.term_count()));
      }
    }
    check_groebner_input_terms(raw_term_bound, limits);
    generators.reserve(generator_count);
    for (const auto& left : basis_) {
      for (const auto& right : other.basis_) {
        check_product_term_bound(left, right, limits);
        generators.push_back(left * right);
      }
    }
    return Ideal(ring_, std::move(generators), limits);
  }

  [[nodiscard]] Ideal square(const GroebnerLimits& limits = {}) const {
    std::vector<PolynomialType> generators;
    const auto generator_count = checked_symmetric_pair_count(basis_.size());
    check_groebner_input_count(generator_count, limits);
    std::size_t raw_term_bound = 0;
    for (std::size_t right = 0; right < basis_.size(); ++right) {
      for (std::size_t left = 0; left <= right; ++left) {
        raw_term_bound = saturated_add(
            raw_term_bound,
            saturated_product(
                basis_[left].term_count(), basis_[right].term_count()));
      }
    }
    check_groebner_input_terms(raw_term_bound, limits);
    generators.reserve(generator_count);
    for (std::size_t right = 0; right < basis_.size(); ++right) {
      for (std::size_t left = 0; left <= right; ++left) {
        check_product_term_bound(basis_[left], basis_[right], limits);
        generators.push_back(basis_[left] * basis_[right]);
      }
    }
    return Ideal(ring_, std::move(generators), limits);
  }

  // Compute (I:g) through the kernel of multiplication by g on P/I.  The
  // generic path requires P/I to be finite-dimensional; zero g, g in I, and
  // the unit ideal are resolved directly even when no matrix is needed.
  [[nodiscard]] Ideal colon(
      const PolynomialType& polynomial,
      const StandardMonomialLimits& monomial_limits = {},
      const ReductionLimits& reduction_limits = {},
      const GroebnerLimits& groebner_limits = {}) const {
    require_polynomial(polynomial);
    if (is_unit()) {
      return *this;
    }

    const auto reduced_polynomial =
        normal_form(polynomial, reduction_limits);
    if (reduced_polynomial.is_zero()) {
      return Ideal(ring_, {ring_.one()});
    }
    if (reduced_polynomial.leading_term()->monomial.total_degree() == 0) {
      return *this;
    }
    if (!is_zero_dimensional()) {
      throw std::domain_error(
          "principal colon currently requires a zero-dimensional quotient");
    }

    const auto standard = standard_monomials(monomial_limits);
    // The current exact kernel path materializes the multiplication map
    // densely. Keep direct public colon calls from attempting an unbounded
    // allocation even when the standard-monomial limits are much larger.
    constexpr std::size_t max_matrix_entries = 5'000'000;
    if (standard.size() != 0 &&
        standard.size() > max_matrix_entries / standard.size()) {
      const auto observed =
          standard.size() >
                  std::numeric_limits<std::size_t>::max() / standard.size()
              ? std::numeric_limits<std::size_t>::max()
              : standard.size() * standard.size();
      throw ResourceLimitExceeded(
          ResourceLimitKind::MatrixEntries, max_matrix_entries, observed);
    }
    const auto monomials = standard.polynomials();
    std::vector<PolynomialType> products;
    products.reserve(monomials.size());
    for (const auto& monomial : monomials) {
      products.push_back(monomial * reduced_polynomial);
    }
    const auto columns = standard.coordinates(products, reduction_limits);

    DenseMatrix<Field> multiplication(
        ring_.field(), standard.size(), standard.size());
    for (std::size_t column = 0; column < columns.size(); ++column) {
      for (std::size_t row = 0; row < columns[column].size(); ++row) {
        multiplication.set(row, column, columns[column][row]);
      }
    }

    const auto kernel = multiplication.right_kernel_basis();
    if (kernel.empty()) {
      return *this;
    }
    std::vector<PolynomialType> generators = basis_;
    if (kernel.size() >
        std::numeric_limits<std::size_t>::max() - generators.size()) {
      throw std::length_error(
          "principal-colon generator count overflows size_t");
    }
    check_groebner_input_count(
        generators.size() + kernel.size(), groebner_limits);
    check_groebner_input_terms(
        saturated_add(
            basis_term_count(),
            saturated_product(kernel.size(), standard.size())),
        groebner_limits);
    generators.reserve(generators.size() + kernel.size());
    for (const auto& vector : kernel) {
      generators.push_back(standard.from_coordinates(vector));
    }
    return Ideal(ring_, std::move(generators), groebner_limits);
  }

  [[nodiscard]] Ideal principal_colon(
      const PolynomialType& polynomial,
      const StandardMonomialLimits& monomial_limits = {},
      const ReductionLimits& reduction_limits = {},
      const GroebnerLimits& groebner_limits = {}) const {
    return colon(
        polynomial, monomial_limits, reduction_limits, groebner_limits);
  }

  // Return the contraction to the retained-variable subring, extended back
  // to this original ring.  A temporary lex ring puts the eliminated
  // variables first; every coefficient and exponent is explicitly remapped.
  [[nodiscard]] Ideal eliminate(
      std::span<const std::size_t> variable_indices) const {
    if (variable_indices.empty() || variable_indices.size() > 3) {
      throw std::invalid_argument(
          "elimination requires between one and three variable indices");
    }

    std::vector<std::size_t> eliminated(
        variable_indices.begin(), variable_indices.end());
    for (const auto variable : eliminated) {
      if (variable >= ring_.variable_count()) {
        throw std::out_of_range(
            "elimination variable index is out of range");
      }
    }
    std::sort(eliminated.begin(), eliminated.end());
    if (std::adjacent_find(eliminated.begin(), eliminated.end()) !=
        eliminated.end()) {
      throw std::invalid_argument(
          "elimination variable indices must be distinct");
    }

    std::vector<std::size_t> temporary_to_original = eliminated;
    for (std::size_t variable = 0; variable < ring_.variable_count();
         ++variable) {
      if (!std::binary_search(
              eliminated.begin(), eliminated.end(), variable)) {
        temporary_to_original.push_back(variable);
      }
    }

    std::vector<std::string> temporary_names;
    temporary_names.reserve(ring_.variable_count());
    for (const auto original : temporary_to_original) {
      temporary_names.push_back(ring_.variable_names()[original]);
    }
    const Ring temporary_ring(
        ring_.field(), std::move(temporary_names), Order::Lex);

    std::vector<PolynomialType> temporary_generators;
    temporary_generators.reserve(basis_.size());
    for (const auto& generator : basis_) {
      temporary_generators.push_back(remap_to_temporary(
          generator, temporary_ring, temporary_to_original));
    }
    const auto temporary_basis =
        laughableengine::groebner_basis(temporary_generators);

    std::vector<PolynomialType> contracted_generators;
    for (const auto& candidate : temporary_basis) {
      bool retained_only = true;
      for (const auto& term : candidate.terms()) {
        for (std::size_t variable = 0; variable < eliminated.size();
             ++variable) {
          if (term.monomial.exponent(variable) != 0) {
            retained_only = false;
            break;
          }
        }
        if (!retained_only) {
          break;
        }
      }
      if (retained_only) {
        contracted_generators.push_back(remap_to_original(
            candidate, temporary_to_original));
      }
    }
    return Ideal(ring_, std::move(contracted_generators));
  }

  [[nodiscard]] Ideal eliminate(
      const std::vector<std::size_t>& variable_indices) const {
    return eliminate(std::span<const std::size_t>(
        variable_indices.data(), variable_indices.size()));
  }

  [[nodiscard]] Ideal eliminate(
      std::initializer_list<std::size_t> variable_indices) const {
    return eliminate(std::span<const std::size_t>(
        variable_indices.begin(), variable_indices.size()));
  }

  [[nodiscard]] Ideal eliminate(std::size_t variable_index) const {
    return eliminate(
        std::initializer_list<std::size_t>{variable_index});
  }

  [[nodiscard]] friend Ideal operator+(const Ideal& left, const Ideal& right) {
    return left.sum(right);
  }

  [[nodiscard]] friend Ideal operator*(const Ideal& left, const Ideal& right) {
    return left.product(right);
  }

  friend bool operator==(const Ideal& left, const Ideal& right) {
    return left.same_ring(right) && left.basis_ == right.basis_;
  }

 private:
  struct CanonicalTag {};

  [[nodiscard]] static std::size_t checked_product_size(
      std::size_t left,
      std::size_t right,
      const char* message) {
    if (right != 0 &&
        left > std::numeric_limits<std::size_t>::max() / right) {
      throw std::length_error(message);
    }
    return left * right;
  }

  [[nodiscard]] static std::size_t checked_sum_size(
      std::size_t left,
      std::size_t right,
      const char* message) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
      throw std::length_error(message);
    }
    return left + right;
  }

  static void check_groebner_input_count(
      std::size_t count,
      const GroebnerLimits& limits) {
    if (limits.max_basis_polynomials.has_value() &&
        count > *limits.max_basis_polynomials) {
      throw GroebnerResourceLimit(
          GroebnerResourceKind::BasisPolynomials,
          *limits.max_basis_polynomials, count);
    }
  }

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

  [[nodiscard]] std::size_t basis_term_count() const noexcept {
    std::size_t result = 0;
    for (const auto& polynomial : basis_) {
      result = saturated_add(result, polynomial.term_count());
    }
    return result;
  }

  static void check_groebner_input_terms(
      std::size_t count,
      const GroebnerLimits& limits) {
    if (limits.max_basis_terms.has_value() &&
        count > *limits.max_basis_terms) {
      throw GroebnerResourceLimit(
          GroebnerResourceKind::BasisTerms,
          *limits.max_basis_terms, count);
    }
  }

  static void check_product_term_bound(
      const PolynomialType& left,
      const PolynomialType& right,
      const GroebnerLimits& limits) {
    if (!limits.max_live_terms.has_value()) {
      return;
    }
    const auto terms =
        right.term_count() != 0 &&
                left.term_count() >
                    std::numeric_limits<std::size_t>::max() /
                        right.term_count()
            ? std::numeric_limits<std::size_t>::max()
            : left.term_count() * right.term_count();
    if (terms > *limits.max_live_terms) {
      throw GroebnerResourceLimit(
          GroebnerResourceKind::LiveTerms, *limits.max_live_terms, terms);
    }
  }

  [[nodiscard]] static std::size_t checked_symmetric_pair_count(
      std::size_t count) {
    if (count == std::numeric_limits<std::size_t>::max()) {
      throw std::length_error(
          "ideal-square generator count overflows size_t");
    }
    const auto successor = count + 1;
    if ((count % 2) == 0) {
      return checked_product_size(
          count / 2, successor,
          "ideal-square generator count overflows size_t");
    }
    return checked_product_size(
        count, successor / 2,
        "ideal-square generator count overflows size_t");
  }

  Ideal(const Ring& ring,
        std::vector<PolynomialType> canonical_basis,
        CanonicalTag)
      : ring_(ring),
        basis_(std::move(canonical_basis)),
        reducer_(CompiledReducer<Field>::groebner(
            ring_, basis_, BasisValidation::AssumeVerified)) {}

  [[nodiscard]] static std::vector<PolynomialType> canonicalize(
      const Ring& ring,
      std::vector<PolynomialType> generators,
      const GroebnerLimits& limits) {
    const auto zero = ring.zero();
    for (const auto& generator : generators) {
      if (!zero.same_ring(generator)) {
        throw std::invalid_argument(
            "ideal generators require the exact explicit ring context");
      }
    }
    return laughableengine::groebner_basis(generators, limits);
  }

  [[nodiscard]] bool same_ring(const Ideal& other) const {
    return ring_.zero().same_ring(other.ring_.zero());
  }

  void require_same_ring(const Ideal& other) const {
    if (!same_ring(other)) {
      throw std::invalid_argument(
          "ideal operations require the exact same ring context");
    }
  }

  void require_polynomial(const PolynomialType& polynomial) const {
    if (!ring_.zero().same_ring(polynomial)) {
      throw std::invalid_argument(
          "ideal membership requires the exact same ring context");
    }
  }

  [[nodiscard]] static PolynomialType remap_to_temporary(
      const PolynomialType& polynomial,
      const Ring& temporary_ring,
      const std::vector<std::size_t>& temporary_to_original) {
    std::vector<typename Ring::TermSpec> terms;
    terms.reserve(polynomial.term_count());
    for (const auto& term : polynomial.terms()) {
      std::vector<std::uint16_t> exponents(temporary_ring.variable_count(), 0);
      for (std::size_t temporary = 0;
           temporary < temporary_to_original.size(); ++temporary) {
        exponents[temporary] =
            term.monomial.exponent(temporary_to_original[temporary]);
      }
      terms.push_back(typename Ring::TermSpec{
          temporary_ring.field().canonical(term.coefficient),
          std::move(exponents)});
    }
    return temporary_ring.from_terms(std::move(terms));
  }

  [[nodiscard]] PolynomialType remap_to_original(
      const PolynomialType& polynomial,
      const std::vector<std::size_t>& temporary_to_original) const {
    std::vector<typename Ring::TermSpec> terms;
    terms.reserve(polynomial.term_count());
    for (const auto& term : polynomial.terms()) {
      std::vector<std::uint16_t> exponents(ring_.variable_count(), 0);
      for (std::size_t temporary = 0;
           temporary < temporary_to_original.size(); ++temporary) {
        exponents[temporary_to_original[temporary]] =
            term.monomial.exponent(temporary);
      }
      terms.push_back(typename Ring::TermSpec{
          ring_.field().canonical(term.coefficient), std::move(exponents)});
    }
    return ring_.from_terms(std::move(terms));
  }

  Ring ring_;
  std::vector<PolynomialType> basis_;
  CompiledReducer<Field> reducer_;
};

}  // namespace laughableengine

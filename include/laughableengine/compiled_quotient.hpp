#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "laughableengine/packed_polynomial.hpp"
#include "laughableengine/quotient_context.hpp"

namespace laughableengine {

class PackedDiscoveryResourceLimit : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct CompiledPrimeQuotientLimits {
  std::optional<std::size_t> max_batch_entries = 10'000'000;
  std::optional<std::size_t> max_cached_monomials = 1'000'000;
  std::optional<std::size_t> max_variable_action_applications = 100'000'000;
};

class PackedCoordinateBatch {
 public:
  PackedCoordinateBatch(
      std::uint32_t modulus,
      std::size_t rows,
      std::size_t columns,
      std::vector<std::uint32_t> entries)
      : modulus_(modulus),
        rows_(rows),
        columns_(columns),
        entries_(std::move(entries)) {
    if (columns_ != 0 &&
        rows_ > std::numeric_limits<std::size_t>::max() / columns_) {
      throw std::length_error(
          "packed coordinate batch dimensions overflow size_t");
    }
    if (entries_.size() != rows_ * columns_) {
      throw std::invalid_argument(
          "packed coordinate batch entries do not match its shape");
    }
  }

  [[nodiscard]] std::uint32_t modulus() const noexcept { return modulus_; }
  [[nodiscard]] std::size_t row_count() const noexcept { return rows_; }
  [[nodiscard]] std::size_t column_count() const noexcept { return columns_; }
  [[nodiscard]] const std::vector<std::uint32_t>& entries() const noexcept {
    return entries_;
  }
  [[nodiscard]] std::span<const std::uint32_t> row(
      std::size_t index) const {
    if (index >= rows_) {
      throw std::out_of_range("packed coordinate batch row is out of range");
    }
    const auto* first = entries_.data();
    if (columns_ != 0) {
      first += index * columns_;
    }
    return std::span<const std::uint32_t>(first, columns_);
  }

 private:
  std::uint32_t modulus_;
  std::size_t rows_;
  std::size_t columns_;
  std::vector<std::uint32_t> entries_;
};

namespace compiled_quotient_detail {

inline void append_u64(std::string& bytes, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    bytes.push_back(static_cast<char>((value >> shift) & 0xffU));
  }
}

inline void append_string(std::string& bytes, std::string_view value) {
  append_u64(bytes, value.size());
  bytes.append(value);
}

inline std::uint64_t fnv1a(std::string_view bytes) noexcept {
  std::uint64_t value = 1469598103934665603ULL;
  for (const auto character : bytes) {
    value ^= static_cast<unsigned char>(character);
    value *= 1099511628211ULL;
  }
  return value;
}

struct PackedWordHash {
  [[nodiscard]] std::size_t operator()(std::uint64_t word) const noexcept {
    word ^= word >> 30U;
    word *= 0xbf58476d1ce4e5b9ULL;
    word ^= word >> 27U;
    word *= 0x94d049bb133111ebULL;
    word ^= word >> 31U;
    return static_cast<std::size_t>(word);
  }
};

inline std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* message) {
  if (right != 0 &&
      left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::length_error(message);
  }
  return left * right;
}

}  // namespace compiled_quotient_detail

// A fixed-quotient GF(p) discovery plan.  Construction performs the only
// scalar Groebner reductions needed to compile border rules. Subsequent batch
// coordinate reductions use raw residues and packed variable actions only.
class CompiledPrimeQuotientPlan {
 public:
  static constexpr std::size_t scalar_normal_form_calls_per_batch = 0;

  explicit CompiledPrimeQuotientPlan(
      Ideal<PrimeField> ideal,
      const StandardMonomialLimits& monomial_limits = {})
      : context_(std::move(ideal), monomial_limits),
        arithmetic_(context_.field().modulus()) {
    standard_words_.reserve(context_.dimension());
    standard_indices_.reserve(context_.dimension());
    for (std::size_t index = 0; index < context_.dimension(); ++index) {
      const auto packed = PackedMonomial::from_exact(
          context_.standard_monomials()[index],
          context_.ring().variable_count());
      standard_words_.push_back(packed.word());
      standard_indices_.emplace(packed.word(), index);
    }
    if (context_.dimension() != 0) {
      const auto one = standard_indices_.find(PackedMonomial{}.word());
      if (one == standard_indices_.end()) {
        throw std::logic_error(
            "compiled quotient standard basis omits one");
      }
      one_index_ = one->second;
    }
    compile_variable_actions();
    compile_signatures();
  }

  [[nodiscard]] const QuotientContext<PrimeField>& reference_context() const
      noexcept {
    return context_;
  }
  [[nodiscard]] std::uint32_t modulus() const noexcept {
    return arithmetic_.modulus();
  }
  [[nodiscard]] std::size_t variable_count() const noexcept {
    return context_.ring().variable_count();
  }
  [[nodiscard]] std::size_t dimension() const noexcept {
    return context_.dimension();
  }
  [[nodiscard]] Order order() const noexcept { return context_.ring().order(); }
  [[nodiscard]] const std::vector<std::uint64_t>& standard_monomial_words()
      const noexcept {
    return standard_words_;
  }
  [[nodiscard]] const std::vector<PackedPrimeMatrix>& variable_actions() const
      noexcept {
    return variable_actions_;
  }
  [[nodiscard]] const std::vector<PackedPrimeSparseMatrix>&
  sparse_variable_actions() const noexcept {
    return sparse_variable_actions_;
  }
  [[nodiscard]] std::uint64_t full_fingerprint() const noexcept {
    return full_fingerprint_;
  }
  [[nodiscard]] std::uint64_t leading_fingerprint() const noexcept {
    return leading_fingerprint_;
  }
  [[nodiscard]] std::string_view full_signature() const noexcept {
    return full_signature_;
  }
  [[nodiscard]] std::string_view leading_signature() const noexcept {
    return leading_signature_;
  }

  [[nodiscard]] PackedCoordinateBatch reduce_coordinates_batch(
      std::span<const PackedPrimePolynomial> polynomials,
      const CompiledPrimeQuotientLimits& limits = {}) const {
    const auto entry_count = compiled_quotient_detail::checked_product(
        polynomials.size(), dimension(),
        "packed coordinate batch dimensions overflow size_t");
    if (limits.max_batch_entries.has_value() &&
        entry_count > *limits.max_batch_entries) {
      throw PackedDiscoveryResourceLimit(
          "packed coordinate batch exceeds max_batch_entries");
    }
    std::vector<std::uint32_t> entries;
    entries.reserve(entry_count);
    MonomialCache cache;
    std::size_t action_applications = 0;
    for (const auto& polynomial : polynomials) {
      require_polynomial(polynomial);
      auto coordinates = reduce_one(
          polynomial, cache, action_applications, limits);
      entries.insert(entries.end(), coordinates.begin(), coordinates.end());
    }
    return PackedCoordinateBatch(
        modulus(), polynomials.size(), dimension(), std::move(entries));
  }

  [[nodiscard]] PackedCoordinateBatch reduce_coordinates_batch(
      std::span<const Polynomial<PrimeField>> polynomials,
      const CompiledPrimeQuotientLimits& limits = {}) const {
    std::vector<PackedPrimePolynomial> packed;
    packed.reserve(polynomials.size());
    for (const auto& polynomial : polynomials) {
      if (!context_.ring().zero().same_ring(polynomial)) {
        throw std::invalid_argument(
            "compiled quotient batch requires its exact reference ring");
      }
      packed.push_back(PackedPrimePolynomial::from_exact(polynomial));
    }
    return reduce_coordinates_batch(packed, limits);
  }

  [[nodiscard]] PackedPrimeMatrix multiplication_matrix(
      const PackedPrimePolynomial& polynomial,
      const CompiledPrimeQuotientLimits& limits = {}) const {
    require_polynomial(polynomial);
    const auto entries_count = compiled_quotient_detail::checked_product(
        dimension(), dimension(),
        "packed multiplication matrix dimensions overflow size_t");
    if (limits.max_batch_entries.has_value() &&
        entries_count > *limits.max_batch_entries) {
      throw PackedDiscoveryResourceLimit(
          "packed multiplication matrix exceeds max_batch_entries");
    }
    PackedPrimeMatrix result(modulus(), dimension(), dimension());
    std::size_t action_applications = 0;
    for (std::size_t column = 0; column < dimension(); ++column) {
      std::vector<std::uint32_t> image(dimension(), 0);
      for (std::size_t term = 0; term < polynomial.term_count(); ++term) {
        auto vector = basis_vector(column);
        apply_monomial_in_place(
            PackedMonomial::from_word(polynomial.monomial_words()[term]),
            vector, action_applications, limits);
        const auto coefficient = polynomial.coefficients()[term];
        for (std::size_t row = 0; row < dimension(); ++row) {
          image[row] = arithmetic_.add(
              image[row], arithmetic_.multiply(coefficient, vector[row]));
        }
      }
      for (std::size_t row = 0; row < dimension(); ++row) {
        result.set(row, column, image[row]);
      }
    }
    return result;
  }

  [[nodiscard]] std::vector<std::uint32_t> multiply_by_monomial(
      PackedMonomial monomial,
      std::span<const std::uint32_t> coordinates,
      const CompiledPrimeQuotientLimits& limits = {}) const {
    std::size_t action_applications = 0;
    return multiply_by_monomial(
        monomial, coordinates, action_applications, limits);
  }

  [[nodiscard]] std::vector<std::uint32_t> multiply_by_monomial(
      PackedMonomial monomial,
      std::span<const std::uint32_t> coordinates,
      std::size_t& action_applications,
      const CompiledPrimeQuotientLimits& limits = {}) const {
    if (coordinates.size() != dimension()) {
      throw std::invalid_argument(
          "compiled monomial action has the wrong coordinate dimension");
    }
    std::vector<std::uint32_t> result;
    result.reserve(coordinates.size());
    for (const auto entry : coordinates) {
      result.push_back(arithmetic_.canonical(entry));
    }
    apply_monomial_in_place(
        monomial, result, action_applications, limits);
    return result;
  }

 private:
  using MonomialCache = std::unordered_map<
      std::uint64_t,
      std::vector<std::uint32_t>,
      compiled_quotient_detail::PackedWordHash>;

  void require_polynomial(const PackedPrimePolynomial& polynomial) const {
    if (polynomial.modulus() != modulus() ||
        polynomial.variable_count() != variable_count() ||
        polynomial.order() != order()) {
      throw std::invalid_argument(
          "packed quotient reduction requires one matching ring");
    }
  }

  [[nodiscard]] std::vector<std::uint32_t> basis_vector(
      std::size_t index) const {
    std::vector<std::uint32_t> result(dimension(), 0);
    if (index >= dimension()) {
      throw std::out_of_range("compiled quotient basis index is out of range");
    }
    result[index] = 1;
    return result;
  }

  void check_action_limit(
      std::size_t action_applications,
      const CompiledPrimeQuotientLimits& limits) const {
    if (limits.max_variable_action_applications.has_value() &&
        action_applications >= *limits.max_variable_action_applications) {
      throw PackedDiscoveryResourceLimit(
          "compiled reduction exceeds max_variable_action_applications");
    }
  }

  void apply_monomial_in_place(
      PackedMonomial monomial,
      std::vector<std::uint32_t>& vector,
      std::size_t& action_applications,
      const CompiledPrimeQuotientLimits& limits) const {
    monomial.validate_for_variable_count(variable_count());
    for (std::size_t variable = 0; variable < variable_count(); ++variable) {
      for (std::uint8_t exponent = 0;
           exponent < monomial.exponent(variable); ++exponent) {
        check_action_limit(action_applications, limits);
        vector = sparse_variable_actions_[variable].multiply_column(vector);
        ++action_applications;
      }
    }
  }

  [[nodiscard]] std::vector<std::uint32_t> monomial_coordinates(
      PackedMonomial monomial,
      MonomialCache& cache,
      std::size_t& action_applications,
      const CompiledPrimeQuotientLimits& limits) const {
    const auto standard = standard_indices_.find(monomial.word());
    if (standard != standard_indices_.end()) {
      return basis_vector(standard->second);
    }
    const auto cached = cache.find(monomial.word());
    if (cached != cache.end()) {
      return cached->second;
    }
    if (limits.max_cached_monomials.has_value() &&
        cache.size() >= *limits.max_cached_monomials) {
      throw PackedDiscoveryResourceLimit(
          "compiled reduction exceeds max_cached_monomials");
    }
    auto result = basis_vector(one_index_);
    apply_monomial_in_place(
        monomial, result, action_applications, limits);
    cache.emplace(monomial.word(), result);
    return result;
  }

  [[nodiscard]] std::vector<std::uint32_t> reduce_one(
      const PackedPrimePolynomial& polynomial,
      MonomialCache& cache,
      std::size_t& action_applications,
      const CompiledPrimeQuotientLimits& limits) const {
    std::vector<std::uint32_t> result(dimension(), 0);
    if (dimension() == 0) {
      return result;
    }
    for (std::size_t term = 0; term < polynomial.term_count(); ++term) {
      const auto monomial = PackedMonomial::from_word(
          polynomial.monomial_words()[term]);
      const auto coordinates = monomial_coordinates(
          monomial, cache, action_applications, limits);
      const auto coefficient = polynomial.coefficients()[term];
      for (std::size_t row = 0; row < dimension(); ++row) {
        result[row] = arithmetic_.add(
            result[row], arithmetic_.multiply(coefficient, coordinates[row]));
      }
    }
    return result;
  }

  void compile_variable_actions() {
    sparse_variable_actions_.reserve(variable_count());
    variable_actions_.reserve(variable_count());
    for (const auto& exact : context_.variable_multiplication_tables()) {
      auto sparse = PackedPrimeSparseMatrix::from_exact(exact);
      variable_actions_.push_back(sparse.to_dense());
      sparse_variable_actions_.push_back(std::move(sparse));
    }
    if (variable_actions_.size() != variable_count() ||
        sparse_variable_actions_.size() != variable_count()) {
      throw std::logic_error(
          "compiled quotient has the wrong number of variable actions");
    }
  }

  void compile_signatures() {
    auto append_ring_shape = [this](std::string& signature) {
      compiled_quotient_detail::append_u64(
          signature, static_cast<std::uint64_t>(order()));
      compiled_quotient_detail::append_u64(signature, variable_count());
      for (const auto& variable : context_.ring().variable_names()) {
        compiled_quotient_detail::append_string(signature, variable);
      }
    };
    compiled_quotient_detail::append_u64(full_signature_, modulus());
    append_ring_shape(full_signature_);
    // The leading signature deliberately omits the modulus so identical
    // initial monomial ideals can be compared across discovery primes.
    append_ring_shape(leading_signature_);
    const auto& basis = context_.ideal().groebner_basis();
    compiled_quotient_detail::append_u64(full_signature_, basis.size());
    compiled_quotient_detail::append_u64(leading_signature_, basis.size());
    for (const auto& polynomial : basis) {
      const auto leading = PackedMonomial::from_exact(
          polynomial.leading_term()->monomial, variable_count());
      compiled_quotient_detail::append_u64(
          leading_signature_, leading.word());
      compiled_quotient_detail::append_u64(
          full_signature_, polynomial.term_count());
      for (const auto& term : polynomial.terms()) {
        compiled_quotient_detail::append_u64(
            full_signature_,
            PackedMonomial::from_exact(term.monomial, variable_count()).word());
        compiled_quotient_detail::append_u64(
            full_signature_, term.coefficient.value());
      }
    }
    full_fingerprint_ = compiled_quotient_detail::fnv1a(full_signature_);
    leading_fingerprint_ =
        compiled_quotient_detail::fnv1a(leading_signature_);
  }

  QuotientContext<PrimeField> context_;
  PackedPrimeArithmetic arithmetic_;
  std::vector<std::uint64_t> standard_words_;
  std::unordered_map<
      std::uint64_t,
      std::size_t,
      compiled_quotient_detail::PackedWordHash>
      standard_indices_;
  std::size_t one_index_ = 0;
  std::vector<PackedPrimeMatrix> variable_actions_;
  std::vector<PackedPrimeSparseMatrix> sparse_variable_actions_;
  std::string full_signature_;
  std::string leading_signature_;
  std::uint64_t full_fingerprint_ = 0;
  std::uint64_t leading_fingerprint_ = 0;
};

}  // namespace laughableengine

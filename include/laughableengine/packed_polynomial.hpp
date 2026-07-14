#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "laughableengine/packed_prime.hpp"
#include "laughableengine/polynomial.hpp"

namespace laughableengine {

// Six 8-bit exponent lanes plus a 16-bit cached total degree.  The discovery
// contract normally uses degrees <=15; values outside this packed envelope
// are rejected and remain available through the reference Polynomial type.
class PackedMonomial {
 public:
  static constexpr std::size_t maximum_variables = 6;
  static constexpr std::uint16_t maximum_exponent = 255;

  PackedMonomial() = default;

  [[nodiscard]] static PackedMonomial from_exact(
      const Monomial& monomial,
      std::size_t variable_count) {
    validate_variable_count(variable_count);
    if (std::any_of(
            monomial.exponents().begin() +
                static_cast<std::ptrdiff_t>(variable_count),
            monomial.exponents().end(),
            [](std::uint16_t exponent) { return exponent != 0; })) {
      throw std::invalid_argument(
          "packed monomial has an exponent outside its ring");
    }
    std::uint64_t word = 0;
    std::uint32_t degree = 0;
    for (std::size_t variable = 0; variable < variable_count; ++variable) {
      const auto exponent = monomial.exponent(variable);
      if (exponent > maximum_exponent) {
        throw std::overflow_error(
            "monomial exponent exceeds the packed 8-bit lane");
      }
      word |= static_cast<std::uint64_t>(exponent) << (8U * variable);
      degree += exponent;
    }
    if (degree > std::numeric_limits<std::uint16_t>::max()) {
      throw std::overflow_error("packed monomial total degree overflows");
    }
    word |= static_cast<std::uint64_t>(degree) << 48U;
    return PackedMonomial(word);
  }

  [[nodiscard]] static PackedMonomial from_exponents(
      std::span<const std::uint16_t> exponents,
      std::size_t variable_count) {
    validate_variable_count(variable_count);
    if (exponents.size() > variable_count) {
      throw std::invalid_argument(
          "too many exponents for a packed monomial");
    }
    std::uint64_t word = 0;
    std::uint32_t degree = 0;
    for (std::size_t variable = 0; variable < exponents.size(); ++variable) {
      if (exponents[variable] > maximum_exponent) {
        throw std::overflow_error(
            "monomial exponent exceeds the packed 8-bit lane");
      }
      word |= static_cast<std::uint64_t>(exponents[variable]) <<
              (8U * variable);
      degree += exponents[variable];
    }
    word |= static_cast<std::uint64_t>(degree) << 48U;
    return PackedMonomial(word);
  }

  [[nodiscard]] static PackedMonomial from_word(std::uint64_t word) {
    std::uint32_t degree = 0;
    for (std::size_t variable = 0; variable < maximum_variables; ++variable) {
      degree += static_cast<std::uint8_t>(word >> (8U * variable));
    }
    if (degree != static_cast<std::uint16_t>(word >> 48U)) {
      throw std::invalid_argument(
          "packed monomial word has an inconsistent total degree");
    }
    return PackedMonomial(word);
  }

  [[nodiscard]] std::uint64_t word() const noexcept { return word_; }
  [[nodiscard]] std::uint8_t exponent(std::size_t variable) const {
    if (variable >= maximum_variables) {
      throw std::out_of_range("packed monomial variable is out of range");
    }
    return static_cast<std::uint8_t>(word_ >> (8U * variable));
  }
  [[nodiscard]] std::uint16_t total_degree() const noexcept {
    return static_cast<std::uint16_t>(word_ >> 48U);
  }

  void validate_for_variable_count(std::size_t variable_count) const {
    validate_variable_count(variable_count);
    for (std::size_t variable = variable_count;
         variable < maximum_variables; ++variable) {
      if (exponent(variable) != 0) {
        throw std::invalid_argument(
            "packed monomial has a nonzero inactive exponent lane");
      }
    }
  }

  [[nodiscard]] PackedMonomial multiplied_by(
      PackedMonomial other,
      std::size_t variable_count) const {
    validate_for_variable_count(variable_count);
    other.validate_for_variable_count(variable_count);
    std::vector<std::uint16_t> exponents(variable_count, 0);
    for (std::size_t variable = 0; variable < variable_count; ++variable) {
      const auto sum = static_cast<std::uint16_t>(exponent(variable)) +
                       other.exponent(variable);
      if (sum > maximum_exponent) {
        throw std::overflow_error(
            "packed monomial multiplication overflows an exponent lane");
      }
      exponents[variable] = sum;
    }
    return from_exponents(exponents, variable_count);
  }

  [[nodiscard]] bool is_divisible_by(
      PackedMonomial divisor,
      std::size_t variable_count) const {
    validate_for_variable_count(variable_count);
    divisor.validate_for_variable_count(variable_count);
    for (std::size_t variable = 0; variable < variable_count; ++variable) {
      if (exponent(variable) < divisor.exponent(variable)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] static int compare(
      PackedMonomial left,
      PackedMonomial right,
      Order order,
      std::size_t variable_count) {
    left.validate_for_variable_count(variable_count);
    right.validate_for_variable_count(variable_count);
    if (order == Order::Lex) {
      for (std::size_t variable = 0; variable < variable_count; ++variable) {
        if (left.exponent(variable) != right.exponent(variable)) {
          return left.exponent(variable) > right.exponent(variable) ? 1 : -1;
        }
      }
      return 0;
    }
    if (left.total_degree() != right.total_degree()) {
      return left.total_degree() > right.total_degree() ? 1 : -1;
    }
    for (std::size_t variable = variable_count; variable-- > 0;) {
      if (left.exponent(variable) != right.exponent(variable)) {
        return left.exponent(variable) < right.exponent(variable) ? 1 : -1;
      }
    }
    return 0;
  }

  friend bool operator==(PackedMonomial, PackedMonomial) = default;

 private:
  explicit PackedMonomial(std::uint64_t word) : word_(word) {}

  static void validate_variable_count(std::size_t variable_count) {
    if (variable_count == 0 || variable_count > maximum_variables) {
      throw std::invalid_argument(
          "packed monomials require between one and six variables");
    }
  }

  std::uint64_t word_ = 0;
};

struct PackedPrimeTerm {
  PackedMonomial monomial;
  std::uint32_t coefficient;

  friend bool operator==(const PackedPrimeTerm&, const PackedPrimeTerm&) =
      default;
};

// A structure-of-arrays finite-field polynomial. Terms are canonical,
// nonzero, and sorted in descending ring order.
class PackedPrimePolynomial {
 public:
  PackedPrimePolynomial(
      std::uint32_t modulus,
      std::size_t variable_count,
      Order order,
      std::vector<PackedPrimeTerm> terms)
      : arithmetic_(modulus),
        variable_count_(variable_count),
        order_(order) {
    if (variable_count_ == 0 ||
        variable_count_ > PackedMonomial::maximum_variables) {
      throw std::invalid_argument(
          "packed polynomials require between one and six variables");
    }
    for (const auto& term : terms) {
      term.monomial.validate_for_variable_count(variable_count_);
    }
    normalize(std::move(terms));
  }

  [[nodiscard]] static PackedPrimePolynomial from_exact(
      const Polynomial<PrimeField>& polynomial) {
    std::vector<PackedPrimeTerm> terms;
    terms.reserve(polynomial.term_count());
    for (const auto& term : polynomial.terms()) {
      terms.push_back(PackedPrimeTerm{
          PackedMonomial::from_exact(
              term.monomial, polynomial.variable_count()),
          term.coefficient.value()});
    }
    return PackedPrimePolynomial(
        polynomial.coefficient_field().modulus(), polynomial.variable_count(),
        polynomial.order(), std::move(terms));
  }

  [[nodiscard]] std::uint32_t modulus() const noexcept {
    return arithmetic_.modulus();
  }
  [[nodiscard]] std::size_t variable_count() const noexcept {
    return variable_count_;
  }
  [[nodiscard]] Order order() const noexcept { return order_; }
  [[nodiscard]] std::size_t term_count() const noexcept {
    return coefficients_.size();
  }
  [[nodiscard]] bool is_zero() const noexcept { return coefficients_.empty(); }
  [[nodiscard]] std::span<const std::uint64_t> monomial_words() const noexcept {
    return monomials_;
  }
  [[nodiscard]] std::span<const std::uint32_t> coefficients() const noexcept {
    return coefficients_;
  }

  [[nodiscard]] PackedPrimeTerm term(std::size_t index) const {
    if (index >= term_count()) {
      throw std::out_of_range("packed polynomial term is out of range");
    }
    return PackedPrimeTerm{
        PackedMonomial::from_word(monomials_[index]), coefficients_[index]};
  }

  [[nodiscard]] Polynomial<PrimeField> to_exact(
      const PolynomialRing<PrimeField>& ring) const {
    if (ring.field().modulus() != modulus() ||
        ring.variable_count() != variable_count_ || ring.order() != order_) {
      throw std::invalid_argument(
          "packed polynomial conversion requires a matching exact ring");
    }
    std::vector<typename PolynomialRing<PrimeField>::TermSpec> terms;
    terms.reserve(term_count());
    for (std::size_t index = 0; index < term_count(); ++index) {
      const auto monomial = PackedMonomial::from_word(monomials_[index]);
      std::vector<std::uint16_t> exponents(variable_count_, 0);
      for (std::size_t variable = 0; variable < variable_count_; ++variable) {
        exponents[variable] = monomial.exponent(variable);
      }
      terms.push_back(typename PolynomialRing<PrimeField>::TermSpec{
          ring.field().from_unsigned(coefficients_[index]),
          std::move(exponents)});
    }
    return ring.from_terms(std::move(terms));
  }

  [[nodiscard]] PackedPrimePolynomial operator+(
      const PackedPrimePolynomial& other) const {
    require_compatible(other);
    std::vector<PackedPrimeTerm> terms;
    terms.reserve(term_count() + other.term_count());
    std::size_t left = 0;
    std::size_t right = 0;
    while (left < term_count() || right < other.term_count()) {
      if (right == other.term_count()) {
        terms.push_back(term(left++));
      } else if (left == term_count()) {
        terms.push_back(other.term(right++));
      } else {
        const auto left_monomial = PackedMonomial::from_word(monomials_[left]);
        const auto right_monomial =
            PackedMonomial::from_word(other.monomials_[right]);
        const auto comparison = PackedMonomial::compare(
            left_monomial, right_monomial, order_, variable_count_);
        if (comparison > 0) {
          terms.push_back(term(left++));
        } else if (comparison < 0) {
          terms.push_back(other.term(right++));
        } else {
          const auto coefficient = arithmetic_.add(
              coefficients_[left], other.coefficients_[right]);
          if (coefficient != 0) {
            terms.push_back(PackedPrimeTerm{left_monomial, coefficient});
          }
          ++left;
          ++right;
        }
      }
    }
    return PackedPrimePolynomial(
        modulus(), variable_count_, order_, std::move(terms));
  }

  [[nodiscard]] PackedPrimePolynomial operator*(
      const PackedPrimePolynomial& other) const {
    require_compatible(other);
    std::unordered_map<std::uint64_t, std::uint32_t> accumulated;
    if (term_count() != 0 &&
        other.term_count() >
            std::numeric_limits<std::size_t>::max() / term_count()) {
      throw std::length_error(
          "packed polynomial product term count overflows size_t");
    }
    accumulated.reserve(term_count() * other.term_count());
    for (std::size_t left = 0; left < term_count(); ++left) {
      const auto left_monomial = PackedMonomial::from_word(monomials_[left]);
      for (std::size_t right = 0; right < other.term_count(); ++right) {
        const auto monomial = left_monomial.multiplied_by(
            PackedMonomial::from_word(other.monomials_[right]),
            variable_count_);
        const auto coefficient = arithmetic_.multiply(
            coefficients_[left], other.coefficients_[right]);
        const auto [iterator, inserted] =
            accumulated.emplace(monomial.word(), coefficient);
        if (!inserted) {
          iterator->second = arithmetic_.add(iterator->second, coefficient);
        }
      }
    }
    std::vector<PackedPrimeTerm> terms;
    terms.reserve(accumulated.size());
    for (const auto& [word, coefficient] : accumulated) {
      if (coefficient != 0) {
        terms.push_back(PackedPrimeTerm{
            PackedMonomial::from_word(word), coefficient});
      }
    }
    return PackedPrimePolynomial(
        modulus(), variable_count_, order_, std::move(terms));
  }

  friend bool operator==(
      const PackedPrimePolynomial& left,
      const PackedPrimePolynomial& right) noexcept {
    return left.modulus() == right.modulus() &&
           left.variable_count_ == right.variable_count_ &&
           left.order_ == right.order_ && left.monomials_ == right.monomials_ &&
           left.coefficients_ == right.coefficients_;
  }

 private:
  void normalize(std::vector<PackedPrimeTerm> terms) {
    std::unordered_map<std::uint64_t, std::uint32_t> accumulated;
    accumulated.reserve(terms.size());
    for (auto& term : terms) {
      const auto coefficient = arithmetic_.canonical(term.coefficient);
      const auto [iterator, inserted] =
          accumulated.emplace(term.monomial.word(), coefficient);
      if (!inserted) {
        iterator->second = arithmetic_.add(iterator->second, coefficient);
      }
    }
    terms.clear();
    terms.reserve(accumulated.size());
    for (const auto& [word, coefficient] : accumulated) {
      if (coefficient != 0) {
        terms.push_back(PackedPrimeTerm{
            PackedMonomial::from_word(word), coefficient});
      }
    }
    std::sort(terms.begin(), terms.end(), [this](const auto& left,
                                                 const auto& right) {
      return PackedMonomial::compare(
                 left.monomial, right.monomial, order_, variable_count_) > 0;
    });
    monomials_.reserve(terms.size());
    coefficients_.reserve(terms.size());
    for (const auto& term : terms) {
      monomials_.push_back(term.monomial.word());
      coefficients_.push_back(term.coefficient);
    }
  }

  void require_compatible(const PackedPrimePolynomial& other) const {
    if (modulus() != other.modulus() ||
        variable_count_ != other.variable_count_ || order_ != other.order_) {
      throw std::invalid_argument(
          "packed polynomial operations require one matching ring");
    }
  }

  PackedPrimeArithmetic arithmetic_;
  std::size_t variable_count_;
  Order order_;
  std::vector<std::uint64_t> monomials_;
  std::vector<std::uint32_t> coefficients_;
};

}  // namespace laughableengine

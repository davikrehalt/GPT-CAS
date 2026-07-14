#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/polynomial.hpp"

namespace laughableengine {

struct TruncatedMonomialSpaceLimits {
  // One million exact monomials use tens of megabytes with the fixed-width
  // Monomial representation.  Larger spaces are possible, but callers must
  // request them deliberately.
  std::size_t maximum_monomials = 1'000'000;
};

class TruncatedMonomialSpaceResourceLimit : public std::runtime_error {
 public:
  TruncatedMonomialSpaceResourceLimit(
      std::size_t limit,
      std::size_t required)
      : std::runtime_error(
            "truncated monomial space needs " + std::to_string(required) +
            " monomials, exceeding the configured limit " +
            std::to_string(limit)),
        limit_(limit),
        required_(required) {}

  [[nodiscard]] std::size_t limit() const noexcept { return limit_; }
  [[nodiscard]] std::size_t required() const noexcept { return required_; }

 private:
  std::size_t limit_;
  std::size_t required_;
};

struct MonomialIndexRange {
  std::size_t first = 0;
  std::size_t past_last = 0;

  [[nodiscard]] std::size_t size() const noexcept {
    return past_last - first;
  }
  [[nodiscard]] bool empty() const noexcept { return first == past_last; }

  friend bool operator==(const MonomialIndexRange&, const MonomialIndexRange&) =
      default;
};

namespace truncated_monomial_detail {

struct BoundedBinomial {
  std::size_t value;
  bool exceeds_bound;
};

// Return C(upper, lower), or report that it is larger than bound.  Cancelling
// the denominator before multiplication avoids both transient overflow and a
// dependency on non-portable wide integers.
[[nodiscard]] inline BoundedBinomial binomial_bounded(
    std::size_t upper,
    std::size_t lower,
    std::size_t bound) noexcept {
  if (lower > upper) {
    return {0, false};
  }
  lower = std::min(lower, upper - lower);
  if (bound == 0) {
    return {0, true};
  }

  std::size_t result = 1;
  for (std::size_t step = 1; step <= lower; ++step) {
    std::size_t numerator = upper - lower + step;
    std::size_t denominator = step;

    auto factor = std::gcd(numerator, denominator);
    numerator /= factor;
    denominator /= factor;
    factor = std::gcd(result, denominator);
    result /= factor;
    denominator /= factor;

    // The binomial recurrence is integral at every step.  Reaching this
    // branch would therefore indicate an implementation error, not bad input.
    if (denominator != 1) {
      return {0, true};
    }
    if (result > bound / numerator) {
      return {bound, true};
    }
    result *= numerator;
  }
  return {result, false};
}

}  // namespace truncated_monomial_detail

// Coordinates for k[x_0,...,x_{n-1}] modulo all monomials of degree at least
// cutoff.  The order is deliberately simple and stable: total degree first,
// ascending, then exponent tuples in descending lexicographic order.
class TruncatedMonomialSpace {
 public:
  using Index = std::size_t;

  TruncatedMonomialSpace(
      std::size_t variable_count,
      std::size_t cutoff,
      TruncatedMonomialSpaceLimits limits = {})
      : variable_count_(variable_count), cutoff_(cutoff) {
    validate_shape();

    const auto dimension = truncated_monomial_detail::binomial_bounded(
        variable_count_ + cutoff_ - 1, variable_count_,
        std::numeric_limits<std::size_t>::max());
    if (dimension.exceeds_bound) {
      throw std::overflow_error(
          "truncated monomial space dimension does not fit in size_t");
    }
    if (dimension.value > limits.maximum_monomials) {
      throw TruncatedMonomialSpaceResourceLimit(
          limits.maximum_monomials, dimension.value);
    }

    monomials_.reserve(dimension.value);
    degree_ranges_.reserve(cutoff_);
    Monomial::Exponents exponents{};
    for (std::size_t degree = 0; degree < cutoff_; ++degree) {
      const auto first = monomials_.size();
      append_fixed_degree(
          0, static_cast<std::uint32_t>(degree), exponents);
      degree_ranges_.push_back({first, monomials_.size()});
    }
    if (monomials_.size() != dimension.value) {
      throw std::logic_error(
          "internal error: truncated monomial enumeration changed size");
    }
  }

  [[nodiscard]] std::size_t variable_count() const noexcept {
    return variable_count_;
  }
  [[nodiscard]] std::size_t cutoff() const noexcept { return cutoff_; }
  [[nodiscard]] std::size_t size() const noexcept { return monomials_.size(); }

  [[nodiscard]] const std::vector<Monomial>& monomials() const noexcept {
    return monomials_;
  }
  [[nodiscard]] const Monomial& monomial(Index index) const {
    if (index >= size()) {
      throw std::out_of_range("truncated monomial index is out of range");
    }
    return monomials_[index];
  }

  [[nodiscard]] const std::vector<MonomialIndexRange>& degree_ranges()
      const noexcept {
    return degree_ranges_;
  }
  [[nodiscard]] MonomialIndexRange degree_range(std::size_t degree) const {
    if (degree >= cutoff_) {
      throw std::out_of_range(
          "degree is outside the truncated monomial space");
    }
    return degree_ranges_[degree];
  }
  [[nodiscard]] std::span<const Monomial> monomials_of_degree(
      std::size_t degree) const {
    const auto range = degree_range(degree);
    return std::span<const Monomial>(
        monomials_.data() + range.first, range.size());
  }

  // Optional lookup is useful while truncating sparse input.  index_of is the
  // strict form for code that expects membership as an invariant.
  [[nodiscard]] std::optional<Index> find_index(
      const Monomial& monomial) const noexcept {
    for (std::size_t variable = variable_count_;
         variable < Monomial::maximum_variables; ++variable) {
      if (monomial.exponents()[variable] != 0) {
        return std::nullopt;
      }
    }
    if (monomial.total_degree() >= cutoff_) {
      return std::nullopt;
    }
    return index_from_exponents(
        monomial.exponents(), monomial.total_degree());
  }

  [[nodiscard]] Index index_of(const Monomial& monomial) const {
    const auto index = find_index(monomial);
    if (!index.has_value()) {
      throw std::out_of_range(
          "monomial is outside the truncated monomial space");
    }
    return *index;
  }

  // Return no index precisely when the product is killed by the degree
  // cutoff.  Invalid input indices remain programmer errors and throw.
  [[nodiscard]] std::optional<Index> product_index(
      Index left,
      Index right) const {
    const auto& left_monomial = monomial(left);
    const auto& right_monomial = monomial(right);
    const auto degree = static_cast<std::size_t>(
                            left_monomial.total_degree()) +
                        right_monomial.total_degree();
    if (degree >= cutoff_) {
      return std::nullopt;
    }

    Monomial::Exponents product{};
    for (std::size_t variable = 0; variable < variable_count_; ++variable) {
      product[variable] = static_cast<std::uint16_t>(
          left_monomial.exponents()[variable] +
          right_monomial.exponents()[variable]);
    }
    return index_from_exponents(product, degree);
  }

  [[nodiscard]] std::optional<Index> multiply_by_variable(
      Index source,
      std::size_t variable) const {
    require_variable(variable);
    const auto& monomial_value = monomial(source);
    const auto degree =
        static_cast<std::size_t>(monomial_value.total_degree()) + 1;
    if (degree >= cutoff_) {
      return std::nullopt;
    }

    auto product = monomial_value.exponents();
    ++product[variable];
    return index_from_exponents(product, degree);
  }

  // The coefficient is the ordinary derivative coefficient.  Divided-power
  // callers can deliberately ignore it instead of inheriting hidden semantics.
  [[nodiscard]] std::optional<std::pair<Index, std::uint16_t>> differentiate(
      Index source,
      std::size_t variable) const {
    require_variable(variable);
    const auto& monomial_value = monomial(source);
    const auto coefficient = monomial_value.exponents()[variable];
    if (coefficient == 0) {
      return std::nullopt;
    }

    auto derivative = monomial_value.exponents();
    --derivative[variable];
    return std::pair{
        index_from_exponents(
            derivative,
            static_cast<std::size_t>(monomial_value.total_degree()) - 1),
        coefficient};
  }

  template <typename Field>
  [[nodiscard]] Polynomial<Field> as_polynomial(
      const PolynomialRing<Field>& ring,
      Index index) const {
    if (ring.variable_count() != variable_count_) {
      throw std::invalid_argument(
          "monomial conversion requires a ring with the same variable count");
    }

    const auto& source = monomial(index);
    std::vector<std::uint16_t> exponents(
        source.exponents().begin(),
        source.exponents().begin() +
            static_cast<std::ptrdiff_t>(variable_count_));
    using TermSpec = typename PolynomialRing<Field>::TermSpec;
    std::vector<TermSpec> term;
    term.push_back(TermSpec{ring.field().one(), std::move(exponents)});
    return ring.from_terms(std::move(term));
  }

 private:
  void require_variable(std::size_t variable) const {
    if (variable >= variable_count_) {
      throw std::out_of_range(
          "truncated monomial variable index is out of range");
    }
  }

  void validate_shape() const {
    if (variable_count_ == 0 ||
        variable_count_ > Monomial::maximum_variables) {
      throw std::invalid_argument(
          "a truncated monomial space requires between one and " +
          std::to_string(Monomial::maximum_variables) + " variables");
    }
    if (cutoff_ == 0) {
      throw std::invalid_argument(
          "a truncated monomial space requires a positive degree cutoff");
    }
    if (cutoff_ - 1 > std::numeric_limits<std::uint16_t>::max()) {
      throw std::invalid_argument(
          "truncated monomial degree exceeds the exact exponent limit");
    }
  }

  void append_fixed_degree(
      std::size_t variable,
      std::uint32_t remaining,
      Monomial::Exponents& exponents) {
    if (variable + 1 == variable_count_) {
      exponents[variable] = static_cast<std::uint16_t>(remaining);
      monomials_.push_back(Monomial::from_exponents(
          std::span<const std::uint16_t>(
              exponents.data(), variable_count_),
          variable_count_));
      return;
    }

    for (std::uint32_t value = remaining;; --value) {
      exponents[variable] = static_cast<std::uint16_t>(value);
      append_fixed_degree(variable + 1, remaining - value, exponents);
      if (value == 0) {
        break;
      }
    }
  }

  [[nodiscard]] Index index_from_exponents(
      const Monomial::Exponents& exponents,
      std::size_t degree) const noexcept {
    const auto degree_offset = truncated_monomial_detail::binomial_bounded(
        variable_count_ + degree - 1, variable_count_,
        std::numeric_limits<std::size_t>::max());
    Index result = degree_offset.value;
    auto remaining = degree;
    for (std::size_t variable = 0; variable + 1 < variable_count_;
         ++variable) {
      const auto exponent = static_cast<std::size_t>(exponents[variable]);
      if (exponent < remaining) {
        const auto remaining_variables = variable_count_ - variable - 1;
        const auto skipped = truncated_monomial_detail::binomial_bounded(
            remaining - exponent - 1 + remaining_variables,
            remaining_variables, std::numeric_limits<std::size_t>::max());
        result += skipped.value;
      }
      remaining -= exponent;
    }
    return result;
  }

  std::size_t variable_count_;
  std::size_t cutoff_;
  std::vector<Monomial> monomials_;
  std::vector<MonomialIndexRange> degree_ranges_;
};

}  // namespace laughableengine

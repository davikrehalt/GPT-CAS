#pragma once

#include <algorithm>
#include <cstddef>
#include <deque>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/polynomial.hpp"

namespace laughableengine {

enum class GroebnerResourceKind {
  CriticalPairs,
  BasisPolynomials,
  ReductionSteps,
  LiveTerms,
  BasisTerms,
};

class GroebnerResourceLimit : public std::runtime_error {
 public:
  GroebnerResourceLimit(
      GroebnerResourceKind kind,
      std::size_t limit,
      std::size_t observed)
      : std::runtime_error(make_message(kind, limit, observed)),
        kind_(kind), limit_(limit), observed_(observed) {}

  [[nodiscard]] GroebnerResourceKind kind() const noexcept { return kind_; }
  [[nodiscard]] std::size_t limit() const noexcept { return limit_; }
  [[nodiscard]] std::size_t observed() const noexcept { return observed_; }

 private:
  [[nodiscard]] static const char* resource_name(
      GroebnerResourceKind kind) noexcept {
    switch (kind) {
      case GroebnerResourceKind::CriticalPairs:
        return "Buchberger critical pairs";
      case GroebnerResourceKind::BasisPolynomials:
        return "Groebner basis polynomials";
      case GroebnerResourceKind::ReductionSteps:
        return "Groebner reduction steps";
      case GroebnerResourceKind::LiveTerms:
        return "live Groebner polynomial terms";
      case GroebnerResourceKind::BasisTerms:
        return "generated Groebner basis terms";
    }
    return "Groebner resource";
  }

  [[nodiscard]] static std::string make_message(
      GroebnerResourceKind kind,
      std::size_t limit,
      std::size_t observed) {
    return std::string("Groebner resource limit exceeded for ") +
           resource_name(kind) + ": limit " + std::to_string(limit) +
           ", attempted " + std::to_string(observed);
  }

  GroebnerResourceKind kind_;
  std::size_t limit_;
  std::size_t observed_;
};

struct GroebnerLimits {
  std::optional<std::size_t> max_critical_pairs = 1'000'000;
  std::optional<std::size_t> max_basis_polynomials = 100'000;
  std::optional<std::size_t> max_reduction_steps = 10'000'000;
  std::optional<std::size_t> max_live_terms = 1'000'000;
  std::optional<std::size_t> max_basis_terms = 2'000'000;
};

template <typename Field>
struct DivisionResult {
  std::vector<Polynomial<Field>> quotients;
  Polynomial<Field> remainder;
};

// Ordered multivariate division.  The first divisor whose leading monomial
// divides the current leading monomial is used at each step.
template <typename Field>
[[nodiscard]] DivisionResult<Field> divide(
    const Polynomial<Field>& dividend,
    std::span<const Polynomial<Field>> ordered_divisors) {
  for (const auto& divisor : ordered_divisors) {
    if (!dividend.same_ring(divisor)) {
      throw std::invalid_argument(
          "polynomial division requires one shared ring context");
    }
    if (divisor.is_zero()) {
      throw std::invalid_argument("a polynomial divisor cannot be zero");
    }
  }

  std::vector<Polynomial<Field>> quotients;
  quotients.reserve(ordered_divisors.size());
  for (std::size_t index = 0; index < ordered_divisors.size(); ++index) {
    quotients.push_back(dividend.zero_like());
  }

  auto pending = dividend;
  auto remainder = dividend.zero_like();
  while (!pending.is_zero()) {
    const auto leading = *pending.leading_term();
    bool divided = false;

    for (std::size_t index = 0; index < ordered_divisors.size(); ++index) {
      const auto divisor_leading = *ordered_divisors[index].leading_term();
      if (!leading.monomial.is_divisible_by(divisor_leading.monomial)) {
        continue;
      }

      const auto monomial =
          leading.monomial.quotient_by(divisor_leading.monomial);
      const auto coefficient = dividend.coefficient_field().divide(
          leading.coefficient, divisor_leading.coefficient);
      const auto quotient_term = dividend.term_like(coefficient, monomial);

      quotients[index] = quotients[index] + quotient_term;
      pending = pending -
                ordered_divisors[index].multiply_by_term(coefficient, monomial);
      divided = true;
      break;
    }

    if (!divided) {
      const auto leading_polynomial =
          dividend.term_like(leading.coefficient, leading.monomial);
      remainder = remainder + leading_polynomial;
      pending = pending - leading_polynomial;
    }
  }

  return DivisionResult<Field>{std::move(quotients), std::move(remainder)};
}

template <typename Field>
[[nodiscard]] DivisionResult<Field> divide(
    const Polynomial<Field>& dividend,
    const std::vector<Polynomial<Field>>& ordered_divisors) {
  return divide(
      dividend,
      std::span<const Polynomial<Field>>(ordered_divisors.data(),
                                         ordered_divisors.size()));
}

template <typename Field>
[[nodiscard]] DivisionResult<Field> divide(
    const Polynomial<Field>& dividend,
    std::initializer_list<Polynomial<Field>> ordered_divisors) {
  return divide(
      dividend,
      std::span<const Polynomial<Field>>(ordered_divisors.begin(),
                                         ordered_divisors.size()));
}

template <typename Field>
[[nodiscard]] Polynomial<Field> normal_form(
    const Polynomial<Field>& polynomial,
    std::span<const Polynomial<Field>> ordered_divisors) {
  return divide(polynomial, ordered_divisors).remainder;
}

template <typename Field>
[[nodiscard]] Polynomial<Field> normal_form(
    const Polynomial<Field>& polynomial,
    const std::vector<Polynomial<Field>>& ordered_divisors) {
  return normal_form(
      polynomial,
      std::span<const Polynomial<Field>>(ordered_divisors.data(),
                                         ordered_divisors.size()));
}

template <typename Field>
[[nodiscard]] Polynomial<Field> normal_form(
    const Polynomial<Field>& polynomial,
    std::initializer_list<Polynomial<Field>> ordered_divisors) {
  return normal_form(
      polynomial,
      std::span<const Polynomial<Field>>(ordered_divisors.begin(),
                                         ordered_divisors.size()));
}

template <typename Field>
[[nodiscard]] Polynomial<Field> s_polynomial(
    const Polynomial<Field>& left,
    const Polynomial<Field>& right) {
  if (!left.same_ring(right)) {
    throw std::invalid_argument(
        "an S-polynomial requires one shared ring context");
  }
  if (left.is_zero() || right.is_zero()) {
    throw std::invalid_argument("an S-polynomial requires nonzero inputs");
  }

  const auto left_leading = *left.leading_term();
  const auto right_leading = *right.leading_term();
  const auto common = left_leading.monomial.lcm_with(right_leading.monomial);
  const auto left_monomial = common.quotient_by(left_leading.monomial);
  const auto right_monomial = common.quotient_by(right_leading.monomial);
  const auto& field = left.coefficient_field();

  return left.multiply_by_term(field.inverse(left_leading.coefficient),
                               left_monomial) -
         right.multiply_by_term(field.inverse(right_leading.coefficient),
                                right_monomial);
}

namespace groebner_detail {

template <typename Field>
[[nodiscard]] bool is_nonzero_constant(
    const Polynomial<Field>& polynomial) noexcept {
  return !polynomial.is_zero() &&
         polynomial.leading_term()->monomial.total_degree() == 0;
}

class Meter {
 public:
  explicit Meter(const GroebnerLimits& limits) : limits_(limits) {}

  [[nodiscard]] static std::size_t saturated_add(
      std::size_t left,
      std::size_t right) noexcept {
    return left > std::numeric_limits<std::size_t>::max() - right
               ? std::numeric_limits<std::size_t>::max()
               : left + right;
  }

  void schedule_pair() {
    scheduled_pairs_ = saturated_add(scheduled_pairs_, 1);
    check(
        GroebnerResourceKind::CriticalPairs, scheduled_pairs_,
        limits_.max_critical_pairs);
  }

  void add_reduction_step() {
    reduction_steps_ = saturated_add(reduction_steps_, 1);
    check(
        GroebnerResourceKind::ReductionSteps, reduction_steps_,
        limits_.max_reduction_steps);
  }

  template <typename Field>
  void add_basis_polynomial(const Polynomial<Field>& polynomial) {
    basis_polynomials_ = saturated_add(basis_polynomials_, 1);
    check(
        GroebnerResourceKind::BasisPolynomials, basis_polynomials_,
        limits_.max_basis_polynomials);
    basis_terms_ = saturated_add(basis_terms_, polynomial.term_count());
    check(
        GroebnerResourceKind::BasisTerms, basis_terms_,
        limits_.max_basis_terms);
  }

  void check_live_terms(std::size_t observed) const {
    check(
        GroebnerResourceKind::LiveTerms, observed, limits_.max_live_terms);
  }

 private:
  static void check(
      GroebnerResourceKind kind,
      std::size_t observed,
      const std::optional<std::size_t>& limit) {
    if (limit.has_value() && observed > *limit) {
      throw GroebnerResourceLimit(kind, *limit, observed);
    }
  }

  const GroebnerLimits& limits_;
  std::size_t scheduled_pairs_ = 0;
  std::size_t basis_polynomials_ = 0;
  std::size_t reduction_steps_ = 0;
  std::size_t basis_terms_ = 0;
};

template <typename Field>
[[nodiscard]] Polynomial<Field> bounded_normal_form(
    const Polynomial<Field>& polynomial,
    std::span<const Polynomial<Field>> ordered_divisors,
    Meter& meter) {
  for (const auto& divisor : ordered_divisors) {
    if (!polynomial.same_ring(divisor)) {
      throw std::invalid_argument(
          "polynomial division requires one shared ring context");
    }
    if (divisor.is_zero()) {
      throw std::invalid_argument("a polynomial divisor cannot be zero");
    }
  }

  meter.check_live_terms(polynomial.term_count());
  auto pending = polynomial;
  auto remainder = polynomial.zero_like();
  meter.check_live_terms(pending.term_count());
  while (!pending.is_zero()) {
    meter.add_reduction_step();
    const auto leading = *pending.leading_term();
    bool divided = false;
    for (const auto& divisor : ordered_divisors) {
      const auto divisor_leading = *divisor.leading_term();
      if (!leading.monomial.is_divisible_by(divisor_leading.monomial)) {
        continue;
      }
      const auto workspace = Meter::saturated_add(
          remainder.term_count(),
          Meter::saturated_add(
              pending.term_count(),
              Meter::saturated_add(
                  pending.term_count(), divisor.term_count())));
      meter.check_live_terms(workspace);
      const auto monomial =
          leading.monomial.quotient_by(divisor_leading.monomial);
      const auto coefficient = polynomial.coefficient_field().divide(
          leading.coefficient, divisor_leading.coefficient);
      pending = pending -
                divisor.multiply_by_term(coefficient, monomial);
      meter.check_live_terms(Meter::saturated_add(
          pending.term_count(), remainder.term_count()));
      divided = true;
      break;
    }
    if (!divided) {
      const auto workspace = Meter::saturated_add(
          remainder.term_count(),
          Meter::saturated_add(
              pending.term_count(), pending.term_count()));
      meter.check_live_terms(workspace);
      const auto leading_polynomial =
          polynomial.term_like(leading.coefficient, leading.monomial);
      remainder = remainder + leading_polynomial;
      pending = pending - leading_polynomial;
      meter.check_live_terms(Meter::saturated_add(
          pending.term_count(), remainder.term_count()));
    }
  }
  return remainder;
}

template <typename Field>
[[nodiscard]] Polynomial<Field> bounded_normal_form(
    const Polynomial<Field>& polynomial,
    const std::vector<Polynomial<Field>>& ordered_divisors,
    Meter& meter) {
  return bounded_normal_form(
      polynomial,
      std::span<const Polynomial<Field>>(
          ordered_divisors.data(), ordered_divisors.size()),
      meter);
}

template <typename Field>
[[nodiscard]] std::vector<Polynomial<Field>> reduce_basis(
    const std::vector<Polynomial<Field>>& basis,
    Meter& meter) {
  if (basis.empty()) {
    return {};
  }

  std::vector<bool> keep(basis.size(), true);
  for (std::size_t index = 0; index < basis.size(); ++index) {
    const auto& leading = basis[index].leading_term()->monomial;
    for (std::size_t other = 0; other < basis.size(); ++other) {
      if (index == other) {
        continue;
      }
      const auto& other_leading = basis[other].leading_term()->monomial;
      if (!leading.is_divisible_by(other_leading)) {
        continue;
      }
      if (!(leading == other_leading) || other < index) {
        keep[index] = false;
        break;
      }
    }
  }

  std::vector<Polynomial<Field>> reduced;
  reduced.reserve(basis.size());
  for (std::size_t index = 0; index < basis.size(); ++index) {
    if (keep[index]) {
      reduced.push_back(basis[index].monic());
    }
  }

  // Pairwise-minimal leading monomials cannot disappear here.  Replacing a
  // basis element by this remainder preserves both its leading monomial and
  // the generated ideal.
  for (std::size_t index = 0; index < reduced.size(); ++index) {
    std::vector<Polynomial<Field>> others;
    others.reserve(reduced.size() - 1);
    for (std::size_t other = 0; other < reduced.size(); ++other) {
      if (index != other) {
        others.push_back(reduced[other]);
      }
    }

    auto remainder = bounded_normal_form(reduced[index], others, meter);
    if (remainder.is_zero()) {
      throw std::logic_error(
          "internal error while reducing a Groebner basis");
    }
    reduced[index] = remainder.monic();
  }

  std::sort(reduced.begin(), reduced.end(), [](const auto& left,
                                                const auto& right) {
    const auto comparison = left.compare_monomials(
        left.leading_term()->monomial, right.leading_term()->monomial);
    if (comparison != 0) {
      return comparison > 0;
    }
    return left.to_string() < right.to_string();
  });
  return reduced;
}

}  // namespace groebner_detail

// A deliberately unoptimized Buchberger implementation.  It is the exact
// reference path for small problems; later fast backends can be checked
// against its deterministic reduced, monic output.
template <typename Field>
[[nodiscard]] std::vector<Polynomial<Field>> groebner_basis(
    std::span<const Polynomial<Field>> generators,
    const GroebnerLimits& limits = {}) {
  if (generators.empty()) {
    return {};
  }
  for (const auto& generator : generators) {
    if (!generators.front().same_ring(generator)) {
      throw std::invalid_argument(
          "Groebner generators require one shared ring context");
    }
  }

  groebner_detail::Meter meter(limits);
  std::vector<Polynomial<Field>> basis;
  if (limits.max_basis_polynomials.has_value() &&
      generators.size() > *limits.max_basis_polynomials) {
    throw GroebnerResourceLimit(
        GroebnerResourceKind::BasisPolynomials,
        *limits.max_basis_polynomials, generators.size());
  }
  basis.reserve(generators.size());
  for (const auto& generator : generators) {
    if (generator.is_zero()) {
      continue;
    }
    meter.check_live_terms(generator.term_count());
    auto remainder = basis.empty()
                         ? generator
                         : groebner_detail::bounded_normal_form(
                               generator, basis, meter);
    meter.check_live_terms(remainder.term_count());
    if (remainder.is_zero()) {
      continue;
    }
    remainder = remainder.monic();
    if (groebner_detail::is_nonzero_constant(remainder)) {
      return std::vector<Polynomial<Field>>{remainder.one_like()};
    }
    meter.add_basis_polynomial(remainder);
    basis.push_back(std::move(remainder));
  }

  if (basis.empty()) {
    return {};
  }

  std::deque<std::pair<std::size_t, std::size_t>> pairs;
  for (std::size_t right = 1; right < basis.size(); ++right) {
    for (std::size_t left = 0; left < right; ++left) {
      meter.schedule_pair();
      pairs.emplace_back(left, right);
    }
  }

  while (!pairs.empty()) {
    const auto [left, right] = pairs.front();
    pairs.pop_front();

    const auto input_terms = groebner_detail::Meter::saturated_add(
        basis[left].term_count(), basis[right].term_count());
    meter.check_live_terms(groebner_detail::Meter::saturated_add(
        input_terms, input_terms));
    auto s = s_polynomial(basis[left], basis[right]);
    meter.check_live_terms(s.term_count());
    auto remainder = groebner_detail::bounded_normal_form(s, basis, meter);
    if (remainder.is_zero()) {
      continue;
    }
    remainder = remainder.monic();
    if (groebner_detail::is_nonzero_constant(remainder)) {
      return std::vector<Polynomial<Field>>{remainder.one_like()};
    }

    const auto new_index = basis.size();
    meter.add_basis_polynomial(remainder);
    basis.push_back(std::move(remainder));
    for (std::size_t index = 0; index < new_index; ++index) {
      meter.schedule_pair();
      pairs.emplace_back(index, new_index);
    }
  }

  return groebner_detail::reduce_basis(basis, meter);
}

template <typename Field>
[[nodiscard]] std::vector<Polynomial<Field>> groebner_basis(
    const std::vector<Polynomial<Field>>& generators,
    const GroebnerLimits& limits = {}) {
  return groebner_basis(
      std::span<const Polynomial<Field>>(generators.data(), generators.size()),
      limits);
}

template <typename Field>
[[nodiscard]] std::vector<Polynomial<Field>> groebner_basis(
    std::initializer_list<Polynomial<Field>> generators,
    const GroebnerLimits& limits = {}) {
  return groebner_basis(
      std::span<const Polynomial<Field>>(generators.begin(), generators.size()),
      limits);
}

template <typename Field>
[[nodiscard]] bool is_groebner_basis(
    std::span<const Polynomial<Field>> candidates,
    const GroebnerLimits& limits = {}) {
  if (candidates.empty()) {
    return true;
  }
  for (const auto& candidate : candidates) {
    if (!candidates.front().same_ring(candidate)) {
      throw std::invalid_argument(
          "Groebner basis checking requires one shared ring context");
    }
  }

  groebner_detail::Meter meter(limits);
  std::vector<Polynomial<Field>> nonzero;
  if (limits.max_basis_polynomials.has_value() &&
      candidates.size() > *limits.max_basis_polynomials) {
    throw GroebnerResourceLimit(
        GroebnerResourceKind::BasisPolynomials,
        *limits.max_basis_polynomials, candidates.size());
  }
  nonzero.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    if (!candidate.is_zero()) {
      meter.check_live_terms(candidate.term_count());
      meter.add_basis_polynomial(candidate);
      nonzero.push_back(candidate);
    }
  }

  for (std::size_t right = 1; right < nonzero.size(); ++right) {
    for (std::size_t left = 0; left < right; ++left) {
      meter.schedule_pair();
      const auto input_terms = groebner_detail::Meter::saturated_add(
          nonzero[left].term_count(), nonzero[right].term_count());
      meter.check_live_terms(groebner_detail::Meter::saturated_add(
          input_terms, input_terms));
      const auto s = s_polynomial(nonzero[left], nonzero[right]);
      if (!groebner_detail::bounded_normal_form(s, nonzero, meter).is_zero()) {
        return false;
      }
    }
  }
  return true;
}

template <typename Field>
[[nodiscard]] bool is_groebner_basis(
    const std::vector<Polynomial<Field>>& candidates,
    const GroebnerLimits& limits = {}) {
  return is_groebner_basis(
      std::span<const Polynomial<Field>>(candidates.data(), candidates.size()),
      limits);
}

template <typename Field>
[[nodiscard]] bool is_groebner_basis(
    std::initializer_list<Polynomial<Field>> candidates,
    const GroebnerLimits& limits = {}) {
  return is_groebner_basis(
      std::span<const Polynomial<Field>>(candidates.begin(), candidates.size()),
      limits);
}

}  // namespace laughableengine

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "laughableengine/groebner.hpp"
#include "laughableengine/polynomial.hpp"

namespace laughableengine {

enum class BasisValidation { Verify, AssumeVerified };

enum class ResourceLimitKind {
  ReductionSteps,
  LiveTerms,
  BatchReductionSteps,
  StandardMonomials,
  BorderTests,
  MatrixEntries,
};

namespace quotient_detail {

inline std::string_view resource_limit_name(ResourceLimitKind kind) noexcept {
  switch (kind) {
    case ResourceLimitKind::ReductionSteps:
      return "reduction steps";
    case ResourceLimitKind::LiveTerms:
      return "live terms";
    case ResourceLimitKind::BatchReductionSteps:
      return "batch reduction steps";
    case ResourceLimitKind::StandardMonomials:
      return "standard monomials";
    case ResourceLimitKind::BorderTests:
      return "standard-monomial border tests";
    case ResourceLimitKind::MatrixEntries:
      return "dense matrix entries";
  }
  return "unknown resource";
}

inline std::size_t attempted_after(std::size_t value) noexcept {
  return value == std::numeric_limits<std::size_t>::max() ? value : value + 1;
}

}  // namespace quotient_detail

class ResourceLimitExceeded : public std::runtime_error {
 public:
  ResourceLimitExceeded(
      ResourceLimitKind kind,
      std::size_t limit,
      std::size_t observed,
      std::optional<std::size_t> batch_index = std::nullopt)
      : std::runtime_error(make_message(kind, limit, observed, batch_index)),
        kind_(kind),
        limit_(limit),
        observed_(observed),
        batch_index_(batch_index) {}

  [[nodiscard]] ResourceLimitKind kind() const noexcept { return kind_; }
  [[nodiscard]] std::size_t limit() const noexcept { return limit_; }
  [[nodiscard]] std::size_t observed() const noexcept { return observed_; }
  [[nodiscard]] std::optional<std::size_t> batch_index() const noexcept {
    return batch_index_;
  }

 private:
  static std::string make_message(
      ResourceLimitKind kind,
      std::size_t limit,
      std::size_t observed,
      std::optional<std::size_t> batch_index) {
    std::string message = "resource limit exceeded for ";
    message += quotient_detail::resource_limit_name(kind);
    message += ": limit ";
    message += std::to_string(limit);
    message += ", attempted ";
    message += std::to_string(observed);
    if (batch_index.has_value()) {
      message += " while reducing batch item ";
      message += std::to_string(*batch_index);
    }
    return message;
  }

  ResourceLimitKind kind_;
  std::size_t limit_;
  std::size_t observed_;
  std::optional<std::size_t> batch_index_;
};

class InfiniteQuotientError : public std::domain_error {
 public:
  InfiniteQuotientError()
      : std::domain_error(
            "standard monomials require a finite-dimensional quotient") {}
};

struct ReductionLimits {
  std::optional<std::size_t> max_steps;
  std::optional<std::size_t> max_live_terms;
  std::optional<std::size_t> max_batch_steps;
};

struct StandardMonomialLimits {
  std::optional<std::size_t> max_monomials = 1'000'000;
  std::optional<std::size_t> max_border_tests = 10'000'000;
};

struct FiniteQuotientInfo {
  bool finite = false;
  bool unit_ideal = false;
  std::size_t variable_count = 0;
  std::array<std::optional<std::uint16_t>, Monomial::maximum_variables>
      pure_power_bounds{};
};

template <typename Field>
class CompiledReducer;

template <typename Field>
class StandardMonomialBasis;

namespace quotient_detail {

struct MonomialGreater {
  Order order;
  std::size_t variable_count;

  [[nodiscard]] bool operator()(
      const Monomial& left,
      const Monomial& right) const noexcept {
    return detail::compare_monomials(left, right, order, variable_count) > 0;
  }
};

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

template <typename Field>
struct CompiledDivisor {
  using Element = typename Field::Element;
  using Term = typename Polynomial<Field>::Term;

  Monomial leading_monomial;
  Element inverse_leading_coefficient;
  std::vector<Term> tail;
};

template <typename Field>
struct CompiledReducerState {
  using PolynomialType = Polynomial<Field>;

  PolynomialRing<Field> ring;
  std::vector<PolynomialType> divisors;
  std::vector<CompiledDivisor<Field>> compiled_divisors;
  std::vector<Monomial> minimal_leading_monomials;
  std::vector<Monomial> variable_monomials;
  bool groebner_mode = false;
};

template <typename Field>
[[nodiscard]] std::vector<Monomial> minimal_leading_monomials(
    const std::vector<CompiledDivisor<Field>>& divisors) {
  std::vector<Monomial> result;
  result.reserve(divisors.size());
  for (std::size_t index = 0; index < divisors.size(); ++index) {
    const auto& leading = divisors[index].leading_monomial;
    bool redundant = false;
    for (std::size_t other = 0; other < divisors.size(); ++other) {
      if (index == other) {
        continue;
      }
      const auto& candidate = divisors[other].leading_monomial;
      if (leading.is_divisible_by(candidate) &&
          (!(leading == candidate) || other < index)) {
        redundant = true;
        break;
      }
    }
    if (!redundant) {
      result.push_back(leading);
    }
  }
  return result;
}

template <typename Field>
[[nodiscard]] std::shared_ptr<const CompiledReducerState<Field>> make_state(
    const PolynomialRing<Field>& ring,
    std::vector<Polynomial<Field>> divisors,
    bool groebner_mode,
    BasisValidation validation) {
  const auto ring_zero = ring.zero();
  for (const auto& divisor : divisors) {
    if (!ring_zero.same_ring(divisor)) {
      throw std::invalid_argument(
          "a compiled reducer requires one shared ring context");
    }
  }

  if (groebner_mode) {
    std::erase_if(divisors, [](const auto& divisor) {
      return divisor.is_zero();
    });
    if (validation == BasisValidation::Verify &&
        !is_groebner_basis(divisors)) {
      throw std::invalid_argument(
          "the supplied polynomials are not a Groebner basis");
    }
  } else {
    for (const auto& divisor : divisors) {
      if (divisor.is_zero()) {
        throw std::invalid_argument(
            "a compiled polynomial divisor cannot be zero");
      }
    }
  }

  auto state = std::make_shared<CompiledReducerState<Field>>(
      CompiledReducerState<Field>{ring, std::move(divisors), {}, {}, {},
                                  groebner_mode});
  state->compiled_divisors.reserve(state->divisors.size());
  for (const auto& divisor : state->divisors) {
    const auto& terms = divisor.terms();
    const auto& leading = terms.front();
    std::vector<typename Polynomial<Field>::Term> tail(
        terms.begin() + 1, terms.end());
    state->compiled_divisors.push_back(CompiledDivisor<Field>{
        leading.monomial,
        ring.field().inverse(leading.coefficient),
        std::move(tail)});
  }

  if (groebner_mode) {
    state->minimal_leading_monomials =
        minimal_leading_monomials(state->compiled_divisors);
  }
  state->variable_monomials.reserve(ring.variable_count());
  for (std::size_t variable = 0; variable < ring.variable_count(); ++variable) {
    state->variable_monomials.push_back(
        ring.gen(variable).leading_term()->monomial);
  }
  return state;
}

template <typename Field>
void require_same_ring(
    const CompiledReducerState<Field>& state,
    const Polynomial<Field>& polynomial) {
  if (!state.ring.zero().same_ring(polynomial)) {
    throw std::invalid_argument(
        "compiled reduction requires its exact ring context");
  }
}

inline void check_live_terms(
    const ReductionLimits& limits,
    std::size_t count,
    std::optional<std::size_t> batch_index) {
  if (limits.max_live_terms.has_value() &&
      count > *limits.max_live_terms) {
    throw ResourceLimitExceeded(ResourceLimitKind::LiveTerms,
                                *limits.max_live_terms, count, batch_index);
  }
}

template <typename Field>
[[nodiscard]] Polynomial<Field> reduce_polynomial(
    const CompiledReducerState<Field>& state,
    const Polynomial<Field>& polynomial,
    const ReductionLimits& limits,
    std::size_t* total_batch_steps,
    std::optional<std::size_t> batch_index) {
  require_same_ring(state, polynomial);
  using Element = typename Field::Element;
  using Term = typename Polynomial<Field>::Term;
  using Pending = std::map<Monomial, Element, MonomialGreater>;

  Pending pending(
      MonomialGreater{state.ring.order(), state.ring.variable_count()});
  for (const auto& term : polynomial.terms()) {
    pending.emplace(term.monomial, term.coefficient);
  }
  std::vector<Term> remainder;
  remainder.reserve(polynomial.term_count());
  check_live_terms(limits, pending.size(), batch_index);

  const auto& field = state.ring.field();
  std::size_t steps = 0;
  while (!pending.empty()) {
    if (limits.max_steps.has_value() && steps >= *limits.max_steps) {
      throw ResourceLimitExceeded(
          ResourceLimitKind::ReductionSteps, *limits.max_steps,
          attempted_after(steps), batch_index);
    }
    if (total_batch_steps != nullptr && limits.max_batch_steps.has_value() &&
        *total_batch_steps >= *limits.max_batch_steps) {
      throw ResourceLimitExceeded(
          ResourceLimitKind::BatchReductionSteps, *limits.max_batch_steps,
          attempted_after(*total_batch_steps), batch_index);
    }
    ++steps;
    if (total_batch_steps != nullptr) {
      ++*total_batch_steps;
    }

    const auto leading_iterator = pending.begin();
    const auto leading_monomial = leading_iterator->first;
    const auto leading_coefficient = leading_iterator->second;

    const CompiledDivisor<Field>* selected = nullptr;
    for (const auto& divisor : state.compiled_divisors) {
      if (leading_monomial.is_divisible_by(divisor.leading_monomial)) {
        selected = &divisor;
        break;
      }
    }

    pending.erase(leading_iterator);
    if (selected == nullptr) {
      remainder.push_back(Term{leading_monomial, leading_coefficient});
      check_live_terms(
          limits, pending.size() + remainder.size(), batch_index);
      continue;
    }

    const auto monomial_factor =
        leading_monomial.quotient_by(selected->leading_monomial);
    const auto coefficient_factor = field.multiply(
        leading_coefficient, selected->inverse_leading_coefficient);
    for (const auto& tail_term : selected->tail) {
      const auto target = tail_term.monomial.multiplied_by(
          monomial_factor, state.ring.variable_count());
      const auto delta = field.negate(
          field.multiply(coefficient_factor, tail_term.coefficient));
      const auto iterator = pending.find(target);
      if (iterator == pending.end()) {
        if (!field.is_zero(delta)) {
          pending.emplace(target, delta);
        }
      } else {
        auto combined = field.add(iterator->second, delta);
        if (field.is_zero(combined)) {
          pending.erase(iterator);
        } else {
          iterator->second = std::move(combined);
        }
      }
      check_live_terms(
          limits, pending.size() + remainder.size(), batch_index);
    }
  }

  return state.ring.zero().from_terms_like(std::move(remainder));
}

template <typename Field>
[[nodiscard]] FiniteQuotientInfo finite_quotient_info(
    const CompiledReducerState<Field>& state) {
  if (!state.groebner_mode) {
    throw std::logic_error(
        "finite-quotient queries require a compiled Groebner basis");
  }

  FiniteQuotientInfo result;
  result.variable_count = state.ring.variable_count();
  for (const auto& leading : state.minimal_leading_monomials) {
    if (leading.total_degree() == 0) {
      result.finite = true;
      result.unit_ideal = true;
      return result;
    }

    std::optional<std::size_t> pure_variable;
    for (std::size_t variable = 0; variable < result.variable_count;
         ++variable) {
      if (leading.exponent(variable) == 0) {
        continue;
      }
      if (pure_variable.has_value()) {
        pure_variable.reset();
        break;
      }
      pure_variable = variable;
    }
    if (!pure_variable.has_value()) {
      continue;
    }
    const auto exponent = leading.exponent(*pure_variable);
    auto& bound = result.pure_power_bounds[*pure_variable];
    if (!bound.has_value() || exponent < *bound) {
      bound = exponent;
    }
  }

  result.finite = true;
  for (std::size_t variable = 0; variable < result.variable_count; ++variable) {
    if (!result.pure_power_bounds[variable].has_value()) {
      result.finite = false;
      break;
    }
  }
  return result;
}

}  // namespace quotient_detail

template <typename Field>
class CompiledReducer {
 public:
  using Element = typename Field::Element;
  using PolynomialType = Polynomial<Field>;

  [[nodiscard]] static CompiledReducer ordered(
      const PolynomialRing<Field>& ring,
      std::vector<PolynomialType> ordered_divisors) {
    return CompiledReducer(quotient_detail::make_state(
        ring, std::move(ordered_divisors), false,
        BasisValidation::AssumeVerified));
  }

  [[nodiscard]] static CompiledReducer ordered(
      const PolynomialRing<Field>& ring,
      std::initializer_list<PolynomialType> ordered_divisors) {
    return ordered(ring, std::vector<PolynomialType>(ordered_divisors));
  }

  [[nodiscard]] static CompiledReducer groebner(
      const PolynomialRing<Field>& ring,
      std::vector<PolynomialType> basis,
      BasisValidation validation = BasisValidation::Verify) {
    return CompiledReducer(quotient_detail::make_state(
        ring, std::move(basis), true, validation));
  }

  [[nodiscard]] static CompiledReducer groebner(
      const PolynomialRing<Field>& ring,
      std::initializer_list<PolynomialType> basis,
      BasisValidation validation = BasisValidation::Verify) {
    return groebner(ring, std::vector<PolynomialType>(basis), validation);
  }

  [[nodiscard]] const PolynomialRing<Field>& ring() const noexcept {
    return state_->ring;
  }
  [[nodiscard]] const std::vector<PolynomialType>& divisors() const noexcept {
    return state_->divisors;
  }
  [[nodiscard]] bool is_groebner_mode() const noexcept {
    return state_->groebner_mode;
  }

  [[nodiscard]] PolynomialType normal_form(
      const PolynomialType& polynomial,
      const ReductionLimits& limits = {}) const {
    return quotient_detail::reduce_polynomial(
        *state_, polynomial, limits, nullptr, std::nullopt);
  }

  [[nodiscard]] std::vector<PolynomialType> normal_forms(
      std::span<const PolynomialType> polynomials,
      const ReductionLimits& limits = {}) const {
    for (const auto& polynomial : polynomials) {
      quotient_detail::require_same_ring(*state_, polynomial);
    }

    std::vector<PolynomialType> result;
    result.reserve(polynomials.size());
    std::size_t total_steps = 0;
    for (std::size_t index = 0; index < polynomials.size(); ++index) {
      result.push_back(quotient_detail::reduce_polynomial(
          *state_, polynomials[index], limits, &total_steps, index));
    }
    return result;
  }

  [[nodiscard]] std::vector<PolynomialType> normal_forms(
      const std::vector<PolynomialType>& polynomials,
      const ReductionLimits& limits = {}) const {
    return normal_forms(
        std::span<const PolynomialType>(polynomials.data(), polynomials.size()),
        limits);
  }

  [[nodiscard]] std::vector<PolynomialType> normal_forms(
      std::initializer_list<PolynomialType> polynomials,
      const ReductionLimits& limits = {}) const {
    return normal_forms(
        std::span<const PolynomialType>(polynomials.begin(),
                                        polynomials.size()),
        limits);
  }

  [[nodiscard]] FiniteQuotientInfo finite_quotient_info() const {
    return quotient_detail::finite_quotient_info(*state_);
  }

  [[nodiscard]] bool is_zero_dimensional() const {
    return finite_quotient_info().finite;
  }

  [[nodiscard]] StandardMonomialBasis<Field> standard_monomials(
      const StandardMonomialLimits& limits = {}) const;

  [[nodiscard]] std::size_t quotient_dimension(
      const StandardMonomialLimits& limits = {}) const;

 private:
  explicit CompiledReducer(
      std::shared_ptr<const quotient_detail::CompiledReducerState<Field>> state)
      : state_(std::move(state)) {}

  std::shared_ptr<const quotient_detail::CompiledReducerState<Field>> state_;

  friend class StandardMonomialBasis<Field>;
};

template <typename Field>
class StandardMonomialBasis {
 public:
  using Element = typename Field::Element;
  using PolynomialType = Polynomial<Field>;

  [[nodiscard]] std::size_t size() const noexcept { return monomials_.size(); }
  [[nodiscard]] bool empty() const noexcept { return monomials_.empty(); }
  [[nodiscard]] const std::vector<Monomial>& monomials() const noexcept {
    return monomials_;
  }

  [[nodiscard]] std::vector<PolynomialType> polynomials() const {
    std::vector<PolynomialType> result;
    result.reserve(monomials_.size());
    const auto one = state_->ring.field().one();
    const auto anchor = state_->ring.zero();
    for (const auto& monomial : monomials_) {
      result.push_back(anchor.term_like(one, monomial));
    }
    return result;
  }

  [[nodiscard]] std::vector<Element> coordinates(
      const PolynomialType& polynomial,
      const ReductionLimits& limits = {}) const {
    const auto remainder = quotient_detail::reduce_polynomial(
        *state_, polynomial, limits, nullptr, std::nullopt);
    return coordinates_of_remainder(remainder);
  }

  [[nodiscard]] std::vector<std::vector<Element>> coordinates(
      std::span<const PolynomialType> polynomials,
      const ReductionLimits& limits = {}) const {
    for (const auto& polynomial : polynomials) {
      quotient_detail::require_same_ring(*state_, polynomial);
    }

    std::vector<std::vector<Element>> result;
    result.reserve(polynomials.size());
    std::size_t total_steps = 0;
    for (std::size_t index = 0; index < polynomials.size(); ++index) {
      const auto remainder = quotient_detail::reduce_polynomial(
          *state_, polynomials[index], limits, &total_steps, index);
      result.push_back(coordinates_of_remainder(remainder));
    }
    return result;
  }

  [[nodiscard]] std::vector<std::vector<Element>> coordinates(
      const std::vector<PolynomialType>& polynomials,
      const ReductionLimits& limits = {}) const {
    return coordinates(
        std::span<const PolynomialType>(polynomials.data(), polynomials.size()),
        limits);
  }

  [[nodiscard]] PolynomialType from_coordinates(
      std::span<const Element> coordinates) const {
    if (coordinates.size() != monomials_.size()) {
      throw std::invalid_argument(
          "a quotient coordinate vector has the wrong dimension");
    }
    std::vector<typename PolynomialType::Term> terms;
    terms.reserve(coordinates.size());
    for (std::size_t index = 0; index < coordinates.size(); ++index) {
      auto coefficient = state_->ring.field().canonical(coordinates[index]);
      if (!state_->ring.field().is_zero(coefficient)) {
        terms.push_back(typename PolynomialType::Term{
            monomials_[index], std::move(coefficient)});
      }
    }
    return state_->ring.zero().from_terms_like(std::move(terms));
  }

  [[nodiscard]] PolynomialType from_coordinates(
      const std::vector<Element>& coordinates) const {
    return from_coordinates(
        std::span<const Element>(coordinates.data(), coordinates.size()));
  }

 private:
  StandardMonomialBasis(
      std::shared_ptr<const quotient_detail::CompiledReducerState<Field>> state,
      std::vector<Monomial> monomials)
      : state_(std::move(state)), monomials_(std::move(monomials)) {
    indices_.reserve(monomials_.size());
    for (std::size_t index = 0; index < monomials_.size(); ++index) {
      indices_.emplace(monomials_[index], index);
    }
  }

  [[nodiscard]] std::vector<Element> coordinates_of_remainder(
      const PolynomialType& remainder) const {
    std::vector<Element> result(
        monomials_.size(), state_->ring.field().zero());
    for (const auto& term : remainder.terms()) {
      const auto iterator = indices_.find(term.monomial);
      if (iterator == indices_.end()) {
        throw std::logic_error(
            "internal error: a normal form is outside its standard basis");
      }
      result[iterator->second] = term.coefficient;
    }
    return result;
  }

  std::shared_ptr<const quotient_detail::CompiledReducerState<Field>> state_;
  std::vector<Monomial> monomials_;
  std::unordered_map<Monomial, std::size_t, quotient_detail::MonomialHash>
      indices_;

  friend class CompiledReducer<Field>;
};

template <typename Field>
StandardMonomialBasis<Field> CompiledReducer<Field>::standard_monomials(
    const StandardMonomialLimits& limits) const {
  const auto info = finite_quotient_info();
  if (!info.finite) {
    throw InfiniteQuotientError();
  }
  if (info.unit_ideal) {
    return StandardMonomialBasis<Field>(state_, {});
  }

  std::vector<Monomial> monomials;
  std::deque<Monomial> frontier;
  std::unordered_set<Monomial, quotient_detail::MonomialHash> seen;

  const Monomial one;
  if (limits.max_monomials.has_value() && *limits.max_monomials == 0) {
    throw ResourceLimitExceeded(ResourceLimitKind::StandardMonomials, 0, 1);
  }
  seen.insert(one);
  frontier.push_back(one);
  std::size_t border_tests = 0;

  while (!frontier.empty()) {
    const auto monomial = frontier.front();
    frontier.pop_front();
    monomials.push_back(monomial);

    for (const auto& variable : state_->variable_monomials) {
      if (limits.max_border_tests.has_value() &&
          border_tests >= *limits.max_border_tests) {
        throw ResourceLimitExceeded(
            ResourceLimitKind::BorderTests, *limits.max_border_tests,
            quotient_detail::attempted_after(border_tests));
      }
      ++border_tests;
      const auto child =
          monomial.multiplied_by(variable, state_->ring.variable_count());

      bool standard = true;
      for (const auto& leading : state_->minimal_leading_monomials) {
        if (child.is_divisible_by(leading)) {
          standard = false;
          break;
        }
      }
      if (!standard || seen.contains(child)) {
        continue;
      }
      if (limits.max_monomials.has_value() &&
          seen.size() >= *limits.max_monomials) {
        throw ResourceLimitExceeded(
            ResourceLimitKind::StandardMonomials, *limits.max_monomials,
            quotient_detail::attempted_after(seen.size()));
      }
      seen.insert(child);
      frontier.push_back(child);
    }
  }

  std::sort(monomials.begin(), monomials.end(), [this](const auto& left,
                                                       const auto& right) {
    return detail::compare_monomials(left, right, state_->ring.order(),
                                     state_->ring.variable_count()) < 0;
  });
  return StandardMonomialBasis<Field>(state_, std::move(monomials));
}

template <typename Field>
std::size_t CompiledReducer<Field>::quotient_dimension(
    const StandardMonomialLimits& limits) const {
  return standard_monomials(limits).size();
}

}  // namespace laughableengine

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/ideal.hpp"
#include "laughableengine/polynomial.hpp"
#include "laughableengine/sparse_matrix.hpp"

namespace laughableengine {

// A polynomial in the dual variables is only a coordinate representation.
// OrdinaryDifferentiation interprets X^a literally, while DividedPowers
// interprets the same displayed monomial as X^[a].  These conventions are
// equivalent after factorial rescaling when all relevant factorials are
// invertible, but they are intentionally not conflated here.
enum class ApolarityConvention {
  OrdinaryDifferentiation,
  DividedPowers,
};

enum class InverseSystemInputIssue {
  NoDualGenerators,
  ZeroDualModule,
  MixedDualRings,
  CoefficientFieldMismatch,
  VariableCountMismatch,
  DegreeTooLarge,
};

class InverseSystemInputError : public std::invalid_argument {
 public:
  explicit InverseSystemInputError(InverseSystemInputIssue issue)
      : std::invalid_argument(message(issue)), issue_(issue) {}

  [[nodiscard]] InverseSystemInputIssue issue() const noexcept {
    return issue_;
  }

 private:
  [[nodiscard]] static const char* message(
      InverseSystemInputIssue issue) noexcept {
    switch (issue) {
      case InverseSystemInputIssue::NoDualGenerators:
        return "a Macaulay inverse system requires at least one dual generator";
      case InverseSystemInputIssue::ZeroDualModule:
        return "a Macaulay inverse system requires a nonzero dual generator";
      case InverseSystemInputIssue::MixedDualRings:
        return "all dual generators must share one exact dual-ring context";
      case InverseSystemInputIssue::CoefficientFieldMismatch:
        return "the operator and dual rings must have the same coefficient field";
      case InverseSystemInputIssue::VariableCountMismatch:
        return "the operator and dual rings must have the same variable count";
      case InverseSystemInputIssue::DegreeTooLarge:
        return "the maximum dual degree plus one exceeds the monomial exponent limit";
    }
    return "invalid Macaulay inverse-system input";
  }

  InverseSystemInputIssue issue_;
};

enum class InverseSystemResourceKind {
  BasisMonomials,
  ActionRows,
  ActionNonzeros,
  DenseMatrixEntries,
  KernelCoordinates,
  EliminationLiveNonzeros,
  EliminationOperations,
  InvariantReplayChecks,
  GroebnerConstruction,
  DimensionOverflow,
};

class InverseSystemResourceLimit : public std::runtime_error {
 public:
  InverseSystemResourceLimit(
      InverseSystemResourceKind kind,
      std::size_t limit,
      std::size_t observed)
      : std::runtime_error(make_message(kind, limit, observed)),
        kind_(kind),
        limit_(limit),
        observed_(observed) {}

  [[nodiscard]] InverseSystemResourceKind kind() const noexcept {
    return kind_;
  }
  [[nodiscard]] std::size_t limit() const noexcept { return limit_; }
  [[nodiscard]] std::size_t observed() const noexcept { return observed_; }

 private:
  [[nodiscard]] static const char* resource_name(
      InverseSystemResourceKind kind) noexcept {
    switch (kind) {
      case InverseSystemResourceKind::BasisMonomials:
        return "truncated monomial basis";
      case InverseSystemResourceKind::ActionRows:
        return "inverse-system action rows";
      case InverseSystemResourceKind::ActionNonzeros:
        return "inverse-system action nonzeros";
      case InverseSystemResourceKind::DenseMatrixEntries:
        return "dense fallback matrix entries";
      case InverseSystemResourceKind::KernelCoordinates:
        return "inverse-system kernel coordinates";
      case InverseSystemResourceKind::EliminationLiveNonzeros:
        return "inverse-system sparse elimination live nonzeros";
      case InverseSystemResourceKind::EliminationOperations:
        return "inverse-system sparse elimination operations";
      case InverseSystemResourceKind::InvariantReplayChecks:
        return "inverse-system invariant replay checks";
      case InverseSystemResourceKind::GroebnerConstruction:
        return "inverse-system Groebner construction";
      case InverseSystemResourceKind::DimensionOverflow:
        return "inverse-system dimensions";
    }
    return "inverse-system resource";
  }

  [[nodiscard]] static std::string make_message(
      InverseSystemResourceKind kind,
      std::size_t limit,
      std::size_t observed) {
    return std::string("Macaulay inverse-system resource limit exceeded for ") +
           resource_name(kind) + ": limit " + std::to_string(limit) +
           ", attempted " + std::to_string(observed);
  }

  InverseSystemResourceKind kind_;
  std::size_t limit_;
  std::size_t observed_;
};

struct InverseSystemLimits {
  std::optional<std::size_t> max_basis_monomials = 1'000'000;
  std::optional<std::size_t> max_action_rows = 1'000'000;
  std::optional<std::size_t> max_action_nonzeros = 10'000'000;

  // This guard is consulted only by a SparseMatrix implementation that
  // advertises an explicit dense fallback for elimination.
  std::optional<std::size_t> max_dense_matrix_entries = 5'000'000;

  // Sparse elimination still returns a dense list of kernel coordinates.
  std::optional<std::size_t> max_kernel_coordinate_entries = 5'000'000;

  // Bound fill/work inside native sparse elimination before it can consume
  // an unbounded workspace on a pathological matrix.
  std::optional<std::size_t> max_elimination_live_nonzeros = 10'000'000;
  std::optional<std::size_t> max_elimination_operations = 100'000'000;

  // Expensive redundant kernel/operator/ideal replay is useful for tiny
  // differential tests but is not part of the production one-kernel path.
  bool verify_invariants = false;
  std::optional<std::size_t> max_invariant_replay_checks = 1'000'000;

  GroebnerLimits groebner{};
};

using InverseSystemExponent = Monomial::Exponents;

template <typename Field>
struct InverseSystemData {
  using PolynomialType = Polynomial<Field>;

  ApolarityConvention convention;
  std::vector<PolynomialType> dual_generators;
  std::uint32_t maximum_degree;

  // Both lists are ordered first by increasing total degree, then by
  // lexicographically decreasing exponent tuple in declared-variable order.
  std::vector<InverseSystemExponent> operator_exponents;
  std::vector<InverseSystemExponent> output_exponents;

  // Rows are (dual_generator, output_exponent) blocks; columns are operator
  // exponents.  Thus its shape is
  //
  //   (#dual_generators * binom(e + D, e)) x binom(e + D + 1, e).
  SparseMatrix<Field> action_matrix;
  std::size_t action_rank;
  std::size_t kernel_dimension;
  std::size_t truncated_kernel_generator_count;

  // Ideal stores the deterministic reduced monic Groebner basis, so the
  // large truncated vector-space kernel is not retained in the result.
  Ideal<Field> annihilator;
  std::size_t quotient_length;
};

namespace inverse_system_detail {

inline void check_limit(
    InverseSystemResourceKind kind,
    std::size_t observed,
    const std::optional<std::size_t>& limit) {
  if (limit.has_value() && observed > *limit) {
    throw InverseSystemResourceLimit(kind, *limit, observed);
  }
}

inline std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    InverseSystemResourceKind kind) {
  if (right != 0 &&
      left > std::numeric_limits<std::size_t>::max() / right) {
    throw InverseSystemResourceLimit(
        InverseSystemResourceKind::DimensionOverflow,
        std::numeric_limits<std::size_t>::max(),
        std::numeric_limits<std::size_t>::max());
  }
  const auto result = left * right;
  (void)kind;
  return result;
}

inline std::size_t saturated_product(
    std::size_t left,
    std::size_t right) noexcept {
  return right != 0 &&
                 left > std::numeric_limits<std::size_t>::max() / right
             ? std::numeric_limits<std::size_t>::max()
             : left * right;
}

// binom(variable_count + degree, variable_count), with cancellation before
// multiplication so the intermediate arithmetic cannot overflow sooner than
// the final result.
inline std::size_t monomial_count(
    std::size_t variable_count,
    std::uint32_t degree) {
  const auto n = variable_count + static_cast<std::size_t>(degree);
  std::size_t result = 1;
  for (std::size_t index = 1; index <= variable_count; ++index) {
    auto numerator = n - variable_count + index;
    auto denominator = index;
    const auto first_gcd = std::gcd(numerator, denominator);
    numerator /= first_gcd;
    denominator /= first_gcd;
    const auto second_gcd = std::gcd(result, denominator);
    result /= second_gcd;
    denominator /= second_gcd;
    if (denominator != 1) {
      throw std::logic_error(
          "internal error while counting inverse-system monomials");
    }
    if (numerator != 0 &&
        result > std::numeric_limits<std::size_t>::max() / numerator) {
      throw InverseSystemResourceLimit(
          InverseSystemResourceKind::DimensionOverflow,
          std::numeric_limits<std::size_t>::max(),
          std::numeric_limits<std::size_t>::max());
    }
    result *= numerator;
  }
  return result;
}

inline void append_fixed_degree_exponents(
    std::size_t variable,
    std::size_t variable_count,
    std::uint16_t remaining,
    InverseSystemExponent& current,
    std::vector<InverseSystemExponent>& output) {
  if (variable + 1 == variable_count) {
    current[variable] = remaining;
    output.push_back(current);
    return;
  }
  for (std::uint32_t value = remaining;; --value) {
    current[variable] = static_cast<std::uint16_t>(value);
    append_fixed_degree_exponents(
        variable + 1, variable_count,
        static_cast<std::uint16_t>(remaining - value), current, output);
    if (value == 0) {
      break;
    }
  }
}

inline std::vector<InverseSystemExponent> enumerate_exponents(
    std::size_t variable_count,
    std::uint32_t maximum_degree,
    std::size_t expected_count) {
  std::vector<InverseSystemExponent> result;
  result.reserve(expected_count);
  InverseSystemExponent current{};
  for (std::uint32_t degree = 0; degree <= maximum_degree; ++degree) {
    append_fixed_degree_exponents(
        0, variable_count, static_cast<std::uint16_t>(degree), current,
        result);
  }
  if (result.size() != expected_count) {
    throw std::logic_error(
        "internal error: inverse-system monomial count changed");
  }
  return result;
}

inline std::uint32_t total_degree(
    const InverseSystemExponent& exponent,
    std::size_t variable_count) noexcept {
  std::uint32_t result = 0;
  for (std::size_t variable = 0; variable < variable_count; ++variable) {
    result += exponent[variable];
  }
  return result;
}

template <typename Function>
void enumerate_divisors(
    const InverseSystemExponent& exponent,
    std::size_t variable,
    std::size_t variable_count,
    InverseSystemExponent& divisor,
    Function& function) {
  if (variable == variable_count) {
    function(divisor);
    return;
  }
  for (std::uint32_t value = 0; value <= exponent[variable]; ++value) {
    divisor[variable] = static_cast<std::uint16_t>(value);
    enumerate_divisors(
        exponent, variable + 1, variable_count, divisor, function);
  }
}

template <typename Field>
[[nodiscard]] typename Field::Element ordinary_derivative_factor(
    const Field& field,
    const InverseSystemExponent& source,
    const InverseSystemExponent& derivative,
    std::size_t variable_count) {
  auto result = field.one();
  for (std::size_t variable = 0; variable < variable_count; ++variable) {
    for (std::uint32_t step = 0; step < derivative[variable]; ++step) {
      result = field.multiply(
          result, field.from_unsigned(source[variable] - step));
    }
  }
  return result;
}

template <typename Field>
[[nodiscard]] Polynomial<Field> polynomial_from_coordinates(
    const PolynomialRing<Field>& ring,
    std::span<const InverseSystemExponent> exponents,
    std::span<const typename Field::Element> coordinates) {
  if (coordinates.size() != exponents.size()) {
    throw std::logic_error(
        "internal error: inverse-system kernel vector has the wrong size");
  }
  std::vector<typename PolynomialRing<Field>::TermSpec> terms;
  terms.reserve(coordinates.size());
  for (std::size_t index = 0; index < coordinates.size(); ++index) {
    if (ring.field().is_zero(coordinates[index])) {
      continue;
    }
    std::vector<std::uint16_t> active_exponents;
    active_exponents.reserve(ring.variable_count());
    for (std::size_t variable = 0; variable < ring.variable_count();
         ++variable) {
      active_exponents.push_back(exponents[index][variable]);
    }
    terms.push_back(typename PolynomialRing<Field>::TermSpec{
        coordinates[index], std::move(active_exponents)});
  }
  return ring.from_terms(std::move(terms));
}

template <typename Field>
[[nodiscard]] Polynomial<Field> monomial_from_exponent(
    const PolynomialRing<Field>& ring,
    const InverseSystemExponent& exponent) {
  std::vector<std::uint16_t> active_exponents;
  active_exponents.reserve(ring.variable_count());
  for (std::size_t variable = 0; variable < ring.variable_count(); ++variable) {
    active_exponents.push_back(exponent[variable]);
  }
  return ring.from_terms({typename PolynomialRing<Field>::TermSpec{
      ring.field().one(), std::move(active_exponents)}});
}

template <typename Field>
[[nodiscard]] Polynomial<Field> apply_operator(
    const Polynomial<Field>& operation,
    const Polynomial<Field>& dual,
    ApolarityConvention convention) {
  const auto& field = dual.coefficient_field();
  std::vector<typename Polynomial<Field>::Term> result_terms;
  for (const auto& operator_term : operation.terms()) {
    for (const auto& dual_term : dual.terms()) {
      if (!dual_term.monomial.is_divisible_by(operator_term.monomial)) {
        continue;
      }
      auto coefficient = field.multiply(
          operator_term.coefficient, dual_term.coefficient);
      if (convention == ApolarityConvention::OrdinaryDifferentiation) {
        InverseSystemExponent source = dual_term.monomial.exponents();
        InverseSystemExponent derivative =
            operator_term.monomial.exponents();
        coefficient = field.multiply(
            coefficient,
            ordinary_derivative_factor(
                field, source, derivative, dual.variable_count()));
      }
      if (!field.is_zero(coefficient)) {
        result_terms.push_back(typename Polynomial<Field>::Term{
            dual_term.monomial.quotient_by(operator_term.monomial),
            std::move(coefficient)});
      }
    }
  }
  return dual.from_terms_like(std::move(result_terms));
}

template <typename Matrix>
[[nodiscard]] constexpr bool uses_dense_elimination_fallback() noexcept {
  if constexpr (requires { Matrix::elimination_uses_dense_fallback; }) {
    return Matrix::elimination_uses_dense_fallback;
  }
  return false;
}

template <typename Field>
[[nodiscard]] SparseMatrix<Field> build_action_matrix(
    std::span<const Polynomial<Field>> dual_generators,
    ApolarityConvention convention,
    std::span<const InverseSystemExponent> operator_exponents,
    std::span<const InverseSystemExponent> output_exponents,
    std::size_t row_count,
    const InverseSystemLimits& limits) {
  using Matrix = SparseMatrix<Field>;
  using Triplet = typename Matrix::Triplet;

  const auto variable_count = dual_generators.front().variable_count();
  const auto& field = dual_generators.front().coefficient_field();
  std::map<InverseSystemExponent, std::size_t> operator_indices;
  std::map<InverseSystemExponent, std::size_t> output_indices;
  for (std::size_t index = 0; index < operator_exponents.size(); ++index) {
    operator_indices.emplace(operator_exponents[index], index);
  }
  for (std::size_t index = 0; index < output_exponents.size(); ++index) {
    output_indices.emplace(output_exponents[index], index);
  }

  std::vector<Triplet> triplets;
  for (std::size_t generator = 0; generator < dual_generators.size();
       ++generator) {
    for (const auto& term : dual_generators[generator].terms()) {
      const InverseSystemExponent source = term.monomial.exponents();
      InverseSystemExponent derivative{};
      auto append_derivative = [&](const InverseSystemExponent& alpha) {
        InverseSystemExponent beta{};
        for (std::size_t variable = 0; variable < variable_count; ++variable) {
          beta[variable] = source[variable] - alpha[variable];
        }

        auto coefficient = term.coefficient;
        if (convention == ApolarityConvention::OrdinaryDifferentiation) {
          coefficient = field.multiply(
              coefficient,
              ordinary_derivative_factor(
                  field, source, alpha, variable_count));
        }
        if (field.is_zero(coefficient)) {
          return;
        }

        const auto operator_iterator = operator_indices.find(alpha);
        const auto output_iterator = output_indices.find(beta);
        if (operator_iterator == operator_indices.end() ||
            output_iterator == output_indices.end()) {
          throw std::logic_error(
              "internal error: inverse-system action left its bases");
        }
        const auto attempted = triplets.size() ==
                                       std::numeric_limits<std::size_t>::max()
                                   ? triplets.size()
                                   : triplets.size() + 1;
        check_limit(
            InverseSystemResourceKind::ActionNonzeros, attempted,
            limits.max_action_nonzeros);
        triplets.push_back(Triplet{
            generator * output_exponents.size() + output_iterator->second,
            operator_iterator->second,
            std::move(coefficient)});
      };
      enumerate_divisors(
          source, 0, variable_count, derivative, append_derivative);
    }
  }
  return Matrix(
      field, row_count, operator_exponents.size(), std::move(triplets));
}

}  // namespace inverse_system_detail

template <typename Field>
[[nodiscard]] InverseSystemData<Field> macaulay_annihilator(
    const PolynomialRing<Field>& operator_ring,
    std::span<const Polynomial<Field>> dual_generators,
    ApolarityConvention convention =
        ApolarityConvention::OrdinaryDifferentiation,
    const InverseSystemLimits& limits = {}) {
  using PolynomialType = Polynomial<Field>;
  using Matrix = SparseMatrix<Field>;

  if (dual_generators.empty()) {
    throw InverseSystemInputError(
        InverseSystemInputIssue::NoDualGenerators);
  }
  for (const auto& generator : dual_generators) {
    if (!dual_generators.front().same_ring(generator)) {
      throw InverseSystemInputError(
          InverseSystemInputIssue::MixedDualRings);
    }
  }
  if (!(operator_ring.field() ==
        dual_generators.front().coefficient_field())) {
    throw InverseSystemInputError(
        InverseSystemInputIssue::CoefficientFieldMismatch);
  }
  if (operator_ring.variable_count() !=
      dual_generators.front().variable_count()) {
    throw InverseSystemInputError(
        InverseSystemInputIssue::VariableCountMismatch);
  }

  std::optional<std::uint32_t> maximum_degree;
  for (const auto& generator : dual_generators) {
    const auto degree = generator.total_degree();
    if (degree.has_value() &&
        (!maximum_degree.has_value() || *degree > *maximum_degree)) {
      maximum_degree = *degree;
    }
  }
  if (!maximum_degree.has_value()) {
    throw InverseSystemInputError(InverseSystemInputIssue::ZeroDualModule);
  }
  if (*maximum_degree >= std::numeric_limits<std::uint16_t>::max()) {
    throw InverseSystemInputError(InverseSystemInputIssue::DegreeTooLarge);
  }

  const auto variable_count = operator_ring.variable_count();
  const auto output_count = inverse_system_detail::monomial_count(
      variable_count, *maximum_degree);
  const auto operator_count = inverse_system_detail::monomial_count(
      variable_count, *maximum_degree + 1);
  inverse_system_detail::check_limit(
      InverseSystemResourceKind::BasisMonomials, output_count,
      limits.max_basis_monomials);
  inverse_system_detail::check_limit(
      InverseSystemResourceKind::BasisMonomials, operator_count,
      limits.max_basis_monomials);

  const auto row_count = inverse_system_detail::checked_product(
      dual_generators.size(), output_count,
      InverseSystemResourceKind::ActionRows);
  inverse_system_detail::check_limit(
      InverseSystemResourceKind::ActionRows, row_count,
      limits.max_action_rows);

  auto output_exponents = inverse_system_detail::enumerate_exponents(
      variable_count, *maximum_degree, output_count);
  auto operator_exponents = inverse_system_detail::enumerate_exponents(
      variable_count, *maximum_degree + 1, operator_count);

  auto action = inverse_system_detail::build_action_matrix(
      dual_generators, convention, operator_exponents, output_exponents,
      row_count, limits);

  if constexpr (
      inverse_system_detail::uses_dense_elimination_fallback<Matrix>()) {
    const auto dense_entries = inverse_system_detail::checked_product(
        row_count, operator_count,
        InverseSystemResourceKind::DenseMatrixEntries);
    inverse_system_detail::check_limit(
        InverseSystemResourceKind::DenseMatrixEntries, dense_entries,
        limits.max_dense_matrix_entries);
  }

  typename Matrix::RankKernelResult elimination{};
  try {
    SparseEliminationLimits elimination_limits;
    elimination_limits.max_live_nonzeros =
        limits.max_elimination_live_nonzeros;
    elimination_limits.max_arithmetic_operations =
        limits.max_elimination_operations;
    elimination_limits.max_kernel_coordinate_entries =
        limits.max_kernel_coordinate_entries;
    elimination = action.rank_and_kernel(elimination_limits);
  } catch (const SparseEliminationResourceLimit& error) {
    auto kind = InverseSystemResourceKind::KernelCoordinates;
    switch (error.kind()) {
      case SparseEliminationResourceKind::LiveNonzeros:
        kind = InverseSystemResourceKind::EliminationLiveNonzeros;
        break;
      case SparseEliminationResourceKind::ArithmeticOperations:
        kind = InverseSystemResourceKind::EliminationOperations;
        break;
      case SparseEliminationResourceKind::KernelCoordinates:
        kind = InverseSystemResourceKind::KernelCoordinates;
        break;
      case SparseEliminationResourceKind::KernelNonzeros:
        kind = InverseSystemResourceKind::KernelCoordinates;
        break;
      case SparseEliminationResourceKind::OutputCoordinates:
        kind = InverseSystemResourceKind::KernelCoordinates;
        break;
    }
    throw InverseSystemResourceLimit(
        kind, error.limit(), error.observed());
  }
  const auto action_rank = elimination.rank;
  if (action_rank > operator_count) {
    throw std::logic_error(
        "internal error: inverse-system action rank exceeds its domain");
  }
  const auto kernel_dimension = operator_count - action_rank;

  auto kernel_coordinates = std::move(elimination.kernel_basis);
  if (kernel_coordinates.size() != kernel_dimension) {
    throw std::logic_error(
        "internal error: inverse-system action violates rank-nullity");
  }
  if (limits.groebner.max_basis_polynomials.has_value() &&
      kernel_dimension > *limits.groebner.max_basis_polynomials) {
    throw InverseSystemResourceLimit(
        InverseSystemResourceKind::GroebnerConstruction,
        *limits.groebner.max_basis_polynomials, kernel_dimension);
  }
  const auto raw_kernel_term_bound =
      inverse_system_detail::saturated_product(
          kernel_dimension, operator_count);
  if (limits.groebner.max_basis_terms.has_value() &&
      raw_kernel_term_bound > *limits.groebner.max_basis_terms) {
    throw InverseSystemResourceLimit(
        InverseSystemResourceKind::GroebnerConstruction,
        *limits.groebner.max_basis_terms, raw_kernel_term_bound);
  }

  std::vector<PolynomialType> kernel_generators;
  kernel_generators.reserve(kernel_coordinates.size());
  for (const auto& coordinates : kernel_coordinates) {
    if (coordinates.size() != operator_count) {
      throw std::logic_error(
          "internal error: inverse-system kernel coordinate width changed");
    }
    kernel_generators.push_back(
        inverse_system_detail::polynomial_from_coordinates(
            operator_ring, operator_exponents, coordinates));
  }

  auto annihilator = [&]() {
    try {
      return Ideal<Field>(
          operator_ring, kernel_generators, limits.groebner);
    } catch (const GroebnerResourceLimit& error) {
      throw InverseSystemResourceLimit(
          InverseSystemResourceKind::GroebnerConstruction,
          error.limit(), error.observed());
    }
  }();

  // Exhaustive replay is intentionally opt-in: the production algorithm is
  // one sparse kernel followed by one compact Groebner basis. Tiny
  // differential tests enable replay to independently exercise each bridge.
  if (limits.verify_invariants) {
    std::size_t replay_checks = 0;
    auto add_replay_work = [&](std::size_t count) {
      replay_checks =
          replay_checks > std::numeric_limits<std::size_t>::max() - count
              ? std::numeric_limits<std::size_t>::max()
              : replay_checks + count;
      inverse_system_detail::check_limit(
          InverseSystemResourceKind::InvariantReplayChecks, replay_checks,
          limits.max_invariant_replay_checks);
    };
    for (std::size_t index = 0; index < kernel_generators.size(); ++index) {
      add_replay_work(action.nnz());
      const auto image = action.multiply_column(kernel_coordinates[index]);
      if (!std::all_of(image.begin(), image.end(), [&](const auto& entry) {
            return operator_ring.field().is_zero(entry);
          })) {
        throw std::logic_error(
            "internal error: returned inverse-system kernel vector is nonzero");
      }
      add_replay_work(kernel_generators[index].term_count());
      if (!annihilator.contains(kernel_generators[index])) {
        throw std::logic_error(
            "internal error: compact annihilator lost a kernel generator");
      }
      for (const auto& dual : dual_generators) {
        add_replay_work(inverse_system_detail::saturated_product(
            kernel_generators[index].term_count(), dual.term_count()));
        if (!inverse_system_detail::apply_operator(
                 kernel_generators[index], dual, convention)
                 .is_zero()) {
          throw std::logic_error(
              "internal error: an inverse-system kernel generator does not annihilate its dual module");
        }
      }
    }
    for (const auto& generator : annihilator.groebner_basis()) {
      for (const auto& dual : dual_generators) {
        add_replay_work(inverse_system_detail::saturated_product(
            generator.term_count(), dual.term_count()));
        if (!inverse_system_detail::apply_operator(
                 generator, dual, convention)
                 .is_zero()) {
          throw std::logic_error(
              "internal error: a compact annihilator generator does not annihilate its dual module");
        }
      }
    }
    for (const auto& exponent : operator_exponents) {
      if (inverse_system_detail::total_degree(exponent, variable_count) ==
          *maximum_degree + 1) {
        add_replay_work(1);
        if (!annihilator.contains(
                inverse_system_detail::monomial_from_exponent(
                    operator_ring, exponent))) {
          throw std::logic_error(
              "internal error: annihilator omitted a top-degree monomial");
        }
      }
    }
  }

  if (!annihilator.is_zero_dimensional()) {
    throw std::logic_error(
        "internal error: a finite inverse system produced a nonfinite quotient");
  }
  const auto quotient_length = annihilator.quotient_dimension();
  if (quotient_length != action_rank) {
    throw std::logic_error(
        "internal error: inverse-system action rank disagrees with quotient length");
  }

  return InverseSystemData<Field>{
      convention,
      std::vector<PolynomialType>(
          dual_generators.begin(), dual_generators.end()),
      *maximum_degree,
      std::move(operator_exponents),
      std::move(output_exponents),
      std::move(action),
      action_rank,
      kernel_dimension,
      kernel_generators.size(),
      std::move(annihilator),
      quotient_length};
}

template <typename Field>
[[nodiscard]] InverseSystemData<Field> macaulay_annihilator(
    const PolynomialRing<Field>& operator_ring,
    const std::vector<Polynomial<Field>>& dual_generators,
    ApolarityConvention convention =
        ApolarityConvention::OrdinaryDifferentiation,
    const InverseSystemLimits& limits = {}) {
  return macaulay_annihilator(
      operator_ring,
      std::span<const Polynomial<Field>>(
          dual_generators.data(), dual_generators.size()),
      convention, limits);
}

template <typename Field>
[[nodiscard]] InverseSystemData<Field> macaulay_annihilator(
    const PolynomialRing<Field>& operator_ring,
    std::initializer_list<Polynomial<Field>> dual_generators,
    ApolarityConvention convention =
        ApolarityConvention::OrdinaryDifferentiation,
    const InverseSystemLimits& limits = {}) {
  return macaulay_annihilator(
      operator_ring,
      std::span<const Polynomial<Field>>(
          dual_generators.begin(), dual_generators.size()),
      convention, limits);
}

}  // namespace laughableengine

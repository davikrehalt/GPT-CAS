#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/ideal.hpp"
#include "laughableengine/matrix.hpp"

namespace laughableengine {

enum class CycleAuditStatus {
  // The applicable computation completed.  Inspect cycle_valid and
  // faithful_cycle separately; Complete is not itself a positive result.
  Complete,
  PolynomialNotInIdeal,
  PositiveDimensional,
  UnsupportedAtOrigin,
  UnitIdeal,
  ResourceLimit,
};

struct CycleAuditLimits {
  StandardMonomialLimits standard_monomials{};
  ReductionLimits reduction{};
  GroebnerLimits groebner{};
  std::optional<std::size_t> max_matrix_entries = 5'000'000;
};

template <typename Field>
class CycleMembershipEvidence {
 public:
  using PolynomialType = Polynomial<Field>;

  CycleMembershipEvidence(
      PolynomialType polynomial,
      std::vector<PolynomialType> derivatives,
      PolynomialType polynomial_remainder,
      std::vector<PolynomialType> derivative_remainders)
      : polynomial_(std::move(polynomial)),
        derivatives_(std::move(derivatives)),
        polynomial_remainder_(std::move(polynomial_remainder)),
        derivative_remainders_(std::move(derivative_remainders)) {}

  [[nodiscard]] const PolynomialType& polynomial() const noexcept {
    return polynomial_;
  }
  [[nodiscard]] const std::vector<PolynomialType>& derivatives() const
      noexcept {
    return derivatives_;
  }
  [[nodiscard]] const PolynomialType& polynomial_remainder() const noexcept {
    return polynomial_remainder_;
  }
  [[nodiscard]] const std::vector<PolynomialType>& derivative_remainders()
      const noexcept {
    return derivative_remainders_;
  }

 private:
  PolynomialType polynomial_;
  std::vector<PolynomialType> derivatives_;
  PolynomialType polynomial_remainder_;
  std::vector<PolynomialType> derivative_remainders_;
};

template <typename Field>
class CycleColonEvidence {
 public:
  using Element = typename Field::Element;
  using PolynomialType = Polynomial<Field>;

  CycleColonEvidence(
      DenseMatrix<Field> multiplication_matrix,
      std::vector<std::vector<Element>> kernel_coordinates,
      std::vector<PolynomialType> kernel_lifts,
      std::vector<PolynomialType> lift_product_remainders,
      Ideal<Field> colon_ideal,
      std::size_t annihilator_dimension,
      std::size_t colon_quotient_length,
      std::vector<PolynomialType> ideal_in_colon_remainders,
      std::vector<PolynomialType> colon_in_ideal_remainders,
      bool colon_equals_ideal,
      bool annihilator_zero)
      : multiplication_matrix_(std::move(multiplication_matrix)),
        kernel_coordinates_(std::move(kernel_coordinates)),
        kernel_lifts_(std::move(kernel_lifts)),
        lift_product_remainders_(std::move(lift_product_remainders)),
        colon_ideal_(std::move(colon_ideal)),
        annihilator_dimension_(annihilator_dimension),
        colon_quotient_length_(colon_quotient_length),
        ideal_in_colon_remainders_(
            std::move(ideal_in_colon_remainders)),
        colon_in_ideal_remainders_(
            std::move(colon_in_ideal_remainders)),
        colon_equals_ideal_(colon_equals_ideal),
        annihilator_zero_(annihilator_zero) {}

  [[nodiscard]] const DenseMatrix<Field>& multiplication_matrix() const
      noexcept {
    return multiplication_matrix_;
  }
  [[nodiscard]] const std::vector<std::vector<Element>>& kernel_coordinates()
      const noexcept {
    return kernel_coordinates_;
  }
  [[nodiscard]] const std::vector<PolynomialType>& kernel_lifts() const
      noexcept {
    return kernel_lifts_;
  }
  [[nodiscard]] const std::vector<PolynomialType>& lift_product_remainders()
      const noexcept {
    return lift_product_remainders_;
  }
  [[nodiscard]] const Ideal<Field>& colon_ideal() const noexcept {
    return colon_ideal_;
  }
  [[nodiscard]] std::size_t annihilator_dimension() const noexcept {
    return annihilator_dimension_;
  }
  [[nodiscard]] std::size_t annihilator_length() const noexcept {
    return annihilator_dimension_;
  }
  [[nodiscard]] std::size_t colon_quotient_length() const noexcept {
    return colon_quotient_length_;
  }
  [[nodiscard]] const std::vector<PolynomialType>&
  ideal_in_colon_remainders() const noexcept {
    return ideal_in_colon_remainders_;
  }
  [[nodiscard]] const std::vector<PolynomialType>&
  colon_in_ideal_remainders() const noexcept {
    return colon_in_ideal_remainders_;
  }
  [[nodiscard]] bool colon_equals_ideal() const noexcept {
    return colon_equals_ideal_;
  }
  [[nodiscard]] bool annihilator_zero() const noexcept {
    return annihilator_zero_;
  }

 private:
  DenseMatrix<Field> multiplication_matrix_;
  std::vector<std::vector<Element>> kernel_coordinates_;
  std::vector<PolynomialType> kernel_lifts_;
  std::vector<PolynomialType> lift_product_remainders_;
  Ideal<Field> colon_ideal_;
  std::size_t annihilator_dimension_;
  std::size_t colon_quotient_length_;
  std::vector<PolynomialType> ideal_in_colon_remainders_;
  std::vector<PolynomialType> colon_in_ideal_remainders_;
  bool colon_equals_ideal_;
  bool annihilator_zero_;
};

template <typename Field>
class CycleAuditResult {
 public:
  CycleAuditResult(
      CycleAuditStatus status,
      Ideal<Field> ideal,
      bool finite_quotient,
      bool supported_at_origin,
      std::optional<std::size_t> quotient_length,
      std::optional<CycleMembershipEvidence<Field>> membership_evidence,
      std::optional<bool> polynomial_in_ideal,
      std::optional<bool> derivatives_in_ideal,
      std::optional<bool> cycle_valid,
      std::optional<Ideal<Field>> maximal_times_ideal,
      std::optional<bool> primitive,
      std::optional<Ideal<Field>> ideal_square,
      std::optional<CycleColonEvidence<Field>> colon_evidence,
      std::optional<bool> colon_equals_ideal,
      std::optional<bool> annihilator_zero,
      bool faithful_cycle,
      std::optional<std::string> resource_detail)
      : status_(status),
        ideal_(std::move(ideal)),
        finite_quotient_(finite_quotient),
        supported_at_origin_(supported_at_origin),
        quotient_length_(quotient_length),
        membership_evidence_(std::move(membership_evidence)),
        polynomial_in_ideal_(polynomial_in_ideal),
        derivatives_in_ideal_(derivatives_in_ideal),
        cycle_valid_(cycle_valid),
        maximal_times_ideal_(std::move(maximal_times_ideal)),
        primitive_(primitive),
        ideal_square_(std::move(ideal_square)),
        colon_evidence_(std::move(colon_evidence)),
        colon_equals_ideal_(colon_equals_ideal),
        annihilator_zero_(annihilator_zero),
        faithful_cycle_(faithful_cycle),
        resource_detail_(std::move(resource_detail)) {}

  [[nodiscard]] CycleAuditStatus status() const noexcept { return status_; }
  [[nodiscard]] const Ideal<Field>& ideal() const noexcept { return ideal_; }
  [[nodiscard]] bool finite_quotient() const noexcept {
    return finite_quotient_;
  }
  [[nodiscard]] bool supported_at_origin() const noexcept {
    return supported_at_origin_;
  }
  [[nodiscard]] const std::optional<std::size_t>& quotient_length() const
      noexcept {
    return quotient_length_;
  }
  [[nodiscard]] const std::optional<CycleMembershipEvidence<Field>>&
  membership_evidence() const noexcept {
    return membership_evidence_;
  }
  [[nodiscard]] const std::optional<bool>& polynomial_in_ideal() const
      noexcept {
    return polynomial_in_ideal_;
  }
  [[nodiscard]] const std::optional<bool>& derivatives_in_ideal() const
      noexcept {
    return derivatives_in_ideal_;
  }
  [[nodiscard]] const std::optional<bool>& cycle_valid() const noexcept {
    return cycle_valid_;
  }
  [[nodiscard]] const std::optional<Ideal<Field>>& maximal_times_ideal() const
      noexcept {
    return maximal_times_ideal_;
  }
  [[nodiscard]] const std::optional<bool>& primitive() const noexcept {
    return primitive_;
  }
  [[nodiscard]] const std::optional<Ideal<Field>>& ideal_square() const
      noexcept {
    return ideal_square_;
  }
  [[nodiscard]] const std::optional<CycleColonEvidence<Field>>& colon_evidence()
      const noexcept {
    return colon_evidence_;
  }
  [[nodiscard]] const std::optional<bool>& colon_equals_ideal() const noexcept {
    return colon_equals_ideal_;
  }
  [[nodiscard]] const std::optional<bool>& annihilator_zero() const noexcept {
    return annihilator_zero_;
  }
  [[nodiscard]] bool conclusive() const noexcept {
    return status_ != CycleAuditStatus::ResourceLimit;
  }
  [[nodiscard]] bool faithful_cycle() const noexcept { return faithful_cycle_; }
  [[nodiscard]] const std::optional<std::string>& resource_detail() const
      noexcept {
    return resource_detail_;
  }

 private:
  CycleAuditStatus status_;
  Ideal<Field> ideal_;
  bool finite_quotient_;
  bool supported_at_origin_;
  std::optional<std::size_t> quotient_length_;
  std::optional<CycleMembershipEvidence<Field>> membership_evidence_;
  std::optional<bool> polynomial_in_ideal_;
  std::optional<bool> derivatives_in_ideal_;
  std::optional<bool> cycle_valid_;
  std::optional<Ideal<Field>> maximal_times_ideal_;
  std::optional<bool> primitive_;
  std::optional<Ideal<Field>> ideal_square_;
  std::optional<CycleColonEvidence<Field>> colon_evidence_;
  std::optional<bool> colon_equals_ideal_;
  std::optional<bool> annihilator_zero_;
  bool faithful_cycle_;
  std::optional<std::string> resource_detail_;
};

namespace cycle_audit_detail {

class ResourceExhausted : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

template <typename PolynomialType>
[[nodiscard]] bool all_zero(const std::vector<PolynomialType>& values) {
  return std::all_of(values.begin(), values.end(), [](const auto& value) {
    return value.is_zero();
  });
}

inline void check_matrix_limit(
    std::size_t rows,
    std::size_t columns,
    const CycleAuditLimits& limits) {
  if (!limits.max_matrix_entries.has_value() || rows == 0 || columns == 0) {
    return;
  }
  const auto limit = *limits.max_matrix_entries;
  if (columns > limit || rows > limit / columns) {
    throw ResourceExhausted(
        "cycle multiplication matrix exceeds max_matrix_entries");
  }
}

template <typename Field>
[[nodiscard]] CycleColonEvidence<Field> construct_colon_evidence(
    const Ideal<Field>& ideal,
    const Polynomial<Field>& polynomial,
    const Ideal<Field>& square,
    std::size_t quotient_length,
    const CycleAuditLimits& limits) {
  using PolynomialType = Polynomial<Field>;

  const auto domain = ideal.standard_monomials(limits.standard_monomials);
  const auto codomain = square.standard_monomials(limits.standard_monomials);
  if (domain.size() != quotient_length) {
    throw std::logic_error(
        "cycle audit domain length changed during construction");
  }
  check_matrix_limit(codomain.size(), domain.size(), limits);

  const auto domain_monomials = domain.polynomials();
  std::vector<PolynomialType> products;
  products.reserve(domain_monomials.size());
  for (const auto& monomial : domain_monomials) {
    products.push_back(monomial * polynomial);
  }
  const auto product_coordinates =
      codomain.coordinates(products, limits.reduction);

  DenseMatrix<Field> multiplication(
      ideal.ring().field(), codomain.size(), domain.size());
  for (std::size_t column = 0; column < product_coordinates.size(); ++column) {
    for (std::size_t row = 0; row < product_coordinates[column].size(); ++row) {
      multiplication.set(row, column, product_coordinates[column][row]);
    }
  }

  auto kernel = multiplication.right_kernel_basis();
  const auto rank = multiplication.rank();
  if (rank > quotient_length ||
      kernel.size() != quotient_length - rank) {
    throw std::logic_error(
        "cycle multiplication matrix failed rank-nullity");
  }

  const auto generator_count = ideal.generators().size() + kernel.size();
  if (generator_count < ideal.generators().size()) {
    throw std::length_error("cycle-colon generator count overflows size_t");
  }
  if (limits.groebner.max_basis_polynomials.has_value() &&
      generator_count > *limits.groebner.max_basis_polynomials) {
    throw GroebnerResourceLimit(
        GroebnerResourceKind::BasisPolynomials,
        *limits.groebner.max_basis_polynomials, generator_count);
  }
  std::size_t raw_term_bound = 0;
  for (const auto& generator : ideal.generators()) {
    raw_term_bound = groebner_detail::Meter::saturated_add(
        raw_term_bound, generator.term_count());
  }
  const auto lift_term_bound =
      kernel.size() != 0 &&
              domain.size() >
                  std::numeric_limits<std::size_t>::max() / kernel.size()
          ? std::numeric_limits<std::size_t>::max()
          : kernel.size() * domain.size();
  raw_term_bound = groebner_detail::Meter::saturated_add(
      raw_term_bound, lift_term_bound);
  if (limits.groebner.max_basis_terms.has_value() &&
      raw_term_bound > *limits.groebner.max_basis_terms) {
    throw GroebnerResourceLimit(
        GroebnerResourceKind::BasisTerms,
        *limits.groebner.max_basis_terms, raw_term_bound);
  }

  std::vector<PolynomialType> lifts;
  lifts.reserve(kernel.size());
  for (const auto& coordinates : kernel) {
    lifts.push_back(domain.from_coordinates(coordinates));
  }

  std::vector<PolynomialType> lift_products;
  lift_products.reserve(lifts.size());
  for (const auto& lift : lifts) {
    lift_products.push_back(lift * polynomial);
  }
  auto lift_product_remainders =
      square.normal_forms(lift_products, limits.reduction);
  if (!all_zero(lift_product_remainders)) {
    throw std::logic_error(
        "a lifted multiplication-kernel vector has nonzero product");
  }

  std::vector<PolynomialType> colon_generators = ideal.generators();
  colon_generators.reserve(generator_count);
  colon_generators.insert(
      colon_generators.end(), lifts.begin(), lifts.end());
  Ideal<Field> colon(
      ideal.ring(), std::move(colon_generators), limits.groebner);
  const auto colon_length =
      colon.quotient_dimension(limits.standard_monomials);
  if (colon_length != rank ||
      kernel.size() + colon_length != quotient_length) {
    throw std::logic_error(
        "cycle colon failed the length-nullity identity");
  }

  auto ideal_in_colon =
      colon.normal_forms(ideal.generators(), limits.reduction);
  auto colon_in_ideal =
      ideal.normal_forms(colon.generators(), limits.reduction);
  if (!all_zero(ideal_in_colon)) {
    throw std::logic_error(
        "constructed principal colon does not contain the input ideal");
  }
  const bool equals = all_zero(colon_in_ideal);
  if (equals != (colon == ideal)) {
    throw std::logic_error(
        "cycle colon containment evidence disagrees with canonical equality");
  }
  const bool annihilator_zero = kernel.empty();
  if (annihilator_zero != equals) {
    throw std::logic_error(
        "cycle colon equality disagrees with multiplication nullity");
  }

  return CycleColonEvidence<Field>(
      std::move(multiplication), std::move(kernel), std::move(lifts),
      std::move(lift_product_remainders), std::move(colon),
      quotient_length - rank, colon_length, std::move(ideal_in_colon),
      std::move(colon_in_ideal), equals, annihilator_zero);
}

template <typename Field>
struct AuditBuilder {
  explicit AuditBuilder(const Ideal<Field>& input_ideal) : ideal(input_ideal) {}

  Ideal<Field> ideal;
  bool finite_quotient = false;
  bool supported_at_origin = false;
  std::optional<std::size_t> quotient_length;
  std::optional<CycleMembershipEvidence<Field>> membership_evidence;
  std::optional<bool> polynomial_in_ideal;
  std::optional<bool> derivatives_in_ideal;
  std::optional<bool> cycle_valid;
  std::optional<Ideal<Field>> maximal_times_ideal;
  std::optional<bool> primitive;
  std::optional<Ideal<Field>> ideal_square;
  std::optional<CycleColonEvidence<Field>> colon_evidence;
  std::optional<bool> colon_equals_ideal;
  std::optional<bool> annihilator_zero;

  [[nodiscard]] CycleAuditResult<Field> finish(
      CycleAuditStatus status,
      bool faithful_cycle,
      std::optional<std::string> resource_detail = std::nullopt) {
    return CycleAuditResult<Field>(
        status, std::move(ideal), finite_quotient, supported_at_origin,
        quotient_length,
        std::move(membership_evidence), polynomial_in_ideal,
        derivatives_in_ideal, cycle_valid, std::move(maximal_times_ideal),
        primitive, std::move(ideal_square), std::move(colon_evidence),
        colon_equals_ideal, annihilator_zero, faithful_cycle,
        std::move(resource_detail));
  }
};

template <typename Field>
[[nodiscard]] std::vector<Polynomial<Field>> derivatives_of(
    const Ideal<Field>& ideal,
    const Polynomial<Field>& polynomial) {
  std::vector<Polynomial<Field>> derivatives;
  derivatives.reserve(ideal.ring().variable_count());
  for (std::size_t variable = 0; variable < ideal.ring().variable_count();
       ++variable) {
    derivatives.push_back(polynomial.derivative(variable));
  }
  return derivatives;
}

template <typename Field>
void require_polynomial_ring(
    const Ideal<Field>& ideal,
    const Polynomial<Field>& polynomial) {
  if (!ideal.ring().zero().same_ring(polynomial)) {
    throw std::invalid_argument(
        "cycle audit requires the ideal's exact polynomial-ring context");
  }
}

}  // namespace cycle_audit_detail

template <typename Field>
[[nodiscard]] CycleAuditResult<Field> audit_cycle(
    const Ideal<Field>& ideal,
    const Polynomial<Field>& polynomial,
    const CycleAuditLimits& limits = {}) {
  using PolynomialType = Polynomial<Field>;
  cycle_audit_detail::require_polynomial_ring(ideal, polynomial);
  cycle_audit_detail::AuditBuilder<Field> result(ideal);

  try {
    auto derivatives =
        cycle_audit_detail::derivatives_of(ideal, polynomial);
    std::vector<PolynomialType> membership_inputs;
    membership_inputs.reserve(derivatives.size() + 1);
    membership_inputs.push_back(polynomial);
    membership_inputs.insert(
        membership_inputs.end(), derivatives.begin(), derivatives.end());
    auto membership_remainders =
        ideal.normal_forms(membership_inputs, limits.reduction);
    auto polynomial_remainder = membership_remainders.front();
    std::vector<PolynomialType> derivative_remainders(
        membership_remainders.begin() + 1, membership_remainders.end());
    const bool polynomial_in_ideal = polynomial_remainder.is_zero();
    const bool derivatives_in_ideal =
        cycle_audit_detail::all_zero(derivative_remainders);
    result.polynomial_in_ideal = polynomial_in_ideal;
    result.derivatives_in_ideal = derivatives_in_ideal;
    result.cycle_valid = polynomial_in_ideal && derivatives_in_ideal;
    result.membership_evidence.emplace(
        polynomial, std::move(derivatives), std::move(polynomial_remainder),
        std::move(derivative_remainders));

    std::vector<PolynomialType> maximal_generators;
    maximal_generators.reserve(ideal.ring().variable_count());
    for (std::size_t variable = 0; variable < ideal.ring().variable_count();
         ++variable) {
      maximal_generators.push_back(ideal.ring().gen(variable));
    }
    const Ideal<Field> maximal(
        ideal.ring(), std::move(maximal_generators), limits.groebner);
    result.maximal_times_ideal = maximal.product(ideal, limits.groebner);
    if (polynomial_in_ideal) {
      result.primitive =
          !result.maximal_times_ideal->contains(polynomial, limits.reduction);
    }

    result.ideal_square = ideal.square(limits.groebner);
    result.finite_quotient = ideal.is_zero_dimensional();
    if (result.finite_quotient) {
      result.quotient_length =
          ideal.quotient_dimension(limits.standard_monomials);
      result.supported_at_origin =
          ideal.supported_at_origin(limits.standard_monomials);
    }

    if (polynomial_in_ideal && result.finite_quotient) {
      result.colon_evidence =
          cycle_audit_detail::construct_colon_evidence(
              ideal, polynomial, *result.ideal_square,
              *result.quotient_length, limits);
      result.colon_equals_ideal =
          result.colon_evidence->colon_equals_ideal();
      result.annihilator_zero =
          result.colon_evidence->annihilator_zero();
    }

    CycleAuditStatus status = CycleAuditStatus::Complete;
    if (ideal.is_unit()) {
      status = CycleAuditStatus::UnitIdeal;
    } else if (!result.finite_quotient) {
      status = CycleAuditStatus::PositiveDimensional;
    } else if (!result.supported_at_origin) {
      status = CycleAuditStatus::UnsupportedAtOrigin;
    } else if (!polynomial_in_ideal) {
      status = CycleAuditStatus::PolynomialNotInIdeal;
    }

    const bool faithful =
        status == CycleAuditStatus::Complete &&
        result.cycle_valid.value_or(false) &&
        result.colon_equals_ideal.value_or(false) &&
        result.annihilator_zero.value_or(false);
    return result.finish(status, faithful);
  } catch (const ResourceLimitExceeded& error) {
    return result.finish(
        CycleAuditStatus::ResourceLimit, false, std::string(error.what()));
  } catch (const GroebnerResourceLimit& error) {
    return result.finish(
        CycleAuditStatus::ResourceLimit, false, std::string(error.what()));
  } catch (const cycle_audit_detail::ResourceExhausted& error) {
    return result.finish(
        CycleAuditStatus::ResourceLimit, false, std::string(error.what()));
  }
}

enum class ColonClosureStopStatus {
  ProperFixedPoint,
  UnitIdeal,
  ResourceLimit,
  InvalidStart,
};

struct ColonClosureLimits {
  std::size_t max_steps = 16;
  CycleAuditLimits audit{};
};

template <typename Field>
class ColonClosureStep {
 public:
  ColonClosureStep(
      std::size_t index,
      Ideal<Field> ideal,
      std::optional<std::size_t> quotient_length)
      : index_(index),
        ideal_(std::move(ideal)),
        quotient_length_(quotient_length) {}

  [[nodiscard]] std::size_t index() const noexcept { return index_; }
  [[nodiscard]] const Ideal<Field>& ideal() const noexcept { return ideal_; }
  [[nodiscard]] const std::optional<std::size_t>& quotient_length() const
      noexcept {
    return quotient_length_;
  }

 private:
  std::size_t index_;
  Ideal<Field> ideal_;
  std::optional<std::size_t> quotient_length_;
};

template <typename Field>
class ColonClosureTransition {
 public:
  using PolynomialType = Polynomial<Field>;

  ColonClosureTransition(
      std::vector<PolynomialType> current_in_next_remainders,
      std::vector<PolynomialType> next_in_current_remainders,
      bool current_subset_next,
      bool equal)
      : current_in_next_remainders_(
            std::move(current_in_next_remainders)),
        next_in_current_remainders_(
            std::move(next_in_current_remainders)),
        current_subset_next_(current_subset_next),
        equal_(equal) {}

  [[nodiscard]] const std::vector<PolynomialType>&
  current_in_next_remainders() const noexcept {
    return current_in_next_remainders_;
  }
  [[nodiscard]] const std::vector<PolynomialType>&
  next_in_current_remainders() const noexcept {
    return next_in_current_remainders_;
  }
  [[nodiscard]] bool current_subset_next() const noexcept {
    return current_subset_next_;
  }
  [[nodiscard]] bool equal() const noexcept { return equal_; }

 private:
  std::vector<PolynomialType> current_in_next_remainders_;
  std::vector<PolynomialType> next_in_current_remainders_;
  bool current_subset_next_;
  bool equal_;
};

template <typename Field>
class ColonClosureResult {
 public:
  ColonClosureResult(
      ColonClosureStopStatus status,
      std::vector<ColonClosureStep<Field>> steps,
      std::vector<ColonClosureTransition<Field>> transitions,
      std::optional<std::string> resource_detail = std::nullopt)
      : status_(status),
        steps_(std::move(steps)),
        transitions_(std::move(transitions)),
        resource_detail_(std::move(resource_detail)) {}

  [[nodiscard]] ColonClosureStopStatus status() const noexcept {
    return status_;
  }
  [[nodiscard]] const std::vector<ColonClosureStep<Field>>& steps() const
      noexcept {
    return steps_;
  }
  [[nodiscard]] const std::vector<ColonClosureTransition<Field>>& transitions()
      const noexcept {
    return transitions_;
  }
  [[nodiscard]] const std::optional<std::string>& resource_detail() const
      noexcept {
    return resource_detail_;
  }
  [[nodiscard]] bool conclusive() const noexcept {
    return status_ != ColonClosureStopStatus::ResourceLimit;
  }
  [[nodiscard]] bool faithful_fixed_point_found() const noexcept {
    return status_ == ColonClosureStopStatus::ProperFixedPoint;
  }

 private:
  ColonClosureStopStatus status_;
  std::vector<ColonClosureStep<Field>> steps_;
  std::vector<ColonClosureTransition<Field>> transitions_;
  std::optional<std::string> resource_detail_;
};

template <typename Field>
[[nodiscard]] ColonClosureResult<Field> colon_closure(
    const Ideal<Field>& initial_ideal,
    const Polynomial<Field>& polynomial,
    const ColonClosureLimits& limits = {}) {
  using PolynomialType = Polynomial<Field>;
  cycle_audit_detail::require_polynomial_ring(initial_ideal, polynomial);
  std::vector<ColonClosureStep<Field>> steps;
  std::vector<ColonClosureTransition<Field>> transitions;

  try {
    const bool finite = initial_ideal.is_zero_dimensional();
    std::optional<std::size_t> initial_length;
    if (finite) {
      initial_length = initial_ideal.quotient_dimension(
          limits.audit.standard_monomials);
    }
    steps.emplace_back(0, initial_ideal, initial_length);

    if (initial_ideal.is_unit()) {
      return ColonClosureResult<Field>(
          ColonClosureStopStatus::UnitIdeal, std::move(steps),
          std::move(transitions));
    }

    auto derivatives =
        cycle_audit_detail::derivatives_of(initial_ideal, polynomial);
    std::vector<PolynomialType> cycle_inputs{polynomial};
    cycle_inputs.insert(
        cycle_inputs.end(), derivatives.begin(), derivatives.end());
    const auto cycle_remainders = initial_ideal.normal_forms(
        cycle_inputs, limits.audit.reduction);
    if (!finite ||
        !initial_ideal.supported_at_origin(
            limits.audit.standard_monomials) ||
        !cycle_audit_detail::all_zero(cycle_remainders)) {
      return ColonClosureResult<Field>(
          ColonClosureStopStatus::InvalidStart, std::move(steps),
          std::move(transitions));
    }

    auto current = initial_ideal;
    auto current_length = *initial_length;
    for (std::size_t transition = 0; transition < limits.max_steps;
         ++transition) {
      const auto square = current.square(limits.audit.groebner);
      auto evidence = cycle_audit_detail::construct_colon_evidence(
          current, polynomial, square, current_length, limits.audit);
      auto next = evidence.colon_ideal();

      const bool included = cycle_audit_detail::all_zero(
          evidence.ideal_in_colon_remainders());
      const bool equal = evidence.colon_equals_ideal();
      if (!included) {
        throw std::logic_error(
            "colon closure is not ascending despite g lying in the ideal");
      }
      transitions.emplace_back(
          evidence.ideal_in_colon_remainders(),
          evidence.colon_in_ideal_remainders(), included, equal);
      const auto next_length = evidence.colon_quotient_length();
      if (equal && next_length != current_length) {
        throw std::logic_error(
            "a fixed colon-closure step changed quotient length");
      }
      if (!equal && !next.is_unit() && next_length >= current_length) {
        throw std::logic_error(
            "a strict proper colon-closure step did not decrease length");
      }
      current_length = next_length;
      steps.emplace_back(steps.size(), next, current_length);

      if (next.is_unit()) {
        return ColonClosureResult<Field>(
            ColonClosureStopStatus::UnitIdeal, std::move(steps),
            std::move(transitions));
      }
      if (equal) {
        return ColonClosureResult<Field>(
            ColonClosureStopStatus::ProperFixedPoint, std::move(steps),
            std::move(transitions));
      }
      current = std::move(next);
    }

    return ColonClosureResult<Field>(
        ColonClosureStopStatus::ResourceLimit, std::move(steps),
        std::move(transitions),
        "colon closure reached its maximum number of steps");
  } catch (const ResourceLimitExceeded& error) {
    return ColonClosureResult<Field>(
        ColonClosureStopStatus::ResourceLimit, std::move(steps),
        std::move(transitions), std::string(error.what()));
  } catch (const GroebnerResourceLimit& error) {
    return ColonClosureResult<Field>(
        ColonClosureStopStatus::ResourceLimit, std::move(steps),
        std::move(transitions), std::string(error.what()));
  } catch (const cycle_audit_detail::ResourceExhausted& error) {
    return ColonClosureResult<Field>(
        ColonClosureStopStatus::ResourceLimit, std::move(steps),
        std::move(transitions), std::string(error.what()));
  }
}

}  // namespace laughableengine

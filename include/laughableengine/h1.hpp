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

#include "laughableengine/cycle_audit.hpp"
#include "laughableengine/ideal.hpp"
#include "laughableengine/linear_algebra.hpp"
#include "laughableengine/matrix_space.hpp"
#include "laughableengine/quotient_context.hpp"

namespace laughableengine {

enum class FullH1InputIssue {
  UnitQuotient,
  PositiveDimensionalQuotient,
  NotSupportedAtOrigin,
};

class FullH1InputError : public std::domain_error {
 public:
  explicit FullH1InputError(FullH1InputIssue issue)
      : std::domain_error(message(issue)), issue_(issue) {}

  [[nodiscard]] FullH1InputIssue issue() const noexcept { return issue_; }

 private:
  [[nodiscard]] static const char* message(FullH1InputIssue issue) noexcept {
    switch (issue) {
      case FullH1InputIssue::UnitQuotient:
        return "full H1 requires a proper nonzero quotient";
      case FullH1InputIssue::PositiveDimensionalQuotient:
        return "full H1 requires a zero-dimensional quotient";
      case FullH1InputIssue::NotSupportedAtOrigin:
        return "full H1 socle analysis requires support at the origin";
    }
    return "invalid full H1 input";
  }

  FullH1InputIssue issue_;
};

class FullH1ResourceLimit : public std::runtime_error {
 public:
  explicit FullH1ResourceLimit(std::string detail)
      : std::runtime_error(
            "full H1 resource limit exceeded: " + std::move(detail)) {}
};

struct FullH1Options {
  ReductionLimits reduction_limits{};
  StandardMonomialLimits standard_monomial_limits{};
  MatrixSpaceSearchLimits matrix_space_limits{};
  GroebnerLimits groebner_limits{};
  // Maximum number of field elements in any dense matrix materialized by
  // the current correctness backend.  nullopt disables this guard.
  std::optional<std::size_t> max_matrix_entries = 5'000'000;
  // Independent limits for the principal-colon replay of a full-rank witness.
  // A replay that reaches any of them is retained with ResourceLimit status.
  CycleAuditLimits witness_audit_limits{};
};

template <typename Field>
struct H1ActionData {
  using Element = typename Field::Element;
  using PolynomialType = Polynomial<Field>;

  std::size_t length_Q;
  std::size_t length_P_mod_J2;
  std::size_t conormal_dimension;
  std::size_t h1_dimension;
  std::size_t socle_dimension;

  Ideal<Field> ideal;
  Ideal<Field> squared_ideal;
  QuotientContext<Field> quotient;
  QuotientContext<Field> square_quotient;

  DenseMatrix<Field> reduction_map;
  DenseMatrix<Field> differential;
  LinearSubspace<Field> conormal;
  LinearSubspace<Field> h1;
  LinearSubspace<Field> socle;

  std::vector<PolynomialType> conormal_basis;
  std::vector<PolynomialType> h1_basis;
  std::vector<PolynomialType> socle_basis;
  std::vector<DenseMatrix<Field>> variable_multiplication_matrices;

  // Each map has shape dim(J/J^2)-by-length(Q) and represents multiplication
  // by one H1 basis element, with its image written in conormal coordinates.
  std::vector<DenseMatrix<Field>> h1_multiplication_matrices;

  // A_b : Soc(Q) -> J/J^2, one matrix per H1 basis element. Together these
  // matrices are the socle-action tensor.
  std::vector<DenseMatrix<Field>> action_matrices;
  std::size_t common_product_space_dimension;
  std::size_t common_product_space_rank_bound;
  MatrixSpaceRankResult<Field> individual_rank;

  std::optional<std::vector<Element>> best_h1_coefficients;
  std::optional<PolynomialType> best_h1_polynomial;

  // Present when a base-field full-socle-rank witness exists. This is a
  // deliberately independent principal-colon replay of that witness. It may
  // itself have ResourceLimit status; callers must inspect faithful_cycle().
  std::optional<CycleAuditResult<Field>> faithful_witness_audit;

  // Diagnostic only: this is Ann_Q(H1), not evidence that one H1 element is
  // faithful. It is represented by its inverse-image ideal in P.
  Ideal<Field> common_annihilator_diagnostic;
};

namespace h1_detail {

inline std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* description) {
  if (right != 0 &&
      left > std::numeric_limits<std::size_t>::max() / right) {
    throw FullH1ResourceLimit(
        std::string(description) + " dimensions overflow size_t");
  }
  return left * right;
}

inline std::size_t checked_increment(
    std::size_t value,
    const char* description) {
  if (value == std::numeric_limits<std::size_t>::max()) {
    throw FullH1ResourceLimit(
        std::string(description) + " dimensions overflow size_t");
  }
  return value + 1;
}

inline void check_matrix_entries(
    std::size_t rows,
    std::size_t columns,
    const FullH1Options& options,
    const char* description) {
  const auto entries = checked_product(rows, columns, description);
  if (options.max_matrix_entries.has_value() &&
      entries > *options.max_matrix_entries) {
    throw FullH1ResourceLimit(
        std::string(description) + " needs " + std::to_string(entries) +
        " entries, above max_matrix_entries=" +
        std::to_string(*options.max_matrix_entries));
  }
}

inline void require(bool condition, const char* message) {
  if (!condition) {
    throw std::logic_error(message);
  }
}

template <typename Field>
[[nodiscard]] bool is_zero_vector(
    const Field& field,
    std::span<const typename Field::Element> vector) {
  return std::all_of(vector.begin(), vector.end(), [&field](const auto& entry) {
    return field.is_zero(entry);
  });
}

template <typename Field>
[[nodiscard]] bool is_zero_matrix(const DenseMatrix<Field>& matrix) {
  for (std::size_t row = 0; row < matrix.row_count(); ++row) {
    for (std::size_t column = 0; column < matrix.column_count(); ++column) {
      if (!matrix.field().is_zero(matrix(row, column))) {
        return false;
      }
    }
  }
  return true;
}

template <typename Field>
[[nodiscard]] std::vector<typename Field::Element> matrix_column(
    const DenseMatrix<Field>& matrix,
    std::size_t column) {
  if (column >= matrix.column_count()) {
    throw std::out_of_range("matrix column is out of range");
  }
  std::vector<typename Field::Element> result;
  result.reserve(matrix.row_count());
  for (std::size_t row = 0; row < matrix.row_count(); ++row) {
    result.push_back(matrix(row, column));
  }
  return result;
}

template <typename Field>
[[nodiscard]] std::vector<Polynomial<Field>> polynomial_basis(
    const LinearSubspace<Field>& subspace,
    const QuotientContext<Field>& quotient) {
  std::vector<Polynomial<Field>> result;
  result.reserve(subspace.dimension());
  for (const auto& vector : subspace.basis_vectors()) {
    result.push_back(quotient.representative(vector));
  }
  return result;
}

template <typename Field>
[[nodiscard]] DenseMatrix<Field> build_reduction_map(
    const QuotientContext<Field>& quotient,
    const QuotientContext<Field>& square_quotient,
    const FullH1Options& options) {
  check_matrix_entries(
      quotient.dimension(), square_quotient.dimension(), options,
      "P/J^2 to P/J reduction matrix");
  const auto columns = quotient.coordinates(
      square_quotient.polynomial_basis(), options.reduction_limits);
  return matrix_from_columns(
      quotient.field(), quotient.dimension(), columns);
}

template <typename Field>
[[nodiscard]] DenseMatrix<Field> build_differential(
    const QuotientContext<Field>& quotient,
    const QuotientContext<Field>& square_quotient,
    const FullH1Options& options) {
  const auto variable_count = quotient.ring().variable_count();
  const auto length = quotient.dimension();
  const auto square_length = square_quotient.dimension();
  const auto row_count = checked_product(
      variable_count, length, "cotangent differential matrix");
  check_matrix_entries(
      row_count, square_length, options, "cotangent differential matrix");

  std::vector<Polynomial<Field>> derivatives;
  derivatives.reserve(checked_product(
      square_length, variable_count, "derivative batch"));
  for (const auto& basis_element : square_quotient.polynomial_basis()) {
    for (std::size_t variable = 0; variable < variable_count; ++variable) {
      derivatives.push_back(basis_element.derivative(variable));
    }
  }
  const auto derivative_coordinates =
      quotient.coordinates(derivatives, options.reduction_limits);

  DenseMatrix<Field> differential(
      quotient.field(), row_count, square_length);
  for (std::size_t column = 0; column < square_length; ++column) {
    for (std::size_t variable = 0; variable < variable_count; ++variable) {
      const auto& coordinates =
          derivative_coordinates[column * variable_count + variable];
      for (std::size_t row = 0; row < length; ++row) {
        differential.set(variable * length + row, column, coordinates[row]);
      }
    }
  }
  return differential;
}

template <typename Field>
[[nodiscard]] std::optional<Polynomial<Field>> combine_h1_basis(
    const Ideal<Field>& ideal,
    std::span<const Polynomial<Field>> basis,
    const std::optional<std::vector<typename Field::Element>>& coefficients) {
  if (!coefficients.has_value()) {
    return std::nullopt;
  }
  if (coefficients->size() != basis.size()) {
    throw std::logic_error(
        "matrix-space witness has the wrong H1 coefficient count");
  }
  auto result = ideal.ring().zero();
  for (std::size_t index = 0; index < basis.size(); ++index) {
    result = result + basis[index].scaled((*coefficients)[index]);
  }
  return result;
}

}  // namespace h1_detail

template <typename Field>
[[nodiscard]] H1ActionData<Field> full_h1_action(
    Ideal<Field> ideal,
    const FullH1Options& options = {}) {
  using PolynomialType = Polynomial<Field>;

  try {

  if (ideal.is_unit()) {
    throw FullH1InputError(FullH1InputIssue::UnitQuotient);
  }
  if (!ideal.is_zero_dimensional()) {
    throw FullH1InputError(
        FullH1InputIssue::PositiveDimensionalQuotient);
  }

  QuotientContext<Field> quotient(
      ideal, options.standard_monomial_limits);
  if (!quotient.supported_at_origin()) {
    throw FullH1InputError(FullH1InputIssue::NotSupportedAtOrigin);
  }

  auto squared_ideal = ideal.square(options.groebner_limits);
  h1_detail::require(
      squared_ideal.is_subset_of(ideal),
      "internal error: J^2 is not contained in J");
  QuotientContext<Field> square_quotient(
      squared_ideal, options.standard_monomial_limits);
  h1_detail::require(
      square_quotient.supported_at_origin(),
      "internal error: P/J^2 lost origin support");

  const auto length = quotient.dimension();
  const auto square_length = square_quotient.dimension();
  h1_detail::require(
      length > 0, "internal error: proper quotient has length zero");
  h1_detail::require(
      square_length >= length,
      "internal error: P/J^2 is shorter than P/J");

  auto reduction_map = h1_detail::build_reduction_map(
      quotient, square_quotient, options);
  h1_detail::require(
      reduction_map.rank() == length,
      "internal error: P/J^2 -> P/J is not surjective");
  const auto expected_conormal_dimension = square_length - length;
  h1_detail::check_matrix_entries(
      square_length, expected_conormal_dimension, options,
      "conormal basis matrix");
  auto conormal = LinearSubspace<Field>::kernel(reduction_map);
  h1_detail::require(
      conormal.dimension() == expected_conormal_dimension,
      "internal error: conormal dimension disagrees with quotient lengths");
  h1_detail::require(
      h1_detail::is_zero_matrix(
          multiply(reduction_map, conormal.basis_matrix())),
      "internal error: conormal basis is not in ker(R)");

  auto differential = h1_detail::build_differential(
      quotient, square_quotient, options);
  h1_detail::check_matrix_entries(
      differential.row_count(), conormal.dimension(), options,
      "differential restricted to J/J^2");
  const auto restricted_differential =
      multiply(differential, conormal.basis_matrix());
  const auto relative_h1_dimension =
      conormal.dimension() - restricted_differential.rank();
  h1_detail::check_matrix_entries(
      conormal.dimension(), relative_h1_dimension, options,
      "relative H1 kernel basis matrix");
  h1_detail::check_matrix_entries(
      square_length, relative_h1_dimension, options,
      "H1 basis in P/J^2 coordinates");
  const auto relative_h1 =
      LinearSubspace<Field>::kernel(restricted_differential);
  h1_detail::require(
      relative_h1.dimension() == relative_h1_dimension,
      "internal error: restricted differential nullity changed");
  auto h1_basis_matrix =
      multiply(conormal.basis_matrix(), relative_h1.basis_matrix());
  auto h1 = LinearSubspace<Field>::from_independent_columns(
      std::move(h1_basis_matrix));
  h1_detail::require(
      h1.dimension() == relative_h1.dimension(),
      "internal error: conormal embedding changed H1 dimension");
  h1_detail::require(
      h1_detail::is_zero_matrix(
          multiply(reduction_map, h1.basis_matrix())),
      "internal error: H1 basis is not contained in J/J^2");
  h1_detail::require(
      h1_detail::is_zero_matrix(
          multiply(differential, h1.basis_matrix())),
      "internal error: H1 basis is not killed by the differential");

  auto conormal_polynomials =
      h1_detail::polynomial_basis(conormal, square_quotient);
  auto h1_polynomials = h1_detail::polynomial_basis(h1, square_quotient);
  for (const auto& cycle : h1_polynomials) {
    h1_detail::require(
        ideal.normal_form(cycle, options.reduction_limits).is_zero(),
        "internal error: H1 representative is not in J");
    for (std::size_t variable = 0;
         variable < ideal.ring().variable_count(); ++variable) {
      h1_detail::require(
          ideal.normal_form(
                   cycle.derivative(variable), options.reduction_limits)
              .is_zero(),
          "internal error: H1 representative has a nonzero derivative in Q");
    }
  }

  std::vector<DenseMatrix<Field>> variable_multiplications;
  variable_multiplications.reserve(ideal.ring().variable_count());
  h1_detail::check_matrix_entries(
      length, length, options, "quotient variable multiplication matrix");
  for (std::size_t variable = 0;
       variable < ideal.ring().variable_count(); ++variable) {
    variable_multiplications.push_back(
        quotient.dense_variable_multiplication_matrix(
            variable, options.reduction_limits));
  }
  for (std::size_t left = 0; left < variable_multiplications.size(); ++left) {
    for (std::size_t right = left + 1;
         right < variable_multiplications.size(); ++right) {
      h1_detail::require(
          multiply(variable_multiplications[left],
                   variable_multiplications[right]) ==
              multiply(variable_multiplications[right],
                       variable_multiplications[left]),
          "internal error: quotient multiplication matrices do not commute");
    }
  }
  h1_detail::check_matrix_entries(
      h1_detail::checked_product(
          ideal.ring().variable_count(), length,
          "stacked quotient multiplication matrix"),
      length, options, "stacked quotient multiplication matrix");
  const auto multiplication_stack = vertical_stack(
      quotient.field(), length, variable_multiplications);
  auto socle = LinearSubspace<Field>::kernel(multiplication_stack);
  h1_detail::require(
      socle.dimension() > 0,
      "internal error: a nonzero Artin local quotient has zero socle");
  h1_detail::require(
      h1_detail::is_zero_matrix(
          multiply(multiplication_stack, socle.basis_matrix())),
      "internal error: socle basis is not killed by the maximal ideal");
  auto socle_polynomials = h1_detail::polynomial_basis(socle, quotient);
  for (const auto& socle_element : socle_polynomials) {
    for (std::size_t variable = 0;
         variable < ideal.ring().variable_count(); ++variable) {
      h1_detail::require(
          ideal.contains(
              ideal.ring().gen(variable) * socle_element,
              options.reduction_limits),
          "internal error: returned socle representative is not annihilated");
    }
  }

  std::vector<DenseMatrix<Field>> h1_multiplication_matrices;
  std::vector<DenseMatrix<Field>> action_matrices;
  h1_multiplication_matrices.reserve(h1_polynomials.size());
  action_matrices.reserve(h1_polynomials.size());
  h1_detail::check_matrix_entries(
      conormal.dimension(), length, options,
      "one H1 multiplication matrix");
  h1_detail::check_matrix_entries(
      conormal.dimension(), socle.dimension(), options,
      "one socle-action matrix");
  if (!h1_polynomials.empty()) {
    h1_detail::check_matrix_entries(
        square_length,
        h1_detail::checked_increment(
            conormal.dimension(), "conormal coordinate solve"),
        options, "conormal coordinate solve");
  }
  for (const auto& cycle : h1_polynomials) {
    std::vector<PolynomialType> products;
    products.reserve(length);
    for (const auto& quotient_basis_element : quotient.polynomial_basis()) {
      products.push_back(quotient_basis_element * cycle);
    }
    const auto ambient_columns = square_quotient.coordinates(
        products, options.reduction_limits);
    std::vector<std::vector<typename Field::Element>> conormal_columns;
    conormal_columns.reserve(ambient_columns.size());
    for (const auto& ambient_column : ambient_columns) {
      h1_detail::require(
          h1_detail::is_zero_vector<Field>(
              quotient.field(), reduction_map.multiply_column(ambient_column)),
          "internal error: H1 multiplication product is not in J/J^2");
      h1_detail::require(
          conormal.contains(ambient_column),
          "internal error: H1 multiplication product missed the conormal space");
      auto conormal_coordinates = conormal.coordinates(ambient_column);
      h1_detail::require(
          conormal.from_coordinates(conormal_coordinates) == ambient_column,
          "internal error: conormal product coordinates do not reconstruct");
      conormal_columns.push_back(std::move(conormal_coordinates));
    }
    auto multiplication_map = matrix_from_columns(
        quotient.field(), conormal.dimension(), conormal_columns);
    auto action = multiply(multiplication_map, socle.basis_matrix());

    for (std::size_t socle_index = 0;
         socle_index < socle_polynomials.size(); ++socle_index) {
      const auto ambient_product = square_quotient.coordinates(
          socle_polynomials[socle_index] * cycle,
          options.reduction_limits);
      h1_detail::require(
          conormal.contains(ambient_product),
          "internal error: action tensor product is outside J/J^2");
      const auto expected = conormal.coordinates(ambient_product);
      h1_detail::require(
          h1_detail::matrix_column(action, socle_index) == expected,
          "internal error: action tensor column disagrees with multiplication");
    }

    h1_multiplication_matrices.push_back(std::move(multiplication_map));
    action_matrices.push_back(std::move(action));
  }

  const auto action_column_count = h1_detail::checked_product(
      action_matrices.size(), socle.dimension(),
      "combined socle-action matrix");
  h1_detail::check_matrix_entries(
      conormal.dimension(), action_column_count, options,
      "combined socle-action matrix");
  const auto common_products = horizontal_stack(
      quotient.field(), conormal.dimension(), action_matrices);
  const auto common_product_space_dimension = common_products.rank();
  const auto common_product_space_rank_bound =
      std::min(socle.dimension(), common_product_space_dimension);
  auto individual_rank = analyze_matrix_space(
      quotient.field(), conormal.dimension(), socle.dimension(),
      action_matrices, options.matrix_space_limits);
  h1_detail::require(
      individual_rank.upper_bound <= common_product_space_rank_bound,
      "internal error: matrix-space rank exceeds its common image bound");

  const auto annihilator_row_count = h1_detail::checked_product(
      h1_multiplication_matrices.size(), conormal.dimension(),
      "stacked H1 multiplication matrix");
  h1_detail::check_matrix_entries(
      annihilator_row_count, length, options,
      "stacked H1 multiplication matrix");
  const auto common_multiplication_stack = vertical_stack(
      quotient.field(), length, h1_multiplication_matrices);
  const auto common_annihilator_subspace =
      LinearSubspace<Field>::kernel(common_multiplication_stack);
  h1_detail::require(
      h1_detail::is_zero_matrix(multiply(
          common_multiplication_stack,
          common_annihilator_subspace.basis_matrix())),
      "internal error: common H1 annihilator kernel is incorrect");
  std::vector<PolynomialType> common_annihilator_generators =
      ideal.groebner_basis();
  for (const auto& vector : common_annihilator_subspace.basis_vectors()) {
    common_annihilator_generators.push_back(
        quotient.representative(vector));
  }
  Ideal<Field> common_annihilator(
      ideal.ring(), std::move(common_annihilator_generators),
      options.groebner_limits);
  h1_detail::require(
      ideal.is_subset_of(common_annihilator),
      "internal error: Ann_Q(H1) does not contain J");
  for (const auto& annihilator : common_annihilator.groebner_basis()) {
    for (const auto& cycle : h1_polynomials) {
      h1_detail::require(
          squared_ideal.contains(
              annihilator * cycle, options.reduction_limits),
          "internal error: diagnostic common annihilator does not kill H1");
    }
  }

  auto best_h1_polynomial = h1_detail::combine_h1_basis<Field>(
      ideal, h1_polynomials, individual_rank.witness_coefficients);
  if (best_h1_polynomial.has_value()) {
    h1_detail::require(
        ideal.contains(*best_h1_polynomial, options.reduction_limits),
        "internal error: best matrix-space witness is not in J");
    for (std::size_t variable = 0;
         variable < ideal.ring().variable_count(); ++variable) {
      h1_detail::require(
          ideal.contains(
              best_h1_polynomial->derivative(variable),
              options.reduction_limits),
          "internal error: best matrix-space witness is not an H1 cycle");
    }
  }

  std::optional<CycleAuditResult<Field>> faithful_witness_audit;
  if (individual_rank.has_full_column_rank_witness()) {
    h1_detail::require(
        best_h1_polynomial.has_value(),
        "internal error: full-rank matrix witness has no H1 polynomial");
    auto audit_limits = options.witness_audit_limits;
    auto audit = audit_cycle(ideal, *best_h1_polynomial, audit_limits);
    if (audit.status() != CycleAuditStatus::ResourceLimit) {
      h1_detail::require(
          audit.status() == CycleAuditStatus::Complete &&
              audit.faithful_cycle() && audit.cycle_valid().value_or(false) &&
              audit.primitive().value_or(false) &&
              audit.colon_equals_ideal().value_or(false) &&
              audit.annihilator_zero().value_or(false),
          "internal error: full-socle-rank witness failed its colon replay");
    }
    faithful_witness_audit = std::move(audit);
  }

  return H1ActionData<Field>{
      length,
      square_length,
      conormal.dimension(),
      h1.dimension(),
      socle.dimension(),
      std::move(ideal),
      std::move(squared_ideal),
      std::move(quotient),
      std::move(square_quotient),
      std::move(reduction_map),
      std::move(differential),
      std::move(conormal),
      std::move(h1),
      std::move(socle),
      std::move(conormal_polynomials),
      std::move(h1_polynomials),
      std::move(socle_polynomials),
      std::move(variable_multiplications),
      std::move(h1_multiplication_matrices),
      std::move(action_matrices),
      common_product_space_dimension,
      common_product_space_rank_bound,
      individual_rank,
      individual_rank.witness_coefficients,
      std::move(best_h1_polynomial),
      std::move(faithful_witness_audit),
      std::move(common_annihilator)};
  } catch (const GroebnerResourceLimit& error) {
    throw FullH1ResourceLimit(error.what());
  }
}

}  // namespace laughableengine

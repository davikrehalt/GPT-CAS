#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "laughableengine/field.hpp"
#include "laughableengine/matrix.hpp"

namespace laughableengine {

enum class MatrixSpaceRankProof {
  ProvenMaximum,
  ProvenFullColumnRank,
  GenericOnly,
  ResourceLimit,
};

struct MatrixSpaceSearchLimits {
  std::optional<std::size_t> max_minors = 1'000'000;
  std::optional<std::size_t> max_parameter_terms = 1'000'000;
  std::optional<std::size_t> max_determinant_products = 10'000'000;
  std::size_t max_finite_field_evaluations = 1'000'000;
};

template <typename Field>
struct MatrixSpaceRankResult {
  using Element = typename Field::Element;

  std::size_t parameter_count = 0;
  std::size_t row_count = 0;
  std::size_t column_count = 0;
  std::size_t lower_bound = 0;
  std::size_t upper_bound = 0;
  std::optional<std::size_t> generic_maximum;
  std::optional<std::size_t> exact_maximum;
  MatrixSpaceRankProof proof = MatrixSpaceRankProof::ResourceLimit;
  std::optional<std::vector<Element>> witness_coefficients;
  std::size_t minors_tested = 0;
  std::size_t determinant_products_tested = 0;
  std::size_t finite_field_evaluations = 0;

  [[nodiscard]] bool has_full_column_rank_witness() const noexcept {
    return exact_maximum.has_value() &&
           *exact_maximum == column_count &&
           witness_coefficients.has_value();
  }
};

namespace matrix_space_detail {

using ParameterMonomial = std::vector<std::size_t>;

template <typename Field>
using ParameterPolynomial =
    std::map<ParameterMonomial, typename Field::Element>;

template <typename Field>
bool add_term(
    const Field& field,
    ParameterPolynomial<Field>& polynomial,
    ParameterMonomial monomial,
    typename Field::Element coefficient,
    const MatrixSpaceSearchLimits& limits) {
  coefficient = field.canonical(std::move(coefficient));
  if (field.is_zero(coefficient)) {
    return true;
  }
  const auto iterator = polynomial.find(monomial);
  if (iterator == polynomial.end()) {
    if (limits.max_parameter_terms.has_value() &&
        polynomial.size() >= *limits.max_parameter_terms) {
      return false;
    }
    polynomial.emplace(std::move(monomial), std::move(coefficient));
    return true;
  }
  auto combined = field.add(iterator->second, coefficient);
  if (field.is_zero(combined)) {
    polynomial.erase(iterator);
  } else {
    iterator->second = std::move(combined);
  }
  return true;
}

template <typename Field>
std::optional<ParameterPolynomial<Field>> multiply_by_linear_form(
    const Field& field,
    const ParameterPolynomial<Field>& polynomial,
    std::span<const typename Field::Element> linear_form,
    const MatrixSpaceSearchLimits& limits) {
  ParameterPolynomial<Field> result;
  for (const auto& [monomial, coefficient] : polynomial) {
    for (std::size_t parameter = 0; parameter < linear_form.size();
         ++parameter) {
      if (field.is_zero(linear_form[parameter])) {
        continue;
      }
      auto product_monomial = monomial;
      product_monomial.insert(
          std::upper_bound(product_monomial.begin(),
                           product_monomial.end(), parameter),
          parameter);
      const auto product =
          field.multiply(coefficient, linear_form[parameter]);
      if (!add_term(field, result, std::move(product_monomial), product,
                    limits)) {
        return std::nullopt;
      }
    }
  }
  return result;
}

inline bool permutation_is_odd(const std::vector<std::size_t>& permutation) {
  std::size_t inversions = 0;
  for (std::size_t left = 0; left < permutation.size(); ++left) {
    for (std::size_t right = left + 1; right < permutation.size(); ++right) {
      if (permutation[left] > permutation[right]) {
        ++inversions;
      }
    }
  }
  return (inversions % 2) != 0;
}

template <typename Field>
std::optional<ParameterPolynomial<Field>> determinant_polynomial(
    const Field& field,
    std::span<const DenseMatrix<Field>> basis,
    const std::vector<std::size_t>& rows,
    const std::vector<std::size_t>& columns,
    const MatrixSpaceSearchLimits& limits,
    std::size_t& products_tested) {
  const auto size = rows.size();
  std::vector<std::size_t> permutation(size);
  for (std::size_t index = 0; index < size; ++index) {
    permutation[index] = index;
  }

  ParameterPolynomial<Field> determinant;
  do {
    if (limits.max_determinant_products.has_value() &&
        products_tested >= *limits.max_determinant_products) {
      return std::nullopt;
    }
    ++products_tested;
    ParameterPolynomial<Field> product;
    product.emplace(ParameterMonomial{}, field.one());
    for (std::size_t row = 0; row < size && !product.empty(); ++row) {
      std::vector<typename Field::Element> linear_form;
      linear_form.reserve(basis.size());
      for (const auto& matrix : basis) {
        linear_form.push_back(
            matrix(rows[row], columns[permutation[row]]));
      }
      auto multiplied = multiply_by_linear_form<Field>(
          field, product, linear_form, limits);
      if (!multiplied.has_value()) {
        return std::nullopt;
      }
      product = std::move(*multiplied);
    }

    const bool negative = permutation_is_odd(permutation);
    for (auto& [monomial, coefficient] : product) {
      if (negative) {
        coefficient = field.negate(coefficient);
      }
      if (!add_term(field, determinant, std::move(monomial), coefficient,
                    limits)) {
        return std::nullopt;
      }
    }
  } while (std::next_permutation(permutation.begin(), permutation.end()));
  return determinant;
}

inline std::vector<std::size_t> first_combination(std::size_t size) {
  std::vector<std::size_t> result(size);
  for (std::size_t index = 0; index < size; ++index) {
    result[index] = index;
  }
  return result;
}

inline bool next_combination(
    std::vector<std::size_t>& combination,
    std::size_t universe_size) {
  if (combination.empty()) {
    return false;
  }
  for (std::size_t position = combination.size(); position-- > 0;) {
    const auto maximum =
        universe_size - (combination.size() - position);
    if (combination[position] >= maximum) {
      continue;
    }
    ++combination[position];
    for (std::size_t next = position + 1; next < combination.size(); ++next) {
      combination[next] = combination[next - 1] + 1;
    }
    return true;
  }
  return false;
}

template <typename Field>
ParameterPolynomial<Field> specialize(
    const Field& field,
    const ParameterPolynomial<Field>& polynomial,
    std::size_t parameter,
    const typename Field::Element& value,
    const MatrixSpaceSearchLimits& limits) {
  ParameterPolynomial<Field> result;
  for (const auto& [monomial, coefficient] : polynomial) {
    auto first = std::lower_bound(monomial.begin(), monomial.end(), parameter);
    auto last = std::upper_bound(first, monomial.end(), parameter);
    auto specialized_coefficient = coefficient;
    for (auto iterator = first; iterator != last; ++iterator) {
      specialized_coefficient =
          field.multiply(specialized_coefficient, value);
    }
    auto specialized_monomial = monomial;
    specialized_monomial.erase(
        specialized_monomial.begin() + (first - monomial.begin()),
        specialized_monomial.begin() + (last - monomial.begin()));
    const bool inserted = add_term(
        field, result, std::move(specialized_monomial),
        std::move(specialized_coefficient), limits);
    if (!inserted) {
      throw std::logic_error(
          "specialization unexpectedly exceeded an existing term limit");
    }
  }
  return result;
}

template <typename Field>
std::vector<typename Field::Element> symbolic_witness(
    const Field& field,
    const ParameterPolynomial<Field>& polynomial,
    std::size_t parameter_count,
    std::size_t degree,
    const MatrixSpaceSearchLimits& limits) {
  auto current = polynomial;
  std::vector<typename Field::Element> witness(
      parameter_count, field.zero());
  for (std::size_t parameter = 0; parameter < parameter_count; ++parameter) {
    bool found = false;
    for (std::size_t integer = 0; integer <= degree; ++integer) {
      const auto value = field.from_unsigned(integer);
      auto candidate = specialize(
          field, current, parameter, value, limits);
      if (!candidate.empty()) {
        witness[parameter] = value;
        current = std::move(candidate);
        found = true;
        break;
      }
    }
    if (!found) {
      throw std::logic_error(
          "failed to specialize a nonzero determinant polynomial");
    }
  }
  if (current.empty()) {
    throw std::logic_error("determinant witness specialized to zero");
  }
  return witness;
}

template <typename Field>
DenseMatrix<Field> linear_combination(
    const Field& field,
    std::size_t rows,
    std::size_t columns,
    std::span<const DenseMatrix<Field>> basis,
    std::span<const typename Field::Element> coefficients) {
  if (coefficients.size() != basis.size()) {
    throw std::invalid_argument(
        "matrix-space coefficient vector has the wrong dimension");
  }
  DenseMatrix<Field> result(field, rows, columns);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      auto value = field.zero();
      for (std::size_t parameter = 0; parameter < basis.size(); ++parameter) {
        value = field.add(
            value,
            field.multiply(coefficients[parameter],
                           basis[parameter](row, column)));
      }
      result.set(row, column, std::move(value));
    }
  }
  return result;
}

template <typename Field>
std::size_t common_column_space_bound(
    const Field& field,
    std::size_t rows,
    std::size_t columns,
    std::span<const DenseMatrix<Field>> basis) {
  if (columns != 0 &&
      basis.size() > std::numeric_limits<std::size_t>::max() / columns) {
    throw std::length_error("matrix-space horizontal width overflows size_t");
  }
  DenseMatrix<Field> combined(field, rows, columns * basis.size());
  for (std::size_t parameter = 0; parameter < basis.size(); ++parameter) {
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t column = 0; column < columns; ++column) {
        combined.set(
            row, parameter * columns + column,
            basis[parameter](row, column));
      }
    }
  }
  return std::min(columns, combined.rank());
}

template <typename Field>
void record_witness(
    MatrixSpaceRankResult<Field>& result,
    const Field& field,
    std::span<const DenseMatrix<Field>> basis,
    const std::vector<typename Field::Element>& coefficients) {
  const auto rank = linear_combination(
      field, result.row_count, result.column_count, basis, coefficients)
                        .rank();
  if (rank > result.lower_bound || !result.witness_coefficients.has_value()) {
    result.lower_bound = rank;
    result.witness_coefficients = coefficients;
  }
}

template <typename Field>
bool finite_field_evaluation_count(
    std::uint32_t modulus,
    std::size_t parameter_count,
    std::size_t limit,
    std::size_t& count) {
  count = 1;
  for (std::size_t parameter = 0; parameter < parameter_count; ++parameter) {
    if (modulus != 0 && count > limit / modulus) {
      return false;
    }
    count *= modulus;
  }
  return count <= limit;
}

template <typename Field>
void set_exact_result(MatrixSpaceRankResult<Field>& result) {
  result.exact_maximum = result.lower_bound;
  result.upper_bound = result.lower_bound;
  result.proof = result.lower_bound == result.column_count
                     ? MatrixSpaceRankProof::ProvenFullColumnRank
                     : MatrixSpaceRankProof::ProvenMaximum;
}

}  // namespace matrix_space_detail

// Analyze the exact ranks in a linear matrix space
//
//        A(c) = sum_i c_i * basis[i].
//
// A nonzero symbolic minor is interpreted over the base field only when a
// base-field witness is also constructed. In particular, small finite fields
// are exhausted when feasible; otherwise the result is explicitly GenericOnly.
template <typename Field>
[[nodiscard]] MatrixSpaceRankResult<Field> analyze_matrix_space(
    Field field,
    std::size_t rows,
    std::size_t columns,
    std::span<const DenseMatrix<Field>> basis,
    const MatrixSpaceSearchLimits& limits = {}) {
  static_assert(std::is_same_v<Field, RationalField> ||
                std::is_same_v<Field, PrimeField>);

  for (const auto& matrix : basis) {
    if (!(matrix.field() == field)) {
      throw std::invalid_argument(
          "matrix-space bases require one coefficient field");
    }
    if (matrix.row_count() != rows || matrix.column_count() != columns) {
      throw std::invalid_argument(
          "matrix-space basis matrices require one shape");
    }
  }

  MatrixSpaceRankResult<Field> result;
  result.parameter_count = basis.size();
  result.row_count = rows;
  result.column_count = columns;
  result.upper_bound = matrix_space_detail::common_column_space_bound(
      field, rows, columns, basis);

  std::vector<typename Field::Element> zero(
      basis.size(), field.zero());
  result.witness_coefficients = zero;
  for (std::size_t parameter = 0; parameter < basis.size(); ++parameter) {
    auto coefficients = zero;
    coefficients[parameter] = field.one();
    matrix_space_detail::record_witness(result, field, basis, coefficients);
  }
  if (result.lower_bound == result.upper_bound) {
    result.generic_maximum = result.lower_bound;
    matrix_space_detail::set_exact_result(result);
    return result;
  }

  std::optional<matrix_space_detail::ParameterPolynomial<Field>>
      nonzero_minor;
  std::size_t generic_rank = result.upper_bound;
  while (generic_rank > 0) {
    bool found = false;
    auto selected_rows =
        matrix_space_detail::first_combination(generic_rank);
    do {
      auto selected_columns =
          matrix_space_detail::first_combination(generic_rank);
      do {
        if (limits.max_minors.has_value() &&
            result.minors_tested >= *limits.max_minors) {
          result.proof = MatrixSpaceRankProof::ResourceLimit;
          return result;
        }
        ++result.minors_tested;
        auto determinant = matrix_space_detail::determinant_polynomial<Field>(
            field, basis, selected_rows, selected_columns, limits,
            result.determinant_products_tested);
        if (!determinant.has_value()) {
          result.proof = MatrixSpaceRankProof::ResourceLimit;
          return result;
        }
        if (!determinant->empty()) {
          nonzero_minor = std::move(*determinant);
          found = true;
          break;
        }
      } while (matrix_space_detail::next_combination(
          selected_columns, columns));
      if (found) {
        break;
      }
    } while (matrix_space_detail::next_combination(selected_rows, rows));

    if (found) {
      break;
    }
    --generic_rank;
    result.upper_bound = generic_rank;
    if (result.lower_bound == result.upper_bound) {
      result.generic_maximum = result.lower_bound;
      matrix_space_detail::set_exact_result(result);
      return result;
    }
  }

  result.generic_maximum = generic_rank;
  result.upper_bound = generic_rank;
  if (generic_rank == 0) {
    matrix_space_detail::set_exact_result(result);
    return result;
  }
  if (!nonzero_minor.has_value()) {
    throw std::logic_error("missing nonzero minor for a positive generic rank");
  }

  bool symbolic_values_are_sufficient =
      std::is_same_v<Field, RationalField>;
  if constexpr (std::is_same_v<Field, PrimeField>) {
    symbolic_values_are_sufficient = field.modulus() > generic_rank;
  }
  if (symbolic_values_are_sufficient) {
    const auto witness = matrix_space_detail::symbolic_witness(
        field, *nonzero_minor, basis.size(), generic_rank, limits);
    matrix_space_detail::record_witness(result, field, basis, witness);
    if (result.lower_bound != generic_rank) {
      throw std::logic_error("symbolic minor witness has an incorrect rank");
    }
    matrix_space_detail::set_exact_result(result);
    return result;
  }

  if constexpr (std::is_same_v<Field, PrimeField>) {
    std::size_t evaluation_count = 0;
    if (matrix_space_detail::finite_field_evaluation_count<Field>(
            field.modulus(), basis.size(),
            limits.max_finite_field_evaluations, evaluation_count)) {
      std::vector<std::uint32_t> digits(basis.size(), 0);
      for (std::size_t evaluation = 0; evaluation < evaluation_count;
           ++evaluation) {
        std::vector<typename Field::Element> coefficients;
        coefficients.reserve(digits.size());
        for (const auto digit : digits) {
          coefficients.push_back(field.from_unsigned(digit));
        }
        matrix_space_detail::record_witness(
            result, field, basis, coefficients);
        ++result.finite_field_evaluations;

        for (std::size_t position = 0; position < digits.size(); ++position) {
          ++digits[position];
          if (digits[position] < field.modulus()) {
            break;
          }
          digits[position] = 0;
        }
      }
      matrix_space_detail::set_exact_result(result);
      return result;
    }
  }

  if (result.lower_bound == result.upper_bound) {
    matrix_space_detail::set_exact_result(result);
  } else {
    result.proof = MatrixSpaceRankProof::GenericOnly;
  }
  return result;
}

template <typename Field>
[[nodiscard]] MatrixSpaceRankResult<Field> analyze_matrix_space(
    Field field,
    std::size_t rows,
    std::size_t columns,
    const std::vector<DenseMatrix<Field>>& basis,
    const MatrixSpaceSearchLimits& limits = {}) {
  return analyze_matrix_space(
      std::move(field), rows, columns,
      std::span<const DenseMatrix<Field>>(basis.data(), basis.size()), limits);
}

}  // namespace laughableengine

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/compiled_quotient.hpp"
#include "laughableengine/cycle_audit.hpp"
#include "laughableengine/matrix_space.hpp"

namespace laughableengine {

enum class DiscoveryScreenStatus {
  Complete,
  InvalidInput,
  ResourceLimit,
};

struct PackedCycleScreenOptions {
  CompiledPrimeQuotientLimits compiled_limits{};
  StandardMonomialLimits standard_monomial_limits{};
  GroebnerLimits groebner_limits{};
  std::optional<std::size_t> max_matrix_entries = 20'000'000;
  bool certify_hits = true;
  CycleAuditLimits certification_limits{};
};

struct PackedCycleScreenResult {
  DiscoveryScreenStatus status = DiscoveryScreenStatus::InvalidInput;
  std::size_t length_Q = 0;
  std::size_t length_P_mod_J2 = 0;
  bool g_in_J = false;
  bool derivatives_in_J = false;
  bool cycle_valid = false;
  std::size_t multiplication_rank = 0;
  bool full_column_rank_candidate = false;
  bool certified_faithful = false;
  std::optional<CycleAuditResult<PrimeField>> certification;
  std::optional<std::string> detail;
};

struct PackedH1ScreenOptions {
  CompiledPrimeQuotientLimits compiled_limits{};
  StandardMonomialLimits standard_monomial_limits{};
  GroebnerLimits groebner_limits{};
  MatrixSpaceSearchLimits matrix_space_limits{};
  std::optional<std::size_t> max_matrix_entries = 20'000'000;
  bool certify_hits = true;
  CycleAuditLimits certification_limits{};
};

struct PackedH1ScreenResult {
  DiscoveryScreenStatus status = DiscoveryScreenStatus::InvalidInput;
  std::size_t length_Q = 0;
  std::size_t length_P_mod_J2 = 0;
  std::size_t conormal_dimension = 0;
  std::size_t h1_dimension = 0;
  std::size_t socle_dimension = 0;
  std::uint32_t modulus = 0;
  std::uint64_t leading_ideal_fingerprint = 0;
  std::string leading_ideal_signature;
  std::size_t maximum_individual_rank_lower_bound = 0;
  std::size_t maximum_individual_rank_upper_bound = 0;
  MatrixSpaceRankProof rank_proof = MatrixSpaceRankProof::ResourceLimit;
  bool full_socle_rank_candidate = false;
  bool certified_faithful = false;
  std::optional<std::vector<std::uint32_t>> witness_coefficients;
  std::optional<Polynomial<PrimeField>> witness;
  std::optional<CycleAuditResult<PrimeField>> certification;
  std::optional<std::string> detail;
};

namespace discovery_detail {

inline std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* description) {
  if (right != 0 &&
      left > std::numeric_limits<std::size_t>::max() / right) {
    throw PackedDiscoveryResourceLimit(
        std::string(description) + " dimensions overflow size_t");
  }
  return left * right;
}

inline void check_matrix_entries(
    std::size_t rows,
    std::size_t columns,
    std::optional<std::size_t> maximum,
    const char* description) {
  const auto entries = checked_product(rows, columns, description);
  if (maximum.has_value() && entries > *maximum) {
    throw PackedDiscoveryResourceLimit(
        std::string(description) + " needs " + std::to_string(entries) +
        " packed entries, above max_matrix_entries=" +
        std::to_string(*maximum));
  }
}

inline PackedPrimeMatrix coordinate_batch_as_columns(
    const PackedCoordinateBatch& batch) {
  PackedPrimeMatrix result(
      batch.modulus(), batch.column_count(), batch.row_count());
  for (std::size_t column = 0; column < batch.row_count(); ++column) {
    const auto coordinates = batch.row(column);
    for (std::size_t row = 0; row < batch.column_count(); ++row) {
      result.set(row, column, coordinates[row]);
    }
  }
  return result;
}

inline std::vector<std::uint32_t> matrix_column(
    const PackedPrimeMatrix& matrix,
    std::size_t column) {
  if (column >= matrix.column_count()) {
    throw std::out_of_range("packed matrix column is out of range");
  }
  std::vector<std::uint32_t> result;
  result.reserve(matrix.row_count());
  for (std::size_t row = 0; row < matrix.row_count(); ++row) {
    result.push_back(matrix(row, column));
  }
  return result;
}

inline bool all_zero(std::span<const std::uint32_t> values) {
  return std::all_of(values.begin(), values.end(), [](auto value) {
    return value == 0;
  });
}

inline PackedPrimeMatrix inverse(const PackedPrimeMatrix& matrix) {
  if (matrix.row_count() != matrix.column_count()) {
    throw std::invalid_argument("packed matrix inverse requires a square matrix");
  }
  const auto size = matrix.row_count();
  PackedPrimeMatrix augmented(matrix.modulus(), size, size * 2);
  for (std::size_t row = 0; row < size; ++row) {
    for (std::size_t column = 0; column < size; ++column) {
      augmented.set(row, column, matrix(row, column));
    }
    augmented.set(row, size + row, 1);
  }
  auto reduction = augmented.rref();
  if (reduction.pivot_columns.size() != size) {
    throw std::domain_error("packed matrix is singular");
  }
  for (std::size_t index = 0; index < size; ++index) {
    if (reduction.pivot_columns[index] != index) {
      throw std::domain_error("packed matrix is singular");
    }
  }
  PackedPrimeMatrix result(matrix.modulus(), size, size);
  for (std::size_t row = 0; row < size; ++row) {
    for (std::size_t column = 0; column < size; ++column) {
      result.set(row, column, reduction.reduced(row, size + column));
    }
  }
  return result;
}

inline PackedPrimeMatrix conormal_coordinate_map(
    const PackedPrimeMatrix& conormal_basis) {
  const auto dimension = conormal_basis.column_count();
  if (dimension == 0) {
    return PackedPrimeMatrix(
        conormal_basis.modulus(), 0, conormal_basis.row_count());
  }
  const auto transpose_reduction = conormal_basis.transpose().rref();
  if (transpose_reduction.pivot_columns.size() != dimension) {
    throw std::logic_error("conormal basis lost column independence");
  }
  PackedPrimeMatrix selected(
      conormal_basis.modulus(), dimension, dimension);
  for (std::size_t row = 0; row < dimension; ++row) {
    const auto source_row = transpose_reduction.pivot_columns[row];
    for (std::size_t column = 0; column < dimension; ++column) {
      selected.set(row, column, conormal_basis(source_row, column));
    }
  }
  const auto selected_inverse = inverse(selected);
  PackedPrimeMatrix result(
      conormal_basis.modulus(), dimension, conormal_basis.row_count());
  for (std::size_t selected_row = 0; selected_row < dimension;
       ++selected_row) {
    const auto ambient_row = transpose_reduction.pivot_columns[selected_row];
    for (std::size_t coordinate = 0; coordinate < dimension; ++coordinate) {
      result.set(
          coordinate, ambient_row,
          selected_inverse(coordinate, selected_row));
    }
  }
  return result;
}

inline std::vector<PrimeField::Element> exact_coordinates(
    const PrimeField& field,
    std::span<const std::uint32_t> raw) {
  std::vector<PrimeField::Element> result;
  result.reserve(raw.size());
  for (const auto entry : raw) {
    result.push_back(field.from_unsigned(entry));
  }
  return result;
}

inline std::vector<DenseMatrix<PrimeField>> exact_matrices(
    std::span<const PackedPrimeMatrix> matrices) {
  std::vector<DenseMatrix<PrimeField>> result;
  result.reserve(matrices.size());
  for (const auto& matrix : matrices) {
    result.push_back(matrix.to_exact());
  }
  return result;
}

inline Polynomial<PrimeField> combine_h1_witness(
    const CompiledPrimeQuotientPlan& square_plan,
    const PackedPrimeMatrix& h1_basis,
    std::span<const std::uint32_t> coefficients) {
  if (coefficients.size() != h1_basis.column_count()) {
    throw std::logic_error("packed H1 witness has the wrong coefficient count");
  }
  PackedPrimeArithmetic arithmetic(square_plan.modulus());
  std::vector<std::uint32_t> ambient(h1_basis.row_count(), 0);
  for (std::size_t column = 0; column < h1_basis.column_count(); ++column) {
    for (std::size_t row = 0; row < h1_basis.row_count(); ++row) {
      ambient[row] = arithmetic.add(
          ambient[row],
          arithmetic.multiply(coefficients[column], h1_basis(row, column)));
    }
  }
  const auto exact = exact_coordinates(
      square_plan.reference_context().field(), ambient);
  return square_plan.reference_context().representative(exact);
}

}  // namespace discovery_detail

class PackedCycleDiscoverySession {
 public:
  explicit PackedCycleDiscoverySession(
      Ideal<PrimeField> ideal,
      const PackedCycleScreenOptions& options = {})
      : options_(options),
        ideal_(std::move(ideal)),
        ideal_plan_(ideal_, options.standard_monomial_limits),
        squared_ideal_(ideal_.square(options.groebner_limits)),
        square_plan_(squared_ideal_, options.standard_monomial_limits) {
    if (ideal_.is_unit() || !ideal_.is_zero_dimensional() ||
        !ideal_plan_.reference_context().supported_at_origin()) {
      throw std::domain_error(
          "packed cycle discovery requires a proper origin-supported finite quotient");
    }
  }

  [[nodiscard]] const Ideal<PrimeField>& ideal() const noexcept {
    return ideal_;
  }
  [[nodiscard]] const CompiledPrimeQuotientPlan& ideal_plan() const noexcept {
    return ideal_plan_;
  }
  [[nodiscard]] const CompiledPrimeQuotientPlan& square_plan() const noexcept {
    return square_plan_;
  }

  [[nodiscard]] PackedCycleScreenResult screen(
      const Polynomial<PrimeField>& polynomial) const {
    PackedCycleScreenResult result;
    result.length_Q = ideal_plan_.dimension();
    result.length_P_mod_J2 = square_plan_.dimension();
    try {
      if (!ideal_.ring().zero().same_ring(polynomial)) {
        throw std::invalid_argument(
            "packed cycle candidate requires the session ring");
      }
      std::vector<Polynomial<PrimeField>> membership;
      membership.reserve(ideal_.ring().variable_count() + 1);
      membership.push_back(polynomial);
      for (std::size_t variable = 0;
           variable < ideal_.ring().variable_count(); ++variable) {
        membership.push_back(polynomial.derivative(variable));
      }
      const auto membership_coordinates =
          ideal_plan_.reduce_coordinates_batch(
              membership, options_.compiled_limits);
      result.g_in_J = discovery_detail::all_zero(
          membership_coordinates.row(0));
      result.derivatives_in_J = true;
      for (std::size_t derivative = 1;
           derivative < membership_coordinates.row_count(); ++derivative) {
        result.derivatives_in_J =
            result.derivatives_in_J && discovery_detail::all_zero(
                membership_coordinates.row(derivative));
      }
      result.cycle_valid = result.g_in_J && result.derivatives_in_J;
      if (!result.g_in_J) {
        result.status = DiscoveryScreenStatus::Complete;
        return result;
      }

      discovery_detail::check_matrix_entries(
          square_plan_.dimension(), ideal_plan_.dimension(),
          options_.max_matrix_entries, "packed cycle multiplication matrix");
      std::vector<Polynomial<PrimeField>> products;
      products.reserve(ideal_plan_.dimension());
      for (const auto& basis :
           ideal_plan_.reference_context().polynomial_basis()) {
        products.push_back(basis * polynomial);
      }
      const auto product_coordinates =
          square_plan_.reduce_coordinates_batch(
              products, options_.compiled_limits);
      const auto multiplication =
          discovery_detail::coordinate_batch_as_columns(product_coordinates);
      result.multiplication_rank = multiplication.rank();
      result.full_column_rank_candidate =
          result.cycle_valid &&
          result.multiplication_rank == ideal_plan_.dimension();
      if (result.full_column_rank_candidate && options_.certify_hits) {
        result.certification = audit_cycle(
            ideal_, polynomial, options_.certification_limits);
        result.certified_faithful =
            result.certification->faithful_cycle();
        if (!result.certification->conclusive()) {
          result.status = DiscoveryScreenStatus::ResourceLimit;
          result.detail = result.certification->resource_detail();
          return result;
        }
        if (!result.certified_faithful) {
          throw std::logic_error(
              "packed full-rank cycle failed exact colon certification");
        }
      }
      result.status = DiscoveryScreenStatus::Complete;
      return result;
    } catch (const PackedDiscoveryResourceLimit& error) {
      result.status = DiscoveryScreenStatus::ResourceLimit;
      result.detail = error.what();
      return result;
    } catch (const ResourceLimitExceeded& error) {
      result.status = DiscoveryScreenStatus::ResourceLimit;
      result.detail = error.what();
      return result;
    } catch (const GroebnerResourceLimit& error) {
      result.status = DiscoveryScreenStatus::ResourceLimit;
      result.detail = error.what();
      return result;
    } catch (const std::overflow_error& error) {
      result.status = DiscoveryScreenStatus::ResourceLimit;
      result.detail = error.what();
      return result;
    } catch (const std::invalid_argument& error) {
      result.status = DiscoveryScreenStatus::InvalidInput;
      result.detail = error.what();
      return result;
    }
  }

 private:
  PackedCycleScreenOptions options_;
  Ideal<PrimeField> ideal_;
  CompiledPrimeQuotientPlan ideal_plan_;
  Ideal<PrimeField> squared_ideal_;
  CompiledPrimeQuotientPlan square_plan_;
};

inline PackedCycleScreenResult screen_cycle(
    Ideal<PrimeField> ideal,
    const Polynomial<PrimeField>& polynomial,
    const PackedCycleScreenOptions& options = {}) {
  try {
    return PackedCycleDiscoverySession(std::move(ideal), options)
        .screen(polynomial);
  } catch (const PackedDiscoveryResourceLimit& error) {
    PackedCycleScreenResult result;
    result.status = DiscoveryScreenStatus::ResourceLimit;
    result.detail = error.what();
    return result;
  } catch (const GroebnerResourceLimit& error) {
    PackedCycleScreenResult result;
    result.status = DiscoveryScreenStatus::ResourceLimit;
    result.detail = error.what();
    return result;
  } catch (const std::domain_error& error) {
    PackedCycleScreenResult result;
    result.status = DiscoveryScreenStatus::InvalidInput;
    result.detail = error.what();
    return result;
  } catch (const std::overflow_error& error) {
    PackedCycleScreenResult result;
    result.status = DiscoveryScreenStatus::ResourceLimit;
    result.detail = error.what();
    return result;
  } catch (const std::invalid_argument& error) {
    PackedCycleScreenResult result;
    result.status = DiscoveryScreenStatus::InvalidInput;
    result.detail = error.what();
    return result;
  }
}

inline PackedH1ScreenResult screen_full_h1(
    Ideal<PrimeField> ideal,
    const PackedH1ScreenOptions& options = {}) {
  PackedH1ScreenResult result;
  try {
    if (ideal.is_unit() || !ideal.is_zero_dimensional()) {
      throw std::domain_error(
          "packed full-H1 discovery requires a proper finite quotient");
    }
    CompiledPrimeQuotientPlan quotient(
        ideal, options.standard_monomial_limits);
    result.modulus = quotient.modulus();
    result.leading_ideal_fingerprint = quotient.leading_fingerprint();
    result.leading_ideal_signature = quotient.leading_signature();
    if (!quotient.reference_context().supported_at_origin()) {
      throw std::domain_error(
          "packed full-H1 discovery requires support at the origin");
    }
    auto square = ideal.square(options.groebner_limits);
    CompiledPrimeQuotientPlan square_quotient(
        square, options.standard_monomial_limits);
    const auto length = quotient.dimension();
    const auto square_length = square_quotient.dimension();
    const auto variable_count = ideal.ring().variable_count();
    result.length_Q = length;
    result.length_P_mod_J2 = square_length;

    discovery_detail::check_matrix_entries(
        length, square_length, options.max_matrix_entries,
        "packed H1 reduction map");
    const auto reduction_batch = quotient.reduce_coordinates_batch(
        square_quotient.reference_context().polynomial_basis(),
        options.compiled_limits);
    const auto reduction =
        discovery_detail::coordinate_batch_as_columns(reduction_batch);
    auto reduction_elimination = reduction.rank_and_kernel();
    if (reduction_elimination.rank != length) {
      throw std::logic_error("packed P/J2 to P/J map is not surjective");
    }
    result.conormal_dimension = reduction_elimination.kernel_basis.size();
    discovery_detail::check_matrix_entries(
        square_length, result.conormal_dimension,
        options.max_matrix_entries, "packed conormal basis");
    const auto conormal = packed_matrix_from_columns(
        ideal.ring().field().modulus(), square_length,
        reduction_elimination.kernel_basis);

    const auto differential_rows = discovery_detail::checked_product(
        variable_count, length, "packed H1 differential");
    discovery_detail::check_matrix_entries(
        differential_rows, square_length, options.max_matrix_entries,
        "packed H1 differential");
    std::vector<Polynomial<PrimeField>> derivatives;
    derivatives.reserve(discovery_detail::checked_product(
        square_length, variable_count, "packed derivative batch"));
    for (const auto& basis :
         square_quotient.reference_context().polynomial_basis()) {
      for (std::size_t variable = 0; variable < variable_count; ++variable) {
        derivatives.push_back(basis.derivative(variable));
      }
    }
    const auto derivative_batch = quotient.reduce_coordinates_batch(
        derivatives, options.compiled_limits);
    std::vector<PackedPrimeSparseMatrix::Triplet> differential_triplets;
    differential_triplets.reserve(static_cast<std::size_t>(std::count_if(
        derivative_batch.entries().begin(), derivative_batch.entries().end(),
        [](std::uint32_t value) { return value != 0; })));
    for (std::size_t column = 0; column < square_length; ++column) {
      for (std::size_t variable = 0; variable < variable_count; ++variable) {
        const auto coordinates =
            derivative_batch.row(column * variable_count + variable);
        for (std::size_t row = 0; row < length; ++row) {
          if (coordinates[row] != 0) {
            differential_triplets.push_back(PackedPrimeSparseMatrix::Triplet{
                variable * length + row, column, coordinates[row]});
          }
        }
      }
    }
    const PackedPrimeSparseMatrix differential(
        ideal.ring().field().modulus(), differential_rows, square_length,
        std::move(differential_triplets));
    const auto restricted_differential = multiply(differential, conormal);
    auto relative_h1 = restricted_differential.rank_and_kernel();
    result.h1_dimension = relative_h1.kernel_basis.size();
    discovery_detail::check_matrix_entries(
        result.conormal_dimension, result.h1_dimension,
        options.max_matrix_entries, "packed relative H1 basis");
    const auto relative_h1_basis = packed_matrix_from_columns(
        ideal.ring().field().modulus(), result.conormal_dimension,
        relative_h1.kernel_basis);
    const auto h1_basis = multiply(conormal, relative_h1_basis);

    discovery_detail::check_matrix_entries(
        variable_count * length, length, options.max_matrix_entries,
        "packed socle multiplication stack");
    const auto multiplication_stack = packed_vertical_stack(
        ideal.ring().field().modulus(), length,
        quotient.variable_actions());
    auto socle_elimination = multiplication_stack.rank_and_kernel();
    result.socle_dimension = socle_elimination.kernel_basis.size();
    const auto socle_basis = packed_matrix_from_columns(
        ideal.ring().field().modulus(), length,
        socle_elimination.kernel_basis);
    if (result.socle_dimension == 0) {
      throw std::logic_error("packed Artin local quotient has zero socle");
    }

    const auto conormal_coordinates =
        discovery_detail::conormal_coordinate_map(conormal);
    const auto action_column_count = discovery_detail::checked_product(
        result.h1_dimension, result.socle_dimension,
        "packed H1 action tensor");
    discovery_detail::check_matrix_entries(
        result.conormal_dimension, action_column_count,
        options.max_matrix_entries, "packed H1 action tensor");
    std::vector<PackedPrimeMatrix> packed_actions;
    packed_actions.reserve(result.h1_dimension);
    PackedPrimeArithmetic packed_arithmetic(ideal.ring().field().modulus());
    std::size_t action_applications = 0;
    for (std::size_t h1_index = 0; h1_index < result.h1_dimension;
         ++h1_index) {
      std::vector<std::vector<std::uint32_t>> action_columns;
      action_columns.reserve(result.socle_dimension);
      const auto cycle = discovery_detail::matrix_column(h1_basis, h1_index);
      for (std::size_t socle_index = 0;
           socle_index < result.socle_dimension; ++socle_index) {
        std::vector<std::uint32_t> ambient(square_length, 0);
        for (std::size_t basis_index = 0; basis_index < length;
             ++basis_index) {
          const auto coefficient = socle_basis(basis_index, socle_index);
          if (coefficient == 0) {
            continue;
          }
          const auto product = square_quotient.multiply_by_monomial(
              PackedMonomial::from_word(
                  quotient.standard_monomial_words()[basis_index]),
              cycle, action_applications, options.compiled_limits);
          for (std::size_t row = 0; row < square_length; ++row) {
            ambient[row] = packed_arithmetic.add(
                ambient[row],
                packed_arithmetic.multiply(coefficient, product[row]));
          }
        }
        action_columns.push_back(
            conormal_coordinates.multiply_column(ambient));
      }
      packed_actions.push_back(packed_matrix_from_columns(
          ideal.ring().field().modulus(), result.conormal_dimension,
          action_columns));
    }

    const auto exact_actions = discovery_detail::exact_matrices(packed_actions);
    const auto rank = analyze_matrix_space(
        ideal.ring().field(), result.conormal_dimension,
        result.socle_dimension, exact_actions, options.matrix_space_limits);
    result.maximum_individual_rank_lower_bound = rank.lower_bound;
    result.maximum_individual_rank_upper_bound = rank.upper_bound;
    result.rank_proof = rank.proof;
    result.full_socle_rank_candidate = rank.has_full_column_rank_witness();
    if (rank.witness_coefficients.has_value()) {
      std::vector<std::uint32_t> coefficients;
      coefficients.reserve(rank.witness_coefficients->size());
      for (const auto entry : *rank.witness_coefficients) {
        coefficients.push_back(entry.value());
      }
      result.witness_coefficients = coefficients;
      result.witness = discovery_detail::combine_h1_witness(
          square_quotient, h1_basis, coefficients);
    }
    if (rank.proof == MatrixSpaceRankProof::ResourceLimit ||
        rank.proof == MatrixSpaceRankProof::GenericOnly) {
      result.status = DiscoveryScreenStatus::ResourceLimit;
      result.detail =
          rank.proof == MatrixSpaceRankProof::ResourceLimit
              ? "exact matrix-space rank search exceeded its configured resource limits"
              : "finite-field matrix-space rank is generic-only; no exact base-field maximum was proved";
      return result;
    }
    if (result.full_socle_rank_candidate && options.certify_hits) {
      if (!result.witness.has_value()) {
        throw std::logic_error("packed full-rank H1 result has no witness");
      }
      result.certification = audit_cycle(
          ideal, *result.witness, options.certification_limits);
      result.certified_faithful = result.certification->faithful_cycle();
      if (!result.certification->conclusive()) {
        result.status = DiscoveryScreenStatus::ResourceLimit;
        result.detail = result.certification->resource_detail();
        return result;
      }
      if (!result.certified_faithful) {
        throw std::logic_error(
            "packed full-rank H1 witness failed exact colon certification");
      }
    }
    result.status = DiscoveryScreenStatus::Complete;
    return result;
  } catch (const PackedDiscoveryResourceLimit& error) {
    result.status = DiscoveryScreenStatus::ResourceLimit;
    result.detail = error.what();
    return result;
  } catch (const ResourceLimitExceeded& error) {
    result.status = DiscoveryScreenStatus::ResourceLimit;
    result.detail = error.what();
    return result;
  } catch (const GroebnerResourceLimit& error) {
    result.status = DiscoveryScreenStatus::ResourceLimit;
    result.detail = error.what();
    return result;
  } catch (const std::overflow_error& error) {
    result.status = DiscoveryScreenStatus::ResourceLimit;
    result.detail = error.what();
    return result;
  } catch (const std::domain_error& error) {
    result.status = DiscoveryScreenStatus::InvalidInput;
    result.detail = error.what();
    return result;
  } catch (const std::invalid_argument& error) {
    result.status = DiscoveryScreenStatus::InvalidInput;
    result.detail = error.what();
    return result;
  }
}

}  // namespace laughableengine

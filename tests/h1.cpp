#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/h1.hpp"

namespace {

using laughableengine::DenseMatrix;
using laughableengine::FullH1InputError;
using laughableengine::FullH1InputIssue;
using laughableengine::FullH1Options;
using laughableengine::FullH1ResourceLimit;
using laughableengine::GF;
using laughableengine::H1ActionData;
using laughableengine::Ideal;
using laughableengine::MatrixSpaceRankProof;
using laughableengine::Order;
using laughableengine::PrimeField;
using laughableengine::QQ;
using laughableengine::RationalField;
using laughableengine::full_h1_action;
using laughableengine::horizontal_stack;
using laughableengine::make_ring;
using laughableengine::matrix_from_columns;
using laughableengine::multiply;

[[noreturn]] void fail(const std::string& message, int line) {
  throw std::runtime_error("line " + std::to_string(line) + ": " + message);
}

#define CHECK(expression)                                                     \
  do {                                                                        \
    if (!(expression)) {                                                      \
      fail("CHECK failed: " #expression, __LINE__);                          \
    }                                                                         \
  } while (false)

template <typename Function>
void expect_h1_issue(
    Function&& function,
    FullH1InputIssue expected,
    int line) {
  try {
    std::invoke(std::forward<Function>(function));
  } catch (const FullH1InputError& error) {
    if (error.issue() != expected) {
      fail("full-H1 input error reported the wrong issue", line);
    }
    return;
  } catch (const std::exception& error) {
    fail("unexpected exception type: " + std::string(error.what()), line);
  }
  fail("expected full-H1 input error was not thrown", line);
}

#define EXPECT_H1_ISSUE(issue, expression)                                    \
  expect_h1_issue(                                                            \
      [&] { static_cast<void>(expression); }, issue, __LINE__)

template <typename Field>
std::vector<typename Field::Element> matrix_column(
    const DenseMatrix<Field>& matrix,
    std::size_t column) {
  std::vector<typename Field::Element> result;
  result.reserve(matrix.row_count());
  for (std::size_t row = 0; row < matrix.row_count(); ++row) {
    result.push_back(matrix(row, column));
  }
  return result;
}

template <typename Field>
DenseMatrix<Field> linear_combination(
    const Field& field,
    std::size_t rows,
    std::size_t columns,
    const std::vector<DenseMatrix<Field>>& matrices,
    const std::vector<typename Field::Element>& coefficients) {
  CHECK(matrices.size() == coefficients.size());
  DenseMatrix<Field> result(field, rows, columns);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      auto entry = field.zero();
      for (std::size_t index = 0; index < matrices.size(); ++index) {
        entry = field.add(
            entry,
            field.multiply(coefficients[index], matrices[index](row, column)));
      }
      result.set(row, column, std::move(entry));
    }
  }
  return result;
}

template <typename Field>
void check_structural_invariants(const H1ActionData<Field>& data) {
  const auto& field = data.quotient.field();
  const auto variable_count = data.ideal.ring().variable_count();
  const auto length = data.length_Q;
  const auto square_length = data.length_P_mod_J2;
  const auto conormal_dimension = data.conormal_dimension;
  const auto h1_dimension = data.h1_dimension;
  const auto socle_dimension = data.socle_dimension;

  CHECK(data.quotient.ideal() == data.ideal);
  CHECK(data.square_quotient.ideal() == data.squared_ideal);
  CHECK(data.length_Q == data.quotient.dimension());
  CHECK(data.length_P_mod_J2 == data.square_quotient.dimension());
  CHECK(data.conormal_dimension == square_length - length);
  CHECK(data.conormal.dimension() == conormal_dimension);
  CHECK(data.h1.dimension() == h1_dimension);
  CHECK(data.socle.dimension() == socle_dimension);
  CHECK(data.conormal.ambient_dimension() == square_length);
  CHECK(data.h1.ambient_dimension() == square_length);
  CHECK(data.socle.ambient_dimension() == length);

  CHECK(data.reduction_map.row_count() == length);
  CHECK(data.reduction_map.column_count() == square_length);
  CHECK(data.reduction_map.rank() == length);
  CHECK(data.differential.row_count() == variable_count * length);
  CHECK(data.differential.column_count() == square_length);
  CHECK(multiply(data.reduction_map, data.conormal.basis_matrix()) ==
        DenseMatrix<Field>(field, length, conormal_dimension));
  CHECK(multiply(data.reduction_map, data.h1.basis_matrix()) ==
        DenseMatrix<Field>(field, length, h1_dimension));
  CHECK(multiply(data.differential, data.h1.basis_matrix()) ==
        DenseMatrix<Field>(field, variable_count * length, h1_dimension));

  const auto expected_reduction_columns = data.quotient.coordinates(
      data.square_quotient.polynomial_basis());
  CHECK(data.reduction_map == matrix_from_columns(
                                  field, length,
                                  expected_reduction_columns));

  CHECK(data.conormal_basis.size() == conormal_dimension);
  CHECK(data.h1_basis.size() == h1_dimension);
  CHECK(data.socle_basis.size() == socle_dimension);
  const auto conormal_vectors = data.conormal.basis_vectors();
  const auto h1_vectors = data.h1.basis_vectors();
  const auto socle_vectors = data.socle.basis_vectors();
  for (std::size_t index = 0; index < conormal_dimension; ++index) {
    CHECK(data.conormal_basis[index] ==
          data.square_quotient.representative(conormal_vectors[index]));
  }
  for (std::size_t index = 0; index < h1_dimension; ++index) {
    CHECK(data.conormal.contains(h1_vectors[index]));
    CHECK(data.h1_basis[index] ==
          data.square_quotient.representative(h1_vectors[index]));
    CHECK(data.ideal.contains(data.h1_basis[index]));
    for (std::size_t variable = 0; variable < variable_count; ++variable) {
      CHECK(data.ideal.contains(data.h1_basis[index].derivative(variable)));
    }
  }
  for (std::size_t index = 0; index < socle_dimension; ++index) {
    CHECK(data.socle_basis[index] ==
          data.quotient.representative(socle_vectors[index]));
    for (std::size_t variable = 0; variable < variable_count; ++variable) {
      CHECK(data.ideal.contains(
          data.ideal.ring().gen(variable) * data.socle_basis[index]));
    }
  }

  CHECK(data.variable_multiplication_matrices.size() == variable_count);
  for (std::size_t variable = 0; variable < variable_count; ++variable) {
    const auto& multiplication =
        data.variable_multiplication_matrices[variable];
    CHECK(multiplication.row_count() == length);
    CHECK(multiplication.column_count() == length);
    CHECK(multiply(multiplication, data.socle.basis_matrix()) ==
          DenseMatrix<Field>(field, length, socle_dimension));
  }
  for (std::size_t left = 0; left < variable_count; ++left) {
    for (std::size_t right = left + 1; right < variable_count; ++right) {
      CHECK(multiply(data.variable_multiplication_matrices[left],
                     data.variable_multiplication_matrices[right]) ==
            multiply(data.variable_multiplication_matrices[right],
                     data.variable_multiplication_matrices[left]));
    }
  }

  CHECK(data.h1_multiplication_matrices.size() == h1_dimension);
  CHECK(data.action_matrices.size() == h1_dimension);
  for (std::size_t index = 0; index < h1_dimension; ++index) {
    const auto& multiplication = data.h1_multiplication_matrices[index];
    const auto& action = data.action_matrices[index];
    CHECK(multiplication.row_count() == conormal_dimension);
    CHECK(multiplication.column_count() == length);
    CHECK(action.row_count() == conormal_dimension);
    CHECK(action.column_count() == socle_dimension);
    CHECK(action == multiply(multiplication, data.socle.basis_matrix()));
    for (std::size_t socle_index = 0; socle_index < socle_dimension;
         ++socle_index) {
      const auto product = data.square_quotient.coordinates(
          data.h1_basis[index] * data.socle_basis[socle_index]);
      CHECK(data.conormal.contains(product));
      CHECK(matrix_column(action, socle_index) ==
            data.conormal.coordinates(product));
    }
  }

  const auto common_products = horizontal_stack(
      field, conormal_dimension, data.action_matrices);
  CHECK(data.common_product_space_dimension == common_products.rank());
  CHECK(data.common_product_space_rank_bound ==
        std::min(socle_dimension, data.common_product_space_dimension));
  CHECK(data.individual_rank.parameter_count == h1_dimension);
  CHECK(data.individual_rank.row_count == conormal_dimension);
  CHECK(data.individual_rank.column_count == socle_dimension);
  CHECK(data.individual_rank.lower_bound <= data.individual_rank.upper_bound);
  CHECK(data.individual_rank.upper_bound <=
        data.common_product_space_rank_bound);

  CHECK(data.best_h1_coefficients ==
        data.individual_rank.witness_coefficients);
  CHECK(data.best_h1_coefficients.has_value() ==
        data.best_h1_polynomial.has_value());
  if (data.best_h1_coefficients.has_value()) {
    CHECK(data.best_h1_coefficients->size() == h1_dimension);
    const auto witness_action = linear_combination(
        field, conormal_dimension, socle_dimension, data.action_matrices,
        *data.best_h1_coefficients);
    CHECK(witness_action.rank() == data.individual_rank.lower_bound);

    auto expected_polynomial = data.ideal.ring().zero();
    for (std::size_t index = 0; index < h1_dimension; ++index) {
      expected_polynomial =
          expected_polynomial +
          data.h1_basis[index].scaled((*data.best_h1_coefficients)[index]);
    }
    CHECK(expected_polynomial == *data.best_h1_polynomial);
    CHECK(data.ideal.contains(*data.best_h1_polynomial));
    for (std::size_t variable = 0; variable < variable_count; ++variable) {
      CHECK(data.ideal.contains(
          data.best_h1_polynomial->derivative(variable)));
    }
  }

  CHECK(data.faithful_witness_audit.has_value() ==
        data.individual_rank.has_full_column_rank_witness());
  if (data.faithful_witness_audit.has_value()) {
    const auto& audit = *data.faithful_witness_audit;
    CHECK(audit.status() == laughableengine::CycleAuditStatus::Complete);
    CHECK(audit.faithful_cycle());
    CHECK(audit.cycle_valid() == true);
    CHECK(audit.primitive() == true);
    CHECK(audit.colon_equals_ideal() == true);
    CHECK(audit.annihilator_zero() == true);
    CHECK(audit.ideal() == data.ideal);
    CHECK(audit.ideal_square().has_value());
    CHECK(*audit.ideal_square() == data.squared_ideal);
  }

  CHECK(data.ideal.is_subset_of(data.common_annihilator_diagnostic));
}

void test_three_variable_seed() {
  const auto ring = make_ring(QQ(), {"x", "y", "z"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto z = ring.gen("z");
  const auto seed = y.pow(5) + z.pow(5) + y.pow(2) * z.pow(2);
  const Ideal<RationalField> ideal(
      ring,
      {x.pow(2), x * y, x * z, seed, seed.derivative("y"),
       seed.derivative("z")});

  const auto data = full_h1_action(ideal);
  check_structural_invariants(data);
  CHECK(data.length_Q == 11);
  CHECK(data.length_P_mod_J2 == 48);
  CHECK(data.conormal_dimension == 37);
  CHECK(data.h1_dimension == 19);
  CHECK(data.socle_dimension == 3);
  CHECK(data.common_product_space_dimension == 1);
  CHECK(data.common_product_space_rank_bound == 1);
  CHECK(data.individual_rank.exact_maximum == 1);
  CHECK(data.individual_rank.lower_bound == 1);
  CHECK(data.individual_rank.upper_bound == 1);
  CHECK(data.individual_rank.proof == MatrixSpaceRankProof::ProvenMaximum);
  CHECK(!data.individual_rank.has_full_column_rank_witness());
}

void test_lifted_seed_agrees_across_two_primes() {
  for (const auto prime : {101U, 103U}) {
    const auto ring =
        make_ring(GF(prime), {"x", "y", "z"}, Order::Grevlex);
    const auto x = ring.gen("x");
    const auto y = ring.gen("y");
    const auto z = ring.gen("z");
    const auto seed = y.pow(5) + z.pow(5) + y.pow(2) * z.pow(2);
    const Ideal<PrimeField> ideal(
        ring,
        {x.pow(2), x * y, x * z, seed, seed.derivative("y"),
         seed.derivative("z")});
    const auto data = full_h1_action(ideal);
    CHECK(data.length_Q == 11);
    CHECK(data.length_P_mod_J2 == 48);
    CHECK(data.h1_dimension == 19);
    CHECK(data.socle_dimension == 3);
    CHECK(data.individual_rank.exact_maximum == 1);
    CHECK(data.individual_rank.proof == MatrixSpaceRankProof::ProvenMaximum);
  }
}

void test_tjurina_near_hit() {
  const auto ring = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto potential = x.pow(3) + y.pow(7) + x * y.pow(5);
  const Ideal<RationalField> ideal(
      ring,
      {potential, potential.derivative("x"), potential.derivative("y")});

  const auto data = full_h1_action(ideal);
  check_structural_invariants(data);
  CHECK(data.length_Q == 11);
  CHECK(data.length_P_mod_J2 == 34);
  CHECK(data.conormal_dimension == 23);
  CHECK(data.h1_dimension == 15);
  CHECK(data.socle_dimension == 2);
  CHECK(data.common_product_space_dimension == 1);
  CHECK(data.common_product_space_rank_bound == 1);
  CHECK(data.individual_rank.exact_maximum == 1);
  CHECK(data.individual_rank.proof == MatrixSpaceRankProof::ProvenMaximum);
  CHECK(!data.individual_rank.has_full_column_rank_witness());

  const auto potential_coordinates =
      data.square_quotient.coordinates(potential);
  CHECK(data.h1.contains(potential_coordinates));
  const auto h1_coordinates = data.h1.coordinates(potential_coordinates);
  const auto distinguished_action = linear_combination(
      data.quotient.field(), data.conormal_dimension, data.socle_dimension,
      data.action_matrices, h1_coordinates);
  CHECK(distinguished_action.rank() == 1);
}

void test_homogeneous_characteristic_zero_action_is_zero() {
  const auto ring = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const Ideal<RationalField> ideal(ring, {x.pow(2), y.pow(2)});

  const auto data = full_h1_action(ideal);
  check_structural_invariants(data);
  CHECK(data.common_product_space_dimension == 0);
  CHECK(data.common_product_space_rank_bound == 0);
  CHECK(data.individual_rank.exact_maximum == 0);
  CHECK(data.individual_rank.proof == MatrixSpaceRankProof::ProvenMaximum);
  CHECK(!data.individual_rank.has_full_column_rank_witness());
  for (const auto& action : data.action_matrices) {
    CHECK(action == DenseMatrix<RationalField>(
                        data.quotient.field(), data.conormal_dimension,
                        data.socle_dimension));
  }
}

void test_frobenius_cycle_has_full_rank() {
  const auto ring = make_ring(GF(5), {"x"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const Ideal<PrimeField> ideal(ring, {x.pow(5)});

  const auto data = full_h1_action(ideal);
  check_structural_invariants(data);
  CHECK(data.length_Q == 5);
  CHECK(data.length_P_mod_J2 == 10);
  CHECK(data.conormal_dimension == 5);
  CHECK(data.h1_dimension == 5);
  CHECK(data.socle_dimension == 1);
  CHECK(data.common_product_space_dimension == 1);
  CHECK(data.common_product_space_rank_bound == 1);
  CHECK(data.individual_rank.exact_maximum == 1);
  CHECK(data.individual_rank.proof ==
        MatrixSpaceRankProof::ProvenFullColumnRank);
  CHECK(data.individual_rank.has_full_column_rank_witness());
  CHECK(data.best_h1_polynomial.has_value());
  CHECK(data.faithful_witness_audit.has_value());
  CHECK(data.faithful_witness_audit->faithful_cycle());
  CHECK(data.faithful_witness_audit->colon_evidence().has_value());
  CHECK(data.faithful_witness_audit->colon_evidence()->colon_ideal() ==
        data.ideal);

  const auto witness_coordinates =
      data.square_quotient.coordinates(*data.best_h1_polynomial);
  CHECK(data.h1.contains(witness_coordinates));
  const auto witness_h1_coordinates = data.h1.coordinates(witness_coordinates);
  const auto witness_action = linear_combination(
      data.quotient.field(), data.conormal_dimension, data.socle_dimension,
      data.action_matrices, witness_h1_coordinates);
  CHECK(witness_action.rank() == data.socle_dimension);
  CHECK(data.common_annihilator_diagnostic == data.ideal);
}

void test_rejected_inputs_are_distinguished() {
  const auto bivariate = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  const auto x = bivariate.gen("x");
  const Ideal<RationalField> unit(bivariate, {bivariate.one()});
  const Ideal<RationalField> positive_dimensional(bivariate, {x});

  EXPECT_H1_ISSUE(
      FullH1InputIssue::UnitQuotient,
      full_h1_action(unit));
  EXPECT_H1_ISSUE(
      FullH1InputIssue::PositiveDimensionalQuotient,
      full_h1_action(positive_dimensional));

  const auto univariate = make_ring(QQ(), {"x"}, Order::Grevlex);
  const auto ux = univariate.gen("x");
  const Ideal<RationalField> away_from_origin(
      univariate, {ux.pow(2) - ux});
  EXPECT_H1_ISSUE(
      FullH1InputIssue::NotSupportedAtOrigin,
      full_h1_action(away_from_origin));
}

void test_resource_limits_remain_inconclusive() {
  const auto ring = make_ring(GF(5), {"x"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const Ideal<PrimeField> ideal(ring, {x.pow(5)});

  FullH1Options matrix_limited;
  matrix_limited.max_matrix_entries = 0;
  bool caught_matrix_limit = false;
  try {
    static_cast<void>(full_h1_action(ideal, matrix_limited));
  } catch (const FullH1ResourceLimit& error) {
    caught_matrix_limit =
        std::string(error.what()).find("max_matrix_entries=0") !=
        std::string::npos;
  }
  CHECK(caught_matrix_limit);

  const auto bivariate = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  const auto bx = bivariate.gen("x");
  const auto by = bivariate.gen("y");
  const Ideal<RationalField> solve_fixture(
      bivariate, {bx, by.pow(2)});
  FullH1Options solve_limited;
  solve_limited.max_matrix_entries = 29;
  bool caught_augmented_solve = false;
  try {
    static_cast<void>(full_h1_action(solve_fixture, solve_limited));
  } catch (const FullH1ResourceLimit& error) {
    caught_augmented_solve =
        std::string(error.what()).find("conormal coordinate solve needs 30") !=
        std::string::npos;
  }
  CHECK(caught_augmented_solve);

  FullH1Options replay_limited;
  replay_limited.witness_audit_limits.max_matrix_entries = 0;
  const auto data = full_h1_action(ideal, replay_limited);
  CHECK(data.individual_rank.has_full_column_rank_witness());
  CHECK(data.best_h1_polynomial.has_value());
  CHECK(data.faithful_witness_audit.has_value());
  CHECK(data.faithful_witness_audit->status() ==
        laughableengine::CycleAuditStatus::ResourceLimit);
  CHECK(!data.faithful_witness_audit->conclusive());
  CHECK(!data.faithful_witness_audit->faithful_cycle());

  FullH1Options replay_monomial_limited;
  replay_monomial_limited.witness_audit_limits.standard_monomials
      .max_monomials = 0;
  const auto monomial_limited =
      full_h1_action(ideal, replay_monomial_limited);
  CHECK(monomial_limited.faithful_witness_audit.has_value());
  CHECK(monomial_limited.faithful_witness_audit->status() ==
        laughableengine::CycleAuditStatus::ResourceLimit);
}

}  // namespace

int main() {
  try {
    test_lifted_seed_agrees_across_two_primes();
    test_three_variable_seed();
    test_tjurina_near_hit();
    test_homogeneous_characteristic_zero_action_is_zero();
    test_frobenius_cycle_has_full_rank();
    test_rejected_inputs_are_distinguished();
    test_resource_limits_remain_inconclusive();
    std::cout << "H1 action tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

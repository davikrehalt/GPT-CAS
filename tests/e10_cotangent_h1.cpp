#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "laughableengine/cotangent_h1.hpp"
#include "laughableengine/field.hpp"
#include "laughableengine/polynomial.hpp"
#include "support/e10_fixture.hpp"

namespace {

using laughableengine::CotangentClassStatus;
using laughableengine::CotangentH1Spec;
using laughableengine::GF;
using laughableengine::Order;
using laughableengine::Polynomial;
using laughableengine::PolynomialRing;
using laughableengine::QQ;
using laughableengine::SparseEliminationLimits;
using laughableengine::cotangent_h1;
using laughableengine::tests::support::make_e10_fixture;

[[noreturn]] void fail(const std::string& message, int line) {
  throw std::runtime_error(
      "line " + std::to_string(line) + ": " + message);
}

#define CHECK(expression)                                                     \
  do {                                                                        \
    if (!(expression)) {                                                      \
      fail("CHECK failed: " #expression, __LINE__);                          \
    }                                                                         \
  } while (false)

[[nodiscard]] std::vector<std::string> variable_names() {
  std::vector<std::string> result;
  for (std::size_t index = 1; index <= 10; ++index) {
    result.push_back("x" + std::to_string(index));
  }
  return result;
}

template <typename Field>
void run_e10(Field field) {
  const PolynomialRing<Field> ring(
      std::move(field), variable_names(), Order::Grevlex);
  const auto fixture = make_e10_fixture(ring);
  std::vector<Polynomial<Field>> partials(
      fixture.derivatives.begin(), fixture.derivatives.end());

  const auto h1 = cotangent_h1(CotangentH1Spec{
      ring, std::move(partials), std::size_t{4}});
  CHECK(h1.length_Q() == 176);
  CHECK(h1.length_P_mod_J2() == 2728);
  CHECK(h1.conormal_dimension() == 2552);
  CHECK(h1.h1_dimension() == 1873);
  CHECK(h1.ideal_generators().size() == 725);
  CHECK(h1.reduction_matrix().rank() == 176);
  CHECK(h1.h1_relation_matrix().column_count() == 2728);
  CHECK(h1.h1_relation_matrix().rank() == 855);

  const auto proof = h1.verify_class(fixture.potential);
  CHECK(proof.status == CotangentClassStatus::Complete);
  CHECK(proof.in_ideal);
  CHECK(proof.cycle_valid);
  CHECK(proof.multiplication_rank == 176);
  CHECK(proof.annihilator_dimension == 0);
  CHECK(proof.annihilator_coordinates.empty());
  CHECK(proof.annihilator_basis.empty());
  CHECK(proof.colon_generators.size() == 725);
  CHECK(proof.faithful);
  CHECK(proof.colon_equals_ideal);

  SparseEliminationLimits basis_limits;
  basis_limits.max_kernel_nonzeros = 20'000'000;
  const auto coordinates = h1.h1_kernel_coordinates(basis_limits);
  const auto basis = h1.h1_basis(basis_limits);
  CHECK(coordinates.size() == 1873);
  CHECK(basis.size() == 1873);
  std::size_t coordinate_nonzeros = 0;
  std::size_t polynomial_terms = 0;
  for (std::size_t index = 0; index < basis.size(); ++index) {
    coordinate_nonzeros += coordinates[index].size();
    polynomial_terms += basis[index].term_count();
    CHECK(h1.quotient_remainder(basis[index]).is_zero());
    for (std::size_t variable = 0; variable < 10; ++variable) {
      CHECK(h1.quotient_remainder(
                   basis[index].derivative(variable))
                .is_zero());
    }
  }
  CHECK(coordinate_nonzeros == 2092);
  CHECK(polynomial_terms == 2092);
}

}  // namespace

int main(int argument_count, char** arguments) {
  if (argument_count == 1) {
    run_e10(GF(101));
    return 0;
  }
  if (argument_count == 2 && std::string_view(arguments[1]) == "--qq") {
    run_e10(QQ());
    return 0;
  }
  std::cerr << "usage: e10_cotangent_h1 [--qq]\n";
  return 2;
}

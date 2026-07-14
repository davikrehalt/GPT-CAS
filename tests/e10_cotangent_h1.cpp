#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "laughableengine/field.hpp"
#include "laughableengine/finite_algebra.hpp"
#include "laughableengine/polynomial.hpp"
#include "support/e10_fixture.hpp"

namespace {

using laughableengine::GF;
using laughableengine::Order;
using laughableengine::Polynomial;
using laughableengine::PolynomialRing;
using laughableengine::QQ;
using laughableengine::SparseEliminationLimits;
using laughableengine::annihilator;
using laughableengine::origin_power_ideal;
using laughableengine::preimage;
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

  const auto J = origin_power_ideal(
      ring, std::move(partials), std::size_t{4});
  const auto R = J.quotient();
  const auto conormal = R.conormal_module();
  const auto derivative = conormal.derivative_map();
  const auto h1 = derivative.kernel();

  CHECK(R.dimension() == 176);
  CHECK(R.square_quotient_dimension() == 2728);
  CHECK(conormal.dimension() == 2552);
  CHECK(h1.dimension() == 1873);
  CHECK(J.generators().size() == 725);
  CHECK(conormal.defining_matrix().rank() == 176);
  CHECK(h1.defining_matrix().column_count() == 2728);
  CHECK(h1.defining_matrix().rank() == 855);

  const auto xi = h1.class_of(fixture.potential);
  const auto ann = annihilator(xi);
  CHECK(ann.dimension() == 0);
  CHECK(ann.basis_coordinates().empty());
  CHECK(ann.lift_basis().empty());
  CHECK(ann == R.zero_ideal());

  const auto colon = preimage(ann);
  CHECK(colon == J);
  CHECK(colon.equals_source_ideal());
  CHECK(colon.generators().size() == 725);

  SparseEliminationLimits basis_limits;
  basis_limits.max_kernel_nonzeros = 20'000'000;
  const auto coordinates = h1.basis_coordinates(basis_limits);
  const auto basis = h1.representative_basis(basis_limits);
  CHECK(coordinates.size() == 1873);
  CHECK(basis.size() == 1873);
  std::size_t coordinate_nonzeros = 0;
  std::size_t polynomial_terms = 0;
  for (std::size_t index = 0; index < basis.size(); ++index) {
    coordinate_nonzeros += coordinates[index].size();
    polynomial_terms += basis[index].term_count();
    CHECK(R.remainder(basis[index]).is_zero());
    for (std::size_t variable = 0; variable < 10; ++variable) {
      CHECK(R.remainder(basis[index].derivative(variable))
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

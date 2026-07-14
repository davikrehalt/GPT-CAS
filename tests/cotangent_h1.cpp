#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "laughableengine/cotangent_h1.hpp"
#include "laughableengine/field.hpp"
#include "laughableengine/polynomial.hpp"

namespace {

using laughableengine::CotangentClassStatus;
using laughableengine::CotangentH1Spec;
using laughableengine::GF;
using laughableengine::QQ;
using laughableengine::cotangent_h1;
using laughableengine::make_ring;

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

void characteristic_p_faithful_cycle() {
  const auto ring = make_ring(GF(5), {"x"});
  const auto x = ring.gen(0);
  auto h1 = cotangent_h1(CotangentH1Spec{
      ring, std::vector<typename decltype(ring)::PolynomialType>{}, std::size_t{5}});

  CHECK(h1.length_Q() == 5);
  CHECK(h1.length_P_mod_J2() == 10);
  CHECK(h1.conormal_dimension() == 5);
  CHECK(h1.h1_dimension() == 5);
  CHECK(h1.reduction_matrix().row_count() == 5);
  CHECK(h1.reduction_matrix().column_count() == 10);
  CHECK(h1.derivative_matrix().row_count() == 5);
  CHECK(h1.cycle_matrix().row_count() == 10);

  const auto proof = h1.verify_class(x.pow(5));
  CHECK(proof.status == CotangentClassStatus::Complete);
  CHECK(proof.in_ideal);
  CHECK(proof.cycle_valid);
  CHECK(proof.multiplication_rank == 5);
  CHECK(proof.annihilator_dimension == 0);
  CHECK(proof.annihilator_basis.empty());
  CHECK(proof.colon_generators.size() == 1);
  CHECK(proof.colon_generators[0] == x.pow(5));
  CHECK(proof.faithful);
  CHECK(proof.colon_equals_ideal);
}

void rational_negative_states_and_kernel() {
  const auto ring = make_ring(QQ(), {"x"});
  const auto x = ring.gen(0);
  auto h1 = cotangent_h1(CotangentH1Spec{
      ring, std::vector<typename decltype(ring)::PolynomialType>{}, std::size_t{2}});

  CHECK(h1.length_Q() == 2);
  CHECK(h1.length_P_mod_J2() == 4);
  CHECK(h1.conormal_dimension() == 2);
  CHECK(h1.h1_dimension() == 1);

  const auto outside = h1.verify_class(x);
  CHECK(outside.status == CotangentClassStatus::NotInIdeal);
  CHECK(!outside.in_ideal);
  CHECK(!outside.cycle_valid);
  CHECK(!outside.multiplication_matrix.has_value());

  const auto noncycle = h1.verify_class(x.pow(2));
  CHECK(noncycle.status == CotangentClassStatus::NotCycle);
  CHECK(noncycle.in_ideal);
  CHECK(!noncycle.cycle_valid);

  const auto cycle = h1.verify_class(x.pow(3));
  CHECK(cycle.status == CotangentClassStatus::Complete);
  CHECK(cycle.multiplication_rank == 1);
  CHECK(cycle.annihilator_dimension == 1);
  CHECK(cycle.annihilator_basis.size() == 1);
  CHECK(cycle.annihilator_basis[0] == x);
  CHECK(cycle.colon_generators.size() == 2);
  CHECK(!cycle.faithful);
  CHECK(!cycle.colon_equals_ideal);

  laughableengine::SparseEliminationLimits limits;
  limits.max_kernel_coordinate_entries = 100;
  const auto basis = h1.h1_basis(limits);
  CHECK(basis.size() == 1);
  CHECK(h1.quotient_remainder(basis[0]).is_zero());
  CHECK(h1.quotient_remainder(basis[0].derivative(0)).is_zero());
}

void explicit_lower_generators() {
  const auto ring = make_ring(QQ(), {"x", "y"});
  const auto x = ring.gen(0);
  const auto y = ring.gen(1);
  auto h1 = cotangent_h1(CotangentH1Spec{
      ring,
      std::vector<typename decltype(ring)::PolynomialType>{x.pow(2) - y.pow(2)},
      std::size_t{3}});

  CHECK(h1.length_Q() == 5);
  CHECK(h1.reduction_matrix().rank() == h1.length_Q());
  CHECK(h1.h1_dimension() <= h1.conormal_dimension());

  const auto zero = h1.verify_class(ring.zero());
  CHECK(zero.status == CotangentClassStatus::Complete);
  CHECK(zero.multiplication_rank == 0);
  CHECK(zero.annihilator_dimension == h1.length_Q());
  CHECK(!zero.faithful);
}

}  // namespace

int main() {
  characteristic_p_faithful_cycle();
  rational_negative_states_and_kernel();
  explicit_lower_generators();
}

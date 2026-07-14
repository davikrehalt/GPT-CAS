#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "laughableengine/discovery.hpp"
#include "laughableengine/h1.hpp"
#include "laughableengine/ideal.hpp"

namespace {

using laughableengine::DiscoveryScreenStatus;
using laughableengine::GF;
using laughableengine::Ideal;
using laughableengine::Order;
using laughableengine::PackedCycleDiscoverySession;
using laughableengine::PackedCycleScreenOptions;
using laughableengine::PackedH1ScreenOptions;
using laughableengine::PolynomialRing;
using laughableengine::PrimeField;
using laughableengine::full_h1_action;
using laughableengine::screen_cycle;
using laughableengine::screen_full_h1;

[[noreturn]] void fail(const std::string& message, int line) {
  throw std::runtime_error("line " + std::to_string(line) + ": " + message);
}

#define CHECK(expression)                                                     \
  do {                                                                        \
    if (!(expression)) {                                                      \
      fail("CHECK failed: " #expression, __LINE__);                          \
    }                                                                         \
  } while (false)

template <typename Exception, typename Function>
void expect_throw(Function&& function, int line) {
  try {
    std::invoke(std::forward<Function>(function));
  } catch (const Exception&) {
    return;
  } catch (const std::exception& error) {
    fail("unexpected exception type: " + std::string(error.what()), line);
  }
  fail("expected exception was not thrown", line);
}

#define EXPECT_THROW(exception, expression)                                   \
  expect_throw<exception>([&] { static_cast<void>(expression); }, __LINE__)

Ideal<PrimeField> seed_ideal(const PolynomialRing<PrimeField>& ring) {
  const auto x = ring.gen(0);
  const auto y = ring.gen(1);
  const auto z = ring.gen(2);
  const auto f = y.pow(5) + z.pow(5) + y.pow(2) * z.pow(2);
  return Ideal<PrimeField>(
      ring,
      {x.pow(2), x * y, x * z, f, f.derivative(1), f.derivative(2)});
}

void test_packed_cycle_screen_and_certification() {
  const PolynomialRing<PrimeField> ring(GF(5), {"x"}, Order::Grevlex);
  const auto x = ring.gen(0);
  const Ideal<PrimeField> ideal(ring, {x.pow(5)});
  const auto result = screen_cycle(ideal, x.pow(5));
  CHECK(result.status == DiscoveryScreenStatus::Complete);
  CHECK(result.length_Q == 5);
  CHECK(result.length_P_mod_J2 == 10);
  CHECK(result.cycle_valid);
  CHECK(result.multiplication_rank == 5);
  CHECK(result.full_column_rank_candidate);
  CHECK(result.certified_faithful);
  CHECK(result.certification.has_value());

  const PackedCycleDiscoverySession session(ideal);
  const auto repeated = session.screen(x.pow(5));
  CHECK(repeated.certified_faithful);
  const auto zero = session.screen(ring.zero());
  CHECK(zero.cycle_valid);
  CHECK(zero.multiplication_rank == 0);
  CHECK(!zero.full_column_rank_candidate);
}

void test_cycle_gate_and_resource_semantics() {
  const PolynomialRing<PrimeField> ring(GF(101), {"x"}, Order::Grevlex);
  const auto x = ring.gen(0);
  const Ideal<PrimeField> ideal(ring, {x.pow(2)});
  const auto derivative_failure = screen_cycle(ideal, x.pow(2));
  CHECK(derivative_failure.status == DiscoveryScreenStatus::Complete);
  CHECK(derivative_failure.g_in_J);
  CHECK(!derivative_failure.derivatives_in_J);
  CHECK(!derivative_failure.cycle_valid);
  CHECK(!derivative_failure.full_column_rank_candidate);

  PackedCycleScreenOptions options;
  options.max_matrix_entries = 0;
  const auto limited = screen_cycle(ideal, x.pow(2), options);
  CHECK(limited.status == DiscoveryScreenStatus::ResourceLimit);
  CHECK(limited.detail.has_value());

  PackedCycleScreenOptions groebner_limited_options;
  groebner_limited_options.groebner_limits.max_basis_polynomials = 0;
  const auto groebner_limited_cycle = screen_cycle(
      ideal, x.pow(2), groebner_limited_options);
  CHECK(groebner_limited_cycle.status ==
        DiscoveryScreenStatus::ResourceLimit);

  PackedH1ScreenOptions groebner_limited_h1_options;
  groebner_limited_h1_options.groebner_limits.max_basis_polynomials = 0;
  const auto groebner_limited_h1 = screen_full_h1(
      ideal, groebner_limited_h1_options);
  CHECK(groebner_limited_h1.status ==
        DiscoveryScreenStatus::ResourceLimit);

  const Ideal<PrimeField> positive_dimensional(ring, {});
  const auto invalid = screen_cycle(positive_dimensional, ring.zero());
  CHECK(invalid.status == DiscoveryScreenStatus::InvalidInput);

  const PolynomialRing<PrimeField> other_ring(
      GF(101), {"x"}, Order::Grevlex);
  const auto mixed = screen_cycle(ideal, other_ring.gen(0));
  CHECK(mixed.status == DiscoveryScreenStatus::InvalidInput);

  const auto outside_packed_envelope = screen_cycle(ideal, x.pow(256));
  CHECK(outside_packed_envelope.status ==
        DiscoveryScreenStatus::ResourceLimit);
  CHECK(outside_packed_envelope.detail.has_value());
}

void test_full_h1_matches_reference_at_two_primes() {
  for (const auto prime : {101U, 103U}) {
    const PolynomialRing<PrimeField> ring(
        GF(prime), {"x", "y", "z"}, Order::Grevlex);
    const auto ideal = seed_ideal(ring);
    const auto packed = screen_full_h1(ideal);
    const auto reference = full_h1_action(ideal);
    CHECK(packed.status == DiscoveryScreenStatus::Complete);
    CHECK(packed.length_Q == reference.length_Q);
    CHECK(packed.length_P_mod_J2 == reference.length_P_mod_J2);
    CHECK(packed.conormal_dimension == reference.conormal_dimension);
    CHECK(packed.h1_dimension == reference.h1_dimension);
    CHECK(packed.socle_dimension == reference.socle_dimension);
    CHECK(packed.maximum_individual_rank_lower_bound ==
          reference.individual_rank.lower_bound);
    CHECK(packed.maximum_individual_rank_upper_bound ==
          reference.individual_rank.upper_bound);
    CHECK(!packed.full_socle_rank_candidate);
    CHECK(!packed.certified_faithful);
  }
}

void test_full_h1_faithful_and_homogeneous_cases() {
  {
    const PolynomialRing<PrimeField> ring(GF(5), {"x"}, Order::Grevlex);
    const auto x = ring.gen(0);
    const auto packed = screen_full_h1(Ideal<PrimeField>(ring, {x.pow(5)}));
    CHECK(packed.status == DiscoveryScreenStatus::Complete);
    CHECK(packed.full_socle_rank_candidate);
    CHECK(packed.certified_faithful);
    CHECK(packed.witness.has_value());
  }
  {
    const PolynomialRing<PrimeField> ring(
        GF(101), {"x", "y"}, Order::Grevlex);
    const auto x = ring.gen(0);
    const auto y = ring.gen(1);
    const auto ideal = Ideal<PrimeField>(ring, {x.pow(2), y.pow(2)});
    const auto packed = screen_full_h1(ideal);
    const auto reference = full_h1_action(ideal);
    CHECK(packed.status == DiscoveryScreenStatus::Complete);
    CHECK(packed.h1_dimension == reference.h1_dimension);
    CHECK(packed.socle_dimension == reference.socle_dimension);
    CHECK(packed.maximum_individual_rank_lower_bound == 0);
  }
}

}  // namespace

int main() {
  try {
    test_packed_cycle_screen_and_certification();
    test_cycle_gate_and_resource_semantics();
    test_full_h1_matches_reference_at_two_primes();
    test_full_h1_faithful_and_homogeneous_cases();
    std::cout << "discovery tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "discovery test failure: " << error.what() << '\n';
    return 1;
  }
}

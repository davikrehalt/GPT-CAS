#include <cstddef>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "laughableengine/reconstruction.hpp"

namespace {

using laughableengine::GF;
using laughableengine::Ideal;
using laughableengine::Order;
using laughableengine::PolynomialRing;
using laughableengine::PrimeField;
using laughableengine::QQ;
using laughableengine::RationalField;
using laughableengine::centered_residue;
using laughableengine::compare_two_prime_screen_signatures;
using laughableengine::lift_matching_small_integer_polynomial;
using laughableengine::reduce_mod_prime;
using laughableengine::screen_full_h1;
using laughableengine::screen_signature;
using laughableengine::MatrixSpaceRankProof;

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

void test_modular_reduction_and_two_prime_lift() {
  const PolynomialRing<RationalField> qq(
      QQ(), {"x", "y"}, Order::Grevlex);
  const auto x = qq.gen(0);
  const auto y = qq.gen(1);
  const auto polynomial =
      qq.integer(-7) * x.pow(3) + qq.integer(11) * x * y + qq.integer(2);
  const PolynomialRing<PrimeField> mod101(
      GF(101), {"x", "y"}, Order::Grevlex);
  const PolynomialRing<PrimeField> mod103(
      GF(103), {"x", "y"}, Order::Grevlex);
  const auto first = reduce_mod_prime(polynomial, mod101);
  const auto second = reduce_mod_prime(polynomial, mod103);
  const auto lift = lift_matching_small_integer_polynomial(
      first, second, qq, 20);
  CHECK(lift.agreement);
  CHECK(lift.polynomial.has_value());
  CHECK(*lift.polynomial == polynomial);
  CHECK(centered_residue(
            mod101.field().from_integer(-7), mod101.field()) == -7);

  const auto bounded = lift_matching_small_integer_polynomial(
      first, second, qq, 6);
  CHECK(!bounded.agreement);
  CHECK(bounded.detail.has_value());

  const auto mismatch = lift_matching_small_integer_polynomial(
      first, second + mod103.gen(0), qq, 20);
  CHECK(!mismatch.agreement);
  CHECK(mismatch.detail.has_value());
}

void test_rational_denominator_reduction() {
  const PolynomialRing<RationalField> qq(QQ(), {"x"}, Order::Lex);
  const PolynomialRing<PrimeField> mod5(GF(5), {"x"}, Order::Lex);
  const auto half = qq.constant(qq.field().fraction(1, 2)) * qq.gen(0);
  const auto reduced = reduce_mod_prime(half, mod5);
  CHECK(reduced == mod5.integer(3) * mod5.gen(0));

  const auto fifth = qq.constant(qq.field().fraction(1, 5));
  EXPECT_THROW(std::domain_error, reduce_mod_prime(fifth, mod5));
}

void test_two_prime_screen_agreement_requires_proofs_and_initial_ideal() {
  const PolynomialRing<PrimeField> first_ring(
      GF(101), {"x", "y"}, Order::Grevlex);
  const PolynomialRing<PrimeField> second_ring(
      GF(103), {"x", "y"}, Order::Grevlex);
  const auto first_x = first_ring.gen(0);
  const auto first_y = first_ring.gen(1);
  const auto second_x = second_ring.gen(0);
  const auto second_y = second_ring.gen(1);
  const auto first = screen_full_h1(
      Ideal<PrimeField>(first_ring, {first_x.pow(2), first_y.pow(2)}));
  const auto second = screen_full_h1(
      Ideal<PrimeField>(second_ring, {second_x.pow(2), second_y.pow(2)}));
  const auto first_signature = screen_signature(first);
  const auto second_signature = screen_signature(second);
  CHECK(first_signature.proven());
  CHECK(second_signature.proven());
  CHECK(compare_two_prime_screen_signatures(
            first_signature, second_signature).agreement);

  const auto same_prime = compare_two_prime_screen_signatures(
      first_signature, first_signature);
  CHECK(!same_prime.agreement);
  CHECK(same_prime.detail.has_value());

  auto hash_collision = second_signature;
  hash_collision.leading_ideal_signature.push_back('\0');
  const auto collision_agreement = compare_two_prime_screen_signatures(
      first_signature, hash_collision);
  CHECK(!collision_agreement.agreement);
  CHECK(collision_agreement.detail.has_value());

  const auto changed = screen_full_h1(
      Ideal<PrimeField>(second_ring, {second_x.pow(3), second_y.pow(2)}));
  const auto changed_agreement = compare_two_prime_screen_signatures(
      first_signature, screen_signature(changed));
  CHECK(!changed_agreement.agreement);
  CHECK(changed_agreement.detail.has_value());

  auto inconclusive = first_signature;
  inconclusive.rank_proof = MatrixSpaceRankProof::ResourceLimit;
  const auto inconclusive_agreement = compare_two_prime_screen_signatures(
      inconclusive, second_signature);
  CHECK(!inconclusive_agreement.agreement);
  CHECK(inconclusive_agreement.detail.has_value());
}

}  // namespace

int main() {
  try {
    test_modular_reduction_and_two_prime_lift();
    test_rational_denominator_reduction();
    test_two_prime_screen_agreement_requires_proofs_and_initial_ideal();
    std::cout << "reconstruction tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "reconstruction test failure: " << error.what() << '\n';
    return 1;
  }
}

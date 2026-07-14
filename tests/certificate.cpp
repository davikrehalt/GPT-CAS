#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/certificate.hpp"

namespace {

using laughableengine::GF;
using laughableengine::Order;
using laughableengine::Polynomial;
using laughableengine::PolynomialRing;
using laughableengine::PrimeField;
using laughableengine::QQ;
using laughableengine::RationalField;
using laughableengine::make_jg_certificate_json;

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

void test_gf_fixture_serialization() {
  const PolynomialRing<PrimeField> ring(GF(5), {"x"}, Order::Grevlex);
  const auto generator = ring.gen(0).pow(5);
  const std::vector generators{generator};
  const auto json = make_jg_certificate_json(ring, generators, generator);
  CHECK(json ==
        "{\"schema\":\"laughable-jg-v1\",\"field\":{\"kind\":\"GF\","
        "\"modulus\":\"5\"},\"variables\":[\"x\"],\"order\":\"grevlex\","
        "\"ideal_generators\":[[{\"coefficient\":\"1\",\"exponents\":[5]}]],"
        "\"g\":[{\"coefficient\":\"1\",\"exponents\":[5]}]}");
}

void test_qq_coefficients_and_raw_generators() {
  const PolynomialRing<RationalField> ring(QQ(), {"x", "y"}, Order::Lex);
  const auto x = ring.gen(0);
  const auto y = ring.gen(1);
  const auto half = ring.constant(ring.field().fraction(1, 2));
  const std::vector generators{x * x + half * y, y * y};
  const auto json = make_jg_certificate_json(
      ring, generators, generators.front());
  CHECK(json.find("\"kind\":\"QQ\"") != std::string::npos);
  CHECK(json.find(
            "\"coefficient\":{\"numerator\":\"1\",\"denominator\":\"2\"}") !=
        std::string::npos);
  CHECK(json.find("\"ideal_generators\":[[") != std::string::npos);

  const PolynomialRing<RationalField> foreign(QQ(), {"x", "y"}, Order::Lex);
  const std::vector foreign_generators{foreign.gen(0)};
  EXPECT_THROW(
      std::invalid_argument,
      make_jg_certificate_json(ring, foreign_generators, x));
  const std::vector<Polynomial<RationalField>> empty;
  EXPECT_THROW(
      std::invalid_argument,
      make_jg_certificate_json(ring, empty, x));
}

}  // namespace

int main() {
  try {
    test_gf_fixture_serialization();
    test_qq_coefficients_and_raw_generators();
    std::cout << "certificate tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "certificate test failure: " << error.what() << '\n';
    return 1;
  }
}

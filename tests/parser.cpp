#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "laughableengine/laughableengine.hpp"

namespace {

using laughableengine::GF;
using laughableengine::Order;
using laughableengine::ParseError;
using laughableengine::QQ;
using laughableengine::Rational;
using laughableengine::make_ring;
using laughableengine::parse_polynomial;

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
void expect_parse_error(
    Function&& function,
    std::size_t expected_offset,
    std::string_view expected_message,
    int line) {
  try {
    std::invoke(std::forward<Function>(function));
  } catch (const ParseError& error) {
    if (error.offset() != expected_offset) {
      fail("expected parse offset " + std::to_string(expected_offset) +
               ", got " + std::to_string(error.offset()),
           line);
    }
    if (std::string_view(error.what()).find(expected_message) ==
        std::string_view::npos) {
      fail("parse message did not contain '" + std::string(expected_message) +
               "': " + error.what(),
           line);
    }
    return;
  } catch (const std::exception& error) {
    fail("unexpected exception type: " + std::string(error.what()), line);
  }
  fail("expected ParseError was not thrown", line);
}

#define EXPECT_PARSE_ERROR(expression, offset, message)                       \
  expect_parse_error([&] { static_cast<void>(expression); }, offset, message, \
                     __LINE__)

void test_precedence_and_arithmetic() {
  const auto ring = make_ring(QQ(), {"x", "y"}, Order::Lex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");

  CHECK(parse_polynomial(ring, "-x^2") == -x.pow(2));
  CHECK(parse_polynomial(ring, "(-x)^2") == x.pow(2));
  CHECK(parse_polynomial(ring, "-2^2") == ring.integer(-4));
  CHECK(parse_polynomial(ring, "--x") == x);
  CHECK(parse_polynomial(ring, " (x + y) ** 3 ") == (x + y).pow(3));
  CHECK(parse_polynomial(ring, "-x^2 + 2*x*y - (y + 1)^2") ==
        -x.pow(2) + ring.integer(2) * x * y -
            (y + ring.one()).pow(2));
}

void test_scalar_division() {
  const auto ring = make_ring(QQ(), {"x", "y"});
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto half = ring.constant(ring.field().fraction(1, 2));

  CHECK(parse_polynomial(ring, "(x + y) / (1 + 1)") == half * (x + y));
  CHECK(parse_polynomial(ring, "x / -2") == -half * x);
  CHECK(parse_polynomial(ring, "x / (y - y + 2)") == half * x);
  CHECK(parse_polynomial(ring, "1 / 2 / 3 * x") ==
        ring.constant(ring.field().fraction(1, 6)) * x);

  EXPECT_PARSE_ERROR(parse_polynomial(ring, "x / y"), 2,
                     "polynomial denominator");
  EXPECT_PARSE_ERROR(parse_polynomial(ring, "x / (y + 1)"), 2,
                     "polynomial denominator");
  EXPECT_PARSE_ERROR(parse_polynomial(ring, "1/0"), 1, "division by zero");
}

void test_arbitrary_precision_literals() {
  const auto rational_ring = make_ring(QQ(), {"x"});
  const auto x = rational_ring.gen("x");
  const std::string huge = "1234567890123456789012345678901234567890";
  CHECK(parse_polynomial(rational_ring, huge + "*x/37") ==
        rational_ring.constant(Rational::from_fraction(huge, "37")) * x);

  const auto finite_ring = make_ring(GF(101), {"x"});
  const auto finite_x = finite_ring.gen("x");
  std::string huge_modular = "101";
  huge_modular.append(40, '0');
  huge_modular += '3';
  CHECK(parse_polynomial(finite_ring, huge_modular + "*x") ==
        finite_ring.integer(3) * finite_x);
  CHECK(parse_polynomial(finite_ring, "1/2*x") ==
        finite_ring.integer(51) * finite_x);

  const auto characteristic_five = make_ring(GF(5), {"x"});
  EXPECT_PARSE_ERROR(parse_polynomial(characteristic_five, "x/5"), 1,
                     "division by zero");
}

void test_printer_round_trip() {
  const auto rational_ring = make_ring(QQ(), {"x", "y", "z"});
  const auto x = rational_ring.gen("x");
  const auto y = rational_ring.gen("y");
  const auto z = rational_ring.gen("z");
  const auto rational =
      x.pow(4) -
      rational_ring.constant(rational_ring.field().fraction(7, 11)) * x * y +
      y.pow(2) * z - rational_ring.integer(19);
  CHECK(parse_polynomial(rational_ring, rational.to_string()) == rational);

  const auto finite_ring = make_ring(GF(101), {"x", "y"}, Order::Lex);
  const auto finite_x = finite_ring.gen("x");
  const auto finite_y = finite_ring.gen("y");
  const auto finite = finite_ring.integer(100) * finite_x.pow(3) +
                      finite_ring.integer(51) * finite_x * finite_y +
                      finite_y.pow(9) + finite_ring.integer(7);
  CHECK(parse_polynomial(finite_ring, finite.to_string()) == finite);
  CHECK(parse_polynomial(finite_ring, "0") == finite_ring.zero());
}

void test_diagnostics() {
  const auto ring = make_ring(QQ(), {"x", "y"});

  EXPECT_PARSE_ERROR(parse_polynomial(ring, ""), 0,
                     "expected a polynomial expression");
  EXPECT_PARSE_ERROR(parse_polynomial(ring, "x + mystery"), 4,
                     "unknown polynomial variable");
  EXPECT_PARSE_ERROR(parse_polynomial(ring, "2x"), 1,
                     "implicit multiplication");
  EXPECT_PARSE_ERROR(parse_polynomial(ring, "2 x"), 2,
                     "implicit multiplication");
  EXPECT_PARSE_ERROR(parse_polynomial(ring, "x(y)"), 1,
                     "implicit multiplication");
  EXPECT_PARSE_ERROR(parse_polynomial(ring, "x^"), 2,
                     "nonnegative integer exponent");
  EXPECT_PARSE_ERROR(parse_polynomial(ring, "x^-2"), 2,
                     "nonnegative integer exponent");
  EXPECT_PARSE_ERROR(parse_polynomial(ring, "x^(2)"), 2,
                     "nonnegative integer exponent");
  EXPECT_PARSE_ERROR(parse_polynomial(ring, "x^2^3"), 3,
                     "chained exponentiation");
  EXPECT_PARSE_ERROR(parse_polynomial(ring, "x**2**3"), 4,
                     "chained exponentiation");
  EXPECT_PARSE_ERROR(parse_polynomial(ring, "x^65536"), 2,
                     "limit of 65535");
  EXPECT_PARSE_ERROR(
      parse_polynomial(ring, "x^184467440737095516160000"), 2,
      "limit of 65535");
  EXPECT_PARSE_ERROR(parse_polynomial(ring, "(x+y"), 4, "expected ')'");
  EXPECT_PARSE_ERROR(parse_polynomial(ring, "x @ y"), 2,
                     "unexpected token");
  EXPECT_PARSE_ERROR(parse_polynomial(ring, "x)"), 1, "unexpected token");

  const std::string deep_parentheses =
      std::string(257, '(') + "x" + std::string(257, ')');
  EXPECT_PARSE_ERROR(parse_polynomial(ring, deep_parentheses), 256,
                     "nesting exceeds");
  EXPECT_PARSE_ERROR(parse_polynomial(ring, std::string(257, '-') + "x"),
                     256, "nesting exceeds");
}

}  // namespace

int main() {
  try {
    test_precedence_and_arithmetic();
    test_scalar_division();
    test_arbitrary_precision_literals();
    test_printer_round_trip();
    test_diagnostics();
    std::cout << "laughableengine: parser tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "laughableengine parser test failure: " << error.what() << '\n';
    return 1;
  }
}

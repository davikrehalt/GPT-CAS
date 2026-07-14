#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "laughableengine/laughableengine.hpp"

namespace {

using laughableengine::GF;
using laughableengine::Order;
using laughableengine::QQ;
using laughableengine::Rational;
using laughableengine::make_ring;

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

void test_prime_field() {
  for (const std::uint32_t composite : {0U, 1U, 4U, 9U, 561U}) {
    EXPECT_THROW(std::invalid_argument, GF(composite));
  }
  static_cast<void>(GF(2));
  static_cast<void>(GF(101));
  static_cast<void>(GF(2'147'483'647U));

  const auto field = GF(7);
  CHECK(field.from_integer(-1).value() == 6);
  CHECK(field.from_integer(std::numeric_limits<std::int64_t>::min()).value() ==
        6);
  CHECK(field.add(field.from_integer(6), field.from_integer(3)).value() == 2);
  CHECK(field.subtract(field.from_integer(1), field.from_integer(3)).value() ==
        5);
  CHECK(field.multiply(field.from_integer(3), field.from_integer(5)).value() ==
        1);
  CHECK(field.inverse(field.from_integer(3)).value() == 5);
  CHECK(field.divide(field.from_integer(2), field.from_integer(3)).value() == 3);
  EXPECT_THROW(std::domain_error, field.inverse(field.zero()));

  const auto larger_field = GF(101);
  for (std::uint32_t value = 1; value < larger_field.modulus(); ++value) {
    const auto element = larger_field.from_unsigned(value);
    CHECK(larger_field.multiply(element, larger_field.inverse(element)) ==
          larger_field.one());
  }

  const auto mersenne = GF(2'147'483'647U);
  const auto near_modulus = mersenne.from_unsigned(2'147'483'646U);
  CHECK(mersenne.multiply(near_modulus, near_modulus) == mersenne.one());

  const auto foreign_element = GF(5).one();
  EXPECT_THROW(std::invalid_argument, field.add(field.one(), foreign_element));

  const auto gf7_ring = make_ring(field, {"x"});
  EXPECT_THROW(std::invalid_argument, gf7_ring.constant(foreign_element));
}

void test_rationals() {
  const auto field = QQ();
  const auto half = field.fraction(1, 2);
  const auto third = field.fraction(1, 3);
  CHECK(field.add(half, third).to_string() == "5/6");
  CHECK(field.inverse(half).to_string() == "2");
  CHECK(field.fraction(2, -4).to_string() == "-1/2");
  CHECK(field.fraction(-2, -4).to_string() == "1/2");
  CHECK(Rational::from_fraction(
            "123456789012345678901234567890",
            "987654321098765432109876543210")
            .to_string() == "13717421/109739369");
  EXPECT_THROW(std::domain_error, field.fraction(1, 0));
  EXPECT_THROW(std::domain_error, field.inverse(field.zero()));
}

void test_ring_validation() {
  EXPECT_THROW(std::invalid_argument,
               make_ring(GF(5), std::initializer_list<std::string>{}));
  EXPECT_THROW(std::invalid_argument, make_ring(GF(5), {"x", "x"}));
  EXPECT_THROW(std::invalid_argument, make_ring(GF(5), {"1x"}));

  const auto ten_variable_ring = make_ring(
      GF(5),
      {"x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"});
  CHECK(ten_variable_ring.variable_count() == 10);
  const auto x1 = ten_variable_ring.gen("x1");
  const auto x10 = ten_variable_ring.gen("x10");
  CHECK((x1 * x10 + x10.pow(2)).derivative("x10") ==
        x1 + ten_variable_ring.integer(2) * x10);

  EXPECT_THROW(std::invalid_argument,
               make_ring(
                   GF(5),
                   {"x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
                    "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17"}));
}

void test_finite_field_polynomials() {
  const auto ring = make_ring(GF(5), {"x", "y"});
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");

  const auto f = x.pow(2) + ring.integer(3) * x * y + ring.integer(2);
  CHECK(f.to_string() == "x^2 + 3*x*y + 2");
  CHECK(f.total_degree() == 2);
  CHECK(f.term_count() == 3);
  CHECK(f.derivative("x").to_string() == "2*x + 3*y");
  CHECK(f.derivative("y").to_string() == "3*x");
  CHECK((f - f).is_zero());
  CHECK((x * (y + ring.one())) == (x * y + x));
  CHECK((x * y).derivative("x") ==
        x.derivative("x") * y + x * y.derivative("x"));
  CHECK(x.pow(0) == ring.one());
  CHECK(ring.zero().pow(0) == ring.one());
  CHECK(x.pow(5).derivative("x").is_zero());

  const auto normalized = ring.from_terms({
      {ring.field().from_integer(2), {1, 0}},
      {ring.field().from_integer(4), {0, 1}},
      {ring.field().from_integer(3), {1, 0}},
      {ring.field().from_integer(1), {0, 1}},
  });
  CHECK(normalized.is_zero());

  const auto other_ring = make_ring(GF(5), {"x", "y"});
  EXPECT_THROW(std::invalid_argument, x + other_ring.gen(0));

  const auto gf2 = make_ring(GF(2), {"x", "y"});
  const auto a = gf2.gen(0);
  const auto b = gf2.gen(1);
  CHECK((a + b).pow(2) == a.pow(2) + b.pow(2));
  CHECK(a.pow(2).derivative(0).is_zero());

  const auto maximum_power =
      ring.monomial(ring.field().one(), {std::numeric_limits<std::uint16_t>::max(),
                                        0});
  EXPECT_THROW(std::overflow_error, maximum_power * x);
  EXPECT_THROW(std::invalid_argument,
               ring.monomial(ring.field().one(), {1, 2, 3}));

  const auto surviving_polynomial = [] {
    const auto short_lived_ring = make_ring(GF(11), {"t"});
    return short_lived_ring.gen(0).pow(3) + short_lived_ring.one();
  }();
  CHECK(surviving_polynomial.to_string() == "t^3 + 1");
}

void test_orders_and_rational_polynomials() {
  const auto grevlex = make_ring(QQ(), {"x", "y", "z"}, Order::Grevlex);
  const auto x = grevlex.gen(0);
  const auto y = grevlex.gen(1);
  const auto z = grevlex.gen(2);
  const auto degree_two =
      x.pow(2) + x * y + y.pow(2) + x * z + y * z + z.pow(2);
  CHECK(degree_two.to_string() ==
        "x^2 + x*y + y^2 + x*z + y*z + z^2");

  const auto half = grevlex.constant(grevlex.field().fraction(1, 2));
  const auto polynomial = x.pow(3) + y.pow(2) + half * x * y;
  CHECK(polynomial.to_string() == "x^3 + 1/2*x*y + y^2");
  CHECK(polynomial.derivative("x").to_string() == "3*x^2 + 1/2*y");
  CHECK(polynomial.derivative("y").to_string() == "1/2*x + 2*y");

  const auto lex = make_ring(QQ(), {"x", "y"}, Order::Lex);
  const auto lx = lex.gen(0);
  const auto ly = lex.gen(1);
  CHECK((lx + ly.pow(100)).to_string() == "x + y^100");
  CHECK((lx + ly.pow(100)).total_degree() == 100);
}

void test_cotangent_shaped_computations() {
  const auto rational_ring = make_ring(QQ(), {"x", "y", "z"});
  const auto x = rational_ring.gen("x");
  const auto y = rational_ring.gen("y");
  const auto z = rational_ring.gen("z");

  const auto seed = y.pow(5) + z.pow(5) + y.pow(2) * z.pow(2);
  CHECK(seed.to_string() == "y^5 + z^5 + y^2*z^2");
  CHECK(seed.derivative("x").is_zero());
  CHECK(seed.derivative("y").to_string() == "5*y^4 + 2*y*z^2");
  CHECK(seed.derivative("z").to_string() == "5*z^4 + 2*y^2*z");
  CHECK((x.pow(2) + x * y + x * z).derivative("x").to_string() ==
        "2*x + y + z");

  const auto finite_ring = make_ring(GF(101), {"x", "y"});
  const auto fx = finite_ring.gen("x");
  const auto fy = finite_ring.gen("y");
  const auto potential = fx.pow(3) + fy.pow(7) + fx * fy.pow(5);
  CHECK(potential.to_string() == "y^7 + x*y^5 + x^3");
  CHECK(potential.derivative("x").to_string() == "y^5 + 3*x^2");
  CHECK(potential.derivative("y").to_string() == "7*y^6 + 5*x*y^4");

  const auto characteristic_five = make_ring(GF(5), {"x", "y"});
  const auto cx = characteristic_five.gen("x");
  const auto cy = characteristic_five.gen("y");
  CHECK(cx.pow(5).derivative("x").is_zero());
  CHECK((cx + cy).pow(5) == cx.pow(5) + cy.pow(5));
}

}  // namespace

int main() {
  try {
    test_prime_field();
    test_rationals();
    test_ring_validation();
    test_finite_field_polynomials();
    test_orders_and_rational_polynomials();
    test_cotangent_shaped_computations();
    std::cout << "laughableengine: all tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "laughableengine test failure: " << error.what() << '\n';
    return 1;
  }
}

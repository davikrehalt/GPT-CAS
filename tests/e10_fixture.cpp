#include <iostream>
#include <stdexcept>
#include <string>

#include "laughableengine/field.hpp"
#include "laughableengine/polynomial.hpp"
#include "support/e10_fixture.hpp"

namespace {

using laughableengine::GF;
using laughableengine::QQ;
using laughableengine::make_ring;
using laughableengine::tests::support::e10_variable_count;
using laughableengine::tests::support::make_e10_fixture;

[[noreturn]] void fail(const std::string& message, int line) {
  throw std::runtime_error("line " + std::to_string(line) + ": " + message);
}

#define CHECK(expression)                                                     \
  do {                                                                        \
    if (!(expression)) {                                                      \
      fail("CHECK failed: " #expression, __LINE__);                          \
    }                                                                         \
  } while (false)

template <typename Field>
void check_fixture(const laughableengine::PolynomialRing<Field>& ring) {
  const auto fixture = make_e10_fixture(ring);

  CHECK(fixture.potential.term_count() == 220);
  CHECK(fixture.derivatives.size() == e10_variable_count);
  for (std::size_t variable = 0; variable < e10_variable_count; ++variable) {
    CHECK(fixture.derivatives[variable].term_count() == 85);
    CHECK(fixture.derivatives[variable] ==
          fixture.potential.derivative(variable));
  }
}

void test_over_qq_and_gf101() {
  check_fixture(make_ring(
      QQ(),
      {"x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"}));
  check_fixture(make_ring(
      GF(101),
      {"x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"}));
}

}  // namespace

int main() {
  try {
    test_over_qq_and_gf101();
    std::cout << "laughableengine E10 fixture tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "laughableengine E10 fixture test failure: " << error.what()
              << '\n';
    return 1;
  }
}

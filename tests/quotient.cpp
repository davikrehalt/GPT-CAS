#include <functional>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/laughableengine.hpp"

namespace {

using laughableengine::BasisValidation;
using laughableengine::CompiledReducer;
using laughableengine::GF;
using laughableengine::InfiniteQuotientError;
using laughableengine::Order;
using laughableengine::Polynomial;
using laughableengine::QQ;
using laughableengine::RationalField;
using laughableengine::ReductionLimits;
using laughableengine::ResourceLimitExceeded;
using laughableengine::ResourceLimitKind;
using laughableengine::StandardMonomialLimits;
using laughableengine::groebner_basis;
using laughableengine::make_ring;
using laughableengine::normal_form;

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

template <typename Function>
void expect_resource_limit(
    Function&& function,
    ResourceLimitKind kind,
    std::optional<std::size_t> batch_index,
    int line) {
  try {
    std::invoke(std::forward<Function>(function));
  } catch (const ResourceLimitExceeded& error) {
    if (error.kind() != kind || error.batch_index() != batch_index ||
        error.observed() <= error.limit()) {
      fail("resource-limit metadata was incorrect", line);
    }
    return;
  } catch (const std::exception& error) {
    fail("unexpected exception type: " + std::string(error.what()), line);
  }
  fail("expected a resource limit to be exceeded", line);
}

#define EXPECT_RESOURCE(expression, kind, batch_index)                        \
  expect_resource_limit([&] { static_cast<void>(expression); }, kind,         \
                        batch_index, __LINE__)

template <typename Field>
std::vector<Polynomial<Field>> ideal_square_generators(
    const std::vector<Polynomial<Field>>& generators) {
  std::vector<Polynomial<Field>> result;
  result.reserve(generators.size() * (generators.size() + 1) / 2);
  for (std::size_t right = 0; right < generators.size(); ++right) {
    for (std::size_t left = 0; left <= right; ++left) {
      result.push_back(generators[left] * generators[right]);
    }
  }
  return result;
}

void test_compiled_ordered_reduction_and_limits() {
  const auto ring = make_ring(QQ(), {"x", "y"}, Order::Lex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto f = x.pow(2) * y + x * y.pow(2) + y.pow(2);
  const auto g1 = x * y - ring.one();
  const auto g2 = y.pow(2) - ring.one();

  const auto reducer = CompiledReducer<RationalField>::ordered(ring, {g1, g2});
  CHECK(!reducer.is_groebner_mode());
  CHECK(reducer.normal_form(f) == x + y + ring.one());

  const auto reversed =
      CompiledReducer<RationalField>::ordered(ring, {g2, g1});
  CHECK(reversed.normal_form(f) == ring.integer(2) * x + ring.one());

  const std::vector<Polynomial<RationalField>> batch{
      f, x * g1 + ring.one(), ring.zero()};
  const auto reduced = reducer.normal_forms(batch);
  CHECK((reduced == std::vector<Polynomial<RationalField>>{
                        x + y + ring.one(), ring.one(), ring.zero()}));

  const auto empty = CompiledReducer<RationalField>::ordered(
      ring, std::vector<Polynomial<RationalField>>{});
  CHECK(empty.normal_form(f) == f);
  CHECK(empty.normal_forms(batch) == batch);
  EXPECT_THROW(std::logic_error, empty.is_zero_dimensional());
  EXPECT_THROW(std::invalid_argument,
               CompiledReducer<RationalField>::ordered(ring, {ring.zero()}));

  const auto foreign = make_ring(QQ(), {"x", "y"}, Order::Lex);
  EXPECT_THROW(std::invalid_argument,
               CompiledReducer<RationalField>::ordered(
                   ring, {foreign.gen("x")}));
  EXPECT_THROW(std::invalid_argument, reducer.normal_form(foreign.gen("x")));

  ReductionLimits no_steps;
  no_steps.max_steps = 0;
  EXPECT_RESOURCE(reducer.normal_form(f, no_steps),
                  ResourceLimitKind::ReductionSteps, std::nullopt);

  ReductionLimits two_terms;
  two_terms.max_live_terms = 2;
  EXPECT_RESOURCE(reducer.normal_form(f, two_terms),
                  ResourceLimitKind::LiveTerms, std::nullopt);

  ReductionLimits no_batch_steps;
  no_batch_steps.max_batch_steps = 0;
  EXPECT_RESOURCE(reducer.normal_forms(
                      std::vector<Polynomial<RationalField>>{ring.zero(), f},
                      no_batch_steps),
                  ResourceLimitKind::BatchReductionSteps,
                  std::optional<std::size_t>(1));
}

void test_monomial_complete_intersection(Order order) {
  const auto ring = make_ring(QQ(), {"x", "y"}, order);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto reducer =
      CompiledReducer<RationalField>::groebner(ring, {x.pow(2), y.pow(3)});

  const auto info = reducer.finite_quotient_info();
  CHECK(info.finite);
  CHECK(!info.unit_ideal);
  CHECK(info.variable_count == 2);
  CHECK(info.pure_power_bounds[0] == std::optional<std::uint16_t>(2));
  CHECK(info.pure_power_bounds[1] == std::optional<std::uint16_t>(3));
  CHECK(reducer.is_zero_dimensional());
  CHECK(reducer.quotient_dimension() == 6);

  const auto basis = reducer.standard_monomials();
  std::vector<Polynomial<RationalField>> expected;
  if (order == Order::Lex) {
    expected = {ring.one(), y, y.pow(2), x, x * y, x * y.pow(2)};
  } else {
    expected = {ring.one(), y, x, y.pow(2), x * y, x * y.pow(2)};
  }
  CHECK(basis.polynomials() == expected);

  const auto polynomial = ring.integer(7) + ring.integer(2) * x +
                          ring.integer(3) * y.pow(2) + x * y.pow(2) +
                          x.pow(4) + y.pow(3);
  const auto normal = reducer.normal_form(polynomial);
  const auto coordinates = basis.coordinates(polynomial);
  CHECK(basis.from_coordinates(coordinates) == normal);

  const auto coordinate_batch = basis.coordinates(
      std::vector<Polynomial<RationalField>>{x.pow(2) + x * y, y.pow(3),
                                             polynomial});
  CHECK(basis.from_coordinates(coordinate_batch[0]) == x * y);
  CHECK(basis.from_coordinates(coordinate_batch[1]).is_zero());
  CHECK(basis.from_coordinates(coordinate_batch[2]) == normal);

  auto short_vector = coordinates;
  short_vector.pop_back();
  EXPECT_THROW(std::invalid_argument, basis.from_coordinates(short_vector));

  StandardMonomialLimits five_only;
  five_only.max_monomials = 5;
  EXPECT_RESOURCE(reducer.standard_monomials(five_only),
                  ResourceLimitKind::StandardMonomials, std::nullopt);
  StandardMonomialLimits no_border;
  no_border.max_border_tests = 0;
  EXPECT_RESOURCE(reducer.standard_monomials(no_border),
                  ResourceLimitKind::BorderTests, std::nullopt);
}

void test_zero_unit_and_invalid_bases() {
  const auto ring = make_ring(QQ(), {"x", "y"}, Order::Lex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");

  const auto zero_ideal = CompiledReducer<RationalField>::groebner(
      ring, std::vector<Polynomial<RationalField>>{});
  CHECK(!zero_ideal.is_zero_dimensional());
  EXPECT_THROW(InfiniteQuotientError, zero_ideal.standard_monomials());
  EXPECT_THROW(InfiniteQuotientError, zero_ideal.quotient_dimension());

  const auto unit_ideal =
      CompiledReducer<RationalField>::groebner(ring, {ring.integer(-7)});
  const auto unit_info = unit_ideal.finite_quotient_info();
  CHECK(unit_info.finite);
  CHECK(unit_info.unit_ideal);
  CHECK(unit_ideal.quotient_dimension() == 0);
  const auto unit_basis = unit_ideal.standard_monomials();
  CHECK(unit_basis.empty());
  CHECK(unit_basis.coordinates(x + y).empty());
  CHECK(unit_basis.from_coordinates(std::vector<RationalField::Element>{}) ==
        ring.zero());

  const std::vector<Polynomial<RationalField>> not_a_basis{
      x.pow(2) - y, x * y - ring.one()};
  EXPECT_THROW(std::invalid_argument,
               CompiledReducer<RationalField>::groebner(ring, not_a_basis));
  const auto explicitly_unchecked = CompiledReducer<RationalField>::groebner(
      ring, not_a_basis, BasisValidation::AssumeVerified);
  CHECK(explicitly_unchecked.is_groebner_mode());

  const auto with_zero = CompiledReducer<RationalField>::groebner(
      ring, {ring.zero(), x.pow(2), y.pow(3)});
  CHECK(with_zero.divisors().size() == 2);
  CHECK(with_zero.quotient_dimension() == 6);
}

void test_nonmonomial_basis_and_batch_coordinates() {
  const auto ring = make_ring(QQ(), {"x", "y"}, Order::Lex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto reducer = CompiledReducer<RationalField>::groebner(
      ring, {x - y.pow(2), y.pow(3) - ring.one()});
  const auto basis = reducer.standard_monomials();
  CHECK((basis.polynomials() ==
         std::vector<Polynomial<RationalField>>{ring.one(), y, y.pow(2)}));

  const auto values =
      std::vector<Polynomial<RationalField>>{x, x.pow(2), x * y};
  const auto normal = reducer.normal_forms(values);
  CHECK((normal == std::vector<Polynomial<RationalField>>{
                       y.pow(2), y, ring.one()}));
  const auto coordinates = basis.coordinates(values);
  for (std::size_t index = 0; index < values.size(); ++index) {
    CHECK(basis.from_coordinates(coordinates[index]) == normal[index]);
  }

  ReductionLimits two_total_steps;
  two_total_steps.max_batch_steps = 2;
  EXPECT_RESOURCE(basis.coordinates(values, two_total_steps),
                  ResourceLimitKind::BatchReductionSteps,
                  std::optional<std::size_t>(1));
}

void test_characteristic_p_basis() {
  const auto ring = make_ring(GF(5), {"x"}, Order::Lex);
  const auto x = ring.gen("x");
  const auto reducer =
      CompiledReducer<laughableengine::PrimeField>::groebner(ring, {x.pow(5)});
  const auto basis = reducer.standard_monomials();
  CHECK(basis.size() == 5);
  CHECK((basis.polynomials() ==
         std::vector{ring.one(), x, x.pow(2), x.pow(3), x.pow(4)}));
  CHECK(reducer.normal_form(x.pow(7)).is_zero());
  CHECK(basis.from_coordinates(basis.coordinates(x.pow(7))).is_zero());
}

void test_randomized_reducer_equivalence(Order order) {
  const auto ring = make_ring(GF(101), {"x", "y"}, order);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const std::vector<Polynomial<laughableengine::PrimeField>> divisors{
      x.pow(2) + y + ring.one(),
      x * y + y.pow(2) + ring.integer(2)};
  const auto reducer =
      CompiledReducer<laughableengine::PrimeField>::ordered(ring, divisors);

  std::mt19937 generator(order == Order::Lex ? 9127U : 442671U);
  std::uniform_int_distribution<int> term_count(0, 9);
  std::uniform_int_distribution<int> coefficient(1, 100);
  std::uniform_int_distribution<int> exponent(0, 6);
  std::vector<Polynomial<laughableengine::PrimeField>> inputs;
  inputs.reserve(250);
  for (std::size_t sample = 0; sample < 250; ++sample) {
    std::vector<typename decltype(ring)::TermSpec> terms;
    const auto count = term_count(generator);
    terms.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
      terms.push_back(typename decltype(ring)::TermSpec{
          ring.field().from_integer(coefficient(generator)),
          {static_cast<std::uint16_t>(exponent(generator)),
           static_cast<std::uint16_t>(exponent(generator))}});
    }
    inputs.push_back(ring.from_terms(std::move(terms)));
  }

  const auto batch = reducer.normal_forms(inputs);
  CHECK(batch.size() == inputs.size());
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    CHECK(batch[index] == reducer.normal_form(inputs[index]));
    CHECK(batch[index] == normal_form(inputs[index], divisors));
  }
}

void test_cotangent_seed_dimensions() {
  const auto ring = make_ring(QQ(), {"x", "y", "z"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto z = ring.gen("z");
  const auto seed = y.pow(5) + z.pow(5) + y.pow(2) * z.pow(2);
  const std::vector<Polynomial<RationalField>> generators{
      x.pow(2), x * y, x * z, seed, seed.derivative("y"),
      seed.derivative("z")};
  const auto basis = groebner_basis(generators);
  const auto reducer = CompiledReducer<RationalField>::groebner(
      ring, basis, BasisValidation::AssumeVerified);
  CHECK(reducer.quotient_dimension() == 11);
  CHECK((reducer.standard_monomials().polynomials() ==
         std::vector<Polynomial<RationalField>>{
             ring.one(), z, y, x, z.pow(2), y * z, y.pow(2), z.pow(3),
             y * z.pow(2), y.pow(2) * z, y.pow(3)}));

  const auto square_basis =
      groebner_basis(ideal_square_generators(basis));
  const auto square_reducer = CompiledReducer<RationalField>::groebner(
      ring, square_basis, BasisValidation::AssumeVerified);
  CHECK(square_reducer.quotient_dimension() == 48);
}

void test_tjurina_seed_dimensions() {
  const auto ring = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto potential = x.pow(3) + y.pow(7) + x * y.pow(5);
  const std::vector<Polynomial<RationalField>> generators{
      potential, potential.derivative("x"), potential.derivative("y")};
  const auto basis = groebner_basis(generators);
  const auto reducer = CompiledReducer<RationalField>::groebner(
      ring, basis, BasisValidation::AssumeVerified);
  CHECK(reducer.quotient_dimension() == 11);
  CHECK((reducer.standard_monomials().polynomials() ==
         std::vector<Polynomial<RationalField>>{
             ring.one(), y, x, y.pow(2), x * y, x.pow(2), y.pow(3),
             x * y.pow(2), x.pow(2) * y, y.pow(4), x * y.pow(3)}));

  const auto square_basis =
      groebner_basis(ideal_square_generators(basis));
  const auto square_reducer = CompiledReducer<RationalField>::groebner(
      ring, square_basis, BasisValidation::AssumeVerified);
  CHECK(square_reducer.quotient_dimension() == 34);
}

}  // namespace

int main() {
  try {
    test_compiled_ordered_reduction_and_limits();
    test_monomial_complete_intersection(Order::Lex);
    test_monomial_complete_intersection(Order::Grevlex);
    test_zero_unit_and_invalid_bases();
    test_nonmonomial_basis_and_batch_coordinates();
    test_characteristic_p_basis();
    test_randomized_reducer_equivalence(Order::Lex);
    test_randomized_reducer_equivalence(Order::Grevlex);
    test_cotangent_seed_dimensions();
    test_tjurina_seed_dimensions();
    std::cout << "laughableengine: compiled quotient tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "laughableengine quotient test failure: " << error.what()
              << '\n';
    return 1;
  }
}

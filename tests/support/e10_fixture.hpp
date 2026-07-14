#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "laughableengine/polynomial.hpp"

namespace laughableengine::tests::support {

inline constexpr std::size_t e10_variable_count = 10;

template <typename Field>
struct E10Fixture {
  Polynomial<Field> potential;
  std::array<Polynomial<Field>, e10_variable_count> derivatives;
};

namespace detail {

template <typename Field, std::size_t... Indices>
[[nodiscard]] auto variables(
    const PolynomialRing<Field>& ring,
    std::index_sequence<Indices...>) {
  return std::array<Polynomial<Field>, sizeof...(Indices)>{
      ring.gen(Indices)...};
}

template <typename Field>
[[nodiscard]] Polynomial<Field> derivative_formula(
    const PolynomialRing<Field>& ring,
    const std::array<Polynomial<Field>, e10_variable_count>& x,
    std::size_t omitted) {
  auto result = ring.integer(3) * x[omitted].pow(2);
  for (std::size_t first = 0; first < e10_variable_count; ++first) {
    if (first == omitted) {
      continue;
    }
    for (std::size_t second = first + 1; second < e10_variable_count;
         ++second) {
      if (second == omitted) {
        continue;
      }
      for (std::size_t third = second + 1; third < e10_variable_count;
           ++third) {
        if (third == omitted) {
          continue;
        }
        result = result + x[first] * x[second] * x[third];
      }
    }
  }
  return result;
}

template <typename Field, std::size_t... Indices>
[[nodiscard]] auto derivative_array(
    const PolynomialRing<Field>& ring,
    const std::array<Polynomial<Field>, e10_variable_count>& x,
    std::index_sequence<Indices...>) {
  return std::array<Polynomial<Field>, sizeof...(Indices)>{
      derivative_formula(ring, x, Indices)...};
}

}  // namespace detail

// Construct
//
//   F = sum_i x_i^3 + e_4(x_1, ..., x_10)
//
// and the ten displayed partials
//
//   g_i = 3 x_i^2 + e_3(x_1, ..., omit x_i, ..., x_10).
//
// This fixture intentionally knows nothing about ideals, quotients, or H1.
template <typename Field>
[[nodiscard]] E10Fixture<Field> make_e10_fixture(
    const PolynomialRing<Field>& ring) {
  if (ring.variable_count() != e10_variable_count) {
    throw std::invalid_argument("the E10 fixture requires exactly 10 variables");
  }
  if (ring.field().is_zero(ring.field().from_integer(2)) ||
      ring.field().is_zero(ring.field().from_integer(3))) {
    throw std::invalid_argument(
        "the E10 fixture requires characteristic different from 2 and 3");
  }

  const auto x = detail::variables(
      ring, std::make_index_sequence<e10_variable_count>{});

  auto potential = ring.zero();
  for (const auto& variable : x) {
    potential = potential + variable.pow(3);
  }
  for (std::size_t first = 0; first < e10_variable_count; ++first) {
    for (std::size_t second = first + 1; second < e10_variable_count;
         ++second) {
      for (std::size_t third = second + 1; third < e10_variable_count;
           ++third) {
        for (std::size_t fourth = third + 1; fourth < e10_variable_count;
             ++fourth) {
          potential =
              potential + x[first] * x[second] * x[third] * x[fourth];
        }
      }
    }
  }

  return E10Fixture<Field>{
      std::move(potential),
      detail::derivative_array(
          ring, x, std::make_index_sequence<e10_variable_count>{})};
}

}  // namespace laughableengine::tests::support

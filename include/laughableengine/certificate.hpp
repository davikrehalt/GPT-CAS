#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "laughableengine/field.hpp"
#include "laughableengine/polynomial.hpp"

namespace laughableengine {

namespace certificate_detail {

inline std::string json_string(std::string_view value) {
  std::string result{"\""};
  for (const auto character : value) {
    switch (character) {
      case '\\':
        result += "\\\\";
        break;
      case '"':
        result += "\\\"";
        break;
      case '\b':
        result += "\\b";
        break;
      case '\f':
        result += "\\f";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(character) < 0x20U) {
          throw std::invalid_argument(
              "certificate JSON does not accept control characters");
        }
        result.push_back(character);
    }
  }
  result.push_back('"');
  return result;
}

template <typename Field>
std::string coefficient_json(
    const Field& field,
    const typename Field::Element& coefficient) {
  if constexpr (std::is_same_v<Field, PrimeField>) {
    return json_string(field.to_string(coefficient));
  } else {
    static_assert(std::is_same_v<Field, RationalField>);
    return std::string{"{\"numerator\":"} +
           json_string(coefficient.numerator_string()) +
           ",\"denominator\":" +
           json_string(coefficient.denominator_string()) + "}";
  }
}

template <typename Field>
std::string polynomial_json(
    const Polynomial<Field>& polynomial,
    const PolynomialRing<Field>& ring) {
  if (!ring.zero().same_ring(polynomial)) {
    throw std::invalid_argument(
        "certificate polynomials require the exact supplied ring");
  }
  std::string result{"["};
  for (std::size_t index = 0; index < polynomial.term_count(); ++index) {
    if (index != 0) {
      result.push_back(',');
    }
    const auto& term = polynomial.terms()[index];
    result += "{\"coefficient\":";
    result += coefficient_json(ring.field(), term.coefficient);
    result += ",\"exponents\":[";
    for (std::size_t variable = 0; variable < ring.variable_count();
         ++variable) {
      if (variable != 0) {
        result.push_back(',');
      }
      result += std::to_string(term.monomial.exponent(variable));
    }
    result += "]}";
  }
  result.push_back(']');
  return result;
}

}  // namespace certificate_detail

// Produce the strict raw-input document consumed by tools/laughable_jg_verify.py.
// It intentionally serializes no Groebner bases, remainders, ranks, or claims.
template <typename Field>
[[nodiscard]] std::string make_jg_certificate_json(
    const PolynomialRing<Field>& ring,
    std::span<const Polynomial<Field>> raw_ideal_generators,
    const Polynomial<Field>& polynomial) {
  if (raw_ideal_generators.empty()) {
    throw std::invalid_argument(
        "a J,g certificate requires at least one raw ideal generator");
  }
  for (const auto& generator : raw_ideal_generators) {
    if (!ring.zero().same_ring(generator)) {
      throw std::invalid_argument(
          "certificate generators require the exact supplied ring");
    }
  }
  if (!ring.zero().same_ring(polynomial)) {
    throw std::invalid_argument(
        "certificate polynomial requires the exact supplied ring");
  }

  std::string result{"{\"schema\":\"laughable-jg-v1\",\"field\":"};
  if constexpr (std::is_same_v<Field, PrimeField>) {
    result += "{\"kind\":\"GF\",\"modulus\":";
    result += certificate_detail::json_string(
        std::to_string(ring.field().modulus()));
    result.push_back('}');
  } else {
    static_assert(std::is_same_v<Field, RationalField>);
    result += "{\"kind\":\"QQ\"}";
  }
  result += ",\"variables\":[";
  for (std::size_t variable = 0; variable < ring.variable_count(); ++variable) {
    if (variable != 0) {
      result.push_back(',');
    }
    result += certificate_detail::json_string(
        ring.variable_names()[variable]);
  }
  result += "],\"order\":";
  result += certificate_detail::json_string(
      ring.order() == Order::Grevlex ? "grevlex" : "lex");
  result += ",\"ideal_generators\":[";
  for (std::size_t index = 0; index < raw_ideal_generators.size(); ++index) {
    if (index != 0) {
      result.push_back(',');
    }
    result += certificate_detail::polynomial_json(
        raw_ideal_generators[index], ring);
  }
  result += "],\"g\":";
  result += certificate_detail::polynomial_json(polynomial, ring);
  result.push_back('}');
  return result;
}

template <typename Field>
[[nodiscard]] std::string make_jg_certificate_json(
    const PolynomialRing<Field>& ring,
    const std::vector<Polynomial<Field>>& raw_ideal_generators,
    const Polynomial<Field>& polynomial) {
  return make_jg_certificate_json(
      ring,
      std::span<const Polynomial<Field>>(
          raw_ideal_generators.data(), raw_ideal_generators.size()),
      polynomial);
}

}  // namespace laughableengine

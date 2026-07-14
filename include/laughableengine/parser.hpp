#pragma once

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "laughableengine/field.hpp"
#include "laughableengine/polynomial.hpp"

namespace laughableengine {

class ParseError final : public std::invalid_argument {
 public:
  ParseError(std::size_t offset, std::string message)
      : std::invalid_argument(format_message(offset, message)), offset_(offset) {}

  // Zero-based byte offset into the input expression.
  [[nodiscard]] std::size_t offset() const noexcept { return offset_; }

 private:
  [[nodiscard]] static std::string format_message(
      std::size_t offset,
      const std::string& message) {
    return "parse error at byte " + std::to_string(offset) + ": " + message;
  }

  std::size_t offset_;
};

namespace parser_detail {

inline PrimeField::Element decimal_element(
    const PrimeField& field,
    std::string_view digits) noexcept {
  std::uint64_t remainder = 0;
  for (const char character : digits) {
    remainder =
        (remainder * 10U + static_cast<unsigned>(character - '0')) %
        field.modulus();
  }
  return field.from_unsigned(remainder);
}

inline RationalField::Element decimal_element(
    const RationalField&,
    std::string_view digits) {
  return Rational::from_fraction(digits, "1");
}

template <typename Field>
[[nodiscard]] std::optional<typename Field::Element> scalar_value(
    const Polynomial<Field>& polynomial) {
  if (polynomial.is_zero()) {
    return polynomial.coefficient_field().zero();
  }
  if (polynomial.term_count() != 1 ||
      polynomial.leading_term()->monomial.total_degree() != 0) {
    return std::nullopt;
  }
  return polynomial.leading_term()->coefficient;
}

template <typename Field>
class PolynomialParser {
 public:
  using Ring = PolynomialRing<Field>;
  using PolynomialType = Polynomial<Field>;

  PolynomialParser(const Ring& ring, std::string_view input)
      : ring_(ring), input_(input) {}

  [[nodiscard]] PolynomialType parse() {
    auto result = parse_sum();
    skip_whitespace();
    if (!at_end()) {
      if (starts_primary()) {
        fail(position_, "implicit multiplication is not supported; use '*'");
      }
      fail(position_, "unexpected token " + describe_current());
    }
    return result;
  }

 private:
  class RecursionGuard {
   public:
    RecursionGuard(std::size_t& depth, std::size_t offset) : depth_(depth) {
      if (depth_ >= maximum_recursion_depth) {
        fail(offset,
             "expression nesting exceeds laughableengine's limit of " +
                 std::to_string(maximum_recursion_depth));
      }
      ++depth_;
    }

    RecursionGuard(const RecursionGuard&) = delete;
    RecursionGuard& operator=(const RecursionGuard&) = delete;

    ~RecursionGuard() { --depth_; }

   private:
    std::size_t& depth_;
  };

  static constexpr std::size_t maximum_recursion_depth = 256;

  [[noreturn]] static void fail(std::size_t offset, std::string message) {
    throw ParseError(offset, std::move(message));
  }

  [[nodiscard]] bool at_end() const noexcept {
    return position_ == input_.size();
  }

  void skip_whitespace() noexcept {
    while (!at_end() &&
           std::isspace(static_cast<unsigned char>(input_[position_])) != 0) {
      ++position_;
    }
  }

  [[nodiscard]] bool starts_with(std::string_view token) const noexcept {
    return input_.substr(position_, token.size()) == token;
  }

  [[nodiscard]] bool consume(char character) noexcept {
    skip_whitespace();
    if (!at_end() && input_[position_] == character) {
      ++position_;
      return true;
    }
    return false;
  }

  [[nodiscard]] bool starts_primary() const noexcept {
    if (at_end()) {
      return false;
    }
    const auto character = static_cast<unsigned char>(input_[position_]);
    return std::isdigit(character) != 0 || std::isalpha(character) != 0 ||
           input_[position_] == '_' || input_[position_] == '(';
  }

  [[nodiscard]] std::string describe_current() const {
    if (at_end()) {
      return "end of input";
    }
    return "'" + std::string(1, input_[position_]) + "'";
  }

  [[nodiscard]] PolynomialType parse_sum() {
    auto result = parse_product();
    while (true) {
      if (consume('+')) {
        result = result + parse_product();
      } else if (consume('-')) {
        result = result - parse_product();
      } else {
        return result;
      }
    }
  }

  [[nodiscard]] PolynomialType parse_product() {
    auto result = parse_unary();
    while (true) {
      skip_whitespace();
      if (starts_with("**")) {
        // A valid power is consumed by parse_power.  Reaching one here means
        // that it follows an already completed unary expression.
        fail(position_, "unexpected exponentiation operator");
      }
      if (!at_end() && input_[position_] == '*') {
        const auto operator_offset = position_++;
        auto right = parse_unary();
        try {
          result = result * right;
        } catch (const std::overflow_error& error) {
          fail(operator_offset, error.what());
        }
        continue;
      }
      if (!at_end() && input_[position_] == '/') {
        const auto operator_offset = position_++;
        auto denominator = parse_unary();
        const auto scalar = scalar_value(denominator);
        if (!scalar.has_value()) {
          fail(operator_offset,
               "a polynomial denominator is not supported; divide only by a "
               "scalar");
        }
        if (ring_.field().is_zero(*scalar)) {
          fail(operator_offset, "division by zero");
        }
        result = result.scaled(ring_.field().inverse(*scalar));
        continue;
      }
      return result;
    }
  }

  [[nodiscard]] PolynomialType parse_unary() {
    skip_whitespace();
    const auto operator_offset = position_;
    if (consume('+')) {
      RecursionGuard guard(recursion_depth_, operator_offset);
      return parse_unary();
    }
    if (consume('-')) {
      RecursionGuard guard(recursion_depth_, operator_offset);
      return -parse_unary();
    }
    return parse_power();
  }

  [[nodiscard]] PolynomialType parse_power() {
    auto base = parse_primary();
    skip_whitespace();

    std::size_t operator_offset = position_;
    bool has_power = false;
    if (starts_with("**")) {
      position_ += 2;
      has_power = true;
    } else if (!at_end() && input_[position_] == '^') {
      ++position_;
      has_power = true;
    }
    if (!has_power) {
      return base;
    }

    skip_whitespace();
    const auto exponent_offset = position_;
    if (at_end() ||
        std::isdigit(static_cast<unsigned char>(input_[position_])) == 0) {
      fail(exponent_offset, "expected a nonnegative integer exponent");
    }

    constexpr std::uint64_t maximum_exponent =
        std::numeric_limits<std::uint16_t>::max();
    std::uint64_t exponent = 0;
    while (!at_end() &&
           std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
      const auto digit = static_cast<unsigned>(input_[position_] - '0');
      if (exponent > (maximum_exponent - digit) / 10U) {
        fail(exponent_offset,
             "exponent exceeds laughableengine's limit of 65535");
      }
      exponent = exponent * 10U + digit;
      ++position_;
    }

    skip_whitespace();
    if ((!at_end() && input_[position_] == '^') || starts_with("**")) {
      fail(position_, "chained exponentiation is not supported");
    }

    try {
      return base.pow(exponent);
    } catch (const std::overflow_error& error) {
      fail(operator_offset, error.what());
    }
  }

  [[nodiscard]] PolynomialType parse_primary() {
    skip_whitespace();
    if (at_end()) {
      fail(position_, "expected a polynomial expression");
    }

    const auto character = static_cast<unsigned char>(input_[position_]);
    if (std::isdigit(character) != 0) {
      return parse_decimal();
    }
    if (std::isalpha(character) != 0 || input_[position_] == '_') {
      return parse_variable();
    }
    if (input_[position_] == '(') {
      const auto opening_offset = position_++;
      RecursionGuard guard(recursion_depth_, opening_offset);
      auto result = parse_sum();
      skip_whitespace();
      if (at_end()) {
        fail(position_, "expected ')' to match '(' at byte " +
                            std::to_string(opening_offset));
      }
      if (input_[position_] != ')') {
        if (starts_primary()) {
          fail(position_, "implicit multiplication is not supported; use '*'");
        }
        fail(position_, "expected ')'");
      }
      ++position_;
      return result;
    }

    fail(position_, "expected a number, variable, or '('");
  }

  [[nodiscard]] PolynomialType parse_decimal() {
    const auto start = position_;
    while (!at_end() &&
           std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
      ++position_;
    }
    return ring_.constant(
        decimal_element(ring_.field(), input_.substr(start, position_ - start)));
  }

  [[nodiscard]] PolynomialType parse_variable() {
    const auto start = position_;
    ++position_;
    while (!at_end()) {
      const auto character = static_cast<unsigned char>(input_[position_]);
      if (std::isalnum(character) == 0 && input_[position_] != '_') {
        break;
      }
      ++position_;
    }

    const auto name = input_.substr(start, position_ - start);
    try {
      return ring_.gen(name);
    } catch (const std::invalid_argument&) {
      fail(start, "unknown polynomial variable '" + std::string(name) + "'");
    }
  }

  const Ring& ring_;
  std::string_view input_;
  std::size_t position_ = 0;
  std::size_t recursion_depth_ = 0;
};

}  // namespace parser_detail

template <typename Field>
[[nodiscard]] Polynomial<Field> parse_polynomial(
    const PolynomialRing<Field>& ring,
    std::string_view expression) {
  return parser_detail::PolynomialParser<Field>(ring, expression).parse();
}

}  // namespace laughableengine

#pragma once

#include <cstdint>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <gmpxx.h>

namespace laughableengine {

class PrimeField {
 public:
  class Element {
   public:
    [[nodiscard]] std::uint32_t value() const noexcept { return value_; }
    friend bool operator==(const Element&, const Element&) = default;

   private:
    Element(std::uint32_t value, std::uint32_t modulus)
        : value_(value), modulus_(modulus) {}

    std::uint32_t value_ = 0;
    std::uint32_t modulus_ = 0;
    friend class PrimeField;
  };

  static constexpr std::uint32_t maximum_modulus = 2'147'483'647U;

  explicit PrimeField(std::uint32_t modulus) : modulus_(modulus) {
    if (modulus_ > maximum_modulus) {
      throw std::invalid_argument(
          "laughableengine's first GF(p) backend is limited to p < 2^31");
    }
    if (!is_prime(modulus_)) {
      throw std::invalid_argument("GF(p) requires a prime modulus");
    }
  }

  [[nodiscard]] std::uint32_t modulus() const noexcept { return modulus_; }
  [[nodiscard]] Element zero() const noexcept { return Element(0, modulus_); }
  [[nodiscard]] Element one() const noexcept { return Element(1, modulus_); }

  [[nodiscard]] Element from_integer(std::int64_t value) const noexcept {
    const auto modulus = static_cast<std::int64_t>(modulus_);
    auto reduced = value % modulus;
    if (reduced < 0) {
      reduced += modulus;
    }
    return Element(static_cast<std::uint32_t>(reduced), modulus_);
  }

  [[nodiscard]] Element from_unsigned(std::uint64_t value) const noexcept {
    return Element(static_cast<std::uint32_t>(value % modulus_), modulus_);
  }

  [[nodiscard]] Element canonical(Element value) const {
    require_element(value);
    return value;
  }

  [[nodiscard]] Element add(Element left, Element right) const {
    require_element(left);
    require_element(right);
    const auto sum = static_cast<std::uint64_t>(left.value_) + right.value_;
    return Element(
        static_cast<std::uint32_t>(sum >= modulus_ ? sum - modulus_ : sum),
        modulus_);
  }

  [[nodiscard]] Element subtract(Element left, Element right) const {
    require_element(left);
    require_element(right);
    if (left.value_ >= right.value_) {
      return Element(left.value_ - right.value_, modulus_);
    }
    return Element(modulus_ - (right.value_ - left.value_), modulus_);
  }

  [[nodiscard]] Element negate(Element value) const {
    require_element(value);
    return value.value_ == 0 ? value
                             : Element(modulus_ - value.value_, modulus_);
  }

  [[nodiscard]] Element multiply(Element left, Element right) const {
    require_element(left);
    require_element(right);
    const auto product = static_cast<std::uint64_t>(left.value_) * right.value_;
    return Element(static_cast<std::uint32_t>(product % modulus_), modulus_);
  }

  [[nodiscard]] Element power(Element base, std::uint64_t exponent) const {
    require_element(base);
    auto result = one();
    while (exponent != 0) {
      if ((exponent & 1U) != 0) {
        result = multiply(result, base);
      }
      exponent >>= 1U;
      if (exponent != 0) {
        base = multiply(base, base);
      }
    }
    return result;
  }

  [[nodiscard]] Element inverse(Element value) const {
    if (is_zero(value)) {
      throw std::domain_error("zero has no multiplicative inverse");
    }
    return power(value, static_cast<std::uint64_t>(modulus_) - 2U);
  }

  [[nodiscard]] Element divide(Element numerator, Element denominator) const {
    return multiply(numerator, inverse(denominator));
  }

  [[nodiscard]] bool is_zero(Element value) const {
    require_element(value);
    return value.value_ == 0;
  }

  [[nodiscard]] bool is_one(Element value) const {
    require_element(value);
    return value.value_ == 1;
  }

  [[nodiscard]] bool is_negative(Element value) const {
    require_element(value);
    return false;
  }
  [[nodiscard]] Element absolute(Element value) const {
    require_element(value);
    return value;
  }

  [[nodiscard]] std::string to_string(Element value) const {
    require_element(value);
    return std::to_string(value.value_);
  }

  friend bool operator==(const PrimeField&, const PrimeField&) = default;

 private:
  void require_element(Element value) const {
    if (value.modulus_ != modulus_) {
      throw std::invalid_argument("finite-field element belongs to a different GF(p)");
    }
  }

  static bool is_prime(std::uint32_t value) noexcept {
    if (value < 2) {
      return false;
    }
    if ((value % 2U) == 0U) {
      return value == 2U;
    }
    for (std::uint32_t divisor = 3;
         static_cast<std::uint64_t>(divisor) * divisor <= value;
         divisor += 2U) {
      if ((value % divisor) == 0U) {
        return false;
      }
    }
    return true;
  }

  std::uint32_t modulus_;
};

inline PrimeField GF(std::uint32_t modulus) { return PrimeField(modulus); }

class Rational {
 public:
  Rational() = default;

  [[nodiscard]] static Rational from_fraction(
      std::string_view numerator,
      std::string_view denominator) {
    const mpz_class num(std::string(numerator), 10);
    const mpz_class den(std::string(denominator), 10);
    if (den == 0) {
      throw std::domain_error("a rational denominator cannot be zero");
    }
    mpq_class value(num, den);
    value.canonicalize();
    return Rational(std::move(value));
  }

  [[nodiscard]] std::string to_string() const { return value_.get_str(); }
  [[nodiscard]] std::string numerator_string() const {
    return value_.get_num().get_str();
  }
  [[nodiscard]] std::string denominator_string() const {
    return value_.get_den().get_str();
  }

  friend bool operator==(const Rational&, const Rational&) = default;
  friend std::ostream& operator<<(std::ostream& stream, const Rational& value) {
    return stream << value.to_string();
  }

 private:
  explicit Rational(mpq_class value) : value_(std::move(value)) {
    value_.canonicalize();
  }

  mpq_class value_ = 0;
  friend class RationalField;
};

class RationalField {
 public:
  using Element = Rational;

  [[nodiscard]] Element zero() const { return Element(); }
  [[nodiscard]] Element one() const { return from_integer(1); }

  [[nodiscard]] Element from_integer(long value) const {
    mpq_class result(value);
    return Element(std::move(result));
  }

  [[nodiscard]] Element from_unsigned(std::uint64_t value) const {
    mpz_class integer;
    mpz_import(integer.get_mpz_t(), 1, 1, sizeof(value), 0, 0, &value);
    return Element(mpq_class(integer));
  }

  [[nodiscard]] Element fraction(long numerator, long denominator) const {
    if (denominator == 0) {
      throw std::domain_error("a rational denominator cannot be zero");
    }
    mpq_class result{mpz_class(numerator), mpz_class(denominator)};
    result.canonicalize();
    return Element(std::move(result));
  }

  [[nodiscard]] Element canonical(const Element& value) const { return value; }

  [[nodiscard]] Element add(const Element& left, const Element& right) const {
    mpq_class result = left.value_;
    result += right.value_;
    return Element(std::move(result));
  }

  [[nodiscard]] Element subtract(
      const Element& left,
      const Element& right) const {
    mpq_class result = left.value_;
    result -= right.value_;
    return Element(std::move(result));
  }

  [[nodiscard]] Element negate(const Element& value) const {
    mpq_class result = -value.value_;
    return Element(std::move(result));
  }

  [[nodiscard]] Element multiply(
      const Element& left,
      const Element& right) const {
    mpq_class result = left.value_;
    result *= right.value_;
    return Element(std::move(result));
  }

  [[nodiscard]] Element inverse(const Element& value) const {
    if (is_zero(value)) {
      throw std::domain_error("zero has no multiplicative inverse");
    }
    mpq_class result = 1 / value.value_;
    return Element(std::move(result));
  }

  [[nodiscard]] Element divide(
      const Element& numerator,
      const Element& denominator) const {
    if (is_zero(denominator)) {
      throw std::domain_error("division by zero");
    }
    mpq_class result = numerator.value_;
    result /= denominator.value_;
    return Element(std::move(result));
  }

  [[nodiscard]] bool is_zero(const Element& value) const noexcept {
    return value.value_ == 0;
  }

  [[nodiscard]] bool is_one(const Element& value) const noexcept {
    return value.value_ == 1;
  }

  [[nodiscard]] bool is_negative(const Element& value) const noexcept {
    return value.value_ < 0;
  }

  [[nodiscard]] Element absolute(const Element& value) const {
    return is_negative(value) ? negate(value) : value;
  }

  [[nodiscard]] std::string to_string(const Element& value) const {
    return value.to_string();
  }

  friend constexpr bool operator==(RationalField, RationalField) noexcept {
    return true;
  }
};

inline RationalField QQ() { return RationalField(); }

}  // namespace laughableengine

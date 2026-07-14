#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace laughableengine {

enum class Order { Grevlex, Lex };

class TruncatedMonomialSpace;

class Monomial {
 public:
  static constexpr std::size_t maximum_variables = 16;
  using Exponents = std::array<std::uint16_t, maximum_variables>;

  Monomial() = default;

  [[nodiscard]] const Exponents& exponents() const noexcept {
    return exponents_;
  }
  [[nodiscard]] std::uint16_t exponent(std::size_t variable) const {
    if (variable >= maximum_variables) {
      throw std::out_of_range("monomial variable index is out of range");
    }
    return exponents_[variable];
  }
  [[nodiscard]] std::uint32_t total_degree() const noexcept {
    return total_degree_;
  }

  [[nodiscard]] bool is_divisible_by(const Monomial& divisor) const noexcept {
    for (std::size_t index = 0; index < maximum_variables; ++index) {
      if (exponents_[index] < divisor.exponents_[index]) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] Monomial quotient_by(const Monomial& divisor) const {
    if (!is_divisible_by(divisor)) {
      throw std::domain_error("monomial division is not exact");
    }
    Monomial result;
    for (std::size_t index = 0; index < maximum_variables; ++index) {
      result.exponents_[index] = exponents_[index] - divisor.exponents_[index];
      result.total_degree_ += result.exponents_[index];
    }
    return result;
  }

  [[nodiscard]] Monomial lcm_with(const Monomial& other) const noexcept {
    Monomial result;
    for (std::size_t index = 0; index < maximum_variables; ++index) {
      result.exponents_[index] =
          std::max(exponents_[index], other.exponents_[index]);
      result.total_degree_ += result.exponents_[index];
    }
    return result;
  }

  [[nodiscard]] Monomial multiplied_by(
      const Monomial& other,
      std::size_t variable_count) const {
    if (variable_count == 0 || variable_count > maximum_variables) {
      throw std::invalid_argument(
          "a monomial product requires between one and sixteen variables");
    }
    for (std::size_t index = variable_count; index < maximum_variables;
         ++index) {
      if (exponents_[index] != 0 || other.exponents_[index] != 0) {
        throw std::invalid_argument(
            "a monomial product contains an exponent outside its ring");
      }
    }
    return multiply(*this, other, variable_count);
  }

  friend bool operator==(const Monomial&, const Monomial&) = default;

 private:
  static Monomial from_exponents(
      std::span<const std::uint16_t> exponents,
      std::size_t variable_count) {
    if (variable_count == 0 || variable_count > maximum_variables) {
      throw std::invalid_argument(
          "a monomial requires between one and sixteen variables");
    }
    if (exponents.size() > variable_count) {
      throw std::invalid_argument("too many exponents for this polynomial ring");
    }

    Monomial result;
    for (std::size_t index = 0; index < exponents.size(); ++index) {
      result.exponents_[index] = exponents[index];
      result.total_degree_ += exponents[index];
    }
    return result;
  }

  static Monomial multiply(
      const Monomial& left,
      const Monomial& right,
      std::size_t variable_count) {
    Monomial result;
    for (std::size_t index = 0; index < variable_count; ++index) {
      const auto sum = static_cast<std::uint32_t>(left.exponents_[index]) +
                       right.exponents_[index];
      if (sum > std::numeric_limits<std::uint16_t>::max()) {
        throw std::overflow_error("monomial exponent overflow");
      }
      result.exponents_[index] = static_cast<std::uint16_t>(sum);
      result.total_degree_ += sum;
    }
    return result;
  }

  static Monomial differentiated(const Monomial& input, std::size_t variable) {
    Monomial result = input;
    --result.exponents_[variable];
    --result.total_degree_;
    return result;
  }

  Exponents exponents_{};
  std::uint32_t total_degree_ = 0;

  template <typename Field>
  friend class Polynomial;
  template <typename Field>
  friend class PolynomialRing;
  friend class TruncatedMonomialSpace;
};

namespace detail {

inline int compare_monomials(
    const Monomial& left,
    const Monomial& right,
    Order order,
    std::size_t variable_count) noexcept {
  if (order == Order::Lex) {
    for (std::size_t index = 0; index < variable_count; ++index) {
      if (left.exponents()[index] != right.exponents()[index]) {
        return left.exponents()[index] > right.exponents()[index] ? 1 : -1;
      }
    }
    return 0;
  }

  if (left.total_degree() != right.total_degree()) {
    return left.total_degree() > right.total_degree() ? 1 : -1;
  }
  for (std::size_t index = variable_count; index-- > 0;) {
    if (left.exponents()[index] != right.exponents()[index]) {
      return left.exponents()[index] < right.exponents()[index] ? 1 : -1;
    }
  }
  return 0;
}

inline bool valid_variable_name(std::string_view name) {
  if (name.empty()) {
    return false;
  }
  const auto first = static_cast<unsigned char>(name.front());
  if (!(std::isalpha(first) != 0 || name.front() == '_')) {
    return false;
  }
  return std::all_of(name.begin() + 1, name.end(), [](char character) {
    const auto value = static_cast<unsigned char>(character);
    return std::isalnum(value) != 0 || character == '_';
  });
}

template <typename Field>
struct RingContext {
  Field field;
  std::vector<std::string> variables;
  Order order;
};

}  // namespace detail

template <typename Field>
class Polynomial;

template <typename Field>
class PolynomialRing {
 public:
  using Element = typename Field::Element;
  using PolynomialType = Polynomial<Field>;

  struct TermSpec {
    Element coefficient;
    std::vector<std::uint16_t> exponents;
  };

  PolynomialRing(Field field, std::vector<std::string> variables, Order order)
      : context_(std::make_shared<const detail::RingContext<Field>>(
            detail::RingContext<Field>{
                std::move(field), std::move(variables), order})) {
    validate_context();
  }

  [[nodiscard]] const Field& field() const noexcept { return context_->field; }
  [[nodiscard]] const std::vector<std::string>& variable_names() const noexcept {
    return context_->variables;
  }
  [[nodiscard]] std::size_t variable_count() const noexcept {
    return context_->variables.size();
  }
  [[nodiscard]] Order order() const noexcept { return context_->order; }

  [[nodiscard]] PolynomialType zero() const;
  [[nodiscard]] PolynomialType one() const;
  [[nodiscard]] PolynomialType integer(long value) const;
  [[nodiscard]] PolynomialType constant(Element value) const;
  [[nodiscard]] PolynomialType gen(std::size_t variable) const;
  [[nodiscard]] PolynomialType gen(std::string_view variable) const;
  [[nodiscard]] PolynomialType monomial(
      Element coefficient,
      std::initializer_list<std::uint16_t> exponents) const;
  [[nodiscard]] PolynomialType from_terms(std::vector<TermSpec> terms) const;
  [[nodiscard]] PolynomialType from_terms(
      std::initializer_list<TermSpec> terms) const {
    return from_terms(std::vector<TermSpec>(terms));
  }

 private:
  void validate_context() const {
    const auto count = context_->variables.size();
    if (count == 0 || count > Monomial::maximum_variables) {
      throw std::invalid_argument(
          "a polynomial ring requires between one and sixteen variables");
    }
    for (std::size_t index = 0; index < count; ++index) {
      const auto& name = context_->variables[index];
      if (!detail::valid_variable_name(name)) {
        throw std::invalid_argument("invalid polynomial variable name: " + name);
      }
      if (std::find(context_->variables.begin(),
                    context_->variables.begin() +
                        static_cast<std::ptrdiff_t>(index),
                    name) != context_->variables.begin() +
                                static_cast<std::ptrdiff_t>(index)) {
        throw std::invalid_argument("duplicate polynomial variable name: " + name);
      }
    }
  }

  [[nodiscard]] std::size_t variable_index(std::string_view name) const {
    const auto iterator = std::find(context_->variables.begin(),
                                    context_->variables.end(), name);
    if (iterator == context_->variables.end()) {
      throw std::invalid_argument("unknown polynomial variable: " +
                                  std::string(name));
    }
    return static_cast<std::size_t>(iterator - context_->variables.begin());
  }

  std::shared_ptr<const detail::RingContext<Field>> context_;
};

template <typename Field>
class Polynomial {
 public:
  using Element = typename Field::Element;

  struct Term {
    Monomial monomial;
    Element coefficient;

    friend bool operator==(const Term&, const Term&) = default;
  };

  Polynomial() = delete;

  [[nodiscard]] bool is_zero() const noexcept { return terms_.empty(); }
  [[nodiscard]] std::size_t term_count() const noexcept { return terms_.size(); }
  [[nodiscard]] const std::vector<Term>& terms() const noexcept { return terms_; }
  [[nodiscard]] const Term* leading_term() const noexcept {
    return terms_.empty() ? nullptr : &terms_.front();
  }
  [[nodiscard]] bool same_ring(const Polynomial& other) const noexcept {
    return context_.get() == other.context_.get();
  }
  [[nodiscard]] const Field& coefficient_field() const noexcept {
    return context_->field;
  }
  [[nodiscard]] std::size_t variable_count() const noexcept {
    return context_->variables.size();
  }
  [[nodiscard]] Order order() const noexcept { return context_->order; }
  [[nodiscard]] int compare_monomials(
      const Monomial& left,
      const Monomial& right) const noexcept {
    return detail::compare_monomials(left, right, context_->order,
                                     context_->variables.size());
  }
  [[nodiscard]] std::optional<std::uint32_t> total_degree() const noexcept {
    if (terms_.empty()) {
      return std::nullopt;
    }
    std::uint32_t degree = 0;
    for (const auto& term : terms_) {
      degree = std::max(degree, term.monomial.total_degree());
    }
    return degree;
  }

  [[nodiscard]] Polynomial operator+(const Polynomial& other) const {
    require_same_ring(other);
    std::vector<Term> result;
    result.reserve(terms_.size() + other.terms_.size());
    result.insert(result.end(), terms_.begin(), terms_.end());
    result.insert(result.end(), other.terms_.begin(), other.terms_.end());
    return Polynomial(context_, std::move(result));
  }

  [[nodiscard]] Polynomial operator-() const {
    std::vector<Term> result = terms_;
    for (auto& term : result) {
      term.coefficient = context_->field.negate(term.coefficient);
    }
    return Polynomial(context_, std::move(result));
  }

  [[nodiscard]] Polynomial operator-(const Polynomial& other) const {
    return *this + (-other);
  }

  [[nodiscard]] Polynomial operator*(const Polynomial& other) const {
    require_same_ring(other);
    if (is_zero() || other.is_zero()) {
      return Polynomial(context_, {});
    }

    std::vector<Term> result;
    result.reserve(terms_.size() * other.terms_.size());
    for (const auto& left : terms_) {
      for (const auto& right : other.terms_) {
        result.push_back(Term{
            Monomial::multiply(left.monomial, right.monomial,
                               context_->variables.size()),
            context_->field.multiply(left.coefficient, right.coefficient)});
      }
    }
    return Polynomial(context_, std::move(result));
  }

  [[nodiscard]] Polynomial zero_like() const { return Polynomial(context_, {}); }

  [[nodiscard]] Polynomial one_like() const {
    return Polynomial(context_, constant_terms(context_->field.one()));
  }

  [[nodiscard]] Polynomial term_like(
      Element coefficient,
      const Monomial& monomial) const {
    return Polynomial(
        context_, {Term{monomial, std::move(coefficient)}});
  }

  [[nodiscard]] Polynomial from_terms_like(std::vector<Term> terms) const {
    return Polynomial(context_, std::move(terms));
  }

  [[nodiscard]] Polynomial multiply_by_term(
      Element coefficient,
      const Monomial& monomial) const {
    for (std::size_t index = context_->variables.size();
         index < Monomial::maximum_variables; ++index) {
      if (monomial.exponents()[index] != 0) {
        throw std::invalid_argument("monomial lies outside this polynomial ring");
      }
    }
    coefficient = context_->field.canonical(std::move(coefficient));
    if (is_zero() || context_->field.is_zero(coefficient)) {
      return zero_like();
    }

    std::vector<Term> result;
    result.reserve(terms_.size());
    for (const auto& term : terms_) {
      result.push_back(Term{
          Monomial::multiply(term.monomial, monomial,
                             context_->variables.size()),
          context_->field.multiply(term.coefficient, coefficient)});
    }
    return Polynomial(context_, std::move(result));
  }

  [[nodiscard]] Polynomial scaled(Element coefficient) const {
    return multiply_by_term(std::move(coefficient), Monomial());
  }

  [[nodiscard]] Polynomial monic() const {
    if (is_zero()) {
      return *this;
    }
    return scaled(context_->field.inverse(terms_.front().coefficient));
  }

  [[nodiscard]] Polynomial pow(std::uint64_t exponent) const {
    auto result = Polynomial(context_, constant_terms(context_->field.one()));
    auto base = *this;
    while (exponent != 0) {
      if ((exponent & 1U) != 0) {
        result = result * base;
      }
      exponent >>= 1U;
      if (exponent != 0) {
        base = base * base;
      }
    }
    return result;
  }

  [[nodiscard]] Polynomial derivative(std::size_t variable) const {
    if (variable >= context_->variables.size()) {
      throw std::out_of_range("polynomial variable index is out of range");
    }

    std::vector<Term> result;
    result.reserve(terms_.size());
    for (const auto& term : terms_) {
      const auto exponent = term.monomial.exponents()[variable];
      if (exponent == 0) {
        continue;
      }
      result.push_back(Term{
          Monomial::differentiated(term.monomial, variable),
          context_->field.multiply(
              term.coefficient, context_->field.from_unsigned(exponent))});
    }
    return Polynomial(context_, std::move(result));
  }

  [[nodiscard]] Polynomial derivative(std::string_view variable) const {
    const auto iterator =
        std::find(context_->variables.begin(), context_->variables.end(), variable);
    if (iterator == context_->variables.end()) {
      throw std::invalid_argument("unknown polynomial variable: " +
                                  std::string(variable));
    }
    return derivative(
        static_cast<std::size_t>(iterator - context_->variables.begin()));
  }

  [[nodiscard]] std::string to_string() const {
    if (is_zero()) {
      return "0";
    }

    std::string output;
    bool first_term = true;
    for (const auto& term : terms_) {
      const bool negative = context_->field.is_negative(term.coefficient);
      const auto magnitude = context_->field.absolute(term.coefficient);
      if (first_term) {
        if (negative) {
          output += '-';
        }
      } else {
        output += negative ? " - " : " + ";
      }

      const bool constant = term.monomial.total_degree() == 0;
      if (constant || !context_->field.is_one(magnitude)) {
        output += context_->field.to_string(magnitude);
        if (!constant) {
          output += '*';
        }
      }

      if (!constant) {
        bool first_factor = true;
        for (std::size_t variable = 0; variable < context_->variables.size();
             ++variable) {
          const auto exponent = term.monomial.exponents()[variable];
          if (exponent == 0) {
            continue;
          }
          if (!first_factor) {
            output += '*';
          }
          output += context_->variables[variable];
          if (exponent != 1) {
            output += '^';
            output += std::to_string(exponent);
          }
          first_factor = false;
        }
      }
      first_term = false;
    }
    return output;
  }

  friend bool operator==(const Polynomial& left, const Polynomial& right) {
    return left.context_.get() == right.context_.get() && left.terms_ == right.terms_;
  }

  friend std::ostream& operator<<(std::ostream& stream, const Polynomial& value) {
    return stream << value.to_string();
  }

 private:
  Polynomial(
      std::shared_ptr<const detail::RingContext<Field>> context,
      std::vector<Term> terms)
      : context_(std::move(context)), terms_(normalize(context_, std::move(terms))) {}

  static std::vector<Term> constant_terms(Element coefficient) {
    return {Term{Monomial(), std::move(coefficient)}};
  }

  static std::vector<Term> normalize(
      const std::shared_ptr<const detail::RingContext<Field>>& context,
      std::vector<Term> terms) {
    for (auto& term : terms) {
      term.coefficient = context->field.canonical(term.coefficient);
      for (std::size_t index = context->variables.size();
           index < Monomial::maximum_variables; ++index) {
        if (term.monomial.exponents()[index] != 0) {
          throw std::invalid_argument("monomial contains an exponent outside its ring");
        }
      }
    }

    std::erase_if(terms, [&context](const Term& term) {
      return context->field.is_zero(term.coefficient);
    });
    std::sort(terms.begin(), terms.end(), [&context](const Term& left,
                                                     const Term& right) {
      return detail::compare_monomials(left.monomial, right.monomial,
                                       context->order,
                                       context->variables.size()) > 0;
    });

    std::vector<Term> result;
    result.reserve(terms.size());
    for (auto& term : terms) {
      if (!result.empty() && result.back().monomial == term.monomial) {
        result.back().coefficient = context->field.add(
            result.back().coefficient, term.coefficient);
        if (context->field.is_zero(result.back().coefficient)) {
          result.pop_back();
        }
      } else {
        result.push_back(std::move(term));
      }
    }
    return result;
  }

  void require_same_ring(const Polynomial& other) const {
    if (context_.get() != other.context_.get()) {
      throw std::invalid_argument(
          "polynomial arithmetic requires the same ring context");
    }
  }

  std::shared_ptr<const detail::RingContext<Field>> context_;
  std::vector<Term> terms_;

  friend class PolynomialRing<Field>;
};

template <typename Field>
typename PolynomialRing<Field>::PolynomialType PolynomialRing<Field>::zero() const {
  return PolynomialType(context_, {});
}

template <typename Field>
typename PolynomialRing<Field>::PolynomialType PolynomialRing<Field>::one() const {
  return constant(context_->field.one());
}

template <typename Field>
typename PolynomialRing<Field>::PolynomialType PolynomialRing<Field>::integer(
    long value) const {
  return constant(context_->field.from_integer(value));
}

template <typename Field>
typename PolynomialRing<Field>::PolynomialType PolynomialRing<Field>::constant(
    Element value) const {
  using Term = typename PolynomialType::Term;
  return PolynomialType(context_, {Term{Monomial(), std::move(value)}});
}

template <typename Field>
typename PolynomialRing<Field>::PolynomialType PolynomialRing<Field>::gen(
    std::size_t variable) const {
  if (variable >= variable_count()) {
    throw std::out_of_range("polynomial variable index is out of range");
  }
  std::vector<std::uint16_t> exponents(variable_count(), 0);
  exponents[variable] = 1;
  using Term = typename PolynomialType::Term;
  return PolynomialType(
      context_,
      {Term{Monomial::from_exponents(exponents, variable_count()), field().one()}});
}

template <typename Field>
typename PolynomialRing<Field>::PolynomialType PolynomialRing<Field>::gen(
    std::string_view variable) const {
  return gen(variable_index(variable));
}

template <typename Field>
typename PolynomialRing<Field>::PolynomialType PolynomialRing<Field>::monomial(
    Element coefficient,
    std::initializer_list<std::uint16_t> exponents) const {
  using Term = typename PolynomialType::Term;
  return PolynomialType(
      context_,
      {Term{Monomial::from_exponents(
                std::span<const std::uint16_t>(exponents.begin(), exponents.size()),
                variable_count()),
            std::move(coefficient)}});
}

template <typename Field>
typename PolynomialRing<Field>::PolynomialType PolynomialRing<Field>::from_terms(
    std::vector<TermSpec> terms) const {
  using Term = typename PolynomialType::Term;
  std::vector<Term> converted;
  converted.reserve(terms.size());
  for (auto& term : terms) {
    converted.push_back(Term{
        Monomial::from_exponents(term.exponents, variable_count()),
        std::move(term.coefficient)});
  }
  return PolynomialType(context_, std::move(converted));
}

template <typename Field>
[[nodiscard]] PolynomialRing<Field> make_ring(
    Field field,
    std::initializer_list<std::string> variables,
    Order order = Order::Grevlex) {
  return PolynomialRing<Field>(std::move(field),
                               std::vector<std::string>(variables), order);
}

}  // namespace laughableengine

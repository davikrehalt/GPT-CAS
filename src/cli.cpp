#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "laughableengine/laughableengine.hpp"

#ifndef LAUGHABLEENGINE_VERSION
#define LAUGHABLEENGINE_VERSION "development"
#endif

namespace {

using laughableengine::GF;
using laughableengine::ColonClosureLimits;
using laughableengine::ColonClosureStopStatus;
using laughableengine::CycleAuditStatus;
using laughableengine::CotangentClassStatus;
using laughableengine::CotangentH1Spec;
using laughableengine::FullH1ResourceLimit;
using laughableengine::Ideal;
using laughableengine::InverseSystemResourceLimit;
using laughableengine::MatrixSpaceRankProof;
using laughableengine::Order;
using laughableengine::PrimeField;
using laughableengine::QQ;
using laughableengine::ResourceLimitExceeded;
using laughableengine::audit_cycle;
using laughableengine::colon_closure;
using laughableengine::cotangent_h1;
using laughableengine::divide;
using laughableengine::full_h1_action;
using laughableengine::groebner_basis;
using laughableengine::is_groebner_basis;
using laughableengine::macaulay_annihilator;
using laughableengine::make_jg_certificate_json;
using laughableengine::make_ring;
using laughableengine::normal_form;
using laughableengine::parse_polynomial;
using laughableengine::screen_cycle;
using laughableengine::screen_full_h1;

class UsageError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

class InputParseError : public std::invalid_argument {
 public:
  InputParseError(std::string source, std::size_t offset, std::string message)
      : std::invalid_argument(std::move(message)),
        source_(std::move(source)),
        offset_(offset) {}

  [[nodiscard]] const std::string& source() const noexcept { return source_; }
  [[nodiscard]] std::size_t offset() const noexcept { return offset_; }

 private:
  std::string source_;
  std::size_t offset_;
};

struct Options {
  std::string field = "QQ";
  std::string variables;
  std::string order = "grevlex";
  std::string command;
  std::string derivative_variable;
  std::string colon_polynomial;
  std::string elimination_variables;
  std::string apolarity = "ordinary";
  std::vector<std::string> divisors;
  std::vector<std::string> positional;
  std::size_t max_steps = 16;
  std::size_t maximal_power = 0;
  bool help = false;
  bool version = false;
  bool derivative_variable_supplied = false;
  bool colon_polynomial_supplied = false;
  bool elimination_variables_supplied = false;
  bool max_steps_supplied = false;
  bool maximal_power_supplied = false;
  bool apolarity_supplied = false;
};

[[nodiscard]] std::string_view trim(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())) != 0) {
    value.remove_prefix(1);
  }
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.remove_suffix(1);
  }
  return value;
}

[[nodiscard]] std::string lowercase(std::string_view value) {
  std::string result(value);
  for (auto& character : result) {
    character = static_cast<char>(
        std::tolower(static_cast<unsigned char>(character)));
  }
  return result;
}

[[nodiscard]] std::string take_option_value(
    int& index,
    int count,
    char** values,
    std::string_view option,
    std::string_view argument) {
  const std::string prefix = std::string(option) + '=';
  if (argument.starts_with(prefix)) {
    return std::string(argument.substr(prefix.size()));
  }
  if (index + 1 >= count) {
    throw UsageError(std::string(option) + " requires a value");
  }
  ++index;
  return values[index];
}

[[nodiscard]] Options parse_options(int count, char** values) {
  Options options;
  bool positional_only = false;
  for (int index = 1; index < count; ++index) {
    const std::string_view argument(values[index]);
    if (!positional_only && argument == "--") {
      positional_only = true;
      continue;
    }
    if (!positional_only && (argument == "--help" || argument == "-h")) {
      options.help = true;
      continue;
    }
    if (!positional_only && argument == "--version") {
      options.version = true;
      continue;
    }
    if (!positional_only &&
        (argument == "--field" || argument.starts_with("--field="))) {
      options.field = take_option_value(
          index, count, values, "--field", argument);
      continue;
    }
    if (!positional_only &&
        (argument == "--vars" || argument.starts_with("--vars="))) {
      options.variables = take_option_value(
          index, count, values, "--vars", argument);
      continue;
    }
    if (!positional_only &&
        (argument == "--order" || argument.starts_with("--order="))) {
      options.order = take_option_value(
          index, count, values, "--order", argument);
      continue;
    }
    if (!positional_only &&
        (argument == "--by" || argument.starts_with("--by="))) {
      options.divisors.push_back(take_option_value(
          index, count, values, "--by", argument));
      continue;
    }
    if (!positional_only &&
        (argument == "--var" || argument.starts_with("--var="))) {
      if (options.derivative_variable_supplied) {
        throw UsageError("--var may be supplied only once");
      }
      options.derivative_variable_supplied = true;
      options.derivative_variable = take_option_value(
          index, count, values, "--var", argument);
      continue;
    }
    if (!positional_only &&
        (argument == "--g" || argument.starts_with("--g="))) {
      if (options.colon_polynomial_supplied) {
        throw UsageError("--g may be supplied only once");
      }
      options.colon_polynomial_supplied = true;
      options.colon_polynomial = take_option_value(
          index, count, values, "--g", argument);
      continue;
    }
    if (!positional_only &&
        (argument == "--elim" || argument.starts_with("--elim="))) {
      if (options.elimination_variables_supplied) {
        throw UsageError("--elim may be supplied only once");
      }
      options.elimination_variables_supplied = true;
      options.elimination_variables = take_option_value(
          index, count, values, "--elim", argument);
      continue;
    }
    if (!positional_only &&
        (argument == "--apolarity" ||
         argument.starts_with("--apolarity="))) {
      if (options.apolarity_supplied) {
        throw UsageError("--apolarity may be supplied only once");
      }
      options.apolarity_supplied = true;
      options.apolarity = take_option_value(
          index, count, values, "--apolarity", argument);
      continue;
    }
    if (!positional_only &&
        (argument == "--max-steps" ||
         argument.starts_with("--max-steps="))) {
      if (options.max_steps_supplied) {
        throw UsageError("--max-steps may be supplied only once");
      }
      options.max_steps_supplied = true;
      const auto source = take_option_value(
          index, count, values, "--max-steps", argument);
      std::size_t converted = 0;
      const auto conversion = std::from_chars(
          source.data(), source.data() + source.size(), converted);
      if (source.empty() || conversion.ec != std::errc{} ||
          conversion.ptr != source.data() + source.size()) {
        throw UsageError("--max-steps requires a nonnegative decimal integer");
      }
      options.max_steps = converted;
      continue;
    }
    if (!positional_only &&
        (argument == "--maximal-power" ||
         argument.starts_with("--maximal-power="))) {
      if (options.maximal_power_supplied) {
        throw UsageError("--maximal-power may be supplied only once");
      }
      options.maximal_power_supplied = true;
      const auto source = take_option_value(
          index, count, values, "--maximal-power", argument);
      std::size_t converted = 0;
      const auto conversion = std::from_chars(
          source.data(), source.data() + source.size(), converted);
      if (source.empty() || converted == 0 ||
          conversion.ec != std::errc{} ||
          conversion.ptr != source.data() + source.size()) {
        throw UsageError(
            "--maximal-power requires a positive decimal integer");
      }
      options.maximal_power = converted;
      continue;
    }
    if (!positional_only && argument.starts_with("--")) {
      throw UsageError("unknown option: " + std::string(argument));
    }
    if (options.command.empty()) {
      options.command = lowercase(argument);
    } else {
      options.positional.emplace_back(argument);
    }
  }
  return options;
}

[[nodiscard]] std::vector<std::string> parse_variables(
    const std::string& specification) {
  if (specification.empty()) {
    throw UsageError("--vars is required (for example, --vars x,y)");
  }

  std::vector<std::string> variables;
  std::size_t start = 0;
  while (start <= specification.size()) {
    const auto comma = specification.find(',', start);
    const auto end = comma == std::string::npos ? specification.size() : comma;
    const auto name = trim(std::string_view(specification).substr(start, end - start));
    if (name.empty()) {
      throw UsageError("--vars contains an empty variable name");
    }
    variables.emplace_back(name);
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return variables;
}

[[nodiscard]] Order parse_order(std::string_view specification) {
  const auto normalized = lowercase(trim(specification));
  if (normalized == "grevlex" || normalized == "degrevlex") {
    return Order::Grevlex;
  }
  if (normalized == "lex") {
    return Order::Lex;
  }
  throw UsageError("--order must be grevlex or lex");
}

[[nodiscard]] laughableengine::ApolarityConvention parse_apolarity(
    std::string_view specification) {
  const auto normalized = lowercase(trim(specification));
  if (normalized == "ordinary" || normalized == "differentiation") {
    return laughableengine::ApolarityConvention::OrdinaryDifferentiation;
  }
  if (normalized == "divided" || normalized == "divided-powers") {
    return laughableengine::ApolarityConvention::DividedPowers;
  }
  throw UsageError("--apolarity must be ordinary or divided");
}

[[nodiscard]] std::uint32_t parse_prime_field(std::string_view specification) {
  specification = trim(specification);
  const auto normalized = lowercase(specification);
  if (!normalized.starts_with("gf(") || !normalized.ends_with(')')) {
    throw UsageError("--field must be QQ or GF(p)");
  }
  const auto digits = std::string_view(normalized).substr(3, normalized.size() - 4);
  std::uint32_t modulus = 0;
  const auto result =
      std::from_chars(digits.data(), digits.data() + digits.size(), modulus);
  if (digits.empty() || result.ec != std::errc{} ||
      result.ptr != digits.data() + digits.size()) {
    throw UsageError("GF(p) requires a decimal prime modulus");
  }
  return modulus;
}

template <typename Field>
[[nodiscard]] auto parse_input(
    const laughableengine::PolynomialRing<Field>& ring,
    const std::string& source) {
  try {
    return parse_polynomial(ring, source);
  } catch (const laughableengine::ParseError& error) {
    throw InputParseError(source, error.offset(), error.what());
  }
}

template <typename Field>
[[nodiscard]] std::vector<laughableengine::Polynomial<Field>> parse_inputs(
    const laughableengine::PolynomialRing<Field>& ring,
    const std::vector<std::string>& sources) {
  std::vector<laughableengine::Polynomial<Field>> result;
  result.reserve(sources.size());
  for (const auto& source : sources) {
    result.push_back(parse_input(ring, source));
  }
  return result;
}

void require_count(
    const Options& options,
    std::size_t expected,
    std::string_view synopsis) {
  if (options.positional.size() != expected) {
    throw UsageError(std::string(synopsis));
  }
}

void reject_divisors(const Options& options) {
  if (!options.divisors.empty()) {
    throw UsageError("--by is valid only for nf and divide");
  }
}

void reject_derivative_variable(const Options& options) {
  if (options.derivative_variable_supplied) {
    throw UsageError("--var is valid only for diff");
  }
}

void reject_new_options(
    const Options& options,
    bool allow_colon_polynomial = false,
    bool allow_elimination_variables = false,
    bool allow_max_steps = false,
    bool allow_apolarity = false,
    bool allow_maximal_power = false) {
  if (options.colon_polynomial_supplied && !allow_colon_polynomial) {
    throw UsageError(
        "--g is valid only for colon, audit, closure, certificate, "
        "screen-audit, and verify-h1-class");
  }
  if (options.elimination_variables_supplied &&
      !allow_elimination_variables) {
    throw UsageError("--elim is valid only for eliminate");
  }
  if (options.max_steps_supplied && !allow_max_steps) {
    throw UsageError("--max-steps is valid only for closure");
  }
  if (options.apolarity_supplied && !allow_apolarity) {
    throw UsageError("--apolarity is valid only for inverse-system");
  }
  if (options.maximal_power_supplied && !allow_maximal_power) {
    throw UsageError(
        "--maximal-power is valid only for cotangent-h1 and "
        "verify-h1-class");
  }
}

void require_generators(const Options& options, std::string_view command) {
  if (options.positional.empty()) {
    throw UsageError(
        std::string(command) + " requires at least one ideal generator");
  }
}

void require_colon_polynomial(const Options& options) {
  if (!options.colon_polynomial_supplied ||
      trim(options.colon_polynomial).empty()) {
    throw UsageError("this command requires --g POLYNOMIAL");
  }
}

void require_maximal_power(const Options& options) {
  if (!options.maximal_power_supplied) {
    throw UsageError("this command requires --maximal-power N");
  }
}

template <typename Field>
[[nodiscard]] Ideal<Field> parse_ideal(
    const laughableengine::PolynomialRing<Field>& ring,
    const Options& options) {
  require_generators(options, options.command);
  return Ideal<Field>(ring, parse_inputs(ring, options.positional));
}

template <typename Field>
[[nodiscard]] std::vector<std::size_t> parse_elimination_indices(
    const laughableengine::PolynomialRing<Field>& ring,
    const std::string& specification) {
  if (trim(specification).empty()) {
    throw UsageError("eliminate requires --elim x[,y...]");
  }

  std::vector<std::size_t> result;
  std::size_t start = 0;
  while (start <= specification.size()) {
    const auto comma = specification.find(',', start);
    const auto end = comma == std::string::npos ? specification.size() : comma;
    const auto name = trim(
        std::string_view(specification).substr(start, end - start));
    if (name.empty()) {
      throw UsageError("--elim contains an empty variable name");
    }
    const auto iterator = std::find(
        ring.variable_names().begin(), ring.variable_names().end(), name);
    if (iterator == ring.variable_names().end()) {
      throw UsageError(
          "--elim names an unknown variable: " + std::string(name));
    }
    result.push_back(static_cast<std::size_t>(
        iterator - ring.variable_names().begin()));
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return result;
}

[[nodiscard]] std::string bool_text(bool value) {
  return value ? "true" : "false";
}

[[nodiscard]] std::string optional_bool_text(
    const std::optional<bool>& value) {
  return value.has_value() ? bool_text(*value) : "n/a";
}

[[nodiscard]] std::string optional_size_text(
    const std::optional<std::size_t>& value) {
  return value.has_value() ? std::to_string(*value) : "n/a";
}

[[nodiscard]] std::string cycle_status_text(CycleAuditStatus status) {
  switch (status) {
    case CycleAuditStatus::Complete:
      return "complete";
    case CycleAuditStatus::PolynomialNotInIdeal:
      return "polynomial-not-in-ideal";
    case CycleAuditStatus::PositiveDimensional:
      return "positive-dimensional";
    case CycleAuditStatus::UnsupportedAtOrigin:
      return "unsupported-at-origin";
    case CycleAuditStatus::UnitIdeal:
      return "unit-ideal";
    case CycleAuditStatus::ResourceLimit:
      return "resource-limit";
  }
  throw std::logic_error("unknown cycle audit status");
}

[[nodiscard]] std::string closure_status_text(ColonClosureStopStatus status) {
  switch (status) {
    case ColonClosureStopStatus::ProperFixedPoint:
      return "proper-fixed-point";
    case ColonClosureStopStatus::UnitIdeal:
      return "unit-ideal";
    case ColonClosureStopStatus::ResourceLimit:
      return "resource-limit";
    case ColonClosureStopStatus::InvalidStart:
      return "invalid-start";
  }
  throw std::logic_error("unknown colon closure status");
}

[[nodiscard]] std::string rank_proof_text(MatrixSpaceRankProof proof) {
  switch (proof) {
    case MatrixSpaceRankProof::ProvenMaximum:
      return "proven-maximum";
    case MatrixSpaceRankProof::ProvenFullColumnRank:
      return "proven-full-column-rank";
    case MatrixSpaceRankProof::GenericOnly:
      return "generic-only";
    case MatrixSpaceRankProof::ResourceLimit:
      return "resource-limit";
  }
  throw std::logic_error("unknown matrix-space rank proof");
}

[[nodiscard]] std::string discovery_status_text(
    laughableengine::DiscoveryScreenStatus status) {
  switch (status) {
    case laughableengine::DiscoveryScreenStatus::Complete:
      return "complete";
    case laughableengine::DiscoveryScreenStatus::InvalidInput:
      return "invalid-input";
    case laughableengine::DiscoveryScreenStatus::ResourceLimit:
      return "resource-limit";
  }
  throw std::logic_error("unknown discovery status");
}

[[nodiscard]] std::string cotangent_class_status_text(
    CotangentClassStatus status) {
  switch (status) {
    case CotangentClassStatus::Complete:
      return "complete";
    case CotangentClassStatus::NotInIdeal:
      return "not-in-ideal";
    case CotangentClassStatus::NotCycle:
      return "not-cycle";
  }
  throw std::logic_error("unknown cotangent class status");
}

template <typename Field>
[[nodiscard]] std::string basis_text(
    const std::vector<laughableengine::Polynomial<Field>>& basis) {
  std::string result = "{";
  for (std::size_t index = 0; index < basis.size(); ++index) {
    if (index != 0) {
      result += "; ";
    }
    result += basis[index].to_string();
  }
  result += '}';
  return result;
}

template <typename Field>
[[nodiscard]] std::string matrix_entries_text(
    const laughableengine::DenseMatrix<Field>& matrix) {
  std::string result = "{";
  for (std::size_t row = 0; row < matrix.row_count(); ++row) {
    if (row != 0) {
      result += "; ";
    }
    result += '{';
    for (std::size_t column = 0; column < matrix.column_count(); ++column) {
      if (column != 0) {
        result += ", ";
      }
      result += matrix.field().to_string(matrix(row, column));
    }
    result += '}';
  }
  result += '}';
  return result;
}

template <typename Field>
void write_basis(
    std::ostream& output,
    const std::vector<laughableengine::Polynomial<Field>>& basis) {
  for (const auto& polynomial : basis) {
    output << polynomial << '\n';
  }
}

template <typename Field>
[[nodiscard]] std::string run_command(
    const Options& options,
    Field field,
    int& exit_code) {
  const auto variables = parse_variables(options.variables);
  const auto order = parse_order(options.order);
  const auto ring = laughableengine::PolynomialRing<Field>(
      std::move(field), variables, order);
  std::ostringstream output;

  if (options.command == "print" || options.command == "parse") {
    reject_divisors(options);
    reject_derivative_variable(options);
    reject_new_options(options);
    require_count(options, 1, "print requires exactly one polynomial");
    output << parse_input(ring, options.positional.front()) << '\n';
    return output.str();
  }

  if (options.command == "diff" || options.command == "derivative") {
    reject_divisors(options);
    reject_new_options(options);
    std::string variable = options.derivative_variable;
    std::string source;
    if (!options.derivative_variable_supplied) {
      require_count(
          options, 2,
          "diff requires --var NAME and one polynomial, or NAME POLYNOMIAL");
      variable = options.positional[0];
      source = options.positional[1];
    } else {
      if (variable.empty()) {
        throw UsageError("--var requires a nonempty variable name");
      }
      require_count(options, 1, "diff requires exactly one polynomial");
      source = options.positional.front();
    }
    output << parse_input(ring, source).derivative(variable) << '\n';
    return output.str();
  }

  if (options.command == "nf" || options.command == "normal-form") {
    reject_derivative_variable(options);
    reject_new_options(options);
    require_count(options, 1, "nf requires exactly one polynomial");
    if (options.divisors.empty()) {
      throw UsageError("nf requires at least one ordered --by divisor");
    }
    const auto polynomial = parse_input(ring, options.positional.front());
    const auto divisors = parse_inputs(ring, options.divisors);
    output << normal_form(polynomial, divisors) << '\n';
    return output.str();
  }

  if (options.command == "divide") {
    reject_derivative_variable(options);
    reject_new_options(options);
    require_count(options, 1, "divide requires exactly one polynomial");
    if (options.divisors.empty()) {
      throw UsageError("divide requires at least one ordered --by divisor");
    }
    const auto polynomial = parse_input(ring, options.positional.front());
    const auto divisors = parse_inputs(ring, options.divisors);
    const auto result = divide(polynomial, divisors);
    for (std::size_t index = 0; index < result.quotients.size(); ++index) {
      output << "q[" << index << "] = " << result.quotients[index] << '\n';
    }
    output << "r = " << result.remainder << '\n';
    return output.str();
  }

  if (options.command == "gb" || options.command == "groebner") {
    reject_divisors(options);
    reject_derivative_variable(options);
    reject_new_options(options);
    if (options.positional.empty()) {
      throw UsageError("gb requires at least one generator");
    }
    const auto generators = parse_inputs(ring, options.positional);
    for (const auto& polynomial : groebner_basis(generators)) {
      output << polynomial << '\n';
    }
    return output.str();
  }

  if (options.command == "check-gb") {
    reject_divisors(options);
    reject_derivative_variable(options);
    reject_new_options(options);
    if (options.positional.empty()) {
      throw UsageError("check-gb requires at least one candidate");
    }
    output << (is_groebner_basis(parse_inputs(ring, options.positional))
                   ? "true\n"
                   : "false\n");
    return output.str();
  }

  if (options.command == "dim") {
    reject_divisors(options);
    reject_derivative_variable(options);
    reject_new_options(options);
    output << parse_ideal(ring, options).quotient_dimension() << '\n';
    return output.str();
  }

  if (options.command == "std") {
    reject_divisors(options);
    reject_derivative_variable(options);
    reject_new_options(options);
    write_basis(
        output, parse_ideal(ring, options).standard_monomials().polynomials());
    return output.str();
  }

  if (options.command == "colon") {
    reject_divisors(options);
    reject_derivative_variable(options);
    reject_new_options(options, true);
    require_colon_polynomial(options);
    const auto ideal = parse_ideal(ring, options);
    const auto polynomial = parse_input(ring, options.colon_polynomial);
    write_basis(output, ideal.colon(polynomial).groebner_basis());
    return output.str();
  }

  if (options.command == "eliminate") {
    reject_divisors(options);
    reject_derivative_variable(options);
    reject_new_options(options, false, true);
    if (!options.elimination_variables_supplied) {
      throw UsageError("eliminate requires --elim x[,y...]");
    }
    const auto ideal = parse_ideal(ring, options);
    const auto indices =
        parse_elimination_indices(ring, options.elimination_variables);
    write_basis(output, ideal.eliminate(indices).groebner_basis());
    return output.str();
  }

  if (options.command == "certificate") {
    reject_divisors(options);
    reject_derivative_variable(options);
    reject_new_options(options, true);
    require_colon_polynomial(options);
    require_generators(options, options.command);
    const auto generators = parse_inputs(ring, options.positional);
    const auto polynomial = parse_input(ring, options.colon_polynomial);
    output << make_jg_certificate_json(ring, generators, polynomial) << '\n';
    return output.str();
  }

  if (options.command == "cotangent-h1") {
    reject_divisors(options);
    reject_derivative_variable(options);
    reject_new_options(options, false, false, false, false, true);
    require_maximal_power(options);

    auto presentation = cotangent_h1(CotangentH1Spec<Field>{
        ring,
        parse_inputs(ring, options.positional),
        options.maximal_power});
    const auto& reduction = presentation.reduction_matrix();
    const auto& derivative = presentation.derivative_matrix();
    const auto& cycle = presentation.cycle_matrix();

    output << "status=complete\n"
           << "maximal_power=" << presentation.maximal_power() << '\n'
           << "length_Q=" << presentation.length_Q() << '\n'
           << "length_P_mod_J2=" << presentation.length_P_mod_J2() << '\n'
           << "conormal_dimension=" << presentation.conormal_dimension()
           << '\n'
           << "h1_dimension=" << presentation.h1_dimension() << '\n'
           << "reduction_shape=" << reduction.row_count() << 'x'
           << reduction.column_count() << '\n'
           << "reduction_nnz=" << reduction.nnz() << '\n'
           << "derivative_shape=" << derivative.row_count() << 'x'
           << derivative.column_count() << '\n'
           << "derivative_nnz=" << derivative.nnz() << '\n'
           << "cycle_shape=" << cycle.row_count() << 'x'
           << cycle.column_count() << '\n'
           << "cycle_nnz=" << cycle.nnz() << '\n';
    return output.str();
  }

  if (options.command == "verify-h1-class") {
    reject_divisors(options);
    reject_derivative_variable(options);
    reject_new_options(options, true, false, false, false, true);
    require_maximal_power(options);
    require_colon_polynomial(options);

    auto presentation = cotangent_h1(CotangentH1Spec<Field>{
        ring,
        parse_inputs(ring, options.positional),
        options.maximal_power});
    const auto proof = presentation.verify_class(
        parse_input(ring, options.colon_polynomial));

    output << "status=" << cotangent_class_status_text(proof.status) << '\n'
           << "cycle_valid=" << bool_text(proof.cycle_valid) << '\n'
           << "multiplication_rank="
           << optional_size_text(proof.multiplication_rank) << '\n'
           << "annihilator_dimension="
           << optional_size_text(proof.annihilator_dimension) << '\n'
           << "faithful=" << bool_text(proof.faithful) << '\n'
           << "colon_equals_J=" << bool_text(proof.colon_equals_ideal)
           << '\n';
    return output.str();
  }

  if (options.command == "inverse-system") {
    reject_divisors(options);
    reject_derivative_variable(options);
    reject_new_options(options, false, false, false, true);
    if (options.positional.empty()) {
      throw UsageError("inverse-system requires at least one dual generator");
    }
    const auto dual_generators = parse_inputs(ring, options.positional);
    const auto data = macaulay_annihilator(
        ring, dual_generators, parse_apolarity(options.apolarity));
    output << "status=complete\n"
           << "apolarity="
           << (data.convention ==
                       laughableengine::ApolarityConvention::OrdinaryDifferentiation
                   ? "ordinary"
                   : "divided")
           << '\n'
           << "maximum_dual_degree=" << data.maximum_degree << '\n'
           << "action_shape=" << data.action_matrix.row_count() << 'x'
           << data.action_matrix.column_count() << '\n'
           << "action_nonzeros=" << data.action_matrix.nnz() << '\n'
           << "action_rank=" << data.action_rank << '\n'
           << "kernel_dimension=" << data.kernel_dimension << '\n'
           << "truncated_kernel_generators="
           << data.truncated_kernel_generator_count << '\n'
           << "quotient_length=" << data.quotient_length << '\n'
           << "annihilator_gb="
           << basis_text(data.annihilator.groebner_basis()) << '\n';
    return output.str();
  }

  if (options.command == "screen-audit") {
    reject_divisors(options);
    reject_derivative_variable(options);
    reject_new_options(options, true);
    require_colon_polynomial(options);
    if constexpr (!std::is_same_v<Field, PrimeField>) {
      throw UsageError("screen-audit is a finite-field discovery command");
    } else {
      const auto ideal = parse_ideal(ring, options);
      const auto polynomial = parse_input(ring, options.colon_polynomial);
      const auto result = screen_cycle(ideal, polynomial);
      output << "status=" << discovery_status_text(result.status) << '\n'
             << "length_Q=" << result.length_Q << '\n'
             << "length_P_mod_J2=" << result.length_P_mod_J2 << '\n'
             << "g_in_J=" << bool_text(result.g_in_J) << '\n'
             << "derivatives_in_J=" << bool_text(result.derivatives_in_J)
             << '\n'
             << "cycle_valid=" << bool_text(result.cycle_valid) << '\n'
             << "multiplication_rank=" << result.multiplication_rank << '\n'
             << "full_rank_candidate="
             << bool_text(result.full_column_rank_candidate) << '\n'
             << "certified_faithful="
             << bool_text(result.certified_faithful) << '\n';
      if (result.detail.has_value()) {
        output << "detail=" << *result.detail << '\n';
      }
      if (result.status ==
          laughableengine::DiscoveryScreenStatus::ResourceLimit) {
        exit_code = 3;
      } else if (result.status ==
                 laughableengine::DiscoveryScreenStatus::InvalidInput) {
        exit_code = 2;
      }
      return output.str();
    }
  }

  if (options.command == "screen-h1") {
    reject_divisors(options);
    reject_derivative_variable(options);
    reject_new_options(options);
    if constexpr (!std::is_same_v<Field, PrimeField>) {
      throw UsageError("screen-h1 is a finite-field discovery command");
    } else {
      const auto result = screen_full_h1(parse_ideal(ring, options));
      const bool rank_inconclusive =
          result.rank_proof == MatrixSpaceRankProof::ResourceLimit ||
          result.rank_proof == MatrixSpaceRankProof::GenericOnly;
      output << "status=" << discovery_status_text(result.status) << '\n'
             << "conclusive="
             << bool_text(
                    result.status ==
                            laughableengine::DiscoveryScreenStatus::Complete &&
                    !rank_inconclusive)
             << '\n'
             << "length_Q=" << result.length_Q << '\n'
             << "length_P_mod_J2=" << result.length_P_mod_J2 << '\n'
             << "conormal_dimension=" << result.conormal_dimension << '\n'
             << "h1_dimension=" << result.h1_dimension << '\n'
             << "socle_dimension=" << result.socle_dimension << '\n'
             << "individual_rank_lower="
             << result.maximum_individual_rank_lower_bound << '\n'
             << "individual_rank_upper="
             << result.maximum_individual_rank_upper_bound << '\n'
             << "individual_rank_proof="
             << rank_proof_text(result.rank_proof) << '\n'
             << "full_rank_candidate="
             << bool_text(result.full_socle_rank_candidate) << '\n'
             << "certified_faithful="
             << bool_text(result.certified_faithful) << '\n'
             << "witness=";
      if (result.witness.has_value()) {
        output << *result.witness;
      } else {
        output << "n/a";
      }
      output << '\n';
      if (result.detail.has_value()) {
        output << "detail=" << *result.detail << '\n';
      }
      if (result.status ==
              laughableengine::DiscoveryScreenStatus::ResourceLimit ||
          rank_inconclusive) {
        exit_code = 3;
      } else if (result.status ==
                 laughableengine::DiscoveryScreenStatus::InvalidInput) {
        exit_code = 2;
      }
      return output.str();
    }
  }

  if (options.command == "audit") {
    reject_divisors(options);
    reject_derivative_variable(options);
    reject_new_options(options, true);
    require_colon_polynomial(options);
    const auto ideal = parse_ideal(ring, options);
    const auto polynomial = parse_input(ring, options.colon_polynomial);
    const auto audit = audit_cycle(ideal, polynomial);

    std::optional<std::size_t> square_length;
    if (audit.ideal_square().has_value() &&
        audit.ideal_square()->is_zero_dimensional()) {
      square_length = audit.ideal_square()->quotient_dimension();
    }
    std::optional<std::size_t> annihilator_dimension;
    std::optional<std::size_t> colon_length;
    if (audit.colon_evidence().has_value()) {
      annihilator_dimension =
          audit.colon_evidence()->annihilator_dimension();
      colon_length = audit.colon_evidence()->colon_quotient_length();
    }

    output << "status=" << cycle_status_text(audit.status()) << '\n'
           << "conclusive=" << bool_text(audit.conclusive()) << '\n'
           << "finite_quotient=" << bool_text(audit.finite_quotient()) << '\n'
           << "supported_at_origin="
           << bool_text(audit.supported_at_origin()) << '\n'
           << "length_Q=" << optional_size_text(audit.quotient_length())
           << '\n'
           << "length_P_mod_J2=" << optional_size_text(square_length) << '\n'
           << "g_in_J=" << optional_bool_text(audit.polynomial_in_ideal())
           << '\n'
           << "derivatives_in_J="
           << optional_bool_text(audit.derivatives_in_ideal()) << '\n'
           << "cycle_valid=" << optional_bool_text(audit.cycle_valid())
           << '\n'
           << "primitive=" << optional_bool_text(audit.primitive()) << '\n'
           << "annihilator_dimension="
           << optional_size_text(annihilator_dimension) << '\n'
           << "colon_quotient_length=" << optional_size_text(colon_length)
           << '\n'
           << "colon_equals_J="
           << optional_bool_text(audit.colon_equals_ideal()) << '\n'
           << "annihilator_zero="
           << optional_bool_text(audit.annihilator_zero()) << '\n'
           << "faithful_cycle=" << bool_text(audit.faithful_cycle()) << '\n'
           << "ideal_gb=" << basis_text(audit.ideal().groebner_basis())
           << '\n'
           << "ideal_square_gb=";

    if (audit.ideal_square().has_value()) {
      output << basis_text(audit.ideal_square()->groebner_basis());
    } else {
      output << "n/a";
    }
    output << '\n';

    if (audit.membership_evidence().has_value()) {
      output << "g_remainder="
             << audit.membership_evidence()->polynomial_remainder() << '\n';
      const auto& remainders =
          audit.membership_evidence()->derivative_remainders();
      for (std::size_t variable = 0; variable < remainders.size(); ++variable) {
        output << "derivative_remainder[" << ring.variable_names()[variable]
               << "]=" << remainders[variable] << '\n';
      }
    }
    if (audit.colon_evidence().has_value()) {
      output << "colon_gb="
             << basis_text(
                    audit.colon_evidence()->colon_ideal().groebner_basis())
             << '\n'
             << "ideal_in_colon_remainders="
             << basis_text(
                    audit.colon_evidence()->ideal_in_colon_remainders())
             << '\n'
             << "colon_in_ideal_remainders="
             << basis_text(
                    audit.colon_evidence()->colon_in_ideal_remainders())
             << '\n';
    } else {
      output << "colon_gb=n/a\n"
             << "ideal_in_colon_remainders=n/a\n"
             << "colon_in_ideal_remainders=n/a\n";
    }
    if (audit.resource_detail().has_value()) {
      output << "resource_detail=" << *audit.resource_detail() << '\n';
    }
    if (!audit.conclusive()) {
      exit_code = 3;
    }
    return output.str();
  }

  if (options.command == "closure") {
    reject_divisors(options);
    reject_derivative_variable(options);
    reject_new_options(options, true, false, true);
    require_colon_polynomial(options);
    const auto ideal = parse_ideal(ring, options);
    const auto polynomial = parse_input(ring, options.colon_polynomial);
    ColonClosureLimits limits;
    limits.max_steps = options.max_steps;
    const auto closure = colon_closure(ideal, polynomial, limits);

    output << "status=" << closure_status_text(closure.status()) << '\n'
           << "conclusive=" << bool_text(closure.conclusive()) << '\n'
           << "faithful_fixed_point="
           << bool_text(closure.faithful_fixed_point_found()) << '\n'
           << "steps=" << closure.steps().size() << '\n'
           << "transitions=" << closure.transitions().size() << '\n'
           << "length_chain=";
    for (std::size_t index = 0; index < closure.steps().size(); ++index) {
      if (index != 0) {
        output << "->";
      }
      output << optional_size_text(closure.steps()[index].quotient_length());
    }
    output << '\n';
    for (const auto& step : closure.steps()) {
      output << "step[" << step.index() << "].gb="
             << basis_text(step.ideal().groebner_basis()) << '\n';
    }
    for (std::size_t index = 0; index < closure.transitions().size();
         ++index) {
      const auto& transition = closure.transitions()[index];
      output << "transition[" << index << "].current_subset_next="
             << bool_text(transition.current_subset_next()) << '\n'
             << "transition[" << index << "].equal="
             << bool_text(transition.equal()) << '\n'
             << "transition[" << index
             << "].current_in_next_remainders="
             << basis_text(transition.current_in_next_remainders()) << '\n'
             << "transition[" << index
             << "].next_in_current_remainders="
             << basis_text(transition.next_in_current_remainders()) << '\n';
    }
    if (closure.resource_detail().has_value()) {
      output << "resource_detail=" << *closure.resource_detail() << '\n';
    }
    if (!closure.conclusive()) {
      exit_code = 3;
    }
    return output.str();
  }

  if (options.command == "h1") {
    reject_divisors(options);
    reject_derivative_variable(options);
    reject_new_options(options);
    const auto data = full_h1_action(parse_ideal(ring, options));
    const bool faithful_replay =
        data.faithful_witness_audit.has_value() &&
        data.faithful_witness_audit->faithful_cycle();
    const bool rank_inconclusive =
        data.individual_rank.proof == MatrixSpaceRankProof::ResourceLimit ||
        data.individual_rank.proof == MatrixSpaceRankProof::GenericOnly;
    const bool replay_inconclusive =
        data.faithful_witness_audit.has_value() &&
        !data.faithful_witness_audit->conclusive();
    const bool resource_inconclusive =
        data.individual_rank.proof == MatrixSpaceRankProof::ResourceLimit ||
        replay_inconclusive;
    const auto h1_status =
        resource_inconclusive
            ? "resource-limit"
            : (data.individual_rank.proof == MatrixSpaceRankProof::GenericOnly
                   ? "generic-only"
                   : "complete");

    output << "status=" << h1_status << '\n'
           << "conclusive="
           << bool_text(!rank_inconclusive && !replay_inconclusive) << '\n'
           << "length_Q=" << data.length_Q << '\n'
           << "length_P_mod_J2=" << data.length_P_mod_J2 << '\n'
           << "conormal_dimension=" << data.conormal_dimension << '\n'
           << "h1_dimension=" << data.h1_dimension << '\n'
           << "socle_dimension=" << data.socle_dimension << '\n'
           << "common_product_space_dimension="
           << data.common_product_space_dimension << '\n'
           << "common_product_space_rank_bound="
           << data.common_product_space_rank_bound << '\n'
           << "individual_rank_lower=" << data.individual_rank.lower_bound
           << '\n'
           << "individual_rank_upper=" << data.individual_rank.upper_bound
           << '\n'
           << "individual_rank_exact="
           << optional_size_text(data.individual_rank.exact_maximum) << '\n'
           << "individual_rank_proof="
           << rank_proof_text(data.individual_rank.proof) << '\n'
           << "full_socle_rank_witness="
           << bool_text(data.individual_rank.has_full_column_rank_witness())
           << '\n'
           << "best_h1_witness=";
    if (data.best_h1_polynomial.has_value()) {
      output << *data.best_h1_polynomial;
    } else {
      output << "n/a";
    }
    output << '\n'
           << "independent_faithful_colon_replay="
           << bool_text(faithful_replay) << '\n'
           << "faithful_witness_replay_status=";
    if (data.faithful_witness_audit.has_value()) {
      output << cycle_status_text(data.faithful_witness_audit->status());
    } else {
      output << "n/a";
    }
    output << '\n' << "faithful_witness_replay_conclusive=";
    if (data.faithful_witness_audit.has_value()) {
      output << bool_text(data.faithful_witness_audit->conclusive());
    } else {
      output << "n/a";
    }
    output << '\n' << "faithful_witness_replay_resource_detail=";
    if (data.faithful_witness_audit.has_value() &&
        data.faithful_witness_audit->resource_detail().has_value()) {
      output << *data.faithful_witness_audit->resource_detail();
    } else {
      output << "n/a";
    }
    output << '\n'
           << "common_annihilator_diagnostic_only=true\n"
           << "common_annihilator_diagnostic_gb="
           << basis_text(
                  data.common_annihilator_diagnostic.groebner_basis())
           << '\n'
           << "conormal_basis=" << basis_text(data.conormal_basis) << '\n'
           << "h1_basis=" << basis_text(data.h1_basis) << '\n'
           << "socle_basis=" << basis_text(data.socle_basis) << '\n'
           << "action_matrix_count=" << data.action_matrices.size() << '\n';
    for (std::size_t index = 0; index < data.action_matrices.size(); ++index) {
      const auto& matrix = data.action_matrices[index];
      output << "action_matrix[" << index << "].shape="
             << matrix.row_count() << 'x' << matrix.column_count() << '\n'
             << "action_matrix[" << index << "].entries="
             << matrix_entries_text(matrix) << '\n';
    }
    if (rank_inconclusive || replay_inconclusive) {
      exit_code = 3;
    }
    return output.str();
  }

  throw UsageError("unknown command: " + options.command);
}

[[nodiscard]] std::string execute(const Options& options, int& exit_code) {
  const auto field = lowercase(trim(options.field));
  if (field == "qq") {
    return run_command(options, QQ(), exit_code);
  }
  return run_command(
      options, GF(parse_prime_field(options.field)), exit_code);
}

void print_help(std::ostream& stream) {
  stream <<
      "laughable " LAUGHABLEENGINE_VERSION " - small native exact-algebra engine\n"
      "\n"
      "Algebra commands:\n"
      "  laughable [options] print POLYNOMIAL\n"
      "  laughable [options] diff --var NAME POLYNOMIAL\n"
      "  laughable [options] nf POLYNOMIAL --by DIVISOR [--by DIVISOR ...]\n"
      "  laughable [options] divide POLYNOMIAL --by DIVISOR [--by DIVISOR ...]\n"
      "  laughable [options] gb GENERATOR [GENERATOR ...]\n"
      "  laughable [options] check-gb CANDIDATE [CANDIDATE ...]\n"
      "  laughable [options] dim GENERATOR [GENERATOR ...]\n"
      "  laughable [options] std GENERATOR [GENERATOR ...]\n"
      "  laughable [options] colon --g POLYNOMIAL GENERATOR [GENERATOR ...]\n"
      "  laughable [options] eliminate --elim x[,y...] GENERATOR [...]\n"
      "  laughable [options] inverse-system DUAL [DUAL ...]\n"
      "\n"
      "Structured cotangent commands:\n"
      "  laughable [options] cotangent-h1 --maximal-power N "
      "[GENERATOR ...]\n"
      "  laughable [options] verify-h1-class --maximal-power N "
      "--g POLYNOMIAL [GENERATOR ...]\n"
      "  laughable [options] h1 GENERATOR [GENERATOR ...]\n"
      "\n"
      "Research workflow commands:\n"
      "  laughable [options] audit --g POLYNOMIAL GENERATOR [GENERATOR ...]\n"
      "  laughable [options] screen-audit --g POLYNOMIAL GENERATOR [...]\n"
      "  laughable [options] certificate --g POLYNOMIAL GENERATOR [...]\n"
      "  laughable [options] closure --g POLYNOMIAL [--max-steps N] "
      "GENERATOR [...]\n"
      "  laughable [options] screen-h1 GENERATOR [GENERATOR ...]\n"
      "\n"
      "Options:\n"
      "  --field QQ|GF(p)   coefficient field (default: QQ)\n"
      "  --vars x,y,z       ordered variable names (required)\n"
      "  --order grevlex|lex term order (default: grevlex)\n"
      "  --by POLYNOMIAL    ordered divisor; repeat as needed\n"
      "  --var NAME         differentiation variable\n"
      "  --g POLYNOMIAL     principal-colon or distinguished-cycle element\n"
      "  --elim x[,y...]    one to three variables to eliminate\n"
      "  --max-steps N      closure transition bound (default: 16)\n"
      "  --maximal-power N  N in J=(GENERATORS)+(x_1,...,x_e)^N\n"
      "  --apolarity MODE   ordinary or divided (inverse-system only)\n"
      "  -h, --help         show this help\n"
      "  --version          show the version\n"
      "\n"
      "Exit status:\n"
      "  0  completed command (including a conclusive negative result)\n"
      "  1  unexpected internal failure\n"
      "  2  invalid command, input, or mathematical domain\n"
      "  3  computation was inconclusive because of a resource limit\n"
      "\n"
      "Expressions support integers, field fractions, variables, (), +, -, *,\n"
      "scalar /, and ^ or ** powers. Quote expressions so the shell does not\n"
      "expand '*'. There is deliberately no implicit multiplication.\n"
      "\n"
      "Examples:\n"
      "  laughable --field QQ --vars x,y dim 'x^2' 'y^3'\n"
      "  laughable --field 'GF(5)' --vars x audit --g 'x^5' 'x^5'\n"
      "  laughable --field QQ --vars x,y,z eliminate --elim x "
      "'x*y-1' 'x-z'\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    if (options.version) {
      std::cout << "laughable " LAUGHABLEENGINE_VERSION << '\n';
      return 0;
    }
    if (options.help || options.command.empty()) {
      print_help(std::cout);
      return 0;
    }
    int exit_code = 0;
    std::cout << execute(options, exit_code);
    return exit_code;
  } catch (const InputParseError& error) {
    std::cerr << "error: " << error.what() << '\n'
              << "  " << error.source() << '\n'
              << "  " << std::string(error.offset(), ' ') << "^\n";
    return 2;
  } catch (const UsageError& error) {
    std::cerr << "error: " << error.what() << "\nTry 'laughable --help'.\n";
    return 2;
  } catch (const std::invalid_argument& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  } catch (const std::domain_error& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  } catch (const std::overflow_error& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  } catch (const ResourceLimitExceeded& error) {
    std::cerr << "resource limit: " << error.what() << '\n';
    return 3;
  } catch (const FullH1ResourceLimit& error) {
    std::cerr << "resource limit: " << error.what() << '\n';
    return 3;
  } catch (const InverseSystemResourceLimit& error) {
    std::cerr << "resource limit: " << error.what() << '\n';
    return 3;
  } catch (const laughableengine::CotangentH1ResourceLimit& error) {
    std::cerr << "resource limit: " << error.what() << '\n';
    return 3;
  } catch (const std::exception& error) {
    std::cerr << "internal error: " << error.what() << '\n';
    return 1;
  }
}

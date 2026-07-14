#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "laughableengine/candidate_executor.hpp"
#include "laughableengine/certificate.hpp"
#include "laughableengine/cotangent_h1.hpp"
#include "laughableengine/cycle_audit.hpp"
#include "laughableengine/discovery.hpp"
#include "laughableengine/finite_algebra.hpp"
#include "laughableengine/groebner.hpp"
#include "laughableengine/h1.hpp"
#include "laughableengine/ideal.hpp"
#include "laughableengine/inverse_discovery.hpp"
#include "laughableengine/inverse_system.hpp"
#include "laughableengine/parser.hpp"
#include "laughableengine/polynomial.hpp"
#include "laughableengine/quotient.hpp"

namespace py = pybind11;
namespace le = laughableengine;

namespace {

using QQRing = le::PolynomialRing<le::RationalField>;
using GFRing = le::PolynomialRing<le::PrimeField>;
using QQPolynomial = le::Polynomial<le::RationalField>;
using GFPolynomial = le::Polynomial<le::PrimeField>;
using QQIdeal = le::Ideal<le::RationalField>;
using GFIdeal = le::Ideal<le::PrimeField>;
using QQCotangentH1 = le::CotangentH1Presentation<le::RationalField>;
using GFCotangentH1 = le::CotangentH1Presentation<le::PrimeField>;
using QQCotangentClassProof = le::CotangentClassProof<le::RationalField>;
using GFCotangentClassProof = le::CotangentClassProof<le::PrimeField>;
using QQLocalIdeal = le::OriginPowerIdeal<le::RationalField>;
using GFLocalIdeal = le::OriginPowerIdeal<le::PrimeField>;
using QQFiniteQuotient = le::FiniteQuotient<le::RationalField>;
using GFFiniteQuotient = le::FiniteQuotient<le::PrimeField>;
using QQConormalModule = le::ConormalModule<le::RationalField>;
using GFConormalModule = le::ConormalModule<le::PrimeField>;
using QQConormalDerivativeMap =
    le::ConormalDerivativeMap<le::RationalField>;
using GFConormalDerivativeMap = le::ConormalDerivativeMap<le::PrimeField>;
using QQH1Module = le::CotangentH1Module<le::RationalField>;
using GFH1Module = le::CotangentH1Module<le::PrimeField>;
using QQH1Element = le::CotangentH1Element<le::RationalField>;
using GFH1Element = le::CotangentH1Element<le::PrimeField>;
using QQQuotientIdeal = le::QuotientIdeal<le::RationalField>;
using GFQuotientIdeal = le::QuotientIdeal<le::PrimeField>;
using QQIdealPreimage = le::IdealPreimage<le::RationalField>;
using GFIdealPreimage = le::IdealPreimage<le::PrimeField>;
using QQSparseMatrix = le::SparseMatrix<le::RationalField>;
using GFSparseMatrix = le::SparseMatrix<le::PrimeField>;

class PythonRing;
class PythonIdeal;
class PythonCotangentH1Presentation;
class PythonCotangentClassProof;
class PythonSparseExactMatrix;
class PythonLocalIdeal;
class PythonFiniteQuotient;
class PythonConormalModule;
class PythonConormalDerivativeMap;
class PythonH1Module;
class PythonH1Element;
class PythonQuotientIdeal;
class PythonIdealPreimage;
struct PythonExactMatrix;
struct PythonCycleAuditResult;
struct PythonColonClosureResult;
struct PythonH1ActionResult;
struct PythonInverseSystemResult;
struct PythonCycleScreenResult;
struct PythonH1ScreenResult;
struct PythonInverseSystemDiscoveryRecord;

template <typename Function>
auto without_gil(Function&& function) {
  py::gil_scoped_release release;
  return std::invoke(std::forward<Function>(function));
}

template <typename Function>
auto cotangent_without_gil(Function&& function) {
  try {
    return without_gil(std::forward<Function>(function));
  } catch (const le::TruncatedMonomialSpaceResourceLimit& error) {
    throw le::CotangentH1ResourceLimit(error.what());
  } catch (const le::IncrementalSparseRowSpaceResourceLimit& error) {
    throw le::CotangentH1ResourceLimit(error.what());
  } catch (const le::SparseEliminationResourceLimit& error) {
    throw le::CotangentH1ResourceLimit(error.what());
  }
}

class PythonPolynomial {
 public:
  using Storage = std::variant<QQPolynomial, GFPolynomial>;

  [[nodiscard]] const std::shared_ptr<PythonRing>& ring() const noexcept {
    return ring_;
  }

  [[nodiscard]] std::string to_string() const {
    return std::visit(
        [](const auto& polynomial) { return polynomial.to_string(); }, value_);
  }

  [[nodiscard]] bool is_zero() const noexcept {
    return std::visit(
        [](const auto& polynomial) { return polynomial.is_zero(); }, value_);
  }

  [[nodiscard]] std::size_t term_count() const noexcept {
    return std::visit(
        [](const auto& polynomial) { return polynomial.term_count(); }, value_);
  }

  [[nodiscard]] std::optional<std::uint32_t> degree() const noexcept {
    return std::visit(
        [](const auto& polynomial) { return polynomial.total_degree(); }, value_);
  }

  [[nodiscard]] PythonPolynomial add(const PythonPolynomial& other) const;
  [[nodiscard]] PythonPolynomial subtract(const PythonPolynomial& other) const;
  [[nodiscard]] PythonPolynomial multiply(const PythonPolynomial& other) const;
  [[nodiscard]] PythonPolynomial negate() const;
  [[nodiscard]] PythonPolynomial pow(std::uint64_t exponent) const;
  [[nodiscard]] PythonPolynomial monic() const;
  [[nodiscard]] PythonPolynomial derivative(std::size_t variable) const;
  [[nodiscard]] PythonPolynomial derivative(std::string_view variable) const;
  [[nodiscard]] PythonPolynomial derivative(
      const PythonPolynomial& variable) const;
  [[nodiscard]] bool equals(const PythonPolynomial& other) const noexcept;

 private:
  PythonPolynomial(std::shared_ptr<PythonRing> ring, Storage value)
      : ring_(std::move(ring)), value_(std::move(value)) {}

  void require_same_ring(const PythonPolynomial& other) const;

  std::shared_ptr<PythonRing> ring_;
  Storage value_;

  friend class PythonRing;
  friend class PythonIdeal;
  friend class PythonCotangentH1Presentation;
  friend class PythonFiniteQuotient;
  friend class PythonH1Module;
};

struct PythonDivisionResult {
  std::vector<PythonPolynomial> quotients;
  PythonPolynomial remainder;
};

class PythonRing : public std::enable_shared_from_this<PythonRing> {
 public:
  using Storage = std::variant<QQRing, GFRing>;

  static std::shared_ptr<PythonRing> make_qq(
      std::vector<std::string> variables,
      le::Order order) {
    return std::shared_ptr<PythonRing>(new PythonRing(
        QQRing(le::QQ(), std::move(variables), order)));
  }

  static std::shared_ptr<PythonRing> make_gf(
      std::uint32_t modulus,
      std::vector<std::string> variables,
      le::Order order) {
    return std::shared_ptr<PythonRing>(new PythonRing(
        GFRing(le::GF(modulus), std::move(variables), order)));
  }

  [[nodiscard]] const std::vector<std::string>& variable_names() const {
    return std::visit(
        [](const auto& ring) -> const std::vector<std::string>& {
          return ring.variable_names();
        },
        value_);
  }

  [[nodiscard]] le::Order order_value() const noexcept {
    return std::visit([](const auto& ring) { return ring.order(); }, value_);
  }

  [[nodiscard]] std::string order_name() const {
    return order_value() == le::Order::Lex ? "lex" : "grevlex";
  }

  [[nodiscard]] std::uint32_t characteristic() const noexcept {
    if (const auto* ring = std::get_if<GFRing>(&value_)) {
      return ring->field().modulus();
    }
    return 0;
  }

  [[nodiscard]] std::string coefficient_field() const {
    if (const auto* ring = std::get_if<GFRing>(&value_)) {
      return "GF(" + std::to_string(ring->field().modulus()) + ")";
    }
    return "QQ";
  }

  [[nodiscard]] std::string to_string() const {
    std::ostringstream output;
    output << coefficient_field() << '[';
    const auto& names = variable_names();
    for (std::size_t index = 0; index < names.size(); ++index) {
      if (index != 0) {
        output << ", ";
      }
      output << names[index];
    }
    output << "; order=" << order_name() << ']';
    return output.str();
  }

  [[nodiscard]] PythonPolynomial parse(std::string_view expression) {
    if (const auto* ring = std::get_if<QQRing>(&value_)) {
      return wrap(le::parse_polynomial(*ring, expression));
    }
    return wrap(le::parse_polynomial(std::get<GFRing>(value_), expression));
  }

  [[nodiscard]] PythonPolynomial integer(long value) {
    if (const auto* ring = std::get_if<QQRing>(&value_)) {
      return wrap(ring->integer(value));
    }
    return wrap(std::get<GFRing>(value_).integer(value));
  }

  [[nodiscard]] PythonPolynomial zero() { return integer(0); }
  [[nodiscard]] PythonPolynomial one() { return integer(1); }

  [[nodiscard]] PythonPolynomial gen(std::size_t variable) {
    if (const auto* ring = std::get_if<QQRing>(&value_)) {
      return wrap(ring->gen(variable));
    }
    return wrap(std::get<GFRing>(value_).gen(variable));
  }

  [[nodiscard]] PythonPolynomial gen(std::string_view variable) {
    if (const auto* ring = std::get_if<QQRing>(&value_)) {
      return wrap(ring->gen(variable));
    }
    return wrap(std::get<GFRing>(value_).gen(variable));
  }

  [[nodiscard]] std::vector<PythonPolynomial> gens() {
    std::vector<PythonPolynomial> result;
    result.reserve(variable_names().size());
    for (std::size_t index = 0; index < variable_names().size(); ++index) {
      result.push_back(gen(index));
    }
    return result;
  }

  [[nodiscard]] PythonPolynomial coerce(const PythonPolynomial& polynomial) {
    require_polynomial(polynomial);
    return polynomial;
  }

  [[nodiscard]] std::vector<PythonPolynomial> groebner_basis(
      const std::vector<PythonPolynomial>& generators) {
    if (std::holds_alternative<QQRing>(value_)) {
      const auto concrete = concrete_polynomials<QQPolynomial>(generators);
      auto result = without_gil(
          [&] { return le::groebner_basis(concrete); });
      return wrap_all(std::move(result));
    }
    const auto concrete = concrete_polynomials<GFPolynomial>(generators);
    auto result = without_gil(
        [&] { return le::groebner_basis(concrete); });
    return wrap_all(std::move(result));
  }

  [[nodiscard]] PythonPolynomial normal_form(
      const PythonPolynomial& polynomial,
      const std::vector<PythonPolynomial>& divisors) {
    require_polynomial(polynomial);
    if (std::holds_alternative<QQRing>(value_)) {
      const auto input = std::get<QQPolynomial>(polynomial.value_);
      const auto concrete_divisors =
          concrete_polynomials<QQPolynomial>(divisors);
      auto result = without_gil(
          [&] { return le::normal_form(input, concrete_divisors); });
      return wrap(std::move(result));
    }
    const auto input = std::get<GFPolynomial>(polynomial.value_);
    const auto concrete_divisors =
        concrete_polynomials<GFPolynomial>(divisors);
    auto result = without_gil(
        [&] { return le::normal_form(input, concrete_divisors); });
    return wrap(std::move(result));
  }

  [[nodiscard]] PythonDivisionResult divide(
      const PythonPolynomial& polynomial,
      const std::vector<PythonPolynomial>& divisors) {
    require_polynomial(polynomial);
    if (std::holds_alternative<QQRing>(value_)) {
      const auto input = std::get<QQPolynomial>(polynomial.value_);
      const auto concrete_divisors =
          concrete_polynomials<QQPolynomial>(divisors);
      auto result = without_gil(
          [&] { return le::divide(input, concrete_divisors); });
      return PythonDivisionResult{
          wrap_all(std::move(result.quotients)), wrap(std::move(result.remainder))};
    }
    const auto input = std::get<GFPolynomial>(polynomial.value_);
    const auto concrete_divisors =
        concrete_polynomials<GFPolynomial>(divisors);
    auto result = without_gil(
        [&] { return le::divide(input, concrete_divisors); });
    return PythonDivisionResult{
        wrap_all(std::move(result.quotients)), wrap(std::move(result.remainder))};
  }

  [[nodiscard]] bool is_groebner_basis(
      const std::vector<PythonPolynomial>& candidates) const {
    if (std::holds_alternative<QQRing>(value_)) {
      const auto concrete = concrete_polynomials<QQPolynomial>(candidates);
      return without_gil(
          [&] { return le::is_groebner_basis(concrete); });
    }
    const auto concrete = concrete_polynomials<GFPolynomial>(candidates);
    return without_gil(
        [&] { return le::is_groebner_basis(concrete); });
  }

  [[nodiscard]] std::vector<PythonPolynomial> normal_forms(
      const std::vector<PythonPolynomial>& polynomials,
      const std::vector<PythonPolynomial>& divisors);
  [[nodiscard]] PythonCotangentH1Presentation cotangent_h1(
      const std::vector<PythonPolynomial>& generators,
      std::size_t maximal_power,
      std::size_t max_monomials,
      std::optional<std::size_t> max_generated_rows,
      std::optional<std::size_t> max_matrix_triplets);
  [[nodiscard]] PythonLocalIdeal local_ideal(
      const std::vector<PythonPolynomial>& generators,
      std::size_t maximal_power);
  [[nodiscard]] PythonFiniteQuotient quotient(
      const PythonLocalIdeal& ideal,
      std::size_t max_monomials,
      std::optional<std::size_t> max_generated_rows,
      std::optional<std::size_t> max_matrix_triplets);
  [[nodiscard]] PythonIdeal ideal(
      const std::vector<PythonPolynomial>& generators);
  [[nodiscard]] PythonCycleAuditResult audit_cycle(
      const PythonIdeal& ideal,
      const PythonPolynomial& polynomial,
      std::optional<std::size_t> max_matrix_entries);
  [[nodiscard]] PythonColonClosureResult colon_closure(
      const PythonIdeal& ideal,
      const PythonPolynomial& polynomial,
      std::size_t max_steps,
      std::optional<std::size_t> max_matrix_entries);
  [[nodiscard]] PythonH1ActionResult full_h1_action(
      const PythonIdeal& ideal,
      std::optional<std::size_t> max_matrix_entries,
      std::optional<std::size_t> max_witness_audit_matrix_entries,
      std::optional<std::size_t> max_minors,
      std::optional<std::size_t> max_parameter_terms,
      std::optional<std::size_t> max_determinant_products,
      std::size_t max_finite_field_evaluations,
      std::optional<std::size_t> max_groebner_critical_pairs,
      std::optional<std::size_t> max_groebner_basis_polynomials,
      std::optional<std::size_t> max_groebner_reduction_steps,
      std::optional<std::size_t> max_groebner_live_terms,
      std::optional<std::size_t> max_groebner_basis_terms);
  [[nodiscard]] PythonInverseSystemResult macaulay_annihilator(
      const std::vector<PythonPolynomial>& dual_generators,
      le::ApolarityConvention convention,
      const le::InverseSystemLimits& limits);
  [[nodiscard]] PythonCycleScreenResult screen_cycle(
      const PythonIdeal& ideal,
      const PythonPolynomial& polynomial,
      const le::PackedCycleScreenOptions& options);
  [[nodiscard]] PythonH1ScreenResult screen_full_h1(
      const PythonIdeal& ideal,
      const le::PackedH1ScreenOptions& options);
  [[nodiscard]] std::vector<PythonCycleScreenResult> screen_cycles_parallel(
      const PythonIdeal& ideal,
      const std::vector<PythonPolynomial>& candidates,
      const le::PackedCycleScreenOptions& screen_options,
      const le::CandidateExecutorOptions& executor_options);
  [[nodiscard]] std::vector<PythonInverseSystemDiscoveryRecord>
  search_inverse_systems(
      const std::vector<std::vector<PythonPolynomial>>& candidates,
      const le::InverseSystemDiscoveryOptions& discovery_options,
      const le::CandidateExecutorOptions& executor_options);
  [[nodiscard]] std::string make_jg_certificate(
      const std::vector<PythonPolynomial>& ideal_generators,
      const PythonPolynomial& polynomial);

 private:
  explicit PythonRing(QQRing ring) : value_(std::move(ring)) {}
  explicit PythonRing(GFRing ring) : value_(std::move(ring)) {}

  void require_polynomial(const PythonPolynomial& polynomial) const {
    if (polynomial.ring_.get() != this) {
      throw std::invalid_argument(
          "polynomials must belong to this exact ring context");
    }
  }

  template <typename ConcretePolynomial>
  [[nodiscard]] std::vector<ConcretePolynomial> concrete_polynomials(
      const std::vector<PythonPolynomial>& polynomials) const {
    std::vector<ConcretePolynomial> result;
    result.reserve(polynomials.size());
    for (const auto& polynomial : polynomials) {
      require_polynomial(polynomial);
      result.push_back(std::get<ConcretePolynomial>(polynomial.value_));
    }
    return result;
  }

  void require_ideal(const PythonIdeal& ideal) const;

 public:

  [[nodiscard]] PythonPolynomial wrap(QQPolynomial polynomial) {
    return PythonPolynomial(
        shared_from_this(), PythonPolynomial::Storage(std::move(polynomial)));
  }

  [[nodiscard]] PythonPolynomial wrap(GFPolynomial polynomial) {
    return PythonPolynomial(
        shared_from_this(), PythonPolynomial::Storage(std::move(polynomial)));
  }

  [[nodiscard]] std::vector<PythonPolynomial> wrap_all(
      std::vector<QQPolynomial> polynomials) {
    std::vector<PythonPolynomial> result;
    result.reserve(polynomials.size());
    for (auto& polynomial : polynomials) {
      result.push_back(wrap(std::move(polynomial)));
    }
    return result;
  }

  [[nodiscard]] std::vector<PythonPolynomial> wrap_all(
      std::vector<GFPolynomial> polynomials) {
    std::vector<PythonPolynomial> result;
    result.reserve(polynomials.size());
    for (auto& polynomial : polynomials) {
      result.push_back(wrap(std::move(polynomial)));
    }
    return result;
  }

  [[nodiscard]] PythonIdeal wrap(QQIdeal ideal);
  [[nodiscard]] PythonIdeal wrap(GFIdeal ideal);

 private:
  Storage value_;
};

class PythonIdeal {
 public:
  using Storage = std::variant<QQIdeal, GFIdeal>;

  [[nodiscard]] const std::shared_ptr<PythonRing>& ring() const noexcept {
    return ring_;
  }
  [[nodiscard]] std::vector<PythonPolynomial> generators() const;
  [[nodiscard]] PythonPolynomial normal_form(
      const PythonPolynomial& polynomial) const;
  [[nodiscard]] std::vector<PythonPolynomial> normal_forms(
      const std::vector<PythonPolynomial>& polynomials) const;
  [[nodiscard]] bool contains(const PythonPolynomial& polynomial) const;
  [[nodiscard]] bool is_zero() const noexcept {
    return std::visit([](const auto& ideal) { return ideal.is_zero(); }, value_);
  }
  [[nodiscard]] bool is_unit() const noexcept {
    return std::visit([](const auto& ideal) { return ideal.is_unit(); }, value_);
  }
  [[nodiscard]] bool is_zero_dimensional() const;
  [[nodiscard]] bool supported_at_origin() const;
  [[nodiscard]] std::size_t dimension() const;
  [[nodiscard]] std::vector<PythonPolynomial> standard_monomials() const;
  [[nodiscard]] PythonIdeal add(const PythonIdeal& other) const;
  [[nodiscard]] PythonIdeal multiply(const PythonIdeal& other) const;
  [[nodiscard]] PythonIdeal square() const;
  [[nodiscard]] PythonIdeal colon(const PythonPolynomial& polynomial) const;
  [[nodiscard]] PythonIdeal eliminate(
      const std::vector<std::size_t>& variables) const;
  [[nodiscard]] PythonIdeal eliminate(
      const std::vector<std::string>& variables) const;
  [[nodiscard]] bool equals(const PythonIdeal& other) const noexcept;
  [[nodiscard]] std::string to_string() const;

 private:
  PythonIdeal(std::shared_ptr<PythonRing> ring, Storage value)
      : ring_(std::move(ring)), value_(std::move(value)) {}

  void require_same_ring(const PythonIdeal& other) const;

  std::shared_ptr<PythonRing> ring_;
  Storage value_;

  friend class PythonRing;
};

void require_coordinate_budget(
    std::size_t left,
    std::size_t right,
    std::size_t maximum,
    std::string_view description) {
  const auto required =
      right != 0 && left > std::numeric_limits<std::size_t>::max() / right
          ? std::numeric_limits<std::size_t>::max()
          : left * right;
  if (required > maximum) {
    throw le::CotangentH1ResourceLimit(
        std::string(description) + " needs " + std::to_string(required) +
        " coordinate entries, exceeding max_coordinate_entries=" +
        std::to_string(maximum));
  }
}

[[nodiscard]] py::object python_exact_value(
    const le::RationalField& field,
    const le::RationalField::Element& value) {
  return py::module_::import("fractions")
      .attr("Fraction")(py::str(field.to_string(value)));
}

[[nodiscard]] py::object python_exact_value(
    const le::PrimeField& field,
    const le::PrimeField::Element& value) {
  return py::int_(field.canonical(value).value());
}

// A non-owning matrix view whose aliasing shared_ptr keeps its presentation or
// proof alive.  Entry materialization is deliberately opt-in.
class PythonSparseExactMatrix {
 public:
  using Storage = std::variant<
      std::shared_ptr<const QQSparseMatrix>,
      std::shared_ptr<const GFSparseMatrix>>;

  [[nodiscard]] std::size_t row_count() const noexcept {
    return std::visit(
        [](const auto& matrix) { return matrix->row_count(); }, value_);
  }
  [[nodiscard]] std::size_t column_count() const noexcept {
    return std::visit(
        [](const auto& matrix) { return matrix->column_count(); }, value_);
  }
  [[nodiscard]] std::size_t nnz() const noexcept {
    return std::visit([](const auto& matrix) { return matrix->nnz(); }, value_);
  }
  [[nodiscard]] py::tuple shape() const {
    return py::make_tuple(row_count(), column_count());
  }
  [[nodiscard]] py::object at(std::size_t row, std::size_t column) const {
    return std::visit(
        [row, column](const auto& matrix) {
          return python_exact_value(matrix->field(), matrix->at(row, column));
        },
        value_);
  }
  [[nodiscard]] py::list entries(
      std::size_t max_coordinate_entries) const {
    require_coordinate_budget(
        nnz(), 1, max_coordinate_entries, "sparse matrix entries");
    return std::visit(
        [](const auto& matrix) {
          py::list result;
          const auto& field = matrix->field();
          for (const auto& entry : matrix->triplets()) {
            result.append(py::make_tuple(
                entry.row, entry.column,
                python_exact_value(field, entry.value)));
          }
          return result;
        },
        value_);
  }
  [[nodiscard]] std::string to_string() const {
    return "SparseExactMatrix(shape=(" + std::to_string(row_count()) + ", " +
           std::to_string(column_count()) + "), nnz=" +
           std::to_string(nnz()) + ")";
  }

 private:
  explicit PythonSparseExactMatrix(Storage value) : value_(std::move(value)) {}

  template <typename Owner>
  [[nodiscard]] static PythonSparseExactMatrix from_qq(
      std::shared_ptr<const Owner> owner,
      const QQSparseMatrix& matrix) {
    return PythonSparseExactMatrix(
        Storage(std::shared_ptr<const QQSparseMatrix>(
            std::move(owner), std::addressof(matrix))));
  }

  template <typename Owner>
  [[nodiscard]] static PythonSparseExactMatrix from_gf(
      std::shared_ptr<const Owner> owner,
      const GFSparseMatrix& matrix) {
    return PythonSparseExactMatrix(
        Storage(std::shared_ptr<const GFSparseMatrix>(
            std::move(owner), std::addressof(matrix))));
  }

  Storage value_;

  friend class PythonCotangentH1Presentation;
  friend class PythonCotangentClassProof;
  friend class PythonConormalModule;
  friend class PythonConormalDerivativeMap;
  friend class PythonH1Module;
};

class PythonCotangentClassProof {
 public:
  using Storage = std::variant<
      std::shared_ptr<const QQCotangentClassProof>,
      std::shared_ptr<const GFCotangentClassProof>>;

  [[nodiscard]] const std::shared_ptr<PythonRing>& ring() const noexcept {
    return ring_;
  }
  [[nodiscard]] std::string status() const;
  [[nodiscard]] PythonPolynomial representative() const;
  [[nodiscard]] bool in_ideal() const noexcept;
  [[nodiscard]] std::vector<PythonPolynomial> derivative_remainders() const;
  [[nodiscard]] bool cycle() const noexcept;
  [[nodiscard]] std::optional<PythonSparseExactMatrix>
  multiplication_matrix() const;
  [[nodiscard]] std::optional<std::size_t> rank() const noexcept;
  [[nodiscard]] std::optional<std::size_t> ann() const noexcept;
  [[nodiscard]] std::vector<PythonPolynomial> annihilator_basis() const;
  [[nodiscard]] std::vector<PythonPolynomial> colon_generators() const;
  [[nodiscard]] bool faithful() const noexcept;
  [[nodiscard]] bool colon_equals() const noexcept;
  [[nodiscard]] bool conclusive() const noexcept;
  [[nodiscard]] std::string to_string() const;

 private:
  PythonCotangentClassProof(
      std::shared_ptr<PythonRing> ring,
      Storage value)
      : ring_(std::move(ring)), value_(std::move(value)) {}

  std::shared_ptr<PythonRing> ring_;
  Storage value_;

  friend class PythonCotangentH1Presentation;
};

class PythonCotangentH1Presentation {
 public:
  using Storage = std::variant<
      std::shared_ptr<const QQCotangentH1>,
      std::shared_ptr<const GFCotangentH1>>;

  [[nodiscard]] const std::shared_ptr<PythonRing>& ring() const noexcept {
    return ring_;
  }
  [[nodiscard]] std::vector<PythonPolynomial> generators() const;
  [[nodiscard]] std::size_t maximal_power() const noexcept;
  [[nodiscard]] std::size_t quotient_length() const noexcept;
  [[nodiscard]] std::size_t square_quotient_length() const noexcept;
  [[nodiscard]] std::size_t conormal_dimension() const noexcept;
  [[nodiscard]] std::size_t h1_dimension() const noexcept;
  [[nodiscard]] PythonSparseExactMatrix reduction_matrix() const;
  [[nodiscard]] PythonSparseExactMatrix derivative_matrix() const;
  [[nodiscard]] PythonSparseExactMatrix cycle_matrix() const;
  [[nodiscard]] PythonSparseExactMatrix h1_relation_matrix() const;
  [[nodiscard]] std::vector<PythonPolynomial> ideal_generators(
      std::size_t max_coordinate_entries) const;
  [[nodiscard]] std::vector<PythonPolynomial> quotient_basis(
      std::size_t max_coordinate_entries) const;
  [[nodiscard]] std::vector<PythonPolynomial> square_quotient_basis(
      std::size_t max_coordinate_entries) const;
  [[nodiscard]] std::vector<PythonPolynomial> h1_basis(
      std::size_t max_coordinate_entries) const;
  [[nodiscard]] py::list h1_kernel_coordinates(
      std::size_t max_coordinate_entries) const;
  [[nodiscard]] std::vector<PythonPolynomial> conormal_basis(
      std::size_t max_coordinate_entries) const;
  [[nodiscard]] PythonPolynomial quotient_remainder(
      const PythonPolynomial& polynomial) const;
  [[nodiscard]] PythonCotangentClassProof verify_class(
      const PythonPolynomial& representative) const;
  [[nodiscard]] std::string to_string() const;

 private:
  PythonCotangentH1Presentation(
      std::shared_ptr<PythonRing> ring,
      Storage value)
      : ring_(std::move(ring)), value_(std::move(value)) {}

  void require_polynomial(const PythonPolynomial& polynomial) const {
    if (polynomial.ring_.get() != ring_.get()) {
      throw std::invalid_argument(
          "polynomials must belong to this exact ring context");
    }
  }

  std::shared_ptr<PythonRing> ring_;
  Storage value_;

  friend class PythonRing;
};

// Public finite-algebra handles.  Each object retains the same immutable
// native context, so moving through ideal -> quotient -> module -> map ->
// kernel does not rebuild a Groebner basis or copy the large sparse matrices.
class PythonLocalIdeal {
 public:
  using Storage = std::variant<
      std::shared_ptr<const QQLocalIdeal>,
      std::shared_ptr<const GFLocalIdeal>>;

  [[nodiscard]] const std::shared_ptr<PythonRing>& ring() const noexcept {
    return ring_;
  }
  [[nodiscard]] std::vector<PythonPolynomial> lower_generators() const;
  [[nodiscard]] std::vector<PythonPolynomial> generators() const;
  [[nodiscard]] std::size_t maximal_power() const noexcept;
  [[nodiscard]] PythonFiniteQuotient quotient(
      std::size_t max_monomials,
      std::optional<std::size_t> max_generated_rows,
      std::optional<std::size_t> max_matrix_triplets) const;
  [[nodiscard]] bool equals(const PythonLocalIdeal& other) const noexcept;
  [[nodiscard]] std::string to_string() const;

 private:
  PythonLocalIdeal(std::shared_ptr<PythonRing> ring, Storage value)
      : ring_(std::move(ring)), value_(std::move(value)) {}

  std::shared_ptr<PythonRing> ring_;
  Storage value_;

  friend class PythonRing;
  friend class PythonFiniteQuotient;
  friend class PythonIdealPreimage;
};

class PythonFiniteQuotient {
 public:
  using Storage = std::variant<
      std::shared_ptr<const QQFiniteQuotient>,
      std::shared_ptr<const GFFiniteQuotient>>;

  [[nodiscard]] const std::shared_ptr<PythonRing>& ring() const noexcept {
    return ring_;
  }
  [[nodiscard]] PythonLocalIdeal defining_ideal() const;
  [[nodiscard]] std::size_t length() const noexcept;
  [[nodiscard]] std::size_t square_quotient_length() const;
  [[nodiscard]] PythonPolynomial remainder(
      const PythonPolynomial& polynomial) const;
  [[nodiscard]] std::vector<PythonPolynomial> basis(
      std::size_t max_coordinate_entries) const;
  [[nodiscard]] PythonQuotientIdeal zero_ideal() const;
  [[nodiscard]] PythonConormalModule conormal_module() const;
  [[nodiscard]] std::string to_string() const;

 private:
  PythonFiniteQuotient(std::shared_ptr<PythonRing> ring, Storage value)
      : ring_(std::move(ring)), value_(std::move(value)) {}

  void require_polynomial(const PythonPolynomial& polynomial) const;

  std::shared_ptr<PythonRing> ring_;
  Storage value_;

  friend class PythonRing;
  friend class PythonLocalIdeal;
  friend class PythonConormalModule;
  friend class PythonH1Module;
  friend class PythonQuotientIdeal;
};

class PythonConormalModule {
 public:
  using Storage = std::variant<
      std::shared_ptr<const QQConormalModule>,
      std::shared_ptr<const GFConormalModule>>;

  [[nodiscard]] PythonFiniteQuotient quotient() const;
  [[nodiscard]] std::size_t dimension() const noexcept;
  [[nodiscard]] PythonSparseExactMatrix constraint_matrix() const;
  [[nodiscard]] std::vector<PythonPolynomial> basis(
      std::size_t max_coordinate_entries) const;
  [[nodiscard]] PythonConormalDerivativeMap derivative_map() const;
  [[nodiscard]] std::string to_string() const;

 private:
  PythonConormalModule(std::shared_ptr<PythonRing> ring, Storage value)
      : ring_(std::move(ring)), value_(std::move(value)) {}

  std::shared_ptr<PythonRing> ring_;
  Storage value_;

  friend class PythonFiniteQuotient;
  friend class PythonConormalDerivativeMap;
  friend class PythonH1Module;
};

class PythonConormalDerivativeMap {
 public:
  using Storage = std::variant<
      std::shared_ptr<const QQConormalDerivativeMap>,
      std::shared_ptr<const GFConormalDerivativeMap>>;

  [[nodiscard]] PythonConormalModule domain() const;
  [[nodiscard]] PythonSparseExactMatrix ambient_matrix() const;
  [[nodiscard]] PythonH1Module kernel() const;
  [[nodiscard]] std::string to_string() const;

 private:
  PythonConormalDerivativeMap(
      std::shared_ptr<PythonRing> ring,
      Storage value)
      : ring_(std::move(ring)), value_(std::move(value)) {}

  std::shared_ptr<PythonRing> ring_;
  Storage value_;

  friend class PythonConormalModule;
  friend class PythonH1Module;
};

class PythonH1Module {
 public:
  using Storage = std::variant<
      std::shared_ptr<const QQH1Module>,
      std::shared_ptr<const GFH1Module>>;

  [[nodiscard]] PythonFiniteQuotient quotient() const;
  [[nodiscard]] PythonConormalModule conormal_module() const;
  [[nodiscard]] std::size_t dimension() const noexcept;
  [[nodiscard]] std::size_t ambient_dimension() const noexcept;
  [[nodiscard]] PythonSparseExactMatrix constraint_matrix() const;
  [[nodiscard]] py::list kernel_coordinates(
      std::size_t max_coordinate_entries) const;
  [[nodiscard]] std::vector<PythonPolynomial> basis(
      std::size_t max_coordinate_entries) const;
  [[nodiscard]] PythonH1Element class_of(
      const PythonPolynomial& representative) const;
  [[nodiscard]] std::string to_string() const;

 private:
  PythonH1Module(std::shared_ptr<PythonRing> ring, Storage value)
      : ring_(std::move(ring)), value_(std::move(value)) {}

  void require_polynomial(const PythonPolynomial& polynomial) const;

  std::shared_ptr<PythonRing> ring_;
  Storage value_;

  friend class PythonConormalDerivativeMap;
  friend class PythonH1Element;
};

class PythonH1Element {
 public:
  using Storage = std::variant<
      std::shared_ptr<const QQH1Element>,
      std::shared_ptr<const GFH1Element>>;

  [[nodiscard]] PythonH1Module module() const;
  [[nodiscard]] PythonPolynomial representative() const;
  [[nodiscard]] py::list coordinates() const;
  [[nodiscard]] PythonQuotientIdeal annihilator() const;
  [[nodiscard]] bool equals(const PythonH1Element& other) const noexcept;
  [[nodiscard]] std::string to_string() const;

 private:
  PythonH1Element(std::shared_ptr<PythonRing> ring, Storage value)
      : ring_(std::move(ring)), value_(std::move(value)) {}

  std::shared_ptr<PythonRing> ring_;
  Storage value_;

  friend class PythonH1Module;
};

class PythonQuotientIdeal {
 public:
  using Storage = std::variant<
      std::shared_ptr<const QQQuotientIdeal>,
      std::shared_ptr<const GFQuotientIdeal>>;

  [[nodiscard]] PythonFiniteQuotient quotient() const;
  [[nodiscard]] std::size_t dimension() const noexcept;
  [[nodiscard]] bool is_zero() const noexcept;
  [[nodiscard]] py::list basis_coordinates() const;
  [[nodiscard]] std::vector<PythonPolynomial> generators() const;
  [[nodiscard]] PythonIdealPreimage preimage() const;
  [[nodiscard]] bool equals(const PythonQuotientIdeal& other) const noexcept;
  [[nodiscard]] std::string to_string() const;

 private:
  PythonQuotientIdeal(std::shared_ptr<PythonRing> ring, Storage value)
      : ring_(std::move(ring)), value_(std::move(value)) {}

  std::shared_ptr<PythonRing> ring_;
  Storage value_;

  friend class PythonFiniteQuotient;
  friend class PythonH1Element;
  friend class PythonIdealPreimage;
};

class PythonIdealPreimage {
 public:
  using Storage = std::variant<
      std::shared_ptr<const QQIdealPreimage>,
      std::shared_ptr<const GFIdealPreimage>>;

  [[nodiscard]] PythonQuotientIdeal quotient_ideal() const;
  [[nodiscard]] PythonLocalIdeal source_ideal() const;
  [[nodiscard]] std::vector<PythonPolynomial> generators() const;
  [[nodiscard]] bool equals_source_ideal() const noexcept;
  [[nodiscard]] bool equals(const PythonIdealPreimage& other) const noexcept;
  [[nodiscard]] bool equals(const PythonLocalIdeal& other) const noexcept;
  [[nodiscard]] std::string to_string() const;

 private:
  PythonIdealPreimage(std::shared_ptr<PythonRing> ring, Storage value)
      : ring_(std::move(ring)), value_(std::move(value)) {}

  std::shared_ptr<PythonRing> ring_;
  Storage value_;

  friend class PythonQuotientIdeal;
};

// A language-neutral, shape-preserving view of an exact dense matrix.  The
// native field elements are retained as canonical text and materialized as
// Fraction/int objects only while Python holds the GIL.
struct PythonExactMatrix {
  std::shared_ptr<PythonRing> ring;
  std::size_t row_count;
  std::size_t column_count;
  std::vector<std::string> row_major_entry_text;

  [[nodiscard]] py::tuple shape() const {
    return py::make_tuple(row_count, column_count);
  }
  [[nodiscard]] py::list rows() const;
  [[nodiscard]] std::string to_string() const {
    return "ExactMatrix(shape=(" + std::to_string(row_count) + ", " +
           std::to_string(column_count) + "))";
  }
};

struct PythonCycleAuditResult {
  std::string status;
  PythonIdeal ideal;
  bool finite_quotient;
  bool supported_at_origin;
  std::optional<std::size_t> quotient_length;
  std::optional<PythonPolynomial> polynomial;
  std::optional<PythonPolynomial> polynomial_remainder;
  std::vector<PythonPolynomial> derivatives;
  std::vector<PythonPolynomial> derivative_remainders;
  std::optional<bool> polynomial_in_ideal;
  std::optional<bool> derivatives_in_ideal;
  std::optional<bool> cycle_valid;
  std::optional<PythonIdeal> maximal_times_ideal;
  std::optional<bool> primitive;
  std::optional<PythonIdeal> squared_ideal;
  std::optional<PythonExactMatrix> colon_multiplication_matrix;
  std::optional<PythonExactMatrix> colon_kernel_basis;
  std::vector<PythonPolynomial> colon_kernel_lifts;
  std::vector<PythonPolynomial> colon_lift_product_remainders;
  std::optional<PythonIdeal> colon_ideal;
  std::optional<std::size_t> annihilator_length;
  std::optional<std::size_t> colon_quotient_length;
  std::optional<bool> colon_equals_ideal;
  std::optional<bool> annihilator_zero;
  std::vector<PythonPolynomial> ideal_in_colon_remainders;
  std::vector<PythonPolynomial> colon_in_ideal_remainders;
  bool faithful;
  bool conclusive;
  std::optional<std::string> resource_detail;

  [[nodiscard]] std::string to_string() const {
    return "CycleAuditResult(status='" + status + "', faithful=" +
           (faithful ? "True" : "False") + ")";
  }
};

struct PythonColonClosureStep {
  std::size_t index;
  PythonIdeal ideal;
  std::optional<std::size_t> quotient_length;
};

struct PythonColonClosureTransition {
  std::vector<PythonPolynomial> current_in_next_remainders;
  std::vector<PythonPolynomial> next_in_current_remainders;
  bool current_subset_next;
  bool equal;
};

struct PythonColonClosureResult {
  std::string status;
  std::vector<PythonColonClosureStep> steps;
  std::vector<PythonColonClosureTransition> transitions;
  bool conclusive;
  bool faithful_fixed_point_found;
  std::optional<std::string> resource_detail;

  [[nodiscard]] std::vector<PythonIdeal> ideals() const {
    std::vector<PythonIdeal> result;
    result.reserve(steps.size());
    for (const auto& step : steps) {
      result.push_back(step.ideal);
    }
    return result;
  }

  [[nodiscard]] std::vector<std::optional<std::size_t>> quotient_lengths()
      const {
    std::vector<std::optional<std::size_t>> result;
    result.reserve(steps.size());
    for (const auto& step : steps) {
      result.push_back(step.quotient_length);
    }
    return result;
  }

  [[nodiscard]] std::string to_string() const {
    return "ColonClosureResult(status='" + status + "', steps=" +
           std::to_string(steps.size()) + ")";
  }
};

struct PythonMatrixSpaceRankResult {
  std::shared_ptr<PythonRing> ring;
  std::size_t parameter_count;
  std::size_t row_count;
  std::size_t column_count;
  std::size_t lower_bound;
  std::size_t upper_bound;
  std::optional<std::size_t> generic_maximum;
  std::optional<std::size_t> exact_maximum;
  std::string proof;
  std::optional<std::vector<std::string>> witness_coefficient_text;
  std::size_t minors_tested;
  std::size_t determinant_products_tested;
  std::size_t finite_field_evaluations;
  bool has_full_column_rank_witness;

  [[nodiscard]] py::object witness_coefficients() const;
  [[nodiscard]] std::string to_string() const {
    return "MatrixSpaceRankResult(proof='" + proof + "', bounds=[" +
           std::to_string(lower_bound) + ", " +
           std::to_string(upper_bound) + "])";
  }
};

struct PythonH1ActionResult {
  std::string status;
  std::size_t length_Q;
  std::size_t length_P_mod_J2;
  std::size_t conormal_dimension;
  std::size_t h1_dimension;
  std::size_t socle_dimension;
  PythonIdeal ideal;
  PythonIdeal squared_ideal;
  std::vector<PythonPolynomial> conormal_basis;
  std::vector<PythonPolynomial> h1_basis;
  std::vector<PythonPolynomial> socle_basis;
  PythonExactMatrix reduction_map;
  PythonExactMatrix differential;
  std::vector<PythonExactMatrix> variable_multiplication_matrices;
  std::vector<PythonExactMatrix> h1_multiplication_matrices;
  std::vector<PythonExactMatrix> action_matrices;
  std::size_t common_product_space_dimension;
  std::size_t common_product_space_rank_bound;
  PythonMatrixSpaceRankResult rank;
  std::optional<PythonPolynomial> best_h1_polynomial;
  std::optional<PythonCycleAuditResult> faithful_witness_audit;
  PythonIdeal common_annihilator_diagnostic;

  [[nodiscard]] bool has_faithful_witness() const noexcept {
    return faithful_witness_audit.has_value() &&
           faithful_witness_audit->faithful;
  }
  [[nodiscard]] std::string to_string() const {
    return "H1ActionResult(length_Q=" + std::to_string(length_Q) +
           ", h1_dimension=" + std::to_string(h1_dimension) +
           ", socle_dimension=" + std::to_string(socle_dimension) + ")";
  }
};

struct PythonInverseSystemResult {
  std::string convention;
  std::vector<PythonPolynomial> dual_generators;
  std::uint32_t maximum_degree;
  std::vector<std::vector<std::uint16_t>> operator_exponents;
  std::vector<std::vector<std::uint16_t>> output_exponents;
  std::size_t action_row_count;
  std::size_t action_column_count;
  std::size_t action_nonzeros;
  std::size_t action_rank;
  std::size_t kernel_dimension;
  std::size_t truncated_kernel_generator_count;
  PythonIdeal annihilator;
  std::size_t quotient_length;

  [[nodiscard]] py::tuple action_shape() const {
    return py::make_tuple(action_row_count, action_column_count);
  }
  [[nodiscard]] std::string to_string() const {
    return "InverseSystemResult(convention='" + convention +
           "', quotient_length=" + std::to_string(quotient_length) + ")";
  }
};

struct PythonCycleScreenResult {
  std::string status;
  std::size_t length_Q;
  std::size_t length_P_mod_J2;
  bool g_in_J;
  bool derivatives_in_J;
  bool cycle_valid;
  std::size_t multiplication_rank;
  bool full_column_rank_candidate;
  bool certified_faithful;
  std::optional<PythonCycleAuditResult> certification;
  std::optional<std::string> detail;

  [[nodiscard]] bool conclusive() const noexcept {
    return status == "complete";
  }
  [[nodiscard]] std::string to_string() const {
    return "CycleScreenResult(status='" + status +
           "', full_column_rank_candidate=" +
           (full_column_rank_candidate ? "True" : "False") + ")";
  }
};

struct PythonH1ScreenResult {
  std::string status;
  std::size_t length_Q;
  std::size_t length_P_mod_J2;
  std::size_t conormal_dimension;
  std::size_t h1_dimension;
  std::size_t socle_dimension;
  std::uint32_t modulus;
  std::uint64_t leading_ideal_fingerprint;
  std::string leading_ideal_signature;
  std::size_t maximum_individual_rank_lower_bound;
  std::size_t maximum_individual_rank_upper_bound;
  std::string rank_proof;
  bool full_socle_rank_candidate;
  bool certified_faithful;
  std::optional<std::vector<std::uint32_t>> witness_coefficients;
  std::optional<PythonPolynomial> witness;
  std::optional<PythonCycleAuditResult> certification;
  std::optional<std::string> detail;

  [[nodiscard]] bool conclusive() const noexcept {
    return status == "complete" && rank_proof != "resource_limit" &&
           rank_proof != "generic_only";
  }
  [[nodiscard]] std::string to_string() const {
    return "H1ScreenResult(status='" + status + "', h1_dimension=" +
           std::to_string(h1_dimension) + ")";
  }
};

struct PythonInverseSystemDiscoveryRecord {
  std::string status;
  std::size_t candidate_index;
  std::uint32_t maximum_dual_degree;
  std::size_t action_rank;
  std::size_t kernel_dimension;
  std::size_t quotient_length;
  std::size_t h1_dimension;
  std::size_t socle_dimension;
  std::uint32_t modulus;
  std::uint64_t leading_ideal_fingerprint;
  std::string leading_ideal_signature;
  std::size_t maximum_individual_rank_lower_bound;
  std::size_t maximum_individual_rank_upper_bound;
  std::string rank_proof;
  std::size_t maximum_individual_rank;
  bool full_rank_candidate;
  bool certified_faithful;
  std::vector<PythonPolynomial> retained_dual_generators;
  std::vector<PythonPolynomial> retained_annihilator_basis;
  std::optional<PythonPolynomial> witness;
  std::optional<std::string> detail;

  [[nodiscard]] bool conclusive() const noexcept {
    return status == "complete" && rank_proof != "resource_limit" &&
           rank_proof != "generic_only";
  }
  [[nodiscard]] std::string to_string() const {
    return "InverseSystemDiscoveryRecord(index=" +
           std::to_string(candidate_index) + ", status='" + status + "')";
  }
};

void PythonPolynomial::require_same_ring(
    const PythonPolynomial& other) const {
  if (ring_.get() != other.ring_.get()) {
    throw std::invalid_argument(
        "polynomial arithmetic requires the same exact ring context");
  }
}

PythonPolynomial PythonPolynomial::add(
    const PythonPolynomial& other) const {
  require_same_ring(other);
  if (const auto* polynomial = std::get_if<QQPolynomial>(&value_)) {
    return PythonPolynomial(
        ring_, *polynomial + std::get<QQPolynomial>(other.value_));
  }
  return PythonPolynomial(
      ring_, std::get<GFPolynomial>(value_) +
                 std::get<GFPolynomial>(other.value_));
}

PythonPolynomial PythonPolynomial::subtract(
    const PythonPolynomial& other) const {
  require_same_ring(other);
  if (const auto* polynomial = std::get_if<QQPolynomial>(&value_)) {
    return PythonPolynomial(
        ring_, *polynomial - std::get<QQPolynomial>(other.value_));
  }
  return PythonPolynomial(
      ring_, std::get<GFPolynomial>(value_) -
                 std::get<GFPolynomial>(other.value_));
}

PythonPolynomial PythonPolynomial::multiply(
    const PythonPolynomial& other) const {
  require_same_ring(other);
  if (const auto* polynomial = std::get_if<QQPolynomial>(&value_)) {
    return PythonPolynomial(
        ring_, *polynomial * std::get<QQPolynomial>(other.value_));
  }
  return PythonPolynomial(
      ring_, std::get<GFPolynomial>(value_) *
                 std::get<GFPolynomial>(other.value_));
}

PythonPolynomial PythonPolynomial::negate() const {
  return std::visit(
      [this](const auto& polynomial) {
        return PythonPolynomial(ring_, -polynomial);
      },
      value_);
}

PythonPolynomial PythonPolynomial::pow(std::uint64_t exponent) const {
  return std::visit(
      [this, exponent](const auto& polynomial) {
        return PythonPolynomial(ring_, polynomial.pow(exponent));
      },
      value_);
}

PythonPolynomial PythonPolynomial::monic() const {
  return std::visit(
      [this](const auto& polynomial) {
        return PythonPolynomial(ring_, polynomial.monic());
      },
      value_);
}

PythonPolynomial PythonPolynomial::derivative(std::size_t variable) const {
  return std::visit(
      [this, variable](const auto& polynomial) {
        return PythonPolynomial(ring_, polynomial.derivative(variable));
      },
      value_);
}

PythonPolynomial PythonPolynomial::derivative(std::string_view variable) const {
  return std::visit(
      [this, variable](const auto& polynomial) {
        return PythonPolynomial(ring_, polynomial.derivative(variable));
      },
      value_);
}

PythonPolynomial PythonPolynomial::derivative(
    const PythonPolynomial& variable) const {
  require_same_ring(variable);
  const auto generators = ring_->gens();
  for (std::size_t index = 0; index < generators.size(); ++index) {
    if (variable.equals(generators[index])) {
      return derivative(index);
    }
  }
  throw std::invalid_argument(
      "a polynomial derivative argument must be a ring generator");
}

bool PythonPolynomial::equals(const PythonPolynomial& other) const noexcept {
  if (ring_.get() != other.ring_.get() || value_.index() != other.value_.index()) {
    return false;
  }
  if (const auto* polynomial = std::get_if<QQPolynomial>(&value_)) {
    return *polynomial == std::get<QQPolynomial>(other.value_);
  }
  return std::get<GFPolynomial>(value_) ==
         std::get<GFPolynomial>(other.value_);
}

PythonIdeal PythonRing::wrap(QQIdeal ideal) {
  return PythonIdeal(
      shared_from_this(), PythonIdeal::Storage(std::move(ideal)));
}

PythonIdeal PythonRing::wrap(GFIdeal ideal) {
  return PythonIdeal(
      shared_from_this(), PythonIdeal::Storage(std::move(ideal)));
}

void PythonRing::require_ideal(const PythonIdeal& ideal) const {
  if (ideal.ring_.get() != this) {
    throw std::invalid_argument(
        "ideals must belong to this exact ring context");
  }
}

std::vector<PythonPolynomial> PythonRing::normal_forms(
    const std::vector<PythonPolynomial>& polynomials,
    const std::vector<PythonPolynomial>& divisors) {
  if (std::holds_alternative<QQRing>(value_)) {
    const auto inputs = concrete_polynomials<QQPolynomial>(polynomials);
    const auto concrete_divisors =
        concrete_polynomials<QQPolynomial>(divisors);
    const auto result = without_gil([&] {
      const auto reducer =
          le::CompiledReducer<le::RationalField>::ordered(
              std::get<QQRing>(value_), concrete_divisors);
      return reducer.normal_forms(inputs);
    });
    return wrap_all(result);
  }
  const auto inputs = concrete_polynomials<GFPolynomial>(polynomials);
  const auto concrete_divisors = concrete_polynomials<GFPolynomial>(divisors);
  const auto result = without_gil([&] {
    const auto reducer = le::CompiledReducer<le::PrimeField>::ordered(
        std::get<GFRing>(value_), concrete_divisors);
    return reducer.normal_forms(inputs);
  });
  return wrap_all(result);
}

PythonIdeal PythonRing::ideal(
    const std::vector<PythonPolynomial>& generators) {
  if (std::holds_alternative<QQRing>(value_)) {
    const auto concrete = concrete_polynomials<QQPolynomial>(generators);
    auto result = without_gil(
        [&] { return QQIdeal(std::get<QQRing>(value_), concrete); });
    return wrap(std::move(result));
  }
  const auto concrete = concrete_polynomials<GFPolynomial>(generators);
  auto result = without_gil(
      [&] { return GFIdeal(std::get<GFRing>(value_), concrete); });
  return wrap(std::move(result));
}

PythonCotangentH1Presentation PythonRing::cotangent_h1(
    const std::vector<PythonPolynomial>& generators,
    std::size_t maximal_power,
    std::size_t max_monomials,
    std::optional<std::size_t> max_generated_rows,
    std::optional<std::size_t> max_matrix_triplets) {
  le::CotangentH1Options options;
  options.monomial_space_limits.maximum_monomials = max_monomials;
  options.max_generated_rows = max_generated_rows;
  options.max_matrix_triplets = max_matrix_triplets;

  if (std::holds_alternative<QQRing>(value_)) {
    auto concrete = concrete_polynomials<QQPolynomial>(generators);
    auto presentation = cotangent_without_gil([&] {
      return le::cotangent_h1(le::CotangentH1Spec<le::RationalField>{
          std::get<QQRing>(value_), std::move(concrete), maximal_power},
          options);
    });
    auto native =
        std::make_shared<const QQCotangentH1>(std::move(presentation));
    return PythonCotangentH1Presentation(
        shared_from_this(),
        PythonCotangentH1Presentation::Storage(std::move(native)));
  }

  auto concrete = concrete_polynomials<GFPolynomial>(generators);
  auto presentation = cotangent_without_gil([&] {
    return le::cotangent_h1(le::CotangentH1Spec<le::PrimeField>{
        std::get<GFRing>(value_), std::move(concrete), maximal_power},
        options);
  });
  auto native = std::make_shared<const GFCotangentH1>(std::move(presentation));
  return PythonCotangentH1Presentation(
      shared_from_this(),
      PythonCotangentH1Presentation::Storage(std::move(native)));
}

PythonLocalIdeal PythonRing::local_ideal(
    const std::vector<PythonPolynomial>& generators,
    std::size_t maximal_power) {
  if (std::holds_alternative<QQRing>(value_)) {
    auto concrete = concrete_polynomials<QQPolynomial>(generators);
    auto ideal = le::origin_power_ideal(
        std::get<QQRing>(value_), std::move(concrete), maximal_power);
    auto native = std::make_shared<const QQLocalIdeal>(std::move(ideal));
    return PythonLocalIdeal(
        shared_from_this(), PythonLocalIdeal::Storage(std::move(native)));
  }
  auto concrete = concrete_polynomials<GFPolynomial>(generators);
  auto ideal = le::origin_power_ideal(
      std::get<GFRing>(value_), std::move(concrete), maximal_power);
  auto native = std::make_shared<const GFLocalIdeal>(std::move(ideal));
  return PythonLocalIdeal(
      shared_from_this(), PythonLocalIdeal::Storage(std::move(native)));
}

PythonFiniteQuotient PythonRing::quotient(
    const PythonLocalIdeal& ideal,
    std::size_t max_monomials,
    std::optional<std::size_t> max_generated_rows,
    std::optional<std::size_t> max_matrix_triplets) {
  if (ideal.ring_.get() != this) {
    throw std::invalid_argument(
        "local ideals must belong to this exact ring context");
  }
  return ideal.quotient(
      max_monomials, max_generated_rows, max_matrix_triplets);
}

std::vector<PythonPolynomial> PythonIdeal::generators() const {
  if (const auto* ideal = std::get_if<QQIdeal>(&value_)) {
    return ring_->wrap_all(ideal->generators());
  }
  return ring_->wrap_all(std::get<GFIdeal>(value_).generators());
}

PythonPolynomial PythonIdeal::normal_form(
    const PythonPolynomial& polynomial) const {
  if (polynomial.ring().get() != ring_.get()) {
    throw std::invalid_argument(
        "ideal reduction requires the same exact ring context");
  }
  if (const auto* ideal = std::get_if<QQIdeal>(&value_)) {
    const auto input = std::get<QQPolynomial>(polynomial.value_);
    auto result = without_gil([&] { return ideal->normal_form(input); });
    return ring_->wrap(std::move(result));
  }
  const auto input = std::get<GFPolynomial>(polynomial.value_);
  auto result = without_gil(
      [&] { return std::get<GFIdeal>(value_).normal_form(input); });
  return ring_->wrap(std::move(result));
}

std::vector<PythonPolynomial> PythonIdeal::normal_forms(
    const std::vector<PythonPolynomial>& polynomials) const {
  if (const auto* ideal = std::get_if<QQIdeal>(&value_)) {
    std::vector<QQPolynomial> inputs;
    inputs.reserve(polynomials.size());
    for (const auto& polynomial : polynomials) {
      if (polynomial.ring().get() != ring_.get()) {
        throw std::invalid_argument(
            "ideal reduction requires the same exact ring context");
      }
      inputs.push_back(std::get<QQPolynomial>(polynomial.value_));
    }
    auto result = without_gil([&] { return ideal->normal_forms(inputs); });
    return ring_->wrap_all(std::move(result));
  }
  std::vector<GFPolynomial> inputs;
  inputs.reserve(polynomials.size());
  for (const auto& polynomial : polynomials) {
    if (polynomial.ring().get() != ring_.get()) {
      throw std::invalid_argument(
          "ideal reduction requires the same exact ring context");
    }
    inputs.push_back(std::get<GFPolynomial>(polynomial.value_));
  }
  auto result = without_gil(
      [&] { return std::get<GFIdeal>(value_).normal_forms(inputs); });
  return ring_->wrap_all(std::move(result));
}

bool PythonIdeal::contains(const PythonPolynomial& polynomial) const {
  if (polynomial.ring().get() != ring_.get()) {
    throw std::invalid_argument(
        "ideal membership requires the same exact ring context");
  }
  if (const auto* ideal = std::get_if<QQIdeal>(&value_)) {
    const auto input = std::get<QQPolynomial>(polynomial.value_);
    return without_gil([&] { return ideal->contains(input); });
  }
  const auto input = std::get<GFPolynomial>(polynomial.value_);
  return without_gil(
      [&] { return std::get<GFIdeal>(value_).contains(input); });
}

bool PythonIdeal::is_zero_dimensional() const {
  return without_gil(
      [&] { return std::visit(
          [](const auto& ideal) { return ideal.is_zero_dimensional(); },
          value_); });
}

bool PythonIdeal::supported_at_origin() const {
  return without_gil(
      [&] { return std::visit(
          [](const auto& ideal) { return ideal.supported_at_origin(); },
          value_); });
}

std::size_t PythonIdeal::dimension() const {
  return without_gil(
      [&] { return std::visit(
          [](const auto& ideal) { return ideal.quotient_dimension(); },
          value_); });
}

std::vector<PythonPolynomial> PythonIdeal::standard_monomials() const {
  if (const auto* ideal = std::get_if<QQIdeal>(&value_)) {
    auto result = without_gil(
        [&] { return ideal->standard_monomials().polynomials(); });
    return ring_->wrap_all(std::move(result));
  }
  auto result = without_gil([&] {
    return std::get<GFIdeal>(value_).standard_monomials().polynomials();
  });
  return ring_->wrap_all(std::move(result));
}

void PythonIdeal::require_same_ring(const PythonIdeal& other) const {
  if (ring_.get() != other.ring_.get()) {
    throw std::invalid_argument(
        "ideal operations require the same exact ring context");
  }
}

PythonIdeal PythonIdeal::add(const PythonIdeal& other) const {
  require_same_ring(other);
  if (const auto* ideal = std::get_if<QQIdeal>(&value_)) {
    auto result = without_gil(
        [&] { return ideal->sum(std::get<QQIdeal>(other.value_)); });
    return ring_->wrap(std::move(result));
  }
  auto result = without_gil([&] {
    return std::get<GFIdeal>(value_).sum(std::get<GFIdeal>(other.value_));
  });
  return ring_->wrap(std::move(result));
}

PythonIdeal PythonIdeal::multiply(const PythonIdeal& other) const {
  require_same_ring(other);
  if (const auto* ideal = std::get_if<QQIdeal>(&value_)) {
    auto result = without_gil(
        [&] { return ideal->product(std::get<QQIdeal>(other.value_)); });
    return ring_->wrap(std::move(result));
  }
  auto result = without_gil([&] {
    return std::get<GFIdeal>(value_).product(
        std::get<GFIdeal>(other.value_));
  });
  return ring_->wrap(std::move(result));
}

PythonIdeal PythonIdeal::square() const {
  if (const auto* ideal = std::get_if<QQIdeal>(&value_)) {
    auto result = without_gil([&] { return ideal->square(); });
    return ring_->wrap(std::move(result));
  }
  auto result = without_gil(
      [&] { return std::get<GFIdeal>(value_).square(); });
  return ring_->wrap(std::move(result));
}

PythonIdeal PythonIdeal::colon(const PythonPolynomial& polynomial) const {
  if (polynomial.ring().get() != ring_.get()) {
    throw std::invalid_argument(
        "ideal colon requires the same exact ring context");
  }
  if (const auto* ideal = std::get_if<QQIdeal>(&value_)) {
    const auto input = std::get<QQPolynomial>(polynomial.value_);
    auto result = without_gil([&] { return ideal->colon(input); });
    return ring_->wrap(std::move(result));
  }
  const auto input = std::get<GFPolynomial>(polynomial.value_);
  auto result = without_gil(
      [&] { return std::get<GFIdeal>(value_).colon(input); });
  return ring_->wrap(std::move(result));
}

PythonIdeal PythonIdeal::eliminate(
    const std::vector<std::size_t>& variables) const {
  if (const auto* ideal = std::get_if<QQIdeal>(&value_)) {
    auto result = without_gil([&] { return ideal->eliminate(variables); });
    return ring_->wrap(std::move(result));
  }
  auto result = without_gil(
      [&] { return std::get<GFIdeal>(value_).eliminate(variables); });
  return ring_->wrap(std::move(result));
}

PythonIdeal PythonIdeal::eliminate(
    const std::vector<std::string>& variables) const {
  std::vector<std::size_t> indices;
  indices.reserve(variables.size());
  const auto& names = ring_->variable_names();
  for (const auto& variable : variables) {
    const auto iterator = std::find(names.begin(), names.end(), variable);
    if (iterator == names.end()) {
      throw std::invalid_argument(
          "elimination variable is not in this polynomial ring");
    }
    indices.push_back(static_cast<std::size_t>(iterator - names.begin()));
  }
  return eliminate(indices);
}

bool PythonIdeal::equals(const PythonIdeal& other) const noexcept {
  if (ring_.get() != other.ring_.get() ||
      value_.index() != other.value_.index()) {
    return false;
  }
  if (const auto* ideal = std::get_if<QQIdeal>(&value_)) {
    return *ideal == std::get<QQIdeal>(other.value_);
  }
  return std::get<GFIdeal>(value_) == std::get<GFIdeal>(other.value_);
}

std::string PythonIdeal::to_string() const {
  std::ostringstream output;
  output << "Ideal([";
  std::visit(
      [&output](const auto& ideal) {
        const auto& basis = ideal.groebner_basis();
        for (std::size_t index = 0; index < basis.size(); ++index) {
          if (index != 0) {
            output << ", ";
          }
          output << basis[index].to_string();
        }
      },
      value_);
  output << "], ring=" << ring_->to_string() << ')';
  return output.str();
}

py::list PythonExactMatrix::rows() const {
  py::list result;
  py::object fraction;
  const bool rational = ring->characteristic() == 0;
  if (rational) {
    fraction = py::module_::import("fractions").attr("Fraction");
  }
  for (std::size_t row = 0; row < row_count; ++row) {
    py::list values;
    for (std::size_t column = 0; column < column_count; ++column) {
      const auto& text =
          row_major_entry_text[row * column_count + column];
      if (rational) {
        values.append(fraction(py::str(text)));
      } else {
        values.append(py::int_(std::stoll(text)));
      }
    }
    result.append(std::move(values));
  }
  return result;
}

template <typename Field>
[[nodiscard]] PythonExactMatrix wrap_exact_matrix(
    const std::shared_ptr<PythonRing>& ring,
    const le::DenseMatrix<Field>& matrix) {
  std::vector<std::string> entries;
  entries.reserve(matrix.row_count() * matrix.column_count());
  for (std::size_t row = 0; row < matrix.row_count(); ++row) {
    for (std::size_t column = 0; column < matrix.column_count(); ++column) {
      entries.push_back(matrix.field().to_string(matrix(row, column)));
    }
  }
  return PythonExactMatrix{
      ring, matrix.row_count(), matrix.column_count(), std::move(entries)};
}

template <typename Field>
[[nodiscard]] PythonExactMatrix wrap_exact_rows(
    const std::shared_ptr<PythonRing>& ring,
    const std::vector<std::vector<typename Field::Element>>& rows,
    std::size_t column_count,
    const Field& field) {
  std::vector<std::string> entries;
  entries.reserve(rows.size() * column_count);
  for (const auto& row : rows) {
    if (row.size() != column_count) {
      throw std::logic_error(
          "native exact-matrix rows have inconsistent dimensions");
    }
    for (const auto& entry : row) {
      entries.push_back(field.to_string(entry));
    }
  }
  return PythonExactMatrix{
      ring, rows.size(), column_count, std::move(entries)};
}

template <typename Field>
[[nodiscard]] std::vector<PythonExactMatrix> wrap_exact_matrices(
    const std::shared_ptr<PythonRing>& ring,
    const std::vector<le::DenseMatrix<Field>>& matrices) {
  std::vector<PythonExactMatrix> result;
  result.reserve(matrices.size());
  for (const auto& matrix : matrices) {
    result.push_back(wrap_exact_matrix(ring, matrix));
  }
  return result;
}

[[nodiscard]] std::string cotangent_class_status_name(
    le::CotangentClassStatus status) {
  switch (status) {
    case le::CotangentClassStatus::Complete:
      return "complete";
    case le::CotangentClassStatus::NotInIdeal:
      return "not_in_ideal";
    case le::CotangentClassStatus::NotCycle:
      return "not_cycle";
  }
  throw std::logic_error("unknown cotangent class status");
}

std::vector<PythonPolynomial> PythonCotangentH1Presentation::generators()
    const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQCotangentH1>>(&value_)) {
    const auto generators = (*native)->generators();
    return ring_->wrap_all(
        std::vector<QQPolynomial>(generators.begin(), generators.end()));
  }
  const auto& native = std::get<std::shared_ptr<const GFCotangentH1>>(value_);
  const auto generators = native->generators();
  return ring_->wrap_all(
      std::vector<GFPolynomial>(generators.begin(), generators.end()));
}

std::size_t PythonCotangentH1Presentation::maximal_power() const noexcept {
  return std::visit(
      [](const auto& native) { return native->maximal_power(); }, value_);
}

std::size_t PythonCotangentH1Presentation::quotient_length() const noexcept {
  return std::visit(
      [](const auto& native) { return native->length_Q(); }, value_);
}

std::size_t PythonCotangentH1Presentation::square_quotient_length()
    const noexcept {
  return std::visit(
      [](const auto& native) { return native->length_P_mod_J2(); }, value_);
}

std::size_t PythonCotangentH1Presentation::conormal_dimension()
    const noexcept {
  return std::visit(
      [](const auto& native) { return native->conormal_dimension(); }, value_);
}

std::size_t PythonCotangentH1Presentation::h1_dimension() const noexcept {
  return std::visit(
      [](const auto& native) { return native->h1_dimension(); }, value_);
}

PythonSparseExactMatrix PythonCotangentH1Presentation::reduction_matrix()
    const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQCotangentH1>>(&value_)) {
    return PythonSparseExactMatrix::from_qq(
        *native, (*native)->reduction_matrix());
  }
  const auto& native = std::get<std::shared_ptr<const GFCotangentH1>>(value_);
  return PythonSparseExactMatrix::from_gf(
      native, native->reduction_matrix());
}

PythonSparseExactMatrix PythonCotangentH1Presentation::derivative_matrix()
    const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQCotangentH1>>(&value_)) {
    return PythonSparseExactMatrix::from_qq(
        *native, (*native)->derivative_matrix());
  }
  const auto& native = std::get<std::shared_ptr<const GFCotangentH1>>(value_);
  return PythonSparseExactMatrix::from_gf(
      native, native->derivative_matrix());
}

PythonSparseExactMatrix PythonCotangentH1Presentation::cycle_matrix() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQCotangentH1>>(&value_)) {
    return PythonSparseExactMatrix::from_qq(
        *native, (*native)->cycle_matrix());
  }
  const auto& native = std::get<std::shared_ptr<const GFCotangentH1>>(value_);
  return PythonSparseExactMatrix::from_gf(native, native->cycle_matrix());
}

PythonSparseExactMatrix PythonCotangentH1Presentation::h1_relation_matrix()
    const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQCotangentH1>>(&value_)) {
    return PythonSparseExactMatrix::from_qq(
        *native, (*native)->h1_relation_matrix());
  }
  const auto& native = std::get<std::shared_ptr<const GFCotangentH1>>(value_);
  return PythonSparseExactMatrix::from_gf(
      native, native->h1_relation_matrix());
}

std::vector<PythonPolynomial>
PythonCotangentH1Presentation::ideal_generators(
    std::size_t max_coordinate_entries) const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQCotangentH1>>(&value_)) {
    auto result = without_gil([&] { return (*native)->ideal_generators(); });
    require_coordinate_budget(
        result.size(), 1, max_coordinate_entries, "explicit ideal generators");
    return ring_->wrap_all(std::move(result));
  }
  const auto& native = std::get<std::shared_ptr<const GFCotangentH1>>(value_);
  auto result = without_gil([&] { return native->ideal_generators(); });
  require_coordinate_budget(
      result.size(), 1, max_coordinate_entries, "explicit ideal generators");
  return ring_->wrap_all(std::move(result));
}

std::vector<PythonPolynomial>
PythonCotangentH1Presentation::quotient_basis(
    std::size_t max_coordinate_entries) const {
  require_coordinate_budget(
      quotient_length(), 1, max_coordinate_entries,
      "explicit quotient basis");
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQCotangentH1>>(&value_)) {
    auto result = without_gil([&] { return (*native)->quotient_basis(); });
    return ring_->wrap_all(std::move(result));
  }
  const auto& native = std::get<std::shared_ptr<const GFCotangentH1>>(value_);
  auto result = without_gil([&] { return native->quotient_basis(); });
  return ring_->wrap_all(std::move(result));
}

std::vector<PythonPolynomial>
PythonCotangentH1Presentation::square_quotient_basis(
    std::size_t max_coordinate_entries) const {
  require_coordinate_budget(
      square_quotient_length(), 1, max_coordinate_entries,
      "explicit square-quotient basis");
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQCotangentH1>>(&value_)) {
    auto result =
        without_gil([&] { return (*native)->square_quotient_basis(); });
    return ring_->wrap_all(std::move(result));
  }
  const auto& native = std::get<std::shared_ptr<const GFCotangentH1>>(value_);
  auto result = without_gil([&] { return native->square_quotient_basis(); });
  return ring_->wrap_all(std::move(result));
}

std::vector<PythonPolynomial> PythonCotangentH1Presentation::h1_basis(
    std::size_t max_coordinate_entries) const {
  le::SparseEliminationLimits limits;
  limits.max_kernel_nonzeros = max_coordinate_entries;
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQCotangentH1>>(&value_)) {
    auto result =
        cotangent_without_gil([&] { return (*native)->h1_basis(limits); });
    return ring_->wrap_all(std::move(result));
  }
  const auto& native = std::get<std::shared_ptr<const GFCotangentH1>>(value_);
  auto result =
      cotangent_without_gil([&] { return native->h1_basis(limits); });
  return ring_->wrap_all(std::move(result));
}

py::list PythonCotangentH1Presentation::h1_kernel_coordinates(
    std::size_t max_coordinate_entries) const {
  le::SparseEliminationLimits limits;
  limits.max_kernel_nonzeros = max_coordinate_entries;
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQCotangentH1>>(&value_)) {
    const auto rows = cotangent_without_gil(
        [&] { return (*native)->h1_kernel_coordinates(limits); });
    py::list triples;
    for (std::size_t basis = 0; basis < rows.size(); ++basis) {
      for (const auto& entry : rows[basis]) {
        triples.append(py::make_tuple(
            basis, entry.column,
            python_exact_value((*native)->ring().field(), entry.value)));
      }
    }
    return triples;
  }
  const auto& native = std::get<std::shared_ptr<const GFCotangentH1>>(value_);
  const auto rows = cotangent_without_gil(
      [&] { return native->h1_kernel_coordinates(limits); });
  py::list triples;
  for (std::size_t basis = 0; basis < rows.size(); ++basis) {
    for (const auto& entry : rows[basis]) {
      triples.append(py::make_tuple(
          basis, entry.column,
          python_exact_value(native->ring().field(), entry.value)));
    }
  }
  return triples;
}

std::vector<PythonPolynomial> PythonCotangentH1Presentation::conormal_basis(
    std::size_t max_coordinate_entries) const {
  le::SparseEliminationLimits limits;
  limits.max_kernel_nonzeros = max_coordinate_entries;
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQCotangentH1>>(&value_)) {
    auto result = cotangent_without_gil(
        [&] { return (*native)->conormal_basis(limits); });
    return ring_->wrap_all(std::move(result));
  }
  const auto& native = std::get<std::shared_ptr<const GFCotangentH1>>(value_);
  auto result =
      cotangent_without_gil([&] { return native->conormal_basis(limits); });
  return ring_->wrap_all(std::move(result));
}

PythonPolynomial PythonCotangentH1Presentation::quotient_remainder(
    const PythonPolynomial& polynomial) const {
  require_polynomial(polynomial);
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQCotangentH1>>(&value_)) {
    auto result = without_gil([&] {
      return (*native)->quotient_remainder(
          std::get<QQPolynomial>(polynomial.value_));
    });
    return ring_->wrap(std::move(result));
  }
  const auto& native = std::get<std::shared_ptr<const GFCotangentH1>>(value_);
  auto result = without_gil([&] {
    return native->quotient_remainder(
        std::get<GFPolynomial>(polynomial.value_));
  });
  return ring_->wrap(std::move(result));
}

PythonCotangentClassProof PythonCotangentH1Presentation::verify_class(
    const PythonPolynomial& representative) const {
  require_polynomial(representative);
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQCotangentH1>>(&value_)) {
    auto proof = cotangent_without_gil([&] {
      return (*native)->verify_class(
          std::get<QQPolynomial>(representative.value_));
    });
    auto shared =
        std::make_shared<const QQCotangentClassProof>(std::move(proof));
    return PythonCotangentClassProof(
        ring_, PythonCotangentClassProof::Storage(std::move(shared)));
  }
  const auto& native = std::get<std::shared_ptr<const GFCotangentH1>>(value_);
  auto proof = cotangent_without_gil([&] {
    return native->verify_class(
        std::get<GFPolynomial>(representative.value_));
  });
  auto shared =
      std::make_shared<const GFCotangentClassProof>(std::move(proof));
  return PythonCotangentClassProof(
      ring_, PythonCotangentClassProof::Storage(std::move(shared)));
}

std::string PythonCotangentH1Presentation::to_string() const {
  return "CotangentH1Presentation(length_Q=" +
         std::to_string(quotient_length()) + ", conormal_dimension=" +
         std::to_string(conormal_dimension()) + ", h1_dimension=" +
         std::to_string(h1_dimension()) + ")";
}

std::string PythonCotangentClassProof::status() const {
  return std::visit(
      [](const auto& native) {
        return cotangent_class_status_name(native->status);
      },
      value_);
}

PythonPolynomial PythonCotangentClassProof::representative() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQCotangentClassProof>>(&value_)) {
    return ring_->wrap((*native)->representative);
  }
  return ring_->wrap(
      std::get<std::shared_ptr<const GFCotangentClassProof>>(value_)
          ->representative);
}

bool PythonCotangentClassProof::in_ideal() const noexcept {
  return std::visit(
      [](const auto& native) { return native->in_ideal; }, value_);
}

std::vector<PythonPolynomial>
PythonCotangentClassProof::derivative_remainders() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQCotangentClassProof>>(&value_)) {
    return ring_->wrap_all((*native)->derivative_remainders);
  }
  return ring_->wrap_all(
      std::get<std::shared_ptr<const GFCotangentClassProof>>(value_)
          ->derivative_remainders);
}

bool PythonCotangentClassProof::cycle() const noexcept {
  return std::visit(
      [](const auto& native) { return native->cycle_valid; }, value_);
}

std::optional<PythonSparseExactMatrix>
PythonCotangentClassProof::multiplication_matrix() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQCotangentClassProof>>(&value_)) {
    if (!(*native)->multiplication_matrix.has_value()) {
      return std::nullopt;
    }
    return PythonSparseExactMatrix::from_qq(
        *native, *(*native)->multiplication_matrix);
  }
  const auto& native =
      std::get<std::shared_ptr<const GFCotangentClassProof>>(value_);
  if (!native->multiplication_matrix.has_value()) {
    return std::nullopt;
  }
  return PythonSparseExactMatrix::from_gf(
      native, *native->multiplication_matrix);
}

std::optional<std::size_t> PythonCotangentClassProof::rank() const noexcept {
  return std::visit(
      [](const auto& native) { return native->multiplication_rank; }, value_);
}

std::optional<std::size_t> PythonCotangentClassProof::ann() const noexcept {
  return std::visit(
      [](const auto& native) { return native->annihilator_dimension; }, value_);
}

std::vector<PythonPolynomial>
PythonCotangentClassProof::annihilator_basis() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQCotangentClassProof>>(&value_)) {
    return ring_->wrap_all((*native)->annihilator_basis);
  }
  return ring_->wrap_all(
      std::get<std::shared_ptr<const GFCotangentClassProof>>(value_)
          ->annihilator_basis);
}

std::vector<PythonPolynomial>
PythonCotangentClassProof::colon_generators() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQCotangentClassProof>>(&value_)) {
    return ring_->wrap_all((*native)->colon_generators);
  }
  return ring_->wrap_all(
      std::get<std::shared_ptr<const GFCotangentClassProof>>(value_)
          ->colon_generators);
}

bool PythonCotangentClassProof::faithful() const noexcept {
  return std::visit(
      [](const auto& native) { return native->faithful; }, value_);
}

bool PythonCotangentClassProof::colon_equals() const noexcept {
  return std::visit(
      [](const auto& native) { return native->colon_equals_ideal; }, value_);
}

bool PythonCotangentClassProof::conclusive() const noexcept {
  return std::visit(
      [](const auto& native) { return native->conclusive(); }, value_);
}

std::string PythonCotangentClassProof::to_string() const {
  return "CotangentClassProof(status='" + status() + "', faithful=" +
         std::string(faithful() ? "True" : "False") + ")";
}

std::vector<PythonPolynomial> PythonLocalIdeal::lower_generators() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQLocalIdeal>>(&value_)) {
    const auto values = (*native)->lower_generators();
    return ring_->wrap_all(
        std::vector<QQPolynomial>(values.begin(), values.end()));
  }
  const auto& native = std::get<std::shared_ptr<const GFLocalIdeal>>(value_);
  const auto values = native->lower_generators();
  return ring_->wrap_all(
      std::vector<GFPolynomial>(values.begin(), values.end()));
}

std::vector<PythonPolynomial> PythonLocalIdeal::generators() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQLocalIdeal>>(&value_)) {
    return ring_->wrap_all((*native)->generators());
  }
  return ring_->wrap_all(
      std::get<std::shared_ptr<const GFLocalIdeal>>(value_)->generators());
}

std::size_t PythonLocalIdeal::maximal_power() const noexcept {
  return std::visit(
      [](const auto& native) { return native->maximal_power(); }, value_);
}

PythonFiniteQuotient PythonLocalIdeal::quotient(
    std::size_t max_monomials,
    std::optional<std::size_t> max_generated_rows,
    std::optional<std::size_t> max_matrix_triplets) const {
  le::CotangentH1Options options;
  options.monomial_space_limits.maximum_monomials = max_monomials;
  options.max_generated_rows = max_generated_rows;
  options.max_matrix_triplets = max_matrix_triplets;
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQLocalIdeal>>(&value_)) {
    auto quotient = cotangent_without_gil(
        [&] { return (*native)->quotient(options); });
    auto shared =
        std::make_shared<const QQFiniteQuotient>(std::move(quotient));
    return PythonFiniteQuotient(
        ring_, PythonFiniteQuotient::Storage(std::move(shared)));
  }
  const auto& native = std::get<std::shared_ptr<const GFLocalIdeal>>(value_);
  auto quotient =
      cotangent_without_gil([&] { return native->quotient(options); });
  auto shared =
      std::make_shared<const GFFiniteQuotient>(std::move(quotient));
  return PythonFiniteQuotient(
      ring_, PythonFiniteQuotient::Storage(std::move(shared)));
}

bool PythonLocalIdeal::equals(const PythonLocalIdeal& other) const noexcept {
  if (ring_.get() != other.ring_.get() ||
      value_.index() != other.value_.index()) {
    return false;
  }
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQLocalIdeal>>(&value_)) {
    return (*native)->same_presentation(
        *std::get<std::shared_ptr<const QQLocalIdeal>>(other.value_));
  }
  return std::get<std::shared_ptr<const GFLocalIdeal>>(value_)
      ->same_presentation(
          *std::get<std::shared_ptr<const GFLocalIdeal>>(other.value_));
}

std::string PythonLocalIdeal::to_string() const {
  const auto lower_count = std::visit(
      [](const auto& native) { return native->lower_generators().size(); },
      value_);
  return "LocalIdeal(lower_generators=" +
         std::to_string(lower_count) + ", maximal_power=" +
         std::to_string(maximal_power()) + ")";
}

void PythonFiniteQuotient::require_polynomial(
    const PythonPolynomial& polynomial) const {
  if (polynomial.ring().get() != ring_.get()) {
    throw std::invalid_argument(
        "finite-quotient polynomials must belong to its exact ring context");
  }
}

PythonLocalIdeal PythonFiniteQuotient::defining_ideal() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQFiniteQuotient>>(&value_)) {
    auto ideal = std::make_shared<const QQLocalIdeal>(
        (*native)->defining_ideal());
    return PythonLocalIdeal(
        ring_, PythonLocalIdeal::Storage(std::move(ideal)));
  }
  auto ideal = std::make_shared<const GFLocalIdeal>(
      std::get<std::shared_ptr<const GFFiniteQuotient>>(value_)
          ->defining_ideal());
  return PythonLocalIdeal(
      ring_, PythonLocalIdeal::Storage(std::move(ideal)));
}

std::size_t PythonFiniteQuotient::length() const noexcept {
  return std::visit(
      [](const auto& native) { return native->dimension(); }, value_);
}

std::size_t PythonFiniteQuotient::square_quotient_length() const {
  return cotangent_without_gil([&] {
    return std::visit(
        [](const auto& native) {
          return native->square_quotient_dimension();
        },
        value_);
  });
}

PythonPolynomial PythonFiniteQuotient::remainder(
    const PythonPolynomial& polynomial) const {
  require_polynomial(polynomial);
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQFiniteQuotient>>(&value_)) {
    auto result = without_gil([&] {
      return (*native)->remainder(
          std::get<QQPolynomial>(polynomial.value_));
    });
    return ring_->wrap(std::move(result));
  }
  auto result = without_gil([&] {
    return std::get<std::shared_ptr<const GFFiniteQuotient>>(value_)
        ->remainder(std::get<GFPolynomial>(polynomial.value_));
  });
  return ring_->wrap(std::move(result));
}

std::vector<PythonPolynomial> PythonFiniteQuotient::basis(
    std::size_t max_coordinate_entries) const {
  require_coordinate_budget(
      length(), 1, max_coordinate_entries, "explicit quotient basis");
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQFiniteQuotient>>(&value_)) {
    return ring_->wrap_all((*native)->basis_representatives());
  }
  return ring_->wrap_all(
      std::get<std::shared_ptr<const GFFiniteQuotient>>(value_)
          ->basis_representatives());
}

PythonQuotientIdeal PythonFiniteQuotient::zero_ideal() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQFiniteQuotient>>(&value_)) {
    auto ideal = std::make_shared<const QQQuotientIdeal>(
        (*native)->zero_ideal());
    return PythonQuotientIdeal(
        ring_, PythonQuotientIdeal::Storage(std::move(ideal)));
  }
  auto ideal = std::make_shared<const GFQuotientIdeal>(
      std::get<std::shared_ptr<const GFFiniteQuotient>>(value_)
          ->zero_ideal());
  return PythonQuotientIdeal(
      ring_, PythonQuotientIdeal::Storage(std::move(ideal)));
}

PythonConormalModule PythonFiniteQuotient::conormal_module() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQFiniteQuotient>>(&value_)) {
    auto native_module = cotangent_without_gil(
        [&] { return (*native)->conormal_module(); });
    auto module = std::make_shared<const QQConormalModule>(
        std::move(native_module));
    return PythonConormalModule(
        ring_, PythonConormalModule::Storage(std::move(module)));
  }
  const auto& native =
      std::get<std::shared_ptr<const GFFiniteQuotient>>(value_);
  auto native_module = cotangent_without_gil(
      [&] { return native->conormal_module(); });
  auto module = std::make_shared<const GFConormalModule>(
      std::move(native_module));
  return PythonConormalModule(
      ring_, PythonConormalModule::Storage(std::move(module)));
}

std::string PythonFiniteQuotient::to_string() const {
  return "FiniteQuotient(length=" + std::to_string(length()) + ")";
}

PythonFiniteQuotient PythonConormalModule::quotient() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQConormalModule>>(&value_)) {
    auto quotient =
        std::make_shared<const QQFiniteQuotient>((*native)->algebra());
    return PythonFiniteQuotient(
        ring_, PythonFiniteQuotient::Storage(std::move(quotient)));
  }
  auto quotient = std::make_shared<const GFFiniteQuotient>(
      std::get<std::shared_ptr<const GFConormalModule>>(value_)->algebra());
  return PythonFiniteQuotient(
      ring_, PythonFiniteQuotient::Storage(std::move(quotient)));
}

std::size_t PythonConormalModule::dimension() const noexcept {
  return std::visit(
      [](const auto& native) { return native->dimension(); }, value_);
}

PythonSparseExactMatrix PythonConormalModule::constraint_matrix() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQConormalModule>>(&value_)) {
    return PythonSparseExactMatrix::from_qq(
        *native, (*native)->defining_matrix());
  }
  const auto& native =
      std::get<std::shared_ptr<const GFConormalModule>>(value_);
  return PythonSparseExactMatrix::from_gf(
      native, native->defining_matrix());
}

std::vector<PythonPolynomial> PythonConormalModule::basis(
    std::size_t max_coordinate_entries) const {
  le::SparseEliminationLimits limits;
  limits.max_kernel_nonzeros = max_coordinate_entries;
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQConormalModule>>(&value_)) {
    auto result = cotangent_without_gil(
        [&] { return (*native)->representative_basis(limits); });
    return ring_->wrap_all(std::move(result));
  }
  const auto& native =
      std::get<std::shared_ptr<const GFConormalModule>>(value_);
  auto result = cotangent_without_gil(
      [&] { return native->representative_basis(limits); });
  return ring_->wrap_all(std::move(result));
}

PythonConormalDerivativeMap PythonConormalModule::derivative_map() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQConormalModule>>(&value_)) {
    auto native_map = cotangent_without_gil(
        [&] { return (*native)->derivative_map(); });
    auto map = std::make_shared<const QQConormalDerivativeMap>(
        std::move(native_map));
    return PythonConormalDerivativeMap(
        ring_, PythonConormalDerivativeMap::Storage(std::move(map)));
  }
  const auto& native =
      std::get<std::shared_ptr<const GFConormalModule>>(value_);
  auto native_map = cotangent_without_gil(
      [&] { return native->derivative_map(); });
  auto map = std::make_shared<const GFConormalDerivativeMap>(
      std::move(native_map));
  return PythonConormalDerivativeMap(
      ring_, PythonConormalDerivativeMap::Storage(std::move(map)));
}

std::string PythonConormalModule::to_string() const {
  return "ConormalModule(dimension=" + std::to_string(dimension()) + ")";
}

PythonConormalModule PythonConormalDerivativeMap::domain() const {
  if (const auto* native = std::get_if<
          std::shared_ptr<const QQConormalDerivativeMap>>(&value_)) {
    auto module =
        std::make_shared<const QQConormalModule>((*native)->domain());
    return PythonConormalModule(
        ring_, PythonConormalModule::Storage(std::move(module)));
  }
  auto module = std::make_shared<const GFConormalModule>(
      std::get<std::shared_ptr<const GFConormalDerivativeMap>>(value_)
          ->domain());
  return PythonConormalModule(
      ring_, PythonConormalModule::Storage(std::move(module)));
}

PythonSparseExactMatrix PythonConormalDerivativeMap::ambient_matrix() const {
  if (const auto* native = std::get_if<
          std::shared_ptr<const QQConormalDerivativeMap>>(&value_)) {
    return PythonSparseExactMatrix::from_qq(
        *native, (*native)->ambient_matrix());
  }
  const auto& native =
      std::get<std::shared_ptr<const GFConormalDerivativeMap>>(value_);
  return PythonSparseExactMatrix::from_gf(
      native, native->ambient_matrix());
}

PythonH1Module PythonConormalDerivativeMap::kernel() const {
  if (const auto* native = std::get_if<
          std::shared_ptr<const QQConormalDerivativeMap>>(&value_)) {
    auto native_module = cotangent_without_gil(
        [&] { return (*native)->kernel(); });
    auto module = std::make_shared<const QQH1Module>(
        std::move(native_module));
    return PythonH1Module(
        ring_, PythonH1Module::Storage(std::move(module)));
  }
  const auto& native =
      std::get<std::shared_ptr<const GFConormalDerivativeMap>>(value_);
  auto native_module = cotangent_without_gil(
      [&] { return native->kernel(); });
  auto module = std::make_shared<const GFH1Module>(
      std::move(native_module));
  return PythonH1Module(
      ring_, PythonH1Module::Storage(std::move(module)));
}

std::string PythonConormalDerivativeMap::to_string() const {
  const auto shape = ambient_matrix().shape();
  return "ConormalDerivativeMap(shape=(" +
         std::to_string(shape[0].cast<std::size_t>()) + ", " +
         std::to_string(shape[1].cast<std::size_t>()) + "))";
}

PythonFiniteQuotient PythonH1Module::quotient() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQH1Module>>(&value_)) {
    auto quotient =
        std::make_shared<const QQFiniteQuotient>((*native)->algebra());
    return PythonFiniteQuotient(
        ring_, PythonFiniteQuotient::Storage(std::move(quotient)));
  }
  auto quotient = std::make_shared<const GFFiniteQuotient>(
      std::get<std::shared_ptr<const GFH1Module>>(value_)->algebra());
  return PythonFiniteQuotient(
      ring_, PythonFiniteQuotient::Storage(std::move(quotient)));
}

PythonConormalModule PythonH1Module::conormal_module() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQH1Module>>(&value_)) {
    auto module =
        std::make_shared<const QQConormalModule>((*native)->conormal());
    return PythonConormalModule(
        ring_, PythonConormalModule::Storage(std::move(module)));
  }
  auto module = std::make_shared<const GFConormalModule>(
      std::get<std::shared_ptr<const GFH1Module>>(value_)->conormal());
  return PythonConormalModule(
      ring_, PythonConormalModule::Storage(std::move(module)));
}

std::size_t PythonH1Module::dimension() const noexcept {
  return std::visit(
      [](const auto& native) { return native->dimension(); }, value_);
}

std::size_t PythonH1Module::ambient_dimension() const noexcept {
  return std::visit(
      [](const auto& native) { return native->ambient_dimension(); }, value_);
}

PythonSparseExactMatrix PythonH1Module::constraint_matrix() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQH1Module>>(&value_)) {
    return PythonSparseExactMatrix::from_qq(
        *native, (*native)->defining_matrix());
  }
  const auto& native = std::get<std::shared_ptr<const GFH1Module>>(value_);
  return PythonSparseExactMatrix::from_gf(
      native, native->defining_matrix());
}

py::list PythonH1Module::kernel_coordinates(
    std::size_t max_coordinate_entries) const {
  le::SparseEliminationLimits limits;
  limits.max_kernel_nonzeros = max_coordinate_entries;
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQH1Module>>(&value_)) {
    const auto rows = cotangent_without_gil(
        [&] { return (*native)->basis_coordinates(limits); });
    py::list triples;
    for (std::size_t basis = 0; basis < rows.size(); ++basis) {
      for (const auto& entry : rows[basis]) {
        triples.append(py::make_tuple(
            basis, entry.column,
            python_exact_value(
                (*native)->algebra().field(), entry.value)));
      }
    }
    return triples;
  }
  const auto& native = std::get<std::shared_ptr<const GFH1Module>>(value_);
  const auto rows = cotangent_without_gil(
      [&] { return native->basis_coordinates(limits); });
  py::list triples;
  for (std::size_t basis = 0; basis < rows.size(); ++basis) {
    for (const auto& entry : rows[basis]) {
      triples.append(py::make_tuple(
          basis, entry.column,
          python_exact_value(native->algebra().field(), entry.value)));
    }
  }
  return triples;
}

std::vector<PythonPolynomial> PythonH1Module::basis(
    std::size_t max_coordinate_entries) const {
  le::SparseEliminationLimits limits;
  limits.max_kernel_nonzeros = max_coordinate_entries;
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQH1Module>>(&value_)) {
    auto result = cotangent_without_gil(
        [&] { return (*native)->representative_basis(limits); });
    return ring_->wrap_all(std::move(result));
  }
  const auto& native = std::get<std::shared_ptr<const GFH1Module>>(value_);
  auto result = cotangent_without_gil(
      [&] { return native->representative_basis(limits); });
  return ring_->wrap_all(std::move(result));
}

void PythonH1Module::require_polynomial(
    const PythonPolynomial& polynomial) const {
  if (polynomial.ring().get() != ring_.get()) {
    throw std::invalid_argument(
        "cotangent-H1 representatives must belong to its exact ring context");
  }
}

PythonH1Element PythonH1Module::class_of(
    const PythonPolynomial& representative) const {
  require_polynomial(representative);
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQH1Module>>(&value_)) {
    auto element = cotangent_without_gil([&] {
      return (*native)->class_of(
          std::get<QQPolynomial>(representative.value_));
    });
    auto shared = std::make_shared<const QQH1Element>(std::move(element));
    return PythonH1Element(
        ring_, PythonH1Element::Storage(std::move(shared)));
  }
  const auto& native = std::get<std::shared_ptr<const GFH1Module>>(value_);
  auto element = cotangent_without_gil([&] {
    return native->class_of(
        std::get<GFPolynomial>(representative.value_));
  });
  auto shared = std::make_shared<const GFH1Element>(std::move(element));
  return PythonH1Element(
      ring_, PythonH1Element::Storage(std::move(shared)));
}

std::string PythonH1Module::to_string() const {
  return "H1Module(dimension=" + std::to_string(dimension()) + ")";
}

PythonH1Module PythonH1Element::module() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQH1Element>>(&value_)) {
    auto module = std::make_shared<const QQH1Module>((*native)->module());
    return PythonH1Module(
        ring_, PythonH1Module::Storage(std::move(module)));
  }
  auto module = std::make_shared<const GFH1Module>(
      std::get<std::shared_ptr<const GFH1Element>>(value_)->module());
  return PythonH1Module(
      ring_, PythonH1Module::Storage(std::move(module)));
}

PythonPolynomial PythonH1Element::representative() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQH1Element>>(&value_)) {
    return ring_->wrap((*native)->representative());
  }
  return ring_->wrap(
      std::get<std::shared_ptr<const GFH1Element>>(value_)
          ->representative());
}

py::list PythonH1Element::coordinates() const {
  py::list result;
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQH1Element>>(&value_)) {
    const auto field = (*native)->module().algebra().field();
    for (const auto& entry : (*native)->coordinates()) {
      result.append(py::make_tuple(
          entry.column, python_exact_value(field, entry.value)));
    }
    return result;
  }
  const auto& native = std::get<std::shared_ptr<const GFH1Element>>(value_);
  const auto field = native->module().algebra().field();
  for (const auto& entry : native->coordinates()) {
    result.append(py::make_tuple(
        entry.column, python_exact_value(field, entry.value)));
  }
  return result;
}

PythonQuotientIdeal PythonH1Element::annihilator() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQH1Element>>(&value_)) {
    auto ideal = cotangent_without_gil(
        [&] { return (*native)->annihilator(); });
    auto shared = std::make_shared<const QQQuotientIdeal>(std::move(ideal));
    return PythonQuotientIdeal(
        ring_, PythonQuotientIdeal::Storage(std::move(shared)));
  }
  const auto& native = std::get<std::shared_ptr<const GFH1Element>>(value_);
  auto ideal = cotangent_without_gil([&] { return native->annihilator(); });
  auto shared = std::make_shared<const GFQuotientIdeal>(std::move(ideal));
  return PythonQuotientIdeal(
      ring_, PythonQuotientIdeal::Storage(std::move(shared)));
}

bool PythonH1Element::equals(const PythonH1Element& other) const noexcept {
  if (ring_.get() != other.ring_.get() ||
      value_.index() != other.value_.index()) {
    return false;
  }
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQH1Element>>(&value_)) {
    return **native ==
           *std::get<std::shared_ptr<const QQH1Element>>(other.value_);
  }
  return *std::get<std::shared_ptr<const GFH1Element>>(value_) ==
         *std::get<std::shared_ptr<const GFH1Element>>(other.value_);
}

std::string PythonH1Element::to_string() const {
  // Keep repr bounded even when a representative has hundreds of terms.
  const auto term_count = std::visit(
      [](const auto& native) {
        return native->representative().term_count();
      },
      value_);
  return "H1Element(representative_terms=" +
         std::to_string(term_count) + ")";
}

PythonFiniteQuotient PythonQuotientIdeal::quotient() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQQuotientIdeal>>(&value_)) {
    auto quotient =
        std::make_shared<const QQFiniteQuotient>((*native)->quotient());
    return PythonFiniteQuotient(
        ring_, PythonFiniteQuotient::Storage(std::move(quotient)));
  }
  auto quotient = std::make_shared<const GFFiniteQuotient>(
      std::get<std::shared_ptr<const GFQuotientIdeal>>(value_)->quotient());
  return PythonFiniteQuotient(
      ring_, PythonFiniteQuotient::Storage(std::move(quotient)));
}

std::size_t PythonQuotientIdeal::dimension() const noexcept {
  return std::visit(
      [](const auto& native) { return native->dimension(); }, value_);
}

bool PythonQuotientIdeal::is_zero() const noexcept {
  return std::visit(
      [](const auto& native) { return native->is_zero(); }, value_);
}

py::list PythonQuotientIdeal::basis_coordinates() const {
  py::list rows;
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQQuotientIdeal>>(&value_)) {
    const auto field = (*native)->quotient().field();
    for (const auto& row : (*native)->basis_coordinates()) {
      py::list values;
      for (const auto& value : row) {
        values.append(python_exact_value(field, value));
      }
      rows.append(std::move(values));
    }
    return rows;
  }
  const auto& native =
      std::get<std::shared_ptr<const GFQuotientIdeal>>(value_);
  const auto field = native->quotient().field();
  for (const auto& row : native->basis_coordinates()) {
    py::list values;
    for (const auto& value : row) {
      values.append(python_exact_value(field, value));
    }
    rows.append(std::move(values));
  }
  return rows;
}

std::vector<PythonPolynomial> PythonQuotientIdeal::generators() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQQuotientIdeal>>(&value_)) {
    const auto& values = (*native)->lift_basis();
    return ring_->wrap_all(
        std::vector<QQPolynomial>(values.begin(), values.end()));
  }
  const auto& values =
      std::get<std::shared_ptr<const GFQuotientIdeal>>(value_)->lift_basis();
  return ring_->wrap_all(
      std::vector<GFPolynomial>(values.begin(), values.end()));
}

PythonIdealPreimage PythonQuotientIdeal::preimage() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQQuotientIdeal>>(&value_)) {
    auto preimage =
        std::make_shared<const QQIdealPreimage>((*native)->preimage());
    return PythonIdealPreimage(
        ring_, PythonIdealPreimage::Storage(std::move(preimage)));
  }
  auto preimage = std::make_shared<const GFIdealPreimage>(
      std::get<std::shared_ptr<const GFQuotientIdeal>>(value_)->preimage());
  return PythonIdealPreimage(
      ring_, PythonIdealPreimage::Storage(std::move(preimage)));
}

bool PythonQuotientIdeal::equals(
    const PythonQuotientIdeal& other) const noexcept {
  if (ring_.get() != other.ring_.get() ||
      value_.index() != other.value_.index()) {
    return false;
  }
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQQuotientIdeal>>(&value_)) {
    return **native ==
           *std::get<std::shared_ptr<const QQQuotientIdeal>>(other.value_);
  }
  return *std::get<std::shared_ptr<const GFQuotientIdeal>>(value_) ==
         *std::get<std::shared_ptr<const GFQuotientIdeal>>(other.value_);
}

std::string PythonQuotientIdeal::to_string() const {
  return "QuotientIdeal(dimension=" + std::to_string(dimension()) + ")";
}

PythonQuotientIdeal PythonIdealPreimage::quotient_ideal() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQIdealPreimage>>(&value_)) {
    auto ideal = std::make_shared<const QQQuotientIdeal>(
        (*native)->quotient_ideal());
    return PythonQuotientIdeal(
        ring_, PythonQuotientIdeal::Storage(std::move(ideal)));
  }
  auto ideal = std::make_shared<const GFQuotientIdeal>(
      std::get<std::shared_ptr<const GFIdealPreimage>>(value_)
          ->quotient_ideal());
  return PythonQuotientIdeal(
      ring_, PythonQuotientIdeal::Storage(std::move(ideal)));
}

PythonLocalIdeal PythonIdealPreimage::source_ideal() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQIdealPreimage>>(&value_)) {
    auto ideal =
        std::make_shared<const QQLocalIdeal>((*native)->source_ideal());
    return PythonLocalIdeal(
        ring_, PythonLocalIdeal::Storage(std::move(ideal)));
  }
  auto ideal = std::make_shared<const GFLocalIdeal>(
      std::get<std::shared_ptr<const GFIdealPreimage>>(value_)
          ->source_ideal());
  return PythonLocalIdeal(
      ring_, PythonLocalIdeal::Storage(std::move(ideal)));
}

std::vector<PythonPolynomial> PythonIdealPreimage::generators() const {
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQIdealPreimage>>(&value_)) {
    return ring_->wrap_all((*native)->generators());
  }
  return ring_->wrap_all(
      std::get<std::shared_ptr<const GFIdealPreimage>>(value_)
          ->generators());
}

bool PythonIdealPreimage::equals_source_ideal() const noexcept {
  return std::visit(
      [](const auto& native) { return native->equals_source_ideal(); }, value_);
}

bool PythonIdealPreimage::equals(
    const PythonIdealPreimage& other) const noexcept {
  if (ring_.get() != other.ring_.get() ||
      value_.index() != other.value_.index()) {
    return false;
  }
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQIdealPreimage>>(&value_)) {
    return **native ==
           *std::get<std::shared_ptr<const QQIdealPreimage>>(other.value_);
  }
  return *std::get<std::shared_ptr<const GFIdealPreimage>>(value_) ==
         *std::get<std::shared_ptr<const GFIdealPreimage>>(other.value_);
}

bool PythonIdealPreimage::equals(
    const PythonLocalIdeal& other) const noexcept {
  if (ring_.get() != other.ring_.get() ||
      value_.index() != other.value_.index()) {
    return false;
  }
  if (const auto* native =
          std::get_if<std::shared_ptr<const QQIdealPreimage>>(&value_)) {
    return **native ==
           *std::get<std::shared_ptr<const QQLocalIdeal>>(other.value_);
  }
  return *std::get<std::shared_ptr<const GFIdealPreimage>>(value_) ==
         *std::get<std::shared_ptr<const GFLocalIdeal>>(other.value_);
}

std::string PythonIdealPreimage::to_string() const {
  const auto additional_dimension = std::visit(
      [](const auto& native) {
        return native->quotient_ideal().dimension();
      },
      value_);
  return "IdealPreimage(additional_dimension=" +
         std::to_string(additional_dimension) + ")";
}

[[nodiscard]] std::string cycle_status_name(le::CycleAuditStatus status) {
  switch (status) {
    case le::CycleAuditStatus::Complete:
      return "complete";
    case le::CycleAuditStatus::PolynomialNotInIdeal:
      return "polynomial_not_in_ideal";
    case le::CycleAuditStatus::PositiveDimensional:
      return "positive_dimensional";
    case le::CycleAuditStatus::UnsupportedAtOrigin:
      return "unsupported_at_origin";
    case le::CycleAuditStatus::UnitIdeal:
      return "unit_ideal";
    case le::CycleAuditStatus::ResourceLimit:
      return "resource_limit";
  }
  throw std::logic_error("unknown cycle-audit status");
}

[[nodiscard]] std::string closure_status_name(
    le::ColonClosureStopStatus status) {
  switch (status) {
    case le::ColonClosureStopStatus::ProperFixedPoint:
      return "proper_fixed_point";
    case le::ColonClosureStopStatus::UnitIdeal:
      return "unit_ideal";
    case le::ColonClosureStopStatus::ResourceLimit:
      return "resource_limit";
    case le::ColonClosureStopStatus::InvalidStart:
      return "invalid_start";
  }
  throw std::logic_error("unknown colon-closure status");
}

[[nodiscard]] std::string matrix_rank_proof_name(
    le::MatrixSpaceRankProof proof) {
  switch (proof) {
    case le::MatrixSpaceRankProof::ProvenMaximum:
      return "proven_maximum";
    case le::MatrixSpaceRankProof::ProvenFullColumnRank:
      return "proven_full_column_rank";
    case le::MatrixSpaceRankProof::GenericOnly:
      return "generic_only";
    case le::MatrixSpaceRankProof::ResourceLimit:
      return "resource_limit";
  }
  throw std::logic_error("unknown matrix-space rank proof");
}

[[nodiscard]] std::string discovery_status_name(
    le::DiscoveryScreenStatus status) {
  switch (status) {
    case le::DiscoveryScreenStatus::Complete:
      return "complete";
    case le::DiscoveryScreenStatus::InvalidInput:
      return "invalid_input";
    case le::DiscoveryScreenStatus::ResourceLimit:
      return "resource_limit";
  }
  throw std::logic_error("unknown discovery-screen status");
}

[[nodiscard]] std::string apolarity_convention_name(
    le::ApolarityConvention convention) {
  switch (convention) {
    case le::ApolarityConvention::OrdinaryDifferentiation:
      return "ordinary";
    case le::ApolarityConvention::DividedPowers:
      return "divided_powers";
  }
  throw std::logic_error("unknown apolarity convention");
}

[[nodiscard]] le::ApolarityConvention parse_apolarity_convention(
    std::string_view convention) {
  if (convention == "ordinary" || convention == "ordinary_differentiation") {
    return le::ApolarityConvention::OrdinaryDifferentiation;
  }
  if (convention == "divided" || convention == "divided_power" ||
      convention == "divided_powers") {
    return le::ApolarityConvention::DividedPowers;
  }
  throw std::invalid_argument(
      "convention must be 'ordinary' or 'divided_powers'");
}

[[nodiscard]] std::vector<std::vector<std::uint16_t>> wrap_exponents(
    std::span<const le::InverseSystemExponent> exponents,
    std::size_t variable_count) {
  std::vector<std::vector<std::uint16_t>> result;
  result.reserve(exponents.size());
  for (const auto& exponent : exponents) {
    result.emplace_back(
        exponent.begin(), exponent.begin() + variable_count);
  }
  return result;
}

template <typename Field>
[[nodiscard]] PythonCycleAuditResult wrap_cycle_audit(
    const std::shared_ptr<PythonRing>& ring,
    const le::CycleAuditResult<Field>& result) {
  std::optional<PythonPolynomial> polynomial;
  std::optional<PythonPolynomial> polynomial_remainder;
  std::vector<PythonPolynomial> derivatives;
  std::vector<PythonPolynomial> derivative_remainders;
  if (result.membership_evidence().has_value()) {
    const auto& evidence = *result.membership_evidence();
    polynomial = ring->wrap(evidence.polynomial());
    polynomial_remainder = ring->wrap(evidence.polynomial_remainder());
    derivatives = ring->wrap_all(evidence.derivatives());
    derivative_remainders =
        ring->wrap_all(evidence.derivative_remainders());
  }

  std::optional<PythonIdeal> maximal_times_ideal;
  if (result.maximal_times_ideal().has_value()) {
    maximal_times_ideal = ring->wrap(*result.maximal_times_ideal());
  }
  std::optional<PythonIdeal> squared_ideal;
  if (result.ideal_square().has_value()) {
    squared_ideal = ring->wrap(*result.ideal_square());
  }
  std::optional<PythonExactMatrix> colon_multiplication_matrix;
  std::optional<PythonExactMatrix> colon_kernel_basis;
  std::vector<PythonPolynomial> colon_kernel_lifts;
  std::vector<PythonPolynomial> colon_lift_product_remainders;
  std::optional<PythonIdeal> colon_ideal;
  std::optional<std::size_t> annihilator_length;
  std::optional<std::size_t> colon_quotient_length;
  std::vector<PythonPolynomial> ideal_in_colon_remainders;
  std::vector<PythonPolynomial> colon_in_ideal_remainders;
  if (result.colon_evidence().has_value()) {
    const auto& evidence = *result.colon_evidence();
    colon_multiplication_matrix =
        wrap_exact_matrix(ring, evidence.multiplication_matrix());
    colon_kernel_basis = wrap_exact_rows<Field>(
        ring, evidence.kernel_coordinates(),
        evidence.multiplication_matrix().column_count(),
        evidence.multiplication_matrix().field());
    colon_kernel_lifts = ring->wrap_all(evidence.kernel_lifts());
    colon_lift_product_remainders =
        ring->wrap_all(evidence.lift_product_remainders());
    colon_ideal = ring->wrap(evidence.colon_ideal());
    annihilator_length = evidence.annihilator_length();
    colon_quotient_length = evidence.colon_quotient_length();
    ideal_in_colon_remainders =
        ring->wrap_all(evidence.ideal_in_colon_remainders());
    colon_in_ideal_remainders =
        ring->wrap_all(evidence.colon_in_ideal_remainders());
  }

  return PythonCycleAuditResult{
      cycle_status_name(result.status()),
      ring->wrap(result.ideal()),
      result.finite_quotient(),
      result.supported_at_origin(),
      result.quotient_length(),
      std::move(polynomial),
      std::move(polynomial_remainder),
      std::move(derivatives),
      std::move(derivative_remainders),
      result.polynomial_in_ideal(),
      result.derivatives_in_ideal(),
      result.cycle_valid(),
      std::move(maximal_times_ideal),
      result.primitive(),
      std::move(squared_ideal),
      std::move(colon_multiplication_matrix),
      std::move(colon_kernel_basis),
      std::move(colon_kernel_lifts),
      std::move(colon_lift_product_remainders),
      std::move(colon_ideal),
      annihilator_length,
      colon_quotient_length,
      result.colon_equals_ideal(),
      result.annihilator_zero(),
      std::move(ideal_in_colon_remainders),
      std::move(colon_in_ideal_remainders),
      result.faithful_cycle(),
      result.conclusive(),
      result.resource_detail()};
}

template <typename Field>
[[nodiscard]] PythonColonClosureResult wrap_colon_closure(
    const std::shared_ptr<PythonRing>& ring,
    const le::ColonClosureResult<Field>& result) {
  std::vector<PythonColonClosureStep> steps;
  steps.reserve(result.steps().size());
  for (const auto& step : result.steps()) {
    steps.push_back(PythonColonClosureStep{
        step.index(), ring->wrap(step.ideal()), step.quotient_length()});
  }
  std::vector<PythonColonClosureTransition> transitions;
  transitions.reserve(result.transitions().size());
  for (const auto& transition : result.transitions()) {
    transitions.push_back(PythonColonClosureTransition{
        ring->wrap_all(transition.current_in_next_remainders()),
        ring->wrap_all(transition.next_in_current_remainders()),
        transition.current_subset_next(), transition.equal()});
  }
  return PythonColonClosureResult{
      closure_status_name(result.status()), std::move(steps),
      std::move(transitions), result.conclusive(),
      result.faithful_fixed_point_found(), result.resource_detail()};
}

template <typename Field>
[[nodiscard]] PythonMatrixSpaceRankResult wrap_matrix_rank(
    const std::shared_ptr<PythonRing>& ring,
    const le::MatrixSpaceRankResult<Field>& result,
    const Field& field) {
  std::optional<std::vector<std::string>> coefficient_text;
  if (result.witness_coefficients.has_value()) {
    coefficient_text.emplace();
    coefficient_text->reserve(result.witness_coefficients->size());
    for (const auto& coefficient : *result.witness_coefficients) {
      coefficient_text->push_back(field.to_string(coefficient));
    }
  }
  return PythonMatrixSpaceRankResult{
      ring,
      result.parameter_count,
      result.row_count,
      result.column_count,
      result.lower_bound,
      result.upper_bound,
      result.generic_maximum,
      result.exact_maximum,
      matrix_rank_proof_name(result.proof),
      std::move(coefficient_text),
      result.minors_tested,
      result.determinant_products_tested,
      result.finite_field_evaluations,
      result.has_full_column_rank_witness()};
}

py::object PythonMatrixSpaceRankResult::witness_coefficients() const {
  if (!witness_coefficient_text.has_value()) {
    return py::none();
  }
  py::list result;
  if (ring->characteristic() == 0) {
    const auto fraction = py::module_::import("fractions").attr("Fraction");
    for (const auto& coefficient : *witness_coefficient_text) {
      result.append(fraction(py::str(coefficient)));
    }
  } else {
    for (const auto& coefficient : *witness_coefficient_text) {
      result.append(py::int_(std::stoll(coefficient)));
    }
  }
  return std::move(result);
}

template <typename Field>
[[nodiscard]] PythonH1ActionResult wrap_h1_action(
    const std::shared_ptr<PythonRing>& ring,
    const le::H1ActionData<Field>& result) {
  std::string status = "complete";
  if (result.individual_rank.proof ==
          le::MatrixSpaceRankProof::ResourceLimit ||
      (result.faithful_witness_audit.has_value() &&
       result.faithful_witness_audit->status() ==
           le::CycleAuditStatus::ResourceLimit)) {
    status = "resource_limit";
  } else if (result.individual_rank.proof ==
             le::MatrixSpaceRankProof::GenericOnly) {
    status = "generic_only";
  }
  std::optional<PythonPolynomial> best_polynomial;
  if (result.best_h1_polynomial.has_value()) {
    best_polynomial = ring->wrap(*result.best_h1_polynomial);
  }
  std::optional<PythonCycleAuditResult> faithful_audit;
  if (result.faithful_witness_audit.has_value()) {
    faithful_audit =
        wrap_cycle_audit(ring, *result.faithful_witness_audit);
  }
  return PythonH1ActionResult{
      std::move(status),
      result.length_Q,
      result.length_P_mod_J2,
      result.conormal_dimension,
      result.h1_dimension,
      result.socle_dimension,
      ring->wrap(result.ideal),
      ring->wrap(result.squared_ideal),
      ring->wrap_all(result.conormal_basis),
      ring->wrap_all(result.h1_basis),
      ring->wrap_all(result.socle_basis),
      wrap_exact_matrix(ring, result.reduction_map),
      wrap_exact_matrix(ring, result.differential),
      wrap_exact_matrices(ring, result.variable_multiplication_matrices),
      wrap_exact_matrices(ring, result.h1_multiplication_matrices),
      wrap_exact_matrices(ring, result.action_matrices),
      result.common_product_space_dimension,
      result.common_product_space_rank_bound,
      wrap_matrix_rank(ring, result.individual_rank,
                       result.quotient.field()),
      std::move(best_polynomial),
      std::move(faithful_audit),
      ring->wrap(result.common_annihilator_diagnostic)};
}

template <typename Field>
[[nodiscard]] PythonInverseSystemResult wrap_inverse_system(
    const std::shared_ptr<PythonRing>& operator_ring,
    const std::vector<PythonPolynomial>& dual_generators,
    const le::InverseSystemData<Field>& result) {
  if (result.dual_generators.size() != dual_generators.size()) {
    throw std::logic_error(
        "native inverse-system result changed its dual-generator count");
  }
  return PythonInverseSystemResult{
      apolarity_convention_name(result.convention),
      dual_generators,
      result.maximum_degree,
      wrap_exponents(
          result.operator_exponents, operator_ring->variable_names().size()),
      wrap_exponents(
          result.output_exponents, operator_ring->variable_names().size()),
      result.action_matrix.row_count(),
      result.action_matrix.column_count(),
      result.action_matrix.nnz(),
      result.action_rank,
      result.kernel_dimension,
      result.truncated_kernel_generator_count,
      operator_ring->wrap(result.annihilator),
      result.quotient_length};
}

[[nodiscard]] PythonCycleScreenResult wrap_cycle_screen(
    const std::shared_ptr<PythonRing>& ring,
    const le::PackedCycleScreenResult& result) {
  std::optional<PythonCycleAuditResult> certification;
  if (result.certification.has_value()) {
    certification = wrap_cycle_audit(ring, *result.certification);
  }
  return PythonCycleScreenResult{
      discovery_status_name(result.status),
      result.length_Q,
      result.length_P_mod_J2,
      result.g_in_J,
      result.derivatives_in_J,
      result.cycle_valid,
      result.multiplication_rank,
      result.full_column_rank_candidate,
      result.certified_faithful,
      std::move(certification),
      result.detail};
}

[[nodiscard]] PythonH1ScreenResult wrap_h1_screen(
    const std::shared_ptr<PythonRing>& ring,
    const le::PackedH1ScreenResult& result) {
  auto status = discovery_status_name(result.status);
  auto detail = result.detail;
  if (result.status == le::DiscoveryScreenStatus::Complete &&
      result.rank_proof == le::MatrixSpaceRankProof::ResourceLimit) {
    status = "resource_limit";
    if (!detail.has_value()) {
      detail = "matrix-space rank search exhausted its configured limits";
    }
  } else if (result.status == le::DiscoveryScreenStatus::Complete &&
             result.rank_proof == le::MatrixSpaceRankProof::GenericOnly) {
    status = "generic_only";
    if (!detail.has_value()) {
      detail = "matrix-space rank is generic-only, not a base-field proof";
    }
  }
  std::optional<PythonPolynomial> witness;
  if (result.witness.has_value()) {
    witness = ring->wrap(*result.witness);
  }
  std::optional<PythonCycleAuditResult> certification;
  if (result.certification.has_value()) {
    certification = wrap_cycle_audit(ring, *result.certification);
  }
  return PythonH1ScreenResult{
      std::move(status),
      result.length_Q,
      result.length_P_mod_J2,
      result.conormal_dimension,
      result.h1_dimension,
      result.socle_dimension,
      result.modulus,
      result.leading_ideal_fingerprint,
      result.leading_ideal_signature,
      result.maximum_individual_rank_lower_bound,
      result.maximum_individual_rank_upper_bound,
      matrix_rank_proof_name(result.rank_proof),
      result.full_socle_rank_candidate,
      result.certified_faithful,
      result.witness_coefficients,
      std::move(witness),
      std::move(certification),
      std::move(detail)};
}

[[nodiscard]] PythonInverseSystemDiscoveryRecord wrap_inverse_discovery(
    const std::shared_ptr<PythonRing>& operator_ring,
    const std::vector<std::vector<PythonPolynomial>>& candidates,
    const le::InverseSystemDiscoveryRecord& result) {
  if (result.candidate_index >= candidates.size()) {
    throw std::logic_error(
        "native inverse-system search returned an invalid candidate index");
  }
  const auto& candidate = candidates[result.candidate_index];
  std::vector<PythonPolynomial> retained_dual_generators;
  if (!result.retained_dual_generators.empty()) {
    if (candidate.empty()) {
      throw std::logic_error(
          "native inverse-system search retained an empty candidate");
    }
    retained_dual_generators =
        candidate.front().ring()->wrap_all(result.retained_dual_generators);
  }
  auto retained_annihilator_basis =
      operator_ring->wrap_all(result.retained_annihilator_basis);
  std::optional<PythonPolynomial> witness;
  if (result.witness.has_value()) {
    witness = operator_ring->wrap(*result.witness);
  }
  return PythonInverseSystemDiscoveryRecord{
      discovery_status_name(result.status),
      result.candidate_index,
      result.maximum_dual_degree,
      result.action_rank,
      result.kernel_dimension,
      result.quotient_length,
      result.h1_dimension,
      result.socle_dimension,
      result.modulus,
      result.leading_ideal_fingerprint,
      result.leading_ideal_signature,
      result.maximum_individual_rank_lower_bound,
      result.maximum_individual_rank_upper_bound,
      matrix_rank_proof_name(result.rank_proof),
      result.maximum_individual_rank,
      result.full_rank_candidate,
      result.certified_faithful,
      std::move(retained_dual_generators),
      std::move(retained_annihilator_basis),
      std::move(witness),
      result.detail};
}

PythonCycleAuditResult PythonRing::audit_cycle(
    const PythonIdeal& ideal,
    const PythonPolynomial& polynomial,
    std::optional<std::size_t> max_matrix_entries) {
  require_ideal(ideal);
  require_polynomial(polynomial);
  le::CycleAuditLimits limits;
  limits.max_matrix_entries = max_matrix_entries;
  const auto owner = shared_from_this();
  if (const auto* concrete_ideal = std::get_if<QQIdeal>(&ideal.value_)) {
    const auto concrete_polynomial =
        std::get<QQPolynomial>(polynomial.value_);
    const auto result = without_gil([&] {
      return le::audit_cycle(
          *concrete_ideal, concrete_polynomial, limits);
    });
    return wrap_cycle_audit(owner, result);
  }
  const auto concrete_polynomial = std::get<GFPolynomial>(polynomial.value_);
  const auto result = without_gil([&] {
    return le::audit_cycle(
        std::get<GFIdeal>(ideal.value_), concrete_polynomial, limits);
  });
  return wrap_cycle_audit(owner, result);
}

PythonColonClosureResult PythonRing::colon_closure(
    const PythonIdeal& ideal,
    const PythonPolynomial& polynomial,
    std::size_t max_steps,
    std::optional<std::size_t> max_matrix_entries) {
  require_ideal(ideal);
  require_polynomial(polynomial);
  le::ColonClosureLimits limits;
  limits.max_steps = max_steps;
  limits.audit.max_matrix_entries = max_matrix_entries;
  const auto owner = shared_from_this();
  if (const auto* concrete_ideal = std::get_if<QQIdeal>(&ideal.value_)) {
    const auto concrete_polynomial =
        std::get<QQPolynomial>(polynomial.value_);
    const auto result = without_gil([&] {
      return le::colon_closure(
          *concrete_ideal, concrete_polynomial, limits);
    });
    return wrap_colon_closure(owner, result);
  }
  const auto concrete_polynomial = std::get<GFPolynomial>(polynomial.value_);
  const auto result = without_gil([&] {
    return le::colon_closure(
        std::get<GFIdeal>(ideal.value_), concrete_polynomial, limits);
  });
  return wrap_colon_closure(owner, result);
}

PythonH1ActionResult PythonRing::full_h1_action(
    const PythonIdeal& ideal,
    std::optional<std::size_t> max_matrix_entries,
    std::optional<std::size_t> max_witness_audit_matrix_entries,
    std::optional<std::size_t> max_minors,
    std::optional<std::size_t> max_parameter_terms,
    std::optional<std::size_t> max_determinant_products,
    std::size_t max_finite_field_evaluations,
    std::optional<std::size_t> max_groebner_critical_pairs,
    std::optional<std::size_t> max_groebner_basis_polynomials,
    std::optional<std::size_t> max_groebner_reduction_steps,
    std::optional<std::size_t> max_groebner_live_terms,
    std::optional<std::size_t> max_groebner_basis_terms) {
  require_ideal(ideal);
  le::FullH1Options options;
  options.max_matrix_entries = max_matrix_entries;
  options.witness_audit_limits.max_matrix_entries =
      max_witness_audit_matrix_entries;
  options.matrix_space_limits.max_minors = max_minors;
  options.matrix_space_limits.max_parameter_terms = max_parameter_terms;
  options.matrix_space_limits.max_determinant_products =
      max_determinant_products;
  options.matrix_space_limits.max_finite_field_evaluations =
      max_finite_field_evaluations;
  options.groebner_limits.max_critical_pairs =
      max_groebner_critical_pairs;
  options.groebner_limits.max_basis_polynomials =
      max_groebner_basis_polynomials;
  options.groebner_limits.max_reduction_steps =
      max_groebner_reduction_steps;
  options.groebner_limits.max_live_terms = max_groebner_live_terms;
  options.groebner_limits.max_basis_terms = max_groebner_basis_terms;
  options.witness_audit_limits.groebner = options.groebner_limits;
  const auto owner = shared_from_this();
  if (const auto* concrete_ideal = std::get_if<QQIdeal>(&ideal.value_)) {
    const auto result = without_gil(
        [&] { return le::full_h1_action(*concrete_ideal, options); });
    return wrap_h1_action(owner, result);
  }
  const auto result = without_gil([&] {
    return le::full_h1_action(std::get<GFIdeal>(ideal.value_), options);
  });
  return wrap_h1_action(owner, result);
}

PythonInverseSystemResult PythonRing::macaulay_annihilator(
    const std::vector<PythonPolynomial>& dual_generators,
    le::ApolarityConvention convention,
    const le::InverseSystemLimits& limits) {
  const auto owner = shared_from_this();
  if (std::holds_alternative<QQRing>(value_)) {
    std::vector<QQPolynomial> concrete;
    concrete.reserve(dual_generators.size());
    for (const auto& generator : dual_generators) {
      if (!std::holds_alternative<QQPolynomial>(generator.value_)) {
        throw le::InverseSystemInputError(
            le::InverseSystemInputIssue::CoefficientFieldMismatch);
      }
      concrete.push_back(std::get<QQPolynomial>(generator.value_));
    }
    const auto result = without_gil([&] {
      return le::macaulay_annihilator(
          std::get<QQRing>(value_), concrete, convention, limits);
    });
    return wrap_inverse_system(owner, dual_generators, result);
  }

  std::vector<GFPolynomial> concrete;
  concrete.reserve(dual_generators.size());
  for (const auto& generator : dual_generators) {
    if (!std::holds_alternative<GFPolynomial>(generator.value_)) {
      throw le::InverseSystemInputError(
          le::InverseSystemInputIssue::CoefficientFieldMismatch);
    }
    concrete.push_back(std::get<GFPolynomial>(generator.value_));
  }
  const auto result = without_gil([&] {
    return le::macaulay_annihilator(
        std::get<GFRing>(value_), concrete, convention, limits);
  });
  return wrap_inverse_system(owner, dual_generators, result);
}

PythonCycleScreenResult PythonRing::screen_cycle(
    const PythonIdeal& ideal,
    const PythonPolynomial& polynomial,
    const le::PackedCycleScreenOptions& options) {
  require_ideal(ideal);
  require_polynomial(polynomial);
  if (!std::holds_alternative<GFRing>(value_)) {
    throw std::invalid_argument(
        "packed discovery screens are available only over GF(p)");
  }
  const auto owner = shared_from_this();
  const auto concrete_ideal = std::get<GFIdeal>(ideal.value_);
  const auto concrete_polynomial = std::get<GFPolynomial>(polynomial.value_);
  const auto result = without_gil([&] {
    return le::screen_cycle(concrete_ideal, concrete_polynomial, options);
  });
  return wrap_cycle_screen(owner, result);
}

PythonH1ScreenResult PythonRing::screen_full_h1(
    const PythonIdeal& ideal,
    const le::PackedH1ScreenOptions& options) {
  require_ideal(ideal);
  if (!std::holds_alternative<GFRing>(value_)) {
    throw std::invalid_argument(
        "packed discovery screens are available only over GF(p)");
  }
  const auto owner = shared_from_this();
  const auto concrete_ideal = std::get<GFIdeal>(ideal.value_);
  const auto result = without_gil(
      [&] { return le::screen_full_h1(concrete_ideal, options); });
  return wrap_h1_screen(owner, result);
}

std::vector<PythonCycleScreenResult> PythonRing::screen_cycles_parallel(
    const PythonIdeal& ideal,
    const std::vector<PythonPolynomial>& candidates,
    const le::PackedCycleScreenOptions& screen_options,
    const le::CandidateExecutorOptions& executor_options) {
  require_ideal(ideal);
  if (!std::holds_alternative<GFRing>(value_)) {
    throw std::invalid_argument(
        "packed discovery screens are available only over GF(p)");
  }
  const auto concrete_candidates =
      concrete_polynomials<GFPolynomial>(candidates);
  const auto concrete_ideal = std::get<GFIdeal>(ideal.value_);
  // One release covers session construction, native workers, and every
  // candidate. No pybind-owned object is touched until the batch returns.
  const auto native = without_gil([&] {
    return le::screen_cycles_parallel(
        concrete_ideal, concrete_candidates, screen_options,
        executor_options);
  });
  const auto owner = shared_from_this();
  std::vector<PythonCycleScreenResult> result;
  result.reserve(native.size());
  for (const auto& record : native) {
    result.push_back(wrap_cycle_screen(owner, record));
  }
  return result;
}

std::vector<PythonInverseSystemDiscoveryRecord>
PythonRing::search_inverse_systems(
    const std::vector<std::vector<PythonPolynomial>>& candidates,
    const le::InverseSystemDiscoveryOptions& discovery_options,
    const le::CandidateExecutorOptions& executor_options) {
  if (!std::holds_alternative<GFRing>(value_)) {
    throw std::invalid_argument(
        "packed inverse-system search is available only over GF(p)");
  }
  std::vector<std::vector<GFPolynomial>> concrete_candidates;
  concrete_candidates.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    std::vector<GFPolynomial> concrete;
    concrete.reserve(candidate.size());
    for (const auto& generator : candidate) {
      if (!std::holds_alternative<GFPolynomial>(generator.value_)) {
        throw le::InverseSystemInputError(
            le::InverseSystemInputIssue::CoefficientFieldMismatch);
      }
      concrete.push_back(std::get<GFPolynomial>(generator.value_));
    }
    concrete_candidates.push_back(std::move(concrete));
  }
  // Preserve the same one-release contract as fixed-ideal cycle discovery.
  const auto native = without_gil([&] {
    return le::search_inverse_systems_parallel(
        std::get<GFRing>(value_), concrete_candidates, discovery_options,
        executor_options);
  });
  const auto owner = shared_from_this();
  std::vector<PythonInverseSystemDiscoveryRecord> result;
  result.reserve(native.size());
  for (const auto& record : native) {
    result.push_back(wrap_inverse_discovery(owner, candidates, record));
  }
  return result;
}

std::string PythonRing::make_jg_certificate(
    const std::vector<PythonPolynomial>& ideal_generators,
    const PythonPolynomial& polynomial) {
  require_polynomial(polynomial);
  if (std::holds_alternative<QQRing>(value_)) {
    const auto concrete_generators =
        concrete_polynomials<QQPolynomial>(ideal_generators);
    const auto concrete_polynomial =
        std::get<QQPolynomial>(polynomial.value_);
    return without_gil([&] {
      return le::make_jg_certificate_json(
          std::get<QQRing>(value_), concrete_generators,
          concrete_polynomial);
    });
  }
  const auto concrete_generators =
      concrete_polynomials<GFPolynomial>(ideal_generators);
  const auto concrete_polynomial = std::get<GFPolynomial>(polynomial.value_);
  return without_gil([&] {
    return le::make_jg_certificate_json(
        std::get<GFRing>(value_), concrete_generators, concrete_polynomial);
  });
}

[[nodiscard]] le::Order parse_order(std::string_view order) {
  if (order == "lex") {
    return le::Order::Lex;
  }
  if (order == "grevlex") {
    return le::Order::Grevlex;
  }
  throw std::invalid_argument("order must be 'lex' or 'grevlex'");
}

[[nodiscard]] std::string python_integer_text(const py::int_& value) {
  if (py::isinstance<py::bool_>(value)) {
    return value.cast<bool>() ? "1" : "0";
  }
  return py::str(value).cast<std::string>();
}

[[nodiscard]] PythonPolynomial python_integer_like(
    const PythonPolynomial& polynomial,
    const py::int_& value) {
  return polynomial.ring()->parse(python_integer_text(value));
}

[[nodiscard]] PythonPolynomial python_inverse_integer_like(
    const PythonPolynomial& polynomial,
    const py::int_& value) {
  return polynomial.ring()->parse(
      "1/(" + python_integer_text(value) + ")");
}

[[nodiscard]] PythonIdeal coerce_ideal_argument(
    const std::shared_ptr<PythonRing>& ring,
    const py::object& value) {
  if (py::isinstance<PythonIdeal>(value)) {
    return value.cast<PythonIdeal>();
  }
  try {
    return ring->ideal(value.cast<std::vector<PythonPolynomial>>());
  } catch (const py::cast_error&) {
    throw py::type_error(
        "expected an Ideal or an iterable of polynomials");
  }
}

[[nodiscard]] std::vector<std::vector<PythonPolynomial>>
coerce_inverse_candidates(const py::iterable& values) {
  std::vector<std::vector<PythonPolynomial>> result;
  for (const auto handle : values) {
    const auto value = py::reinterpret_borrow<py::object>(handle);
    if (py::isinstance<PythonPolynomial>(value)) {
      result.push_back({value.cast<PythonPolynomial>()});
      continue;
    }
    try {
      result.push_back(value.cast<std::vector<PythonPolynomial>>());
    } catch (const py::cast_error&) {
      throw py::type_error(
          "each inverse-system candidate must be a polynomial or an "
          "iterable of polynomials");
    }
  }
  return result;
}

void configure_groebner_limits(
    le::GroebnerLimits& limits,
    std::optional<std::size_t> max_critical_pairs,
    std::optional<std::size_t> max_basis_polynomials,
    std::optional<std::size_t> max_reduction_steps,
    std::optional<std::size_t> max_live_terms,
    std::optional<std::size_t> max_basis_terms) {
  limits.max_critical_pairs = max_critical_pairs;
  limits.max_basis_polynomials = max_basis_polynomials;
  limits.max_reduction_steps = max_reduction_steps;
  limits.max_live_terms = max_live_terms;
  limits.max_basis_terms = max_basis_terms;
}

[[nodiscard]] le::InverseSystemLimits make_inverse_system_limits(
    std::optional<std::size_t> max_basis_monomials,
    std::optional<std::size_t> max_action_rows,
    std::optional<std::size_t> max_action_nonzeros,
    std::optional<std::size_t> max_dense_matrix_entries,
    std::optional<std::size_t> max_kernel_coordinate_entries,
    std::optional<std::size_t> max_elimination_live_nonzeros,
    std::optional<std::size_t> max_elimination_operations,
    bool verify_invariants,
    std::optional<std::size_t> max_invariant_replay_checks,
    std::optional<std::size_t> max_groebner_critical_pairs,
    std::optional<std::size_t> max_groebner_basis_polynomials,
    std::optional<std::size_t> max_groebner_reduction_steps,
    std::optional<std::size_t> max_groebner_live_terms,
    std::optional<std::size_t> max_groebner_basis_terms) {
  le::InverseSystemLimits limits;
  limits.max_basis_monomials = max_basis_monomials;
  limits.max_action_rows = max_action_rows;
  limits.max_action_nonzeros = max_action_nonzeros;
  limits.max_dense_matrix_entries = max_dense_matrix_entries;
  limits.max_kernel_coordinate_entries = max_kernel_coordinate_entries;
  limits.max_elimination_live_nonzeros = max_elimination_live_nonzeros;
  limits.max_elimination_operations = max_elimination_operations;
  limits.verify_invariants = verify_invariants;
  limits.max_invariant_replay_checks = max_invariant_replay_checks;
  configure_groebner_limits(
      limits.groebner, max_groebner_critical_pairs,
      max_groebner_basis_polynomials, max_groebner_reduction_steps,
      max_groebner_live_terms, max_groebner_basis_terms);
  return limits;
}

[[nodiscard]] le::PackedCycleScreenOptions make_cycle_screen_options(
    std::optional<std::size_t> max_matrix_entries,
    bool certify_hits,
    std::optional<std::size_t> max_certification_matrix_entries,
    std::optional<std::size_t> max_groebner_critical_pairs,
    std::optional<std::size_t> max_groebner_basis_polynomials,
    std::optional<std::size_t> max_groebner_reduction_steps,
    std::optional<std::size_t> max_groebner_live_terms,
    std::optional<std::size_t> max_groebner_basis_terms) {
  le::PackedCycleScreenOptions options;
  options.max_matrix_entries = max_matrix_entries;
  options.certify_hits = certify_hits;
  options.certification_limits.max_matrix_entries =
      max_certification_matrix_entries;
  configure_groebner_limits(
      options.groebner_limits, max_groebner_critical_pairs,
      max_groebner_basis_polynomials, max_groebner_reduction_steps,
      max_groebner_live_terms, max_groebner_basis_terms);
  options.certification_limits.groebner = options.groebner_limits;
  return options;
}

[[nodiscard]] le::PackedH1ScreenOptions make_h1_screen_options(
    std::optional<std::size_t> max_matrix_entries,
    bool certify_hits,
    std::optional<std::size_t> max_certification_matrix_entries,
    std::optional<std::size_t> max_minors,
    std::optional<std::size_t> max_parameter_terms,
    std::optional<std::size_t> max_determinant_products,
    std::size_t max_finite_field_evaluations,
    std::optional<std::size_t> max_groebner_critical_pairs,
    std::optional<std::size_t> max_groebner_basis_polynomials,
    std::optional<std::size_t> max_groebner_reduction_steps,
    std::optional<std::size_t> max_groebner_live_terms,
    std::optional<std::size_t> max_groebner_basis_terms) {
  le::PackedH1ScreenOptions options;
  options.max_matrix_entries = max_matrix_entries;
  options.certify_hits = certify_hits;
  options.certification_limits.max_matrix_entries =
      max_certification_matrix_entries;
  options.matrix_space_limits.max_minors = max_minors;
  options.matrix_space_limits.max_parameter_terms = max_parameter_terms;
  options.matrix_space_limits.max_determinant_products =
      max_determinant_products;
  options.matrix_space_limits.max_finite_field_evaluations =
      max_finite_field_evaluations;
  configure_groebner_limits(
      options.groebner_limits, max_groebner_critical_pairs,
      max_groebner_basis_polynomials, max_groebner_reduction_steps,
      max_groebner_live_terms, max_groebner_basis_terms);
  options.certification_limits.groebner = options.groebner_limits;
  return options;
}

}  // namespace

PYBIND11_MODULE(_laughableengine, module) {
  module.doc() = "Native exact arithmetic for laughableengine";

  py::register_exception<le::FullH1ResourceLimit>(
      module, "FullH1ResourceLimit", PyExc_RuntimeError);
  py::register_exception<le::CotangentH1InputError>(
      module, "CotangentH1InputError", PyExc_ValueError);
  py::register_exception<le::CotangentClassError>(
      module, "CotangentClassError", PyExc_ValueError);
  py::register_exception<le::CotangentH1ResourceLimit>(
      module, "CotangentH1ResourceLimit", PyExc_RuntimeError);
  py::register_exception<le::InverseSystemResourceLimit>(
      module, "InverseSystemResourceLimit", PyExc_RuntimeError);

  auto polynomial = py::class_<PythonPolynomial>(module, "Polynomial");

  py::class_<PythonDivisionResult>(module, "DivisionResult")
      .def_readonly("quotients", &PythonDivisionResult::quotients)
      .def_readonly("remainder", &PythonDivisionResult::remainder)
      .def("__repr__", [](const PythonDivisionResult& result) {
        return "DivisionResult(quotients=" +
               py::repr(py::cast(result.quotients)).cast<std::string>() +
               ", remainder=" + result.remainder.to_string() + ")";
      });

  auto ideal = py::class_<PythonIdeal>(module, "Ideal")
      .def_property_readonly("ring", &PythonIdeal::ring)
      .def_property_readonly("generators", &PythonIdeal::generators)
      .def_property_readonly(
          "groebner_basis", &PythonIdeal::generators)
      .def("gb", &PythonIdeal::generators)
      .def("nf", &PythonIdeal::normal_form, py::arg("polynomial"))
      .def("NF", &PythonIdeal::normal_form, py::arg("polynomial"))
      .def(
          "normal_form", &PythonIdeal::normal_form,
          py::arg("polynomial"))
      .def("nfs", &PythonIdeal::normal_forms, py::arg("polynomials"))
      .def("NFs", &PythonIdeal::normal_forms, py::arg("polynomials"))
      .def(
          "normal_forms", &PythonIdeal::normal_forms,
          py::arg("polynomials"))
      .def("is_zero", &PythonIdeal::is_zero)
      .def("is_unit", &PythonIdeal::is_unit)
      .def("is_zero_dimensional", &PythonIdeal::is_zero_dimensional)
      .def("supported_at_origin", &PythonIdeal::supported_at_origin)
      .def("dimension", &PythonIdeal::dimension)
      .def("quotient_dimension", &PythonIdeal::dimension)
      .def("standard_monomials", &PythonIdeal::standard_monomials)
      .def("square", &PythonIdeal::square)
      .def("colon", &PythonIdeal::colon, py::arg("polynomial"))
      .def(
          "eliminate",
          py::overload_cast<const std::vector<std::size_t>&>(
              &PythonIdeal::eliminate, py::const_),
          py::arg("variables"))
      .def(
          "eliminate",
          py::overload_cast<const std::vector<std::string>&>(
              &PythonIdeal::eliminate, py::const_),
          py::arg("variables"))
      .def("__contains__", &PythonIdeal::contains)
      .def("__add__", &PythonIdeal::add, py::is_operator())
      .def("__mul__", &PythonIdeal::multiply, py::is_operator())
      .def("__eq__", &PythonIdeal::equals, py::is_operator())
      .def("__ne__", [](const PythonIdeal& left, const PythonIdeal& right) {
        return !left.equals(right);
      }, py::is_operator())
      .def("__str__", &PythonIdeal::to_string)
      .def("__repr__", &PythonIdeal::to_string);
  ideal.attr("__hash__") = py::none();

  py::class_<PythonExactMatrix>(module, "ExactMatrix")
      .def_property_readonly("shape", &PythonExactMatrix::shape)
      .def_readonly("row_count", &PythonExactMatrix::row_count)
      .def_readonly("column_count", &PythonExactMatrix::column_count)
      .def_property_readonly("rows", &PythonExactMatrix::rows)
      .def_property_readonly("entries", &PythonExactMatrix::rows)
      .def("to_list", &PythonExactMatrix::rows)
      .def("__len__", [](const PythonExactMatrix& matrix) {
        return matrix.row_count;
      })
      .def("__repr__", &PythonExactMatrix::to_string);

  py::class_<PythonSparseExactMatrix>(module, "SparseExactMatrix")
      .def_property_readonly("shape", &PythonSparseExactMatrix::shape)
      .def_property_readonly("row_count", &PythonSparseExactMatrix::row_count)
      .def_property_readonly(
          "column_count", &PythonSparseExactMatrix::column_count)
      .def_property_readonly("nnz", &PythonSparseExactMatrix::nnz)
      .def("at", &PythonSparseExactMatrix::at, py::arg("row"),
           py::arg("column"))
      .def("entries", &PythonSparseExactMatrix::entries, py::kw_only(),
           py::arg("max_coordinate_entries") = 1'000'000)
      .def("__getitem__", [](const PythonSparseExactMatrix& matrix,
                              const std::pair<std::size_t, std::size_t>& index) {
        return matrix.at(index.first, index.second);
      })
      .def("__repr__", &PythonSparseExactMatrix::to_string);

  py::class_<PythonCotangentClassProof>(module, "CotangentClassProof")
      .def_property_readonly("ring", &PythonCotangentClassProof::ring)
      .def_property_readonly("status", &PythonCotangentClassProof::status)
      .def_property_readonly(
          "representative", &PythonCotangentClassProof::representative)
      .def_property_readonly("in_ideal", &PythonCotangentClassProof::in_ideal)
      .def_property_readonly(
          "derivative_remainders",
          &PythonCotangentClassProof::derivative_remainders)
      .def_property_readonly("cycle", &PythonCotangentClassProof::cycle)
      .def_property_readonly(
          "cycle_valid", &PythonCotangentClassProof::cycle)
      .def_property_readonly(
          "multiplication_matrix",
          &PythonCotangentClassProof::multiplication_matrix)
      .def_property_readonly("rank", &PythonCotangentClassProof::rank)
      .def_property_readonly(
          "multiplication_rank", &PythonCotangentClassProof::rank)
      .def_property_readonly("ann", &PythonCotangentClassProof::ann)
      .def_property_readonly(
          "annihilator_dimension", &PythonCotangentClassProof::ann)
      .def_property_readonly(
          "annihilator_basis",
          &PythonCotangentClassProof::annihilator_basis)
      .def_property_readonly(
          "annihilator_generators",
          &PythonCotangentClassProof::annihilator_basis)
      .def_property_readonly(
          "colon_generators", &PythonCotangentClassProof::colon_generators)
      .def_property_readonly(
          "faithful", &PythonCotangentClassProof::faithful)
      .def_property_readonly(
          "colon_equals", &PythonCotangentClassProof::colon_equals)
      .def_property_readonly(
          "colon_equals_ideal", &PythonCotangentClassProof::colon_equals)
      .def_property_readonly(
          "conclusive", &PythonCotangentClassProof::conclusive)
      .def("__repr__", &PythonCotangentClassProof::to_string);

  py::class_<PythonCotangentH1Presentation>(
      module, "CotangentH1Presentation")
      .def_property_readonly("ring", &PythonCotangentH1Presentation::ring)
      .def_property_readonly(
          "generators", &PythonCotangentH1Presentation::generators)
      .def_property_readonly(
          "maximal_power", &PythonCotangentH1Presentation::maximal_power)
      .def_property_readonly(
          "quotient_length", &PythonCotangentH1Presentation::quotient_length)
      .def_property_readonly(
          "length_Q", &PythonCotangentH1Presentation::quotient_length)
      .def_property_readonly(
          "square_quotient_length",
          &PythonCotangentH1Presentation::square_quotient_length)
      .def_property_readonly(
          "length_P_mod_J2",
          &PythonCotangentH1Presentation::square_quotient_length)
      .def_property_readonly(
          "conormal_dimension",
          &PythonCotangentH1Presentation::conormal_dimension)
      .def_property_readonly(
          "h1_dimension", &PythonCotangentH1Presentation::h1_dimension)
      .def_property_readonly(
          "reduction_matrix",
          &PythonCotangentH1Presentation::reduction_matrix)
      .def_property_readonly(
          "derivative_matrix",
          &PythonCotangentH1Presentation::derivative_matrix)
      .def_property_readonly(
          "cycle_matrix", &PythonCotangentH1Presentation::cycle_matrix)
      .def_property_readonly(
          "h1_relation_matrix",
          &PythonCotangentH1Presentation::h1_relation_matrix)
      .def("ideal_generators",
           &PythonCotangentH1Presentation::ideal_generators, py::kw_only(),
           py::arg("max_coordinate_entries") = 1'000'000)
      .def("quotient_basis", &PythonCotangentH1Presentation::quotient_basis,
           py::kw_only(),
           py::arg("max_coordinate_entries") = 1'000'000)
      .def("square_quotient_basis",
           &PythonCotangentH1Presentation::square_quotient_basis,
           py::kw_only(),
           py::arg("max_coordinate_entries") = 1'000'000)
      .def("h1_kernel_coordinates",
           &PythonCotangentH1Presentation::h1_kernel_coordinates,
           py::kw_only(),
           py::arg("max_coordinate_entries") = 1'000'000)
      .def("h1_basis", &PythonCotangentH1Presentation::h1_basis,
           py::kw_only(),
           py::arg("max_coordinate_entries") = 1'000'000)
      .def("conormal_basis", &PythonCotangentH1Presentation::conormal_basis,
           py::kw_only(),
           py::arg("max_coordinate_entries") = 1'000'000)
      .def("quotient_remainder",
           &PythonCotangentH1Presentation::quotient_remainder,
           py::arg("polynomial"))
      .def("verify_class", &PythonCotangentH1Presentation::verify_class,
           py::arg("representative"))
      .def("__repr__", &PythonCotangentH1Presentation::to_string);

  auto local_ideal = py::class_<PythonLocalIdeal>(module, "LocalIdeal")
      .def_property_readonly("ring", &PythonLocalIdeal::ring)
      .def_property_readonly(
          "lower_generators", &PythonLocalIdeal::lower_generators)
      .def_property_readonly(
          "maximal_power", &PythonLocalIdeal::maximal_power)
      .def("generators", &PythonLocalIdeal::generators)
      .def(
          "quotient", &PythonLocalIdeal::quotient, py::kw_only(),
          py::arg("max_monomials") = 1'000'000,
          py::arg("max_generated_rows") = 2'000'000,
          py::arg("max_matrix_triplets") = 20'000'000)
      .def(
          "__eq__",
          [](const PythonLocalIdeal& left, const py::object& right) {
            if (py::isinstance<PythonLocalIdeal>(right)) {
              return left.equals(right.cast<PythonLocalIdeal>());
            }
            if (py::isinstance<PythonIdealPreimage>(right)) {
              return right.cast<PythonIdealPreimage>().equals(left);
            }
            return false;
          },
          py::is_operator())
      .def(
          "__ne__",
          [](const PythonLocalIdeal& left, const py::object& right) {
            if (py::isinstance<PythonLocalIdeal>(right)) {
              return !left.equals(right.cast<PythonLocalIdeal>());
            }
            if (py::isinstance<PythonIdealPreimage>(right)) {
              return !right.cast<PythonIdealPreimage>().equals(left);
            }
            return true;
          },
          py::is_operator())
      .def("__repr__", &PythonLocalIdeal::to_string);
  local_ideal.attr("__hash__") = py::none();

  py::class_<PythonFiniteQuotient>(module, "FiniteQuotient")
      .def_property_readonly("ring", &PythonFiniteQuotient::ring)
      .def_property_readonly(
          "defining_ideal", &PythonFiniteQuotient::defining_ideal)
      .def_property_readonly("length", &PythonFiniteQuotient::length)
      .def_property_readonly(
          "square_quotient_length",
          &PythonFiniteQuotient::square_quotient_length)
      .def("remainder", &PythonFiniteQuotient::remainder,
           py::arg("polynomial"))
      .def("basis", &PythonFiniteQuotient::basis, py::kw_only(),
           py::arg("max_coordinate_entries") = 1'000'000)
      .def("zero_ideal", &PythonFiniteQuotient::zero_ideal)
      .def("conormal_module", &PythonFiniteQuotient::conormal_module)
      .def("__repr__", &PythonFiniteQuotient::to_string);

  py::class_<PythonConormalModule>(module, "ConormalModule")
      .def_property_readonly("quotient", &PythonConormalModule::quotient)
      .def_property_readonly("dimension", &PythonConormalModule::dimension)
      .def_property_readonly(
          "constraint_matrix", &PythonConormalModule::constraint_matrix)
      .def("basis", &PythonConormalModule::basis, py::kw_only(),
           py::arg("max_coordinate_entries") = 1'000'000)
      .def("derivative_map", &PythonConormalModule::derivative_map)
      .def("__repr__", &PythonConormalModule::to_string);

  py::class_<PythonConormalDerivativeMap>(
      module, "ConormalDerivativeMap")
      .def_property_readonly(
          "domain", &PythonConormalDerivativeMap::domain)
      .def_property_readonly(
          "ambient_matrix", &PythonConormalDerivativeMap::ambient_matrix)
      .def("kernel", &PythonConormalDerivativeMap::kernel)
      .def("__repr__", &PythonConormalDerivativeMap::to_string);

  py::class_<PythonH1Module>(module, "H1Module")
      .def_property_readonly("quotient", &PythonH1Module::quotient)
      .def_property_readonly(
          "conormal_module", &PythonH1Module::conormal_module)
      .def_property_readonly("dimension", &PythonH1Module::dimension)
      .def_property_readonly(
          "ambient_dimension", &PythonH1Module::ambient_dimension)
      .def_property_readonly(
          "constraint_matrix", &PythonH1Module::constraint_matrix)
      .def("kernel_coordinates", &PythonH1Module::kernel_coordinates,
           py::kw_only(),
           py::arg("max_coordinate_entries") = 1'000'000)
      .def("basis", &PythonH1Module::basis, py::kw_only(),
           py::arg("max_coordinate_entries") = 1'000'000)
      .def("class_of", &PythonH1Module::class_of,
           py::arg("representative"))
      .def("__repr__", &PythonH1Module::to_string);

  auto h1_element = py::class_<PythonH1Element>(module, "H1Element")
      .def_property_readonly("module", &PythonH1Element::module)
      .def_property_readonly(
          "representative", &PythonH1Element::representative)
      .def_property_readonly("coordinates", &PythonH1Element::coordinates)
      .def("annihilator", &PythonH1Element::annihilator)
      .def("__eq__", &PythonH1Element::equals, py::is_operator())
      .def(
          "__ne__",
          [](const PythonH1Element& left, const PythonH1Element& right) {
            return !left.equals(right);
          },
          py::is_operator())
      .def("__repr__", &PythonH1Element::to_string);
  h1_element.attr("__hash__") = py::none();

  auto quotient_ideal =
      py::class_<PythonQuotientIdeal>(module, "QuotientIdeal")
          .def_property_readonly(
              "quotient", &PythonQuotientIdeal::quotient)
          .def_property_readonly(
              "dimension", &PythonQuotientIdeal::dimension)
          .def_property_readonly(
              "basis_coordinates", &PythonQuotientIdeal::basis_coordinates)
          .def_property_readonly(
              "generators", &PythonQuotientIdeal::generators)
          .def("is_zero", &PythonQuotientIdeal::is_zero)
          .def("preimage", &PythonQuotientIdeal::preimage)
          .def("__eq__", &PythonQuotientIdeal::equals, py::is_operator())
          .def(
              "__ne__",
              [](const PythonQuotientIdeal& left,
                 const PythonQuotientIdeal& right) {
                return !left.equals(right);
              },
              py::is_operator())
          .def("__repr__", &PythonQuotientIdeal::to_string);
  quotient_ideal.attr("__hash__") = py::none();

  auto ideal_preimage =
      py::class_<PythonIdealPreimage>(module, "IdealPreimage")
          .def_property_readonly(
              "quotient_ideal", &PythonIdealPreimage::quotient_ideal)
          .def_property_readonly(
              "source_ideal", &PythonIdealPreimage::source_ideal)
          .def("generators", &PythonIdealPreimage::generators)
          .def(
              "__eq__",
              [](const PythonIdealPreimage& left, const py::object& right) {
                if (py::isinstance<PythonIdealPreimage>(right)) {
                  return left.equals(right.cast<PythonIdealPreimage>());
                }
                if (py::isinstance<PythonLocalIdeal>(right)) {
                  return left.equals(right.cast<PythonLocalIdeal>());
                }
                return false;
              },
              py::is_operator())
          .def(
              "__ne__",
              [](const PythonIdealPreimage& left, const py::object& right) {
                if (py::isinstance<PythonIdealPreimage>(right)) {
                  return !left.equals(right.cast<PythonIdealPreimage>());
                }
                if (py::isinstance<PythonLocalIdeal>(right)) {
                  return !left.equals(right.cast<PythonLocalIdeal>());
                }
                return true;
              },
              py::is_operator())
          .def("__repr__", &PythonIdealPreimage::to_string);
  ideal_preimage.attr("__hash__") = py::none();

  py::class_<PythonCycleAuditResult>(module, "CycleAuditResult")
      .def_readonly("status", &PythonCycleAuditResult::status)
      .def_readonly("ideal", &PythonCycleAuditResult::ideal)
      .def_readonly(
          "finite_quotient", &PythonCycleAuditResult::finite_quotient)
      .def_readonly(
          "supported_at_origin",
          &PythonCycleAuditResult::supported_at_origin)
      .def_readonly(
          "quotient_length", &PythonCycleAuditResult::quotient_length)
      .def_property_readonly("length_Q", [](const PythonCycleAuditResult& r) {
        return r.quotient_length;
      })
      .def_readonly("polynomial", &PythonCycleAuditResult::polynomial)
      .def_readonly(
          "polynomial_remainder",
          &PythonCycleAuditResult::polynomial_remainder)
      .def_readonly("derivatives", &PythonCycleAuditResult::derivatives)
      .def_readonly(
          "derivative_remainders",
          &PythonCycleAuditResult::derivative_remainders)
      .def_readonly(
          "polynomial_in_ideal",
          &PythonCycleAuditResult::polynomial_in_ideal)
      .def_property_readonly("g_in_J", [](const PythonCycleAuditResult& r) {
        return r.polynomial_in_ideal;
      })
      .def_readonly(
          "derivatives_in_ideal",
          &PythonCycleAuditResult::derivatives_in_ideal)
      .def_property_readonly(
          "derivatives_in_J", [](const PythonCycleAuditResult& r) {
            return r.derivatives_in_ideal;
          })
      .def_readonly("cycle_valid", &PythonCycleAuditResult::cycle_valid)
      .def_readonly(
          "maximal_times_ideal",
          &PythonCycleAuditResult::maximal_times_ideal)
      .def_readonly("primitive", &PythonCycleAuditResult::primitive)
      .def_readonly("squared_ideal", &PythonCycleAuditResult::squared_ideal)
      .def_property_readonly(
          "ideal_square", [](const PythonCycleAuditResult& r) {
            return r.squared_ideal;
          })
      .def_readonly(
          "colon_multiplication_matrix",
          &PythonCycleAuditResult::colon_multiplication_matrix)
      .def_readonly(
          "colon_kernel_basis",
          &PythonCycleAuditResult::colon_kernel_basis)
      .def_property_readonly(
          "colon_kernel_coordinates", [](const PythonCycleAuditResult& r) {
            return r.colon_kernel_basis;
          })
      .def_readonly(
          "colon_kernel_lifts",
          &PythonCycleAuditResult::colon_kernel_lifts)
      .def_property_readonly(
          "kernel_lifts", [](const PythonCycleAuditResult& r) {
            return r.colon_kernel_lifts;
          })
      .def_readonly(
          "colon_lift_product_remainders",
          &PythonCycleAuditResult::colon_lift_product_remainders)
      .def_property_readonly(
          "lift_product_remainders", [](const PythonCycleAuditResult& r) {
            return r.colon_lift_product_remainders;
          })
      .def_readonly("colon_ideal", &PythonCycleAuditResult::colon_ideal)
      .def_readonly(
          "annihilator_length",
          &PythonCycleAuditResult::annihilator_length)
      .def_readonly(
          "colon_quotient_length",
          &PythonCycleAuditResult::colon_quotient_length)
      .def_readonly(
          "colon_equals_ideal",
          &PythonCycleAuditResult::colon_equals_ideal)
      .def_readonly(
          "annihilator_zero", &PythonCycleAuditResult::annihilator_zero)
      .def_readonly(
          "ideal_in_colon_remainders",
          &PythonCycleAuditResult::ideal_in_colon_remainders)
      .def_readonly(
          "colon_in_ideal_remainders",
          &PythonCycleAuditResult::colon_in_ideal_remainders)
      .def_readonly("faithful", &PythonCycleAuditResult::faithful)
      .def_property_readonly(
          "faithful_cycle", [](const PythonCycleAuditResult& r) {
            return r.faithful;
          })
      .def_readonly("conclusive", &PythonCycleAuditResult::conclusive)
      .def_readonly(
          "resource_detail", &PythonCycleAuditResult::resource_detail)
      .def("__repr__", &PythonCycleAuditResult::to_string);

  py::class_<PythonColonClosureStep>(module, "ColonClosureStep")
      .def_readonly("index", &PythonColonClosureStep::index)
      .def_readonly("ideal", &PythonColonClosureStep::ideal)
      .def_readonly(
          "quotient_length", &PythonColonClosureStep::quotient_length);

  py::class_<PythonColonClosureTransition>(
      module, "ColonClosureTransition")
      .def_readonly(
          "current_in_next_remainders",
          &PythonColonClosureTransition::current_in_next_remainders)
      .def_readonly(
          "next_in_current_remainders",
          &PythonColonClosureTransition::next_in_current_remainders)
      .def_readonly(
          "current_subset_next",
          &PythonColonClosureTransition::current_subset_next)
      .def_readonly("equal", &PythonColonClosureTransition::equal);

  py::class_<PythonColonClosureResult>(module, "ColonClosureResult")
      .def_readonly("status", &PythonColonClosureResult::status)
      .def_readonly("steps", &PythonColonClosureResult::steps)
      .def_readonly(
          "transitions", &PythonColonClosureResult::transitions)
      .def_property_readonly("ideals", &PythonColonClosureResult::ideals)
      .def_property_readonly(
          "quotient_lengths",
          &PythonColonClosureResult::quotient_lengths)
      .def_property_readonly(
          "lengths", &PythonColonClosureResult::quotient_lengths)
      .def_readonly("conclusive", &PythonColonClosureResult::conclusive)
      .def_readonly(
          "faithful_fixed_point_found",
          &PythonColonClosureResult::faithful_fixed_point_found)
      .def_readonly(
          "resource_detail", &PythonColonClosureResult::resource_detail)
      .def("__repr__", &PythonColonClosureResult::to_string);

  py::class_<PythonMatrixSpaceRankResult>(
      module, "MatrixSpaceRankResult")
      .def_readonly(
          "parameter_count", &PythonMatrixSpaceRankResult::parameter_count)
      .def_readonly("row_count", &PythonMatrixSpaceRankResult::row_count)
      .def_readonly(
          "column_count", &PythonMatrixSpaceRankResult::column_count)
      .def_readonly("lower_bound", &PythonMatrixSpaceRankResult::lower_bound)
      .def_readonly("upper_bound", &PythonMatrixSpaceRankResult::upper_bound)
      .def_readonly(
          "generic_maximum", &PythonMatrixSpaceRankResult::generic_maximum)
      .def_readonly(
          "exact_maximum", &PythonMatrixSpaceRankResult::exact_maximum)
      .def_readonly("proof", &PythonMatrixSpaceRankResult::proof)
      .def_property_readonly(
          "witness_coefficients",
          &PythonMatrixSpaceRankResult::witness_coefficients)
      .def_readonly(
          "minors_tested", &PythonMatrixSpaceRankResult::minors_tested)
      .def_readonly(
          "determinant_products_tested",
          &PythonMatrixSpaceRankResult::determinant_products_tested)
      .def_readonly(
          "finite_field_evaluations",
          &PythonMatrixSpaceRankResult::finite_field_evaluations)
      .def_readonly(
          "has_full_column_rank_witness",
          &PythonMatrixSpaceRankResult::has_full_column_rank_witness)
      .def("__repr__", &PythonMatrixSpaceRankResult::to_string);

  py::class_<PythonH1ActionResult>(module, "H1ActionResult")
      .def_readonly("status", &PythonH1ActionResult::status)
      .def_property_readonly(
          "conclusive", [](const PythonH1ActionResult& result) {
            return result.status == "complete";
          })
      .def_readonly("length_Q", &PythonH1ActionResult::length_Q)
      .def_readonly(
          "length_P_mod_J2", &PythonH1ActionResult::length_P_mod_J2)
      .def_readonly(
          "conormal_dimension", &PythonH1ActionResult::conormal_dimension)
      .def_readonly("h1_dimension", &PythonH1ActionResult::h1_dimension)
      .def_readonly(
          "socle_dimension", &PythonH1ActionResult::socle_dimension)
      .def_readonly("ideal", &PythonH1ActionResult::ideal)
      .def_readonly("squared_ideal", &PythonH1ActionResult::squared_ideal)
      .def_readonly(
          "conormal_basis", &PythonH1ActionResult::conormal_basis)
      .def_readonly("h1_basis", &PythonH1ActionResult::h1_basis)
      .def_readonly("socle_basis", &PythonH1ActionResult::socle_basis)
      .def_readonly("reduction_map", &PythonH1ActionResult::reduction_map)
      .def_property_readonly(
          "reduction_matrix", [](const PythonH1ActionResult& result) {
            return result.reduction_map;
          })
      .def_readonly("differential", &PythonH1ActionResult::differential)
      .def_readonly(
          "variable_multiplication_matrices",
          &PythonH1ActionResult::variable_multiplication_matrices)
      .def_readonly(
          "h1_multiplication_matrices",
          &PythonH1ActionResult::h1_multiplication_matrices)
      .def_readonly(
          "action_matrices", &PythonH1ActionResult::action_matrices)
      .def_readonly(
          "common_product_space_dimension",
          &PythonH1ActionResult::common_product_space_dimension)
      .def_readonly(
          "common_product_space_rank_bound",
          &PythonH1ActionResult::common_product_space_rank_bound)
      .def_readonly("individual_rank", &PythonH1ActionResult::rank)
      .def_property_readonly("rank_lower_bound", [](const PythonH1ActionResult& result) {
        return result.rank.lower_bound;
      })
      .def_property_readonly("rank_upper_bound", [](const PythonH1ActionResult& result) {
        return result.rank.upper_bound;
      })
      .def_property_readonly("rank_proof", [](const PythonH1ActionResult& result) {
        return result.rank.proof;
      })
      .def_property_readonly(
          "maximum_individual_socle_action_rank", [](const PythonH1ActionResult& result) {
            return result.rank.exact_maximum;
          })
      .def_property_readonly(
          "witness_coefficients", [](const PythonH1ActionResult& result) {
            return result.rank.witness_coefficients();
          })
      .def_property_readonly(
          "best_h1_coefficients", [](const PythonH1ActionResult& result) {
            return result.rank.witness_coefficients();
          })
      .def_readonly(
          "best_h1_polynomial", &PythonH1ActionResult::best_h1_polynomial)
      .def_property_readonly(
          "witness_polynomial", [](const PythonH1ActionResult& result) {
            return result.best_h1_polynomial;
          })
      .def_readonly(
          "faithful_witness_audit",
          &PythonH1ActionResult::faithful_witness_audit)
      .def_property_readonly(
          "has_faithful_witness",
          &PythonH1ActionResult::has_faithful_witness)
      .def_readonly(
          "common_annihilator_diagnostic",
          &PythonH1ActionResult::common_annihilator_diagnostic)
      .def("__repr__", &PythonH1ActionResult::to_string);

  py::class_<PythonInverseSystemResult>(module, "InverseSystemResult")
      .def_readonly("convention", &PythonInverseSystemResult::convention)
      .def_readonly(
          "dual_generators", &PythonInverseSystemResult::dual_generators)
      .def_readonly(
          "maximum_degree", &PythonInverseSystemResult::maximum_degree)
      .def_property_readonly(
          "maximum_dual_degree", [](const PythonInverseSystemResult& result) {
            return result.maximum_degree;
          })
      .def_readonly(
          "operator_exponents", &PythonInverseSystemResult::operator_exponents)
      .def_readonly(
          "output_exponents", &PythonInverseSystemResult::output_exponents)
      .def_property_readonly(
          "action_shape", &PythonInverseSystemResult::action_shape)
      .def_readonly(
          "action_row_count", &PythonInverseSystemResult::action_row_count)
      .def_readonly(
          "action_column_count",
          &PythonInverseSystemResult::action_column_count)
      .def_readonly(
          "action_nonzeros", &PythonInverseSystemResult::action_nonzeros)
      .def_property_readonly("action_nnz", [](const PythonInverseSystemResult& r) {
        return r.action_nonzeros;
      })
      .def_readonly("action_rank", &PythonInverseSystemResult::action_rank)
      .def_readonly(
          "kernel_dimension", &PythonInverseSystemResult::kernel_dimension)
      .def_readonly(
          "truncated_kernel_generator_count",
          &PythonInverseSystemResult::truncated_kernel_generator_count)
      .def_property_readonly(
          "truncated_kernel_generators",
          [](const PythonInverseSystemResult& result) {
            return result.truncated_kernel_generator_count;
          })
      .def_readonly("annihilator", &PythonInverseSystemResult::annihilator)
      .def_property_readonly("ideal", [](const PythonInverseSystemResult& r) {
        return r.annihilator;
      })
      .def_readonly(
          "quotient_length", &PythonInverseSystemResult::quotient_length)
      .def_property_readonly("length_Q", [](const PythonInverseSystemResult& r) {
        return r.quotient_length;
      })
      .def("__repr__", &PythonInverseSystemResult::to_string);

  py::class_<PythonCycleScreenResult>(module, "CycleScreenResult")
      .def_readonly("status", &PythonCycleScreenResult::status)
      .def_property_readonly(
          "conclusive", &PythonCycleScreenResult::conclusive)
      .def_readonly("length_Q", &PythonCycleScreenResult::length_Q)
      .def_readonly(
          "length_P_mod_J2", &PythonCycleScreenResult::length_P_mod_J2)
      .def_readonly("g_in_J", &PythonCycleScreenResult::g_in_J)
      .def_readonly(
          "derivatives_in_J", &PythonCycleScreenResult::derivatives_in_J)
      .def_readonly("cycle_valid", &PythonCycleScreenResult::cycle_valid)
      .def_readonly(
          "multiplication_rank", &PythonCycleScreenResult::multiplication_rank)
      .def_readonly(
          "full_column_rank_candidate",
          &PythonCycleScreenResult::full_column_rank_candidate)
      .def_readonly(
          "certified_faithful",
          &PythonCycleScreenResult::certified_faithful)
      .def_property_readonly("hit", [](const PythonCycleScreenResult& result) {
        return result.certified_faithful;
      })
      .def_readonly(
          "certification", &PythonCycleScreenResult::certification)
      .def_readonly("detail", &PythonCycleScreenResult::detail)
      .def_property_readonly(
          "resource_detail", [](const PythonCycleScreenResult& result) {
            return result.detail;
          })
      .def("__repr__", &PythonCycleScreenResult::to_string);

  py::class_<PythonH1ScreenResult>(module, "H1ScreenResult")
      .def_readonly("status", &PythonH1ScreenResult::status)
      .def_property_readonly(
          "conclusive", &PythonH1ScreenResult::conclusive)
      .def_readonly("length_Q", &PythonH1ScreenResult::length_Q)
      .def_readonly(
          "length_P_mod_J2", &PythonH1ScreenResult::length_P_mod_J2)
      .def_readonly(
          "conormal_dimension", &PythonH1ScreenResult::conormal_dimension)
      .def_readonly("h1_dimension", &PythonH1ScreenResult::h1_dimension)
      .def_readonly(
          "socle_dimension", &PythonH1ScreenResult::socle_dimension)
      .def_readonly("modulus", &PythonH1ScreenResult::modulus)
      .def_readonly(
          "leading_ideal_fingerprint",
          &PythonH1ScreenResult::leading_ideal_fingerprint)
      .def_property_readonly(
          "leading_ideal_signature",
          [](const PythonH1ScreenResult& result) {
            return py::bytes(result.leading_ideal_signature);
          })
      .def_readonly(
          "maximum_individual_rank_lower_bound",
          &PythonH1ScreenResult::maximum_individual_rank_lower_bound)
      .def_readonly(
          "maximum_individual_rank_upper_bound",
          &PythonH1ScreenResult::maximum_individual_rank_upper_bound)
      .def_property_readonly(
          "rank_lower_bound", [](const PythonH1ScreenResult& result) {
            return result.maximum_individual_rank_lower_bound;
          })
      .def_property_readonly(
          "rank_upper_bound", [](const PythonH1ScreenResult& result) {
            return result.maximum_individual_rank_upper_bound;
          })
      .def_property_readonly(
          "maximum_individual_rank",
          [](const PythonH1ScreenResult& result) -> py::object {
            if (result.conclusive() &&
                result.maximum_individual_rank_lower_bound ==
                result.maximum_individual_rank_upper_bound) {
              return py::cast(result.maximum_individual_rank_lower_bound);
            }
            return py::none();
          })
      .def_readonly("rank_proof", &PythonH1ScreenResult::rank_proof)
      .def_readonly(
          "full_socle_rank_candidate",
          &PythonH1ScreenResult::full_socle_rank_candidate)
      .def_readonly(
          "certified_faithful", &PythonH1ScreenResult::certified_faithful)
      .def_property_readonly("hit", [](const PythonH1ScreenResult& result) {
        return result.certified_faithful;
      })
      .def_readonly(
          "witness_coefficients", &PythonH1ScreenResult::witness_coefficients)
      .def_readonly("witness", &PythonH1ScreenResult::witness)
      .def_property_readonly(
          "witness_polynomial", [](const PythonH1ScreenResult& result) {
            return result.witness;
          })
      .def_readonly(
          "certification", &PythonH1ScreenResult::certification)
      .def_readonly("detail", &PythonH1ScreenResult::detail)
      .def_property_readonly(
          "resource_detail", [](const PythonH1ScreenResult& result) {
            return result.detail;
          })
      .def("__repr__", &PythonH1ScreenResult::to_string);

  py::class_<PythonInverseSystemDiscoveryRecord>(
      module, "InverseSystemDiscoveryRecord")
      .def_readonly(
          "status", &PythonInverseSystemDiscoveryRecord::status)
      .def_property_readonly(
          "conclusive", &PythonInverseSystemDiscoveryRecord::conclusive)
      .def_readonly(
          "candidate_index",
          &PythonInverseSystemDiscoveryRecord::candidate_index)
      .def_readonly(
          "maximum_dual_degree",
          &PythonInverseSystemDiscoveryRecord::maximum_dual_degree)
      .def_readonly(
          "action_rank", &PythonInverseSystemDiscoveryRecord::action_rank)
      .def_readonly(
          "kernel_dimension",
          &PythonInverseSystemDiscoveryRecord::kernel_dimension)
      .def_readonly(
          "quotient_length",
          &PythonInverseSystemDiscoveryRecord::quotient_length)
      .def_readonly(
          "h1_dimension", &PythonInverseSystemDiscoveryRecord::h1_dimension)
      .def_readonly(
          "socle_dimension",
          &PythonInverseSystemDiscoveryRecord::socle_dimension)
      .def_readonly(
          "modulus", &PythonInverseSystemDiscoveryRecord::modulus)
      .def_readonly(
          "leading_ideal_fingerprint",
          &PythonInverseSystemDiscoveryRecord::leading_ideal_fingerprint)
      .def_property_readonly(
          "leading_ideal_signature",
          [](const PythonInverseSystemDiscoveryRecord& result) {
            return py::bytes(result.leading_ideal_signature);
          })
      .def_readonly(
          "maximum_individual_rank_lower_bound",
          &PythonInverseSystemDiscoveryRecord::
              maximum_individual_rank_lower_bound)
      .def_readonly(
          "maximum_individual_rank_upper_bound",
          &PythonInverseSystemDiscoveryRecord::
              maximum_individual_rank_upper_bound)
      .def_property_readonly(
          "rank_lower_bound",
          [](const PythonInverseSystemDiscoveryRecord& result) {
            return result.maximum_individual_rank_lower_bound;
          })
      .def_property_readonly(
          "rank_upper_bound",
          [](const PythonInverseSystemDiscoveryRecord& result) {
            return result.maximum_individual_rank_upper_bound;
          })
      .def_readonly(
          "rank_proof", &PythonInverseSystemDiscoveryRecord::rank_proof)
      .def_readonly(
          "maximum_individual_rank",
          &PythonInverseSystemDiscoveryRecord::maximum_individual_rank)
      .def_readonly(
          "full_rank_candidate",
          &PythonInverseSystemDiscoveryRecord::full_rank_candidate)
      .def_readonly(
          "certified_faithful",
          &PythonInverseSystemDiscoveryRecord::certified_faithful)
      .def_property_readonly(
          "hit", [](const PythonInverseSystemDiscoveryRecord& result) {
            return result.certified_faithful;
          })
      .def_readonly(
          "retained_dual_generators",
          &PythonInverseSystemDiscoveryRecord::retained_dual_generators)
      .def_property_readonly(
          "dual_generators",
          [](const PythonInverseSystemDiscoveryRecord& result) {
            return result.retained_dual_generators;
          })
      .def_readonly(
          "retained_annihilator_basis",
          &PythonInverseSystemDiscoveryRecord::retained_annihilator_basis)
      .def_property_readonly(
          "annihilator_basis",
          [](const PythonInverseSystemDiscoveryRecord& result) {
            return result.retained_annihilator_basis;
          })
      .def_readonly("witness", &PythonInverseSystemDiscoveryRecord::witness)
      .def_readonly("detail", &PythonInverseSystemDiscoveryRecord::detail)
      .def_property_readonly(
          "resource_detail",
          [](const PythonInverseSystemDiscoveryRecord& result) {
            return result.detail;
          })
      .def("__repr__", &PythonInverseSystemDiscoveryRecord::to_string);

  py::class_<PythonRing, std::shared_ptr<PythonRing>>(module, "Ring")
      .def_property_readonly("variables", &PythonRing::variable_names)
      .def_property_readonly("order", &PythonRing::order_name)
      .def_property_readonly("characteristic", &PythonRing::characteristic)
      .def_property_readonly(
          "coefficient_field", &PythonRing::coefficient_field)
      .def("__str__", &PythonRing::to_string)
      .def("__repr__", &PythonRing::to_string)
      .def("__call__", [](const std::shared_ptr<PythonRing>& ring,
                           const py::object& value) {
        if (py::isinstance<py::str>(value)) {
          return ring->parse(value.cast<std::string>());
        }
        if (py::isinstance<py::int_>(value)) {
          return ring->parse(
              python_integer_text(value.cast<py::int_>()));
        }
        if (py::isinstance<PythonPolynomial>(value)) {
          return ring->coerce(value.cast<PythonPolynomial>());
        }
        throw py::type_error(
            "a ring can coerce only a polynomial, integer, or expression string");
      })
      .def("zero", &PythonRing::zero)
      .def("one", &PythonRing::one)
      .def("gen",
           py::overload_cast<std::size_t>(&PythonRing::gen),
           py::arg("index"))
      .def("gen",
           py::overload_cast<std::string_view>(&PythonRing::gen),
           py::arg("name"))
      .def("gens", &PythonRing::gens)
      .def("gb", &PythonRing::groebner_basis, py::arg("generators"))
      .def("GB", &PythonRing::groebner_basis, py::arg("generators"))
      .def("groebner_basis",
           &PythonRing::groebner_basis,
           py::arg("generators"))
      .def("nf",
           &PythonRing::normal_form,
           py::arg("polynomial"),
           py::arg("divisors"))
      .def("NF",
           &PythonRing::normal_form,
           py::arg("polynomial"),
           py::arg("divisors"))
      .def("normal_form",
           &PythonRing::normal_form,
           py::arg("polynomial"),
           py::arg("divisors"))
      .def("nfs", [](const std::shared_ptr<PythonRing>& ring,
                       const std::vector<PythonPolynomial>& polynomials,
                       const py::object& reducers) {
        if (py::isinstance<PythonIdeal>(reducers)) {
          const auto ideal = reducers.cast<PythonIdeal>();
          if (ideal.ring().get() != ring.get()) {
            throw std::invalid_argument(
                "ideals must belong to this exact ring context");
          }
          return ideal.normal_forms(polynomials);
        }
        return ring->normal_forms(
            polynomials,
            reducers.cast<std::vector<PythonPolynomial>>());
      }, py::arg("polynomials"), py::arg("divisors_or_ideal"))
      .def("NFs", [](const std::shared_ptr<PythonRing>& ring,
                       const std::vector<PythonPolynomial>& polynomials,
                       const py::object& reducers) {
        if (py::isinstance<PythonIdeal>(reducers)) {
          const auto ideal = reducers.cast<PythonIdeal>();
          if (ideal.ring().get() != ring.get()) {
            throw std::invalid_argument(
                "ideals must belong to this exact ring context");
          }
          return ideal.normal_forms(polynomials);
        }
        return ring->normal_forms(
            polynomials,
            reducers.cast<std::vector<PythonPolynomial>>());
      }, py::arg("polynomials"), py::arg("divisors_or_ideal"))
      .def("normal_forms",
           [](const std::shared_ptr<PythonRing>& ring,
              const std::vector<PythonPolynomial>& polynomials,
              const py::object& reducers) {
        if (py::isinstance<PythonIdeal>(reducers)) {
          const auto ideal = reducers.cast<PythonIdeal>();
          if (ideal.ring().get() != ring.get()) {
            throw std::invalid_argument(
                "ideals must belong to this exact ring context");
          }
          return ideal.normal_forms(polynomials);
        }
        return ring->normal_forms(
            polynomials,
            reducers.cast<std::vector<PythonPolynomial>>());
      }, py::arg("polynomials"), py::arg("divisors_or_ideal"))
      .def("divide",
           &PythonRing::divide,
           py::arg("polynomial"),
           py::arg("divisors"))
      .def("ideal", &PythonRing::ideal, py::arg("generators"))
      .def("local_ideal", &PythonRing::local_ideal,
           py::arg("generators"), py::kw_only(),
           py::arg("maximal_power"))
      .def("quotient", &PythonRing::quotient,
           py::arg("ideal"), py::kw_only(),
           py::arg("max_monomials") = 1'000'000,
           py::arg("max_generated_rows") = 2'000'000,
           py::arg("max_matrix_triplets") = 20'000'000)
      .def("cotangent_h1", &PythonRing::cotangent_h1,
           py::arg("generators"), py::kw_only(),
           py::arg("maximal_power"),
           py::arg("max_monomials") = 1'000'000,
           py::arg("max_generated_rows") = 2'000'000,
           py::arg("max_matrix_triplets") = 20'000'000)
      .def("audit_cycle",
           [](const std::shared_ptr<PythonRing>& ring,
              const py::object& ideal_or_generators,
              const PythonPolynomial& polynomial,
              std::optional<std::size_t> max_matrix_entries) {
             return ring->audit_cycle(
                 coerce_ideal_argument(ring, ideal_or_generators), polynomial,
                 max_matrix_entries);
           },
           py::arg("ideal_or_generators"), py::arg("polynomial"),
           py::kw_only(), py::arg("max_matrix_entries") = 5'000'000)
      .def("colon_closure",
           [](const std::shared_ptr<PythonRing>& ring,
              const py::object& ideal_or_generators,
              const PythonPolynomial& polynomial,
              std::size_t max_steps,
              std::optional<std::size_t> max_matrix_entries) {
             return ring->colon_closure(
                 coerce_ideal_argument(ring, ideal_or_generators), polynomial,
                 max_steps, max_matrix_entries);
           },
           py::arg("ideal_or_generators"), py::arg("polynomial"),
           py::kw_only(), py::arg("max_steps") = 16,
           py::arg("max_matrix_entries") = 5'000'000)
      .def("full_h1_action",
           [](const std::shared_ptr<PythonRing>& ring,
              const py::object& ideal_or_generators,
              std::optional<std::size_t> max_matrix_entries,
              std::optional<std::size_t>
                  max_witness_audit_matrix_entries,
              std::optional<std::size_t> max_minors,
              std::optional<std::size_t> max_parameter_terms,
              std::optional<std::size_t> max_determinant_products,
              std::size_t max_finite_field_evaluations,
              std::optional<std::size_t> max_groebner_critical_pairs,
              std::optional<std::size_t> max_groebner_basis_polynomials,
              std::optional<std::size_t> max_groebner_reduction_steps,
              std::optional<std::size_t> max_groebner_live_terms,
              std::optional<std::size_t> max_groebner_basis_terms) {
             return ring->full_h1_action(
                 coerce_ideal_argument(ring, ideal_or_generators),
                 max_matrix_entries, max_witness_audit_matrix_entries,
                 max_minors, max_parameter_terms, max_determinant_products,
                 max_finite_field_evaluations,
                 max_groebner_critical_pairs,
                 max_groebner_basis_polynomials,
                 max_groebner_reduction_steps,
                 max_groebner_live_terms, max_groebner_basis_terms);
           },
           py::arg("ideal_or_generators"), py::kw_only(),
           py::arg("max_matrix_entries") = 5'000'000,
           py::arg("max_witness_audit_matrix_entries") = 5'000'000,
           py::arg("max_minors") = 1'000'000,
           py::arg("max_parameter_terms") = 1'000'000,
           py::arg("max_determinant_products") = 10'000'000,
           py::arg("max_finite_field_evaluations") = 1'000'000,
           py::arg("max_groebner_critical_pairs") = 1'000'000,
           py::arg("max_groebner_basis_polynomials") = 100'000,
           py::arg("max_groebner_reduction_steps") = 10'000'000,
           py::arg("max_groebner_live_terms") = 1'000'000,
           py::arg("max_groebner_basis_terms") = 2'000'000)
      .def(
          "macaulay_annihilator",
          [](const std::shared_ptr<PythonRing>& ring,
             const std::vector<PythonPolynomial>& dual_generators,
             const std::string& convention,
             std::optional<std::size_t> max_basis_monomials,
             std::optional<std::size_t> max_action_rows,
             std::optional<std::size_t> max_action_nonzeros,
             std::optional<std::size_t> max_dense_matrix_entries,
             std::optional<std::size_t> max_kernel_coordinate_entries,
             std::optional<std::size_t> max_elimination_live_nonzeros,
             std::optional<std::size_t> max_elimination_operations,
             bool verify_invariants,
             std::optional<std::size_t> max_invariant_replay_checks,
             std::optional<std::size_t> max_groebner_critical_pairs,
             std::optional<std::size_t> max_groebner_basis_polynomials,
             std::optional<std::size_t> max_groebner_reduction_steps,
             std::optional<std::size_t> max_groebner_live_terms,
             std::optional<std::size_t> max_groebner_basis_terms) {
            return ring->macaulay_annihilator(
                dual_generators, parse_apolarity_convention(convention),
                make_inverse_system_limits(
                    max_basis_monomials, max_action_rows,
                    max_action_nonzeros, max_dense_matrix_entries,
                    max_kernel_coordinate_entries,
                    max_elimination_live_nonzeros,
                    max_elimination_operations, verify_invariants,
                    max_invariant_replay_checks,
                    max_groebner_critical_pairs,
                    max_groebner_basis_polynomials,
                    max_groebner_reduction_steps,
                    max_groebner_live_terms, max_groebner_basis_terms));
          },
          py::arg("dual_generators"), py::arg("convention") = "ordinary",
          py::kw_only(), py::arg("max_basis_monomials") = 1'000'000,
          py::arg("max_action_rows") = 1'000'000,
          py::arg("max_action_nonzeros") = 10'000'000,
          py::arg("max_dense_matrix_entries") = 5'000'000,
          py::arg("max_kernel_coordinate_entries") = 5'000'000,
          py::arg("max_elimination_live_nonzeros") = 10'000'000,
          py::arg("max_elimination_operations") = 100'000'000,
          py::arg("verify_invariants") = false,
          py::arg("max_invariant_replay_checks") = 1'000'000,
          py::arg("max_groebner_critical_pairs") = 1'000'000,
          py::arg("max_groebner_basis_polynomials") = 100'000,
          py::arg("max_groebner_reduction_steps") = 10'000'000,
          py::arg("max_groebner_live_terms") = 1'000'000,
          py::arg("max_groebner_basis_terms") = 2'000'000)
      .def(
          "screen_cycle",
          [](const std::shared_ptr<PythonRing>& ring,
             const py::object& ideal_or_generators,
             const PythonPolynomial& polynomial,
             std::optional<std::size_t> max_matrix_entries,
             bool certify_hits,
             std::optional<std::size_t> max_certification_matrix_entries,
             std::optional<std::size_t> max_groebner_critical_pairs,
             std::optional<std::size_t> max_groebner_basis_polynomials,
             std::optional<std::size_t> max_groebner_reduction_steps,
             std::optional<std::size_t> max_groebner_live_terms,
             std::optional<std::size_t> max_groebner_basis_terms) {
            if (ring->characteristic() == 0) {
              throw std::invalid_argument(
                  "packed discovery screens are available only over GF(p)");
            }
            return ring->screen_cycle(
                coerce_ideal_argument(ring, ideal_or_generators), polynomial,
                make_cycle_screen_options(
                    max_matrix_entries, certify_hits,
                    max_certification_matrix_entries,
                    max_groebner_critical_pairs,
                    max_groebner_basis_polynomials,
                    max_groebner_reduction_steps,
                    max_groebner_live_terms, max_groebner_basis_terms));
          },
          py::arg("ideal_generators"), py::arg("g"),
          py::kw_only(), py::arg("max_matrix_entries") = 20'000'000,
          py::arg("certify_hits") = true,
          py::arg("max_certification_matrix_entries") = 5'000'000,
          py::arg("max_groebner_critical_pairs") = 1'000'000,
          py::arg("max_groebner_basis_polynomials") = 100'000,
          py::arg("max_groebner_reduction_steps") = 10'000'000,
          py::arg("max_groebner_live_terms") = 1'000'000,
          py::arg("max_groebner_basis_terms") = 2'000'000)
      .def(
          "screen_full_h1",
          [](const std::shared_ptr<PythonRing>& ring,
             const py::object& ideal_or_generators,
             std::optional<std::size_t> max_matrix_entries,
             bool certify_hits,
             std::optional<std::size_t> max_certification_matrix_entries,
             std::optional<std::size_t> max_minors,
             std::optional<std::size_t> max_parameter_terms,
             std::optional<std::size_t> max_determinant_products,
             std::size_t max_finite_field_evaluations,
             std::optional<std::size_t> max_groebner_critical_pairs,
             std::optional<std::size_t> max_groebner_basis_polynomials,
             std::optional<std::size_t> max_groebner_reduction_steps,
             std::optional<std::size_t> max_groebner_live_terms,
             std::optional<std::size_t> max_groebner_basis_terms) {
            if (ring->characteristic() == 0) {
              throw std::invalid_argument(
                  "packed discovery screens are available only over GF(p)");
            }
            return ring->screen_full_h1(
                coerce_ideal_argument(ring, ideal_or_generators),
                make_h1_screen_options(
                    max_matrix_entries, certify_hits,
                    max_certification_matrix_entries, max_minors,
                    max_parameter_terms, max_determinant_products,
                    max_finite_field_evaluations,
                    max_groebner_critical_pairs,
                    max_groebner_basis_polynomials,
                    max_groebner_reduction_steps,
                    max_groebner_live_terms, max_groebner_basis_terms));
          },
          py::arg("ideal_generators"), py::kw_only(),
          py::arg("max_matrix_entries") = 20'000'000,
          py::arg("certify_hits") = true,
          py::arg("max_certification_matrix_entries") = 5'000'000,
          py::arg("max_minors") = 1'000'000,
          py::arg("max_parameter_terms") = 1'000'000,
          py::arg("max_determinant_products") = 10'000'000,
          py::arg("max_finite_field_evaluations") = 1'000'000,
          py::arg("max_groebner_critical_pairs") = 1'000'000,
          py::arg("max_groebner_basis_polynomials") = 100'000,
          py::arg("max_groebner_reduction_steps") = 10'000'000,
          py::arg("max_groebner_live_terms") = 1'000'000,
          py::arg("max_groebner_basis_terms") = 2'000'000)
      .def(
          "screen_cycles_parallel",
          [](const std::shared_ptr<PythonRing>& ring,
             const py::object& ideal_or_generators,
             const std::vector<PythonPolynomial>& candidates,
             std::size_t workers,
             std::optional<std::size_t> max_matrix_entries,
             bool certify_hits,
             std::optional<std::size_t> max_certification_matrix_entries,
             std::optional<std::size_t> max_groebner_critical_pairs,
             std::optional<std::size_t> max_groebner_basis_polynomials,
             std::optional<std::size_t> max_groebner_reduction_steps,
             std::optional<std::size_t> max_groebner_live_terms,
             std::optional<std::size_t> max_groebner_basis_terms) {
            if (ring->characteristic() == 0) {
              throw std::invalid_argument(
                  "packed discovery screens are available only over GF(p)");
            }
            le::CandidateExecutorOptions executor;
            executor.worker_count = workers;
            return ring->screen_cycles_parallel(
                coerce_ideal_argument(ring, ideal_or_generators), candidates,
                make_cycle_screen_options(
                    max_matrix_entries, certify_hits,
                    max_certification_matrix_entries,
                    max_groebner_critical_pairs,
                    max_groebner_basis_polynomials,
                    max_groebner_reduction_steps,
                    max_groebner_live_terms, max_groebner_basis_terms),
                executor);
          },
          py::arg("ideal_generators"), py::arg("candidates"),
          py::arg("workers") = 0, py::kw_only(),
          py::arg("max_matrix_entries") = 20'000'000,
          py::arg("certify_hits") = true,
          py::arg("max_certification_matrix_entries") = 5'000'000,
          py::arg("max_groebner_critical_pairs") = 1'000'000,
          py::arg("max_groebner_basis_polynomials") = 100'000,
          py::arg("max_groebner_reduction_steps") = 10'000'000,
          py::arg("max_groebner_live_terms") = 1'000'000,
          py::arg("max_groebner_basis_terms") = 2'000'000)
      .def(
          "search_inverse_systems",
          [](const std::shared_ptr<PythonRing>& ring,
             const py::iterable& candidate_values,
             std::size_t workers,
             const std::string& convention,
             bool retain_all,
             std::optional<std::size_t> max_basis_monomials,
             std::optional<std::size_t> max_action_rows,
             std::optional<std::size_t> max_action_nonzeros,
             std::optional<std::size_t> max_dense_matrix_entries,
             std::optional<std::size_t> max_kernel_coordinate_entries,
             std::optional<std::size_t> max_elimination_live_nonzeros,
             std::optional<std::size_t> max_elimination_operations,
             bool verify_invariants,
             std::optional<std::size_t> max_invariant_replay_checks,
             std::optional<std::size_t> max_groebner_critical_pairs,
             std::optional<std::size_t> max_groebner_basis_polynomials,
             std::optional<std::size_t> max_groebner_reduction_steps,
             std::optional<std::size_t> max_groebner_live_terms,
             std::optional<std::size_t> max_groebner_basis_terms,
             std::optional<std::size_t> max_matrix_entries,
             bool certify_hits,
             std::optional<std::size_t> max_certification_matrix_entries,
             std::optional<std::size_t> max_minors,
             std::optional<std::size_t> max_parameter_terms,
             std::optional<std::size_t> max_determinant_products,
             std::size_t max_finite_field_evaluations) {
            if (ring->characteristic() == 0) {
              throw std::invalid_argument(
                  "packed inverse-system search is available only over GF(p)");
            }
            auto candidates = coerce_inverse_candidates(candidate_values);
            le::InverseSystemDiscoveryOptions discovery;
            discovery.convention = parse_apolarity_convention(convention);
            discovery.retain_all_candidates = retain_all;
            discovery.inverse_system_limits = make_inverse_system_limits(
                max_basis_monomials, max_action_rows,
                max_action_nonzeros, max_dense_matrix_entries,
                max_kernel_coordinate_entries,
                max_elimination_live_nonzeros,
                max_elimination_operations, verify_invariants,
                max_invariant_replay_checks,
                max_groebner_critical_pairs,
                max_groebner_basis_polynomials,
                max_groebner_reduction_steps,
                max_groebner_live_terms, max_groebner_basis_terms);
            discovery.h1_options = make_h1_screen_options(
                max_matrix_entries, certify_hits,
                max_certification_matrix_entries, max_minors,
                max_parameter_terms, max_determinant_products,
                max_finite_field_evaluations,
                max_groebner_critical_pairs,
                max_groebner_basis_polynomials,
                max_groebner_reduction_steps,
                max_groebner_live_terms, max_groebner_basis_terms);
            le::CandidateExecutorOptions executor;
            executor.worker_count = workers;
            return ring->search_inverse_systems(
                candidates, discovery, executor);
          },
          py::arg("candidates"), py::arg("workers") = 0,
          py::arg("convention") = "ordinary",
          py::arg("retain_all") = false, py::kw_only(),
          py::arg("max_basis_monomials") = 1'000'000,
          py::arg("max_action_rows") = 1'000'000,
          py::arg("max_action_nonzeros") = 10'000'000,
          py::arg("max_dense_matrix_entries") = 5'000'000,
          py::arg("max_kernel_coordinate_entries") = 5'000'000,
          py::arg("max_elimination_live_nonzeros") = 10'000'000,
          py::arg("max_elimination_operations") = 100'000'000,
          py::arg("verify_invariants") = false,
          py::arg("max_invariant_replay_checks") = 1'000'000,
          py::arg("max_groebner_critical_pairs") = 1'000'000,
          py::arg("max_groebner_basis_polynomials") = 100'000,
          py::arg("max_groebner_reduction_steps") = 10'000'000,
          py::arg("max_groebner_live_terms") = 1'000'000,
          py::arg("max_groebner_basis_terms") = 2'000'000,
          py::arg("max_matrix_entries") = 20'000'000,
          py::arg("certify_hits") = true,
          py::arg("max_certification_matrix_entries") = 5'000'000,
          py::arg("max_minors") = 1'000'000,
          py::arg("max_parameter_terms") = 1'000'000,
          py::arg("max_determinant_products") = 10'000'000,
          py::arg("max_finite_field_evaluations") = 1'000'000)
      .def(
          "make_jg_certificate", &PythonRing::make_jg_certificate,
          py::arg("ideal_generators"), py::arg("g"))
      .def("is_gb", &PythonRing::is_groebner_basis, py::arg("candidates"))
      .def("is_groebner_basis",
           &PythonRing::is_groebner_basis,
           py::arg("candidates"));

  polynomial
      .def_property_readonly("ring", &PythonPolynomial::ring)
      .def("__str__", &PythonPolynomial::to_string)
      .def("__repr__", &PythonPolynomial::to_string)
      .def("__eq__", &PythonPolynomial::equals, py::is_operator())
      .def("__ne__", [](const PythonPolynomial& left,
                         const PythonPolynomial& right) {
        return !left.equals(right);
      }, py::is_operator())
      .def("__add__", [](const PythonPolynomial& left,
                          const PythonPolynomial& right) {
        return left.add(right);
      }, py::is_operator())
      .def("__add__", [](const PythonPolynomial& left,
                          const py::int_& right) {
        return left.add(python_integer_like(left, right));
      }, py::is_operator())
      .def("__radd__", [](const PythonPolynomial& right,
                           const py::int_& left) {
        return right.add(python_integer_like(right, left));
      }, py::is_operator())
      .def("__sub__", [](const PythonPolynomial& left,
                          const PythonPolynomial& right) {
        return left.subtract(right);
      }, py::is_operator())
      .def("__sub__", [](const PythonPolynomial& left,
                          const py::int_& right) {
        return left.subtract(python_integer_like(left, right));
      }, py::is_operator())
      .def("__rsub__", [](const PythonPolynomial& right,
                           const py::int_& left) {
        return python_integer_like(right, left).subtract(right);
      }, py::is_operator())
      .def("__mul__", [](const PythonPolynomial& left,
                          const PythonPolynomial& right) {
        return left.multiply(right);
      }, py::is_operator())
      .def("__mul__", [](const PythonPolynomial& left,
                          const py::int_& right) {
        return left.multiply(python_integer_like(left, right));
      }, py::is_operator())
      .def("__rmul__", [](const PythonPolynomial& right,
                           const py::int_& left) {
        return right.multiply(python_integer_like(right, left));
      }, py::is_operator())
      .def("__truediv__", [](const PythonPolynomial& numerator,
                              const py::int_& denominator) {
        return numerator.multiply(
            python_inverse_integer_like(numerator, denominator));
      }, py::is_operator())
      .def("__neg__", &PythonPolynomial::negate, py::is_operator())
      .def("__pow__", [](const PythonPolynomial& value,
                          const py::int_& exponent,
                          const py::object& modulus) {
        if (!modulus.is_none()) {
          throw py::type_error("modular polynomial powers are not supported");
        }
        const py::int_ zero(0);
        const int is_negative =
            PyObject_RichCompareBool(exponent.ptr(), zero.ptr(), Py_LT);
        if (is_negative < 0) {
          throw py::error_already_set();
        }
        if (is_negative != 0) {
          throw py::value_error("a polynomial exponent must be nonnegative");
        }
        const auto converted = PyLong_AsUnsignedLongLong(exponent.ptr());
        if (converted == static_cast<unsigned long long>(-1) &&
            PyErr_Occurred() != nullptr) {
          throw py::error_already_set();
        }
        return value.pow(static_cast<std::uint64_t>(converted));
      }, py::arg("exponent"), py::arg("modulus") = py::none(),
         py::is_operator())
      .def("derivative",
           py::overload_cast<std::size_t>(&PythonPolynomial::derivative,
                                          py::const_),
           py::arg("variable"))
      .def("derivative",
           py::overload_cast<std::string_view>(&PythonPolynomial::derivative,
                                               py::const_),
           py::arg("variable"))
      .def("derivative",
           py::overload_cast<const PythonPolynomial&>(
               &PythonPolynomial::derivative, py::const_),
           py::arg("variable"))
      .def("diff",
           py::overload_cast<std::size_t>(&PythonPolynomial::derivative,
                                          py::const_),
           py::arg("variable"))
      .def("diff",
           py::overload_cast<std::string_view>(&PythonPolynomial::derivative,
                                               py::const_),
           py::arg("variable"))
      .def("diff",
           py::overload_cast<const PythonPolynomial&>(
               &PythonPolynomial::derivative, py::const_),
           py::arg("variable"))
      .def("degree", &PythonPolynomial::degree)
      .def("term_count", &PythonPolynomial::term_count)
      .def("is_zero", &PythonPolynomial::is_zero)
      .def("monic", &PythonPolynomial::monic);
  polynomial.attr("__hash__") = py::none();

  module.def(
      "_make_qq_ring",
      [](std::vector<std::string> variables, const std::string& order) {
        return PythonRing::make_qq(std::move(variables), parse_order(order));
      },
      py::arg("variables"),
      py::arg("order") = "grevlex");
  module.def(
      "_make_gf_ring",
      [](std::uint32_t modulus,
         std::vector<std::string> variables,
         const std::string& order) {
        return PythonRing::make_gf(
            modulus, std::move(variables), parse_order(order));
      },
      py::arg("modulus"),
      py::arg("variables"),
      py::arg("order") = "grevlex");
}

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "laughableengine/cotangent_h1.hpp"

namespace laughableengine {

template <typename Field>
class OriginPowerIdeal;
template <typename Field>
class FiniteQuotient;
template <typename Field>
class QuotientIdeal;
template <typename Field>
class IdealPreimage;
template <typename Field>
class ConormalModule;
template <typename Field>
class ConormalDerivativeMap;
template <typename Field>
class CotangentH1Module;
template <typename Field>
class CotangentH1Element;

namespace finite_algebra_detail {

template <typename Field>
struct OriginPowerIdealState {
  PolynomialRing<Field> ring;
  std::vector<Polynomial<Field>> lower_generators;
  std::size_t maximal_power;
};

template <typename Field>
struct FiniteAlgebraState {
  OriginPowerIdeal<Field> ideal;
  std::shared_ptr<const OriginPowerQuotientData<Field>> quotient;

  mutable std::once_flag conormal_once;
  mutable std::once_flag derivative_once;
  mutable std::once_flag h1_once;
  mutable std::shared_ptr<const OriginPowerConormalData<Field>> conormal;
  mutable std::shared_ptr<const OriginPowerDerivativeData<Field>> derivative;
  mutable std::shared_ptr<const CotangentH1Presentation<Field>> h1;

  FiniteAlgebraState(
      OriginPowerIdeal<Field> source_ideal,
      std::shared_ptr<const OriginPowerQuotientData<Field>> quotient_data)
      : ideal(std::move(source_ideal)),
        quotient(std::move(quotient_data)) {}

  [[nodiscard]] std::shared_ptr<const OriginPowerConormalData<Field>>
  conormal_data() const {
    std::call_once(conormal_once, [this] {
      conormal = build_origin_power_conormal_data(quotient);
    });
    return conormal;
  }

  [[nodiscard]] std::shared_ptr<const OriginPowerDerivativeData<Field>>
  derivative_data() const {
    const auto conormal_stage = conormal_data();
    std::call_once(derivative_once, [this, conormal_stage] {
      derivative = build_cotangent_derivative_data(conormal_stage);
    });
    return derivative;
  }

  [[nodiscard]] std::shared_ptr<const CotangentH1Presentation<Field>>
  h1_data() const {
    const auto derivative_stage = derivative_data();
    std::call_once(h1_once, [this, derivative_stage] {
      h1 = std::make_shared<const CotangentH1Presentation<Field>>(
          build_cotangent_h1_data(derivative_stage));
    });
    return h1;
  }
};

template <typename Field>
[[nodiscard]] std::size_t coordinate_rank(
    const Field& field,
    std::size_t ambient_dimension,
    std::span<const std::vector<typename Field::Element>> vectors) {
  using RowSpace = IncrementalSparseRowSpace<Field>;
  using SparseVector = typename RowSpace::SparseVector;

  RowSpace rows(field, ambient_dimension);
  for (const auto& vector : vectors) {
    if (vector.size() != ambient_dimension) {
      throw std::invalid_argument(
          "quotient-ideal coordinate vector has the wrong dimension");
    }
    SparseVector sparse;
    for (std::size_t column = 0; column < vector.size(); ++column) {
      auto value = field.canonical(vector[column]);
      if (!field.is_zero(value)) {
        sparse.push_back({column, std::move(value)});
      }
    }
    static_cast<void>(rows.insert(std::move(sparse)));
  }

  return rows.rank();
}

template <typename Field>
[[nodiscard]] bool coordinate_span_contains(
    const Field& field,
    std::size_t ambient_dimension,
    std::span<const std::vector<typename Field::Element>> basis,
    std::span<const typename Field::Element> vector) {
  using RowSpace = IncrementalSparseRowSpace<Field>;
  using SparseVector = typename RowSpace::SparseVector;
  if (vector.size() != ambient_dimension) {
    throw std::invalid_argument(
        "quotient-ideal membership vector has the wrong dimension");
  }

  RowSpace rows(field, ambient_dimension);
  for (const auto& generator : basis) {
    SparseVector sparse;
    for (std::size_t column = 0; column < generator.size(); ++column) {
      if (!field.is_zero(generator[column])) {
        sparse.push_back({column, generator[column]});
      }
    }
    static_cast<void>(rows.insert(std::move(sparse)));
  }

  SparseVector candidate;
  for (std::size_t column = 0; column < vector.size(); ++column) {
    auto value = field.canonical(vector[column]);
    if (!field.is_zero(value)) {
      candidate.push_back({column, std::move(value)});
    }
  }
  return rows.normal_form_readonly(std::move(candidate)).empty();
}

}  // namespace finite_algebra_detail

class CotangentClassError : public std::domain_error {
 public:
  explicit CotangentClassError(CotangentClassStatus status)
      : std::domain_error(message(status)), status_(status) {
    if (status == CotangentClassStatus::Complete) {
      throw std::invalid_argument(
          "CotangentClassError requires an invalid class status");
    }
  }

  [[nodiscard]] CotangentClassStatus status() const noexcept {
    return status_;
  }

 private:
  [[nodiscard]] static const char* message(
      CotangentClassStatus status) noexcept {
    switch (status) {
      case CotangentClassStatus::Complete:
        return "the polynomial represents a cotangent-H1 class";
      case CotangentClassStatus::NotInIdeal:
        return "a cotangent-H1 representative must belong to the defining ideal";
      case CotangentClassStatus::NotCycle:
        return "the derivative of a cotangent-H1 representative must vanish in the quotient";
    }
    return "invalid cotangent-H1 representative";
  }

  CotangentClassStatus status_;
};

// A precise, non-Groebner representation of J=(G)+m^N at the origin.  The
// type deliberately does not pretend to be Ideal<Field>, whose contract is an
// eagerly computed reduced Groebner basis.
template <typename Field>
class OriginPowerIdeal {
 public:
  using Ring = PolynomialRing<Field>;
  using PolynomialType = Polynomial<Field>;

  OriginPowerIdeal(
      Ring ring,
      std::vector<PolynomialType> lower_generators,
      std::size_t maximal_power)
      : state_(std::make_shared<const State>(State{
            std::move(ring), std::move(lower_generators), maximal_power})) {
    validate();
  }

  [[nodiscard]] const Ring& ring() const noexcept { return state_->ring; }
  [[nodiscard]] std::span<const PolynomialType> lower_generators() const
      noexcept {
    return state_->lower_generators;
  }
  [[nodiscard]] std::size_t maximal_power() const noexcept {
    return state_->maximal_power;
  }

  // This is a deterministic displayed generating set, not a reduced
  // Groebner basis: first G, then all degree-N monomials.
  [[nodiscard]] std::vector<PolynomialType> generators() const {
    TruncatedMonomialSpace monomials(
        ring().variable_count(), maximal_power() + 1);
    auto result = state_->lower_generators;
    const auto degree_n = monomials.degree_range(maximal_power());
    result.reserve(result.size() + degree_n.size());
    for (std::size_t index = degree_n.first; index < degree_n.past_last;
         ++index) {
      result.push_back(monomials.as_polynomial(ring(), index));
    }
    return result;
  }

  [[nodiscard]] bool same_presentation(
      const OriginPowerIdeal& other) const noexcept {
    return state_.get() == other.state_.get();
  }

  [[nodiscard]] FiniteQuotient<Field> quotient(
      const CotangentH1Options& options = {}) const;

 private:
  using State = finite_algebra_detail::OriginPowerIdealState<Field>;

  void validate() const {
    if (maximal_power() == 0 ||
        maximal_power() >
            static_cast<std::size_t>(
                std::numeric_limits<std::uint16_t>::max()) /
                2) {
      throw CotangentH1InputError(
          CotangentH1InputIssue::InvalidMaximalPower);
    }
    const auto zero = ring().zero();
    for (const auto& generator : state_->lower_generators) {
      if (!zero.same_ring(generator)) {
        throw CotangentH1InputError(
            CotangentH1InputIssue::GeneratorRingMismatch);
      }
      if (std::any_of(
              generator.terms().begin(), generator.terms().end(),
              [](const auto& term) {
                return term.monomial.total_degree() == 0;
              })) {
        throw CotangentH1InputError(
            CotangentH1InputIssue::GeneratorHasConstantTerm);
      }
    }
  }

  std::shared_ptr<const State> state_;
};

template <typename Field>
[[nodiscard]] OriginPowerIdeal<Field> origin_power_ideal(
    PolynomialRing<Field> ring,
    std::vector<Polynomial<Field>> lower_generators,
    std::size_t maximal_power) {
  return OriginPowerIdeal<Field>(
      std::move(ring), std::move(lower_generators), maximal_power);
}

template <typename Field>
[[nodiscard]] OriginPowerIdeal<Field> origin_power_ideal(
    PolynomialRing<Field> ring,
    std::initializer_list<Polynomial<Field>> lower_generators,
    std::size_t maximal_power) {
  return origin_power_ideal(
      std::move(ring),
      std::vector<Polynomial<Field>>(lower_generators), maximal_power);
}

template <typename Field>
class FiniteQuotient {
 public:
  using Element = typename Field::Element;
  using PolynomialType = Polynomial<Field>;
  using Ring = PolynomialRing<Field>;

  explicit FiniteQuotient(
      OriginPowerIdeal<Field> ideal,
      const CotangentH1Options& options = {}) {
    auto quotient_data = build_origin_power_quotient_data(
        CotangentH1Spec<Field>{
            ideal.ring(),
            std::vector<PolynomialType>(
                ideal.lower_generators().begin(),
                ideal.lower_generators().end()),
            ideal.maximal_power()},
        options);
    state_ = std::make_shared<const State>(
        std::move(ideal), std::move(quotient_data));
  }

  [[nodiscard]] const OriginPowerIdeal<Field>& defining_ideal() const noexcept {
    return state_->ideal;
  }
  [[nodiscard]] const Ring& polynomial_ring() const noexcept {
    return state_->quotient->ring();
  }
  [[nodiscard]] const Field& field() const noexcept {
    return polynomial_ring().field();
  }
  [[nodiscard]] std::size_t dimension() const noexcept {
    return state_->quotient->dimension();
  }
  [[nodiscard]] std::size_t square_quotient_dimension() const {
    return state_->conormal_data()->ambient_dimension();
  }
  [[nodiscard]] PolynomialType remainder(
      const PolynomialType& polynomial) const {
    return state_->quotient->remainder(polynomial);
  }
  [[nodiscard]] std::vector<PolynomialType> basis_representatives() const {
    return state_->quotient->basis();
  }
  [[nodiscard]] PolynomialType representative(
      std::span<const Element> coordinates) const {
    if (coordinates.size() != dimension()) {
      throw std::invalid_argument(
          "finite-quotient coordinate vector has the wrong dimension");
    }
    const auto basis = basis_representatives();
    auto result = polynomial_ring().zero();
    for (std::size_t index = 0; index < coordinates.size(); ++index) {
      result = result + basis[index].scaled(coordinates[index]);
    }
    return result;
  }

  [[nodiscard]] QuotientIdeal<Field> zero_ideal() const;
  [[nodiscard]] ConormalModule<Field> conormal_module() const;

  [[nodiscard]] bool same_context(
      const FiniteQuotient& other) const noexcept {
    return state_.get() == other.state_.get();
  }

 private:
  using State = finite_algebra_detail::FiniteAlgebraState<Field>;

  explicit FiniteQuotient(std::shared_ptr<const State> state)
      : state_(std::move(state)) {}

  std::shared_ptr<const State> state_;

  friend class QuotientIdeal<Field>;
  friend class IdealPreimage<Field>;
  friend class ConormalModule<Field>;
  friend class ConormalDerivativeMap<Field>;
  friend class CotangentH1Module<Field>;
  friend class CotangentH1Element<Field>;
};

template <typename Field>
[[nodiscard]] FiniteQuotient<Field> quotient(
    OriginPowerIdeal<Field> ideal,
    const CotangentH1Options& options = {}) {
  return FiniteQuotient<Field>(std::move(ideal), options);
}

template <typename Field>
FiniteQuotient<Field> OriginPowerIdeal<Field>::quotient(
    const CotangentH1Options& options) const {
  return laughableengine::quotient(*this, options);
}

template <typename Field>
class QuotientIdeal {
 public:
  using Element = typename Field::Element;
  using PolynomialType = Polynomial<Field>;
  using CoordinateVector = std::vector<Element>;

  [[nodiscard]] FiniteQuotient<Field> quotient() const {
    return FiniteQuotient<Field>(state_);
  }
  [[nodiscard]] std::size_t dimension() const noexcept {
    return basis_coordinates_.size();
  }
  [[nodiscard]] bool is_zero() const noexcept {
    return basis_coordinates_.empty();
  }
  [[nodiscard]] const std::vector<CoordinateVector>& basis_coordinates() const
      noexcept {
    return basis_coordinates_;
  }
  [[nodiscard]] const std::vector<PolynomialType>& lift_basis() const
      noexcept {
    return lift_basis_;
  }

  [[nodiscard]] bool contains_coordinates(
      std::span<const Element> coordinates) const {
    return finite_algebra_detail::coordinate_span_contains<Field>(
        state_->quotient->ring().field(),
        state_->quotient->dimension(), basis_coordinates_, coordinates);
  }

  [[nodiscard]] IdealPreimage<Field> preimage() const;

  friend bool operator==(
      const QuotientIdeal& left,
      const QuotientIdeal& right) {
    if (left.state_.get() != right.state_.get() ||
        left.dimension() != right.dimension()) {
      return false;
    }
    for (const auto& vector : left.basis_coordinates_) {
      if (!right.contains_coordinates(vector)) {
        return false;
      }
    }
    return true;
  }

 private:
  using State = finite_algebra_detail::FiniteAlgebraState<Field>;

  QuotientIdeal(
      std::shared_ptr<const State> state,
      std::vector<CoordinateVector> basis_coordinates,
      std::vector<PolynomialType> lift_basis = {})
      : state_(std::move(state)),
        basis_coordinates_(std::move(basis_coordinates)),
        lift_basis_(std::move(lift_basis)) {
    const auto rank = finite_algebra_detail::coordinate_rank<Field>(
        state_->quotient->ring().field(),
        state_->quotient->dimension(), basis_coordinates_);
    if (rank != basis_coordinates_.size()) {
      throw std::invalid_argument(
          "quotient-ideal basis coordinates must be independent");
    }
    if (lift_basis_.empty() && !basis_coordinates_.empty()) {
      lift_basis_.reserve(basis_coordinates_.size());
      const FiniteQuotient<Field> quotient(state_);
      for (const auto& coordinates : basis_coordinates_) {
        lift_basis_.push_back(quotient.representative(coordinates));
      }
    } else if (lift_basis_.size() != basis_coordinates_.size()) {
      throw std::invalid_argument(
          "quotient-ideal lift basis has the wrong size");
    }
  }

  std::shared_ptr<const State> state_;
  std::vector<CoordinateVector> basis_coordinates_;
  std::vector<PolynomialType> lift_basis_;

  friend class FiniteQuotient<Field>;
  friend class CotangentH1Element<Field>;
  friend class IdealPreimage<Field>;
};

template <typename Field>
QuotientIdeal<Field> FiniteQuotient<Field>::zero_ideal() const {
  return QuotientIdeal<Field>(state_, {});
}

template <typename Field>
class IdealPreimage {
 public:
  using PolynomialType = Polynomial<Field>;

  [[nodiscard]] const QuotientIdeal<Field>& quotient_ideal() const noexcept {
    return quotient_ideal_;
  }
  [[nodiscard]] const OriginPowerIdeal<Field>& source_ideal() const noexcept {
    return quotient_ideal_.state_->ideal;
  }
  [[nodiscard]] bool equals_source_ideal() const noexcept {
    return quotient_ideal_.is_zero();
  }
  [[nodiscard]] std::vector<PolynomialType> generators() const {
    auto result = quotient_ideal_.state_->quotient->ideal_generators();
    result.insert(
        result.end(), quotient_ideal_.lift_basis_.begin(),
        quotient_ideal_.lift_basis_.end());
    return result;
  }

  friend bool operator==(
      const IdealPreimage& left,
      const IdealPreimage& right) {
    return left.quotient_ideal_ == right.quotient_ideal_;
  }
  friend bool operator==(
      const IdealPreimage& left,
      const OriginPowerIdeal<Field>& right) noexcept {
    return left.source_ideal().same_presentation(right) &&
           left.equals_source_ideal();
  }
  friend bool operator==(
      const OriginPowerIdeal<Field>& left,
      const IdealPreimage& right) noexcept {
    return right == left;
  }

 private:
  explicit IdealPreimage(QuotientIdeal<Field> quotient_ideal)
      : quotient_ideal_(std::move(quotient_ideal)) {}

  QuotientIdeal<Field> quotient_ideal_;

  friend class QuotientIdeal<Field>;
};

template <typename Field>
IdealPreimage<Field> QuotientIdeal<Field>::preimage() const {
  return IdealPreimage<Field>(*this);
}

template <typename Field>
[[nodiscard]] IdealPreimage<Field> preimage(
    const QuotientIdeal<Field>& ideal) {
  return ideal.preimage();
}

template <typename Field>
class ConormalModule {
 public:
  using PolynomialType = Polynomial<Field>;

  explicit ConormalModule(FiniteQuotient<Field> quotient)
      : state_(std::move(quotient.state_)),
        conormal_(state_->conormal_data()) {}

  [[nodiscard]] FiniteQuotient<Field> algebra() const {
    return FiniteQuotient<Field>(state_);
  }
  [[nodiscard]] std::size_t dimension() const noexcept {
    return conormal_->dimension();
  }
  [[nodiscard]] const SparseMatrix<Field>& defining_matrix() const noexcept {
    return conormal_->reduction_matrix();
  }
  [[nodiscard]] std::vector<PolynomialType> representative_basis(
      const SparseEliminationLimits& limits) const {
    return conormal_->representative_basis(limits);
  }

  [[nodiscard]] ConormalDerivativeMap<Field> derivative_map() const;

 private:
  using State = finite_algebra_detail::FiniteAlgebraState<Field>;
  std::shared_ptr<const State> state_;
  std::shared_ptr<const OriginPowerConormalData<Field>> conormal_;

  friend class ConormalDerivativeMap<Field>;
};

template <typename Field>
ConormalModule<Field> FiniteQuotient<Field>::conormal_module() const {
  return ConormalModule<Field>(*this);
}

template <typename Field>
[[nodiscard]] ConormalModule<Field> conormal_module(
    FiniteQuotient<Field> quotient) {
  return quotient.conormal_module();
}

template <typename Field>
class ConormalDerivativeMap {
 public:
  explicit ConormalDerivativeMap(ConormalModule<Field> domain)
      : state_(std::move(domain.state_)),
        conormal_(std::move(domain.conormal_)),
        derivative_(state_->derivative_data()) {}

  [[nodiscard]] ConormalModule<Field> domain() const {
    return ConormalModule<Field>(FiniteQuotient<Field>(state_));
  }

  // Matrix of d:P/J^2 -> (P/J)^n.  The domain() defining matrix supplies the
  // additional kernel condition selecting J/J^2.
  [[nodiscard]] const SparseMatrix<Field>& ambient_matrix() const noexcept {
    return derivative_->matrix();
  }
  [[nodiscard]] const SparseMatrix<Field>& kernel_defining_matrix() const {
    return state_->h1_data()->cycle_matrix();
  }

  [[nodiscard]] CotangentH1Module<Field> kernel() const;

 private:
  using State = finite_algebra_detail::FiniteAlgebraState<Field>;
  std::shared_ptr<const State> state_;
  std::shared_ptr<const OriginPowerConormalData<Field>> conormal_;
  std::shared_ptr<const OriginPowerDerivativeData<Field>> derivative_;

  friend class CotangentH1Module<Field>;
};

template <typename Field>
ConormalDerivativeMap<Field> ConormalModule<Field>::derivative_map() const {
  return ConormalDerivativeMap<Field>(*this);
}

template <typename Field>
[[nodiscard]] ConormalDerivativeMap<Field> derivative_map(
    ConormalModule<Field> module) {
  return module.derivative_map();
}

template <typename Field>
class CotangentH1Module {
 public:
  using PolynomialType = Polynomial<Field>;
  using SparseKernelVector =
      typename SparseMatrix<Field>::SparseKernelVector;

  explicit CotangentH1Module(ConormalDerivativeMap<Field> map)
      : state_(std::move(map.state_)),
        conormal_(std::move(map.conormal_)),
        presentation_(state_->h1_data()) {}

  [[nodiscard]] FiniteQuotient<Field> algebra() const {
    return FiniteQuotient<Field>(state_);
  }
  [[nodiscard]] ConormalModule<Field> conormal() const {
    return ConormalModule<Field>(algebra());
  }
  [[nodiscard]] std::size_t dimension() const noexcept {
    return presentation_->h1_dimension();
  }
  [[nodiscard]] std::size_t ambient_dimension() const noexcept {
    return presentation_->length_P_mod_J2();
  }
  [[nodiscard]] const SparseMatrix<Field>& defining_matrix() const noexcept {
    return presentation_->cycle_matrix();
  }
  [[nodiscard]] std::vector<SparseKernelVector> basis_coordinates(
      const SparseEliminationLimits& limits) const {
    return presentation_->h1_kernel_coordinates(limits);
  }
  [[nodiscard]] std::vector<PolynomialType> representative_basis(
      const SparseEliminationLimits& limits) const {
    return presentation_->h1_basis(limits);
  }

  [[nodiscard]] CotangentH1Element<Field> class_of(
      const PolynomialType& representative) const;

 private:
  using State = finite_algebra_detail::FiniteAlgebraState<Field>;
  std::shared_ptr<const State> state_;
  std::shared_ptr<const OriginPowerConormalData<Field>> conormal_;
  std::shared_ptr<const CotangentH1Presentation<Field>> presentation_;

  friend class CotangentH1Element<Field>;
};

template <typename Field>
CotangentH1Module<Field> ConormalDerivativeMap<Field>::kernel() const {
  return CotangentH1Module<Field>(*this);
}

template <typename Field>
[[nodiscard]] CotangentH1Module<Field> kernel(
    ConormalDerivativeMap<Field> map) {
  return map.kernel();
}

template <typename Field>
class CotangentH1Element {
 public:
  using PolynomialType = Polynomial<Field>;
  using SparseEntry = typename IncrementalSparseRowSpace<Field>::Entry;

  [[nodiscard]] CotangentH1Module<Field> module() const {
    return ConormalDerivativeMap<Field>(
               ConormalModule<Field>(FiniteQuotient<Field>(state_)))
        .kernel();
  }
  [[nodiscard]] const PolynomialType& representative() const noexcept {
    return representative_;
  }
  [[nodiscard]] std::span<const SparseEntry> coordinates() const noexcept {
    return coordinates_;
  }

  [[nodiscard]] QuotientIdeal<Field> annihilator() const {
    CotangentClassData<Field> class_data{
        CotangentClassStatus::Complete,
        representative_,
        true,
        {},
        std::optional<std::vector<SparseEntry>>(coordinates_)};
    auto data = presentation_->annihilator_data(class_data);
    return QuotientIdeal<Field>(
        state_, std::move(data.kernel_coordinates),
        std::move(data.kernel_lifts));
  }

  friend bool operator==(
      const CotangentH1Element& left,
      const CotangentH1Element& right) {
    return left.state_.get() == right.state_.get() &&
           left.coordinates_ == right.coordinates_;
  }

 private:
  using State = finite_algebra_detail::FiniteAlgebraState<Field>;

  CotangentH1Element(
      std::shared_ptr<const State> state,
      std::shared_ptr<const CotangentH1Presentation<Field>> presentation,
      PolynomialType representative,
      std::vector<SparseEntry> coordinates)
      : state_(std::move(state)),
        presentation_(std::move(presentation)),
        representative_(std::move(representative)),
        coordinates_(std::move(coordinates)) {}

  std::shared_ptr<const State> state_;
  std::shared_ptr<const CotangentH1Presentation<Field>> presentation_;
  PolynomialType representative_;
  std::vector<SparseEntry> coordinates_;

  friend class CotangentH1Module<Field>;
};

template <typename Field>
CotangentH1Element<Field> CotangentH1Module<Field>::class_of(
    const PolynomialType& representative) const {
  auto data = presentation_->inspect_class(representative);
  if (!data.is_class()) {
    throw CotangentClassError(data.status);
  }
  return CotangentH1Element<Field>(
      state_, presentation_, std::move(data.representative),
      std::move(*data.coordinates));
}

template <typename Field>
[[nodiscard]] QuotientIdeal<Field> annihilator(
    const CotangentH1Element<Field>& element) {
  return element.annihilator();
}

}  // namespace laughableengine

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "laughableengine/field.hpp"
#include "laughableengine/finite_algebra.hpp"
#include "laughableengine/polynomial.hpp"

namespace {

using laughableengine::CotangentClassError;
using laughableengine::CotangentClassStatus;
using laughableengine::CotangentH1ResourceLimit;
using laughableengine::CotangentH1Spec;
using laughableengine::GF;
using laughableengine::QQ;
using laughableengine::CotangentH1Options;
using laughableengine::SparseEliminationLimits;
using laughableengine::SparseEliminationResourceLimit;
using laughableengine::annihilator;
using laughableengine::conormal_module;
using laughableengine::cotangent_h1;
using laughableengine::derivative_map;
using laughableengine::kernel;
using laughableengine::make_ring;
using laughableengine::origin_power_ideal;
using laughableengine::preimage;
using laughableengine::quotient;

[[noreturn]] void fail(const std::string& message, int line) {
  throw std::runtime_error(
      "line " + std::to_string(line) + ": " + message);
}

#define CHECK(expression)                                                     \
  do {                                                                        \
    if (!(expression)) {                                                      \
      fail("CHECK failed: " #expression, __LINE__);                          \
    }                                                                         \
  } while (false)

void explicit_object_chain_over_gf() {
  const auto P = make_ring(GF(5), {"x"});
  const auto x = P.gen(0);
  const auto J = origin_power_ideal(
      P, std::vector<typename decltype(P)::PolynomialType>{},
      std::size_t{5});
  const auto R = quotient(J);
  const auto conormal = conormal_module(R);
  const auto d = derivative_map(conormal);
  const auto H = kernel(d);

  CHECK(R.dimension() == 5);
  CHECK(R.square_quotient_dimension() == 10);
  CHECK(conormal.dimension() == 5);
  CHECK(H.dimension() == 5);
  CHECK(H.ambient_dimension() == 10);
  CHECK(conormal.defining_matrix().row_count() == 5);
  CHECK(d.ambient_matrix().row_count() == 5);
  CHECK(H.defining_matrix().row_count() == 10);

  // Constructing xi performs membership and cycle checks only.  Its
  // annihilator is a separate, explicit operation below.
  const auto xi = H.class_of(x.pow(5));
  const auto ann = annihilator(xi);
  const auto zero = R.zero_ideal();
  CHECK(ann == zero);
  CHECK(ann.is_zero());
  CHECK(ann.dimension() == 0);

  const auto colon = preimage(ann);
  CHECK(colon == J);
  CHECK(colon.equals_source_ideal());
  CHECK(colon.generators().size() == 1);
  CHECK(colon.generators().front() == x.pow(5));
}

void nonzero_annihilator_and_class_errors_over_qq() {
  const auto P = make_ring(QQ(), {"x"});
  const auto x = P.gen(0);
  const auto J = origin_power_ideal(P, {}, 2);
  const auto R = J.quotient();
  const auto H = R.conormal_module().derivative_map().kernel();

  bool outside_rejected = false;
  try {
    static_cast<void>(H.class_of(x));
  } catch (const CotangentClassError& error) {
    outside_rejected = error.status() == CotangentClassStatus::NotInIdeal;
  }
  CHECK(outside_rejected);

  bool noncycle_rejected = false;
  try {
    static_cast<void>(H.class_of(x.pow(2)));
  } catch (const CotangentClassError& error) {
    noncycle_rejected = error.status() == CotangentClassStatus::NotCycle;
  }
  CHECK(noncycle_rejected);

  const auto xi = H.class_of(x.pow(3));
  CHECK(xi == H.class_of(x.pow(3) + x.pow(4)));
  const auto ann = xi.annihilator();
  CHECK(!ann.is_zero());
  CHECK(ann.dimension() == 1);
  CHECK(ann.lift_basis().size() == 1);
  CHECK(ann.lift_basis().front() == x);
  CHECK(ann != R.zero_ideal());

  const auto colon = ann.preimage();
  CHECK(!(colon == J));
  CHECK(!colon.equals_source_ideal());
  CHECK(colon.generators().size() == 2);
  CHECK(colon.generators()[0] == x.pow(2));
  CHECK(colon.generators()[1] == x);
}

void exact_context_identity_and_materialized_bases() {
  const auto P = make_ring(QQ(), {"x", "y"});
  const auto x = P.gen(0);
  const auto y = P.gen(1);
  const auto J = origin_power_ideal(P, {x.pow(2) - y.pow(2)}, 3);
  const auto R1 = J.quotient();
  const auto R2 = J.quotient();

  CHECK(R1.dimension() == 5);
  CHECK(R1.basis_representatives().size() == R1.dimension());
  CHECK(R1.zero_ideal() != R2.zero_ideal());

  const auto H = R1.conormal_module().derivative_map().kernel();
  SparseEliminationLimits limits;
  limits.max_kernel_nonzeros = 10'000;
  const auto coordinates = H.basis_coordinates(limits);
  const auto representatives = H.representative_basis(limits);
  CHECK(coordinates.size() == H.dimension());
  CHECK(representatives.size() == H.dimension());
  for (const auto& representative : representatives) {
    CHECK(R1.remainder(representative).is_zero());
    CHECK(R1.remainder(representative.derivative(0)).is_zero());
    CHECK(R1.remainder(representative.derivative(1)).is_zero());
  }
}

void class_construction_does_not_compute_an_annihilator() {
  const auto P = make_ring(QQ(), {"x"});
  const auto x = P.gen(0);
  const auto J = origin_power_ideal(P, {}, 2);

  CotangentH1Options options;
  options.matrix_elimination_limits.max_kernel_coordinate_entries = 0;
  const auto H = J.quotient(options)
                     .conormal_module()
                     .derivative_map()
                     .kernel();

  // This succeeds under a zero kernel-output budget because class_of only
  // constructs the class. The separately requested annihilator has a
  // one-dimensional kernel and therefore reaches that budget.
  const auto xi = H.class_of(x.pow(3));
  bool annihilator_was_separate = false;
  try {
    static_cast<void>(xi.annihilator());
  } catch (const SparseEliminationResourceLimit&) {
    annihilator_was_separate = true;
  }
  CHECK(annihilator_was_separate);
}

void stages_obey_independent_resource_boundaries() {
  const auto P = make_ring(GF(5), {"x", "y"});
  const auto x = P.gen(0);
  const auto y = P.gen(1);
  const auto J = origin_power_ideal(P, {}, 2);

  CotangentH1Options derivative_limited;
  derivative_limited.max_matrix_triplets = 3;
  const auto R = J.quotient(derivative_limited);
  CHECK(R.dimension() == 3);
  CHECK(R.remainder(x * y).is_zero());

  // Stage two has exactly three reduction triplets and succeeds. Stage three
  // needs six derivative and nine stacked triplets, so it fails only when the
  // derivative map is explicitly requested.
  const auto C = R.conormal_module();
  CHECK(C.dimension() == 7);
  CHECK(C.defining_matrix().nnz() == 3);
  CHECK(R.square_quotient_dimension() == 10);
  bool derivative_failed = false;
  try {
    static_cast<void>(C.derivative_map());
  } catch (const CotangentH1ResourceLimit&) {
    derivative_failed = true;
  }
  CHECK(derivative_failed);

  CotangentH1Options h1_limited;
  h1_limited.max_matrix_triplets = 6;
  const auto derivative_ready_quotient = J.quotient(h1_limited);
  const auto derivative_ready_conormal =
      derivative_ready_quotient.conormal_module();
  const auto derivative_ready = derivative_ready_conormal.derivative_map();
  CHECK(derivative_ready.ambient_matrix().nnz() == 6);
  bool h1_failed = false;
  try {
    static_cast<void>(derivative_ready.kernel());
  } catch (const CotangentH1ResourceLimit&) {
    h1_failed = true;
  }
  CHECK(h1_failed);

  const auto unrestricted = J.quotient();
  const auto unrestricted_conormal = unrestricted.conormal_module();
  const auto unrestricted_derivative =
      unrestricted_conormal.derivative_map();
  const auto unrestricted_h1 = unrestricted_derivative.kernel();
  CHECK(unrestricted_derivative.ambient_matrix().nnz() == 6);
  CHECK(unrestricted_h1.defining_matrix().nnz() == 9);
  CHECK(unrestricted_h1.dimension() == 4);

  // Repeated traversals reuse the immutable stage objects rather than
  // rebuilding matrices behind equivalent-looking handles.
  CHECK(&unrestricted.conormal_module().defining_matrix() ==
        &unrestricted_conormal.defining_matrix());
  CHECK(&unrestricted_conormal.derivative_map().ambient_matrix() ==
        &unrestricted_derivative.ambient_matrix());
  CHECK(&unrestricted_derivative.kernel().defining_matrix() ==
        &unrestricted_h1.defining_matrix());

  CotangentH1Options square_space_limited;
  square_space_limited.monomial_space_limits.maximum_monomials = 3;
  const auto quotient_only = J.quotient(square_space_limited);
  CHECK(quotient_only.dimension() == 3);
  CHECK(quotient_only.remainder(x.pow(2) + y.pow(2)).is_zero());
  bool conormal_failed = false;
  try {
    static_cast<void>(quotient_only.conormal_module());
  } catch (const laughableengine::TruncatedMonomialSpaceResourceLimit&) {
    conormal_failed = true;
  }
  CHECK(conormal_failed);

  // The generated-row budget remains cumulative across stages. Stage one
  // offers two rows and stage two offers exactly six. A reset stage-two
  // counter would fit this limit, while the correct cumulative count does not.
  CotangentH1Options row_limited;
  row_limited.max_generated_rows = 6;
  const auto P1 = make_ring(GF(5), {"t"});
  const auto t = P1.gen(0);
  const auto with_lower_generator = origin_power_ideal(P1, {t}, 3);
  const auto row_bounded_quotient =
      with_lower_generator.quotient(row_limited);
  CHECK(row_bounded_quotient.dimension() == 1);
  bool cumulative_limit_failed = false;
  try {
    static_cast<void>(row_bounded_quotient.conormal_module());
  } catch (const CotangentH1ResourceLimit&) {
    cumulative_limit_failed = true;
  }
  CHECK(cumulative_limit_failed);
}

void staged_and_legacy_presentations_are_identical() {
  const auto P = make_ring(QQ(), {"x", "y"});
  const auto x = P.gen(0);
  const auto y = P.gen(1);
  const std::vector<typename decltype(P)::PolynomialType> generators{
      x.pow(2) - y.pow(2)};

  const auto legacy = cotangent_h1(CotangentH1Spec{
      P, generators, std::size_t{3}});
  const auto J = origin_power_ideal(P, generators, 3);
  const auto R = J.quotient();
  const auto C = R.conormal_module();
  const auto d = C.derivative_map();
  const auto H = d.kernel();

  CHECK(R.dimension() == legacy.length_Q());
  CHECK(R.square_quotient_dimension() == legacy.length_P_mod_J2());
  CHECK(C.dimension() == legacy.conormal_dimension());
  CHECK(H.dimension() == legacy.h1_dimension());
  CHECK(C.defining_matrix() == legacy.reduction_matrix());
  CHECK(d.ambient_matrix() == legacy.derivative_matrix());
  CHECK(H.defining_matrix() == legacy.cycle_matrix());

  const auto g = x.pow(3) * y.pow(3);
  const auto staged_class = H.class_of(g);
  const auto staged_annihilator = staged_class.annihilator();
  const auto legacy_proof = legacy.verify_class(g);
  CHECK(legacy_proof.status == CotangentClassStatus::Complete);
  CHECK(staged_annihilator.dimension() ==
        *legacy_proof.annihilator_dimension);
  CHECK(staged_annihilator.lift_basis() ==
        legacy_proof.annihilator_basis);
}

}  // namespace

int main() {
  explicit_object_chain_over_gf();
  nonzero_annihilator_and_class_errors_over_qq();
  exact_context_identity_and_materialized_bases();
  class_construction_does_not_compute_an_annihilator();
  stages_obey_independent_resource_boundaries();
  staged_and_legacy_presentations_are_identical();
}

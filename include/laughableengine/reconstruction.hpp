#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gmpxx.h>

#include "laughableengine/discovery.hpp"

namespace laughableengine {

inline std::int64_t centered_residue(
    PrimeField::Element coefficient,
    const PrimeField& field) {
  const auto residue = coefficient.value();
  const auto modulus = field.modulus();
  if (residue <= modulus / 2U) {
    return static_cast<std::int64_t>(residue);
  }
  return static_cast<std::int64_t>(residue) -
         static_cast<std::int64_t>(modulus);
}

inline Polynomial<PrimeField> reduce_mod_prime(
    const Polynomial<RationalField>& polynomial,
    const PolynomialRing<PrimeField>& target_ring) {
  if (polynomial.variable_count() != target_ring.variable_count() ||
      polynomial.order() != target_ring.order()) {
    throw std::invalid_argument(
        "modular reduction requires matching variables and monomial order");
  }
  const auto modulus = target_ring.field().modulus();
  std::vector<typename PolynomialRing<PrimeField>::TermSpec> terms;
  terms.reserve(polynomial.term_count());
  for (const auto& term : polynomial.terms()) {
    const mpz_class numerator(term.coefficient.numerator_string(), 10);
    const mpz_class denominator(term.coefficient.denominator_string(), 10);
    const auto numerator_residue = mpz_fdiv_ui(numerator.get_mpz_t(), modulus);
    const auto denominator_residue =
        mpz_fdiv_ui(denominator.get_mpz_t(), modulus);
    if (denominator_residue == 0) {
      throw std::domain_error(
          "rational coefficient denominator is not invertible modulo p");
    }
    const auto coefficient = target_ring.field().multiply(
        target_ring.field().from_unsigned(numerator_residue),
        target_ring.field().inverse(
            target_ring.field().from_unsigned(denominator_residue)));
    std::vector<std::uint16_t> exponents(target_ring.variable_count(), 0);
    for (std::size_t variable = 0; variable < target_ring.variable_count();
         ++variable) {
      exponents[variable] = term.monomial.exponent(variable);
    }
    terms.push_back(typename PolynomialRing<PrimeField>::TermSpec{
        coefficient, std::move(exponents)});
  }
  return target_ring.from_terms(std::move(terms));
}

struct SmallIntegerLiftResult {
  bool agreement = false;
  std::optional<Polynomial<RationalField>> polynomial;
  std::optional<std::string> detail;
};

// Safely reconstruct the common small-integer representative of the same
// supported polynomial seen at two primes. This deliberately does not attempt
// to CRT unrelated H1 basis coordinates. The returned QQ polynomial must still
// pass exact certification before it is called a hit.
inline SmallIntegerLiftResult lift_matching_small_integer_polynomial(
    const Polynomial<PrimeField>& first,
    const Polynomial<PrimeField>& second,
    const PolynomialRing<RationalField>& target_ring,
    std::uint64_t coefficient_bound) {
  SmallIntegerLiftResult result;
  if (first.variable_count() != second.variable_count() ||
      first.variable_count() != target_ring.variable_count() ||
      first.order() != second.order() || first.order() != target_ring.order()) {
    result.detail = "rings or monomial orders do not match";
    return result;
  }
  if (first.coefficient_field().modulus() ==
      second.coefficient_field().modulus()) {
    result.detail = "two-prime reconstruction requires distinct primes";
    return result;
  }
  if (first.term_count() != second.term_count()) {
    result.detail = "modular polynomial supports do not match";
    return result;
  }

  std::vector<typename PolynomialRing<RationalField>::TermSpec> terms;
  terms.reserve(first.term_count());
  for (std::size_t index = 0; index < first.term_count(); ++index) {
    const auto& left = first.terms()[index];
    const auto& right = second.terms()[index];
    if (!(left.monomial == right.monomial)) {
      result.detail = "modular polynomial supports do not match";
      return result;
    }
    const auto left_value = centered_residue(
        left.coefficient, first.coefficient_field());
    const auto right_value = centered_residue(
        right.coefficient, second.coefficient_field());
    if (left_value != right_value) {
      result.detail = "centered coefficients disagree across primes";
      return result;
    }
    const auto magnitude = left_value < 0
                               ? static_cast<std::uint64_t>(-left_value)
                               : static_cast<std::uint64_t>(left_value);
    if (magnitude > coefficient_bound) {
      result.detail = "coefficient exceeds the declared reconstruction bound";
      return result;
    }
    std::vector<std::uint16_t> exponents(target_ring.variable_count(), 0);
    for (std::size_t variable = 0; variable < target_ring.variable_count();
         ++variable) {
      exponents[variable] = left.monomial.exponent(variable);
    }
    terms.push_back(typename PolynomialRing<RationalField>::TermSpec{
        target_ring.field().from_integer(static_cast<long>(left_value)),
        std::move(exponents)});
  }
  result.agreement = true;
  result.polynomial = target_ring.from_terms(std::move(terms));
  return result;
}

struct TwoPrimeScreenSignature {
  DiscoveryScreenStatus status = DiscoveryScreenStatus::InvalidInput;
  std::uint32_t modulus = 0;
  std::size_t length_Q = 0;
  std::size_t length_P_mod_J2 = 0;
  std::size_t conormal_dimension = 0;
  std::size_t h1_dimension = 0;
  std::size_t socle_dimension = 0;
  std::size_t maximum_individual_rank_lower_bound = 0;
  std::size_t maximum_individual_rank_upper_bound = 0;
  MatrixSpaceRankProof rank_proof = MatrixSpaceRankProof::ResourceLimit;
  std::uint64_t leading_ideal_fingerprint = 0;
  std::string leading_ideal_signature;

  [[nodiscard]] bool proven() const noexcept {
    return status == DiscoveryScreenStatus::Complete &&
           maximum_individual_rank_lower_bound ==
               maximum_individual_rank_upper_bound &&
           (rank_proof == MatrixSpaceRankProof::ProvenMaximum ||
            rank_proof == MatrixSpaceRankProof::ProvenFullColumnRank);
  }
};

inline TwoPrimeScreenSignature screen_signature(
    const PackedH1ScreenResult& result) {
  return TwoPrimeScreenSignature{
      result.status,
      result.modulus,
      result.length_Q,
      result.length_P_mod_J2,
      result.conormal_dimension,
      result.h1_dimension,
      result.socle_dimension,
      result.maximum_individual_rank_lower_bound,
      result.maximum_individual_rank_upper_bound,
      result.rank_proof,
      result.leading_ideal_fingerprint,
      result.leading_ideal_signature};
}

struct TwoPrimeScreenAgreement {
  bool agreement = false;
  std::optional<std::string> detail;
};

// Agreement is intentionally stronger than structural equality: both screens
// must have proved an exact base-field maximum and must share the same initial
// monomial ideal. Resource-limited/generic-only records can never agree.
inline TwoPrimeScreenAgreement compare_two_prime_screen_signatures(
    const TwoPrimeScreenSignature& first,
    const TwoPrimeScreenSignature& second) {
  TwoPrimeScreenAgreement result;
  if (!first.proven() || !second.proven()) {
    result.detail =
        "two-prime agreement requires two complete, proven matrix-space screens";
    return result;
  }
  if (first.modulus == second.modulus) {
    result.detail = "two-prime agreement requires two distinct primes";
    return result;
  }
  if (first.leading_ideal_fingerprint == 0 ||
      second.leading_ideal_fingerprint == 0 ||
      first.leading_ideal_signature.empty() ||
      second.leading_ideal_signature.empty()) {
    result.detail = "two-prime agreement requires initial-ideal fingerprints";
    return result;
  }
  if (first.length_Q != second.length_Q ||
      first.length_P_mod_J2 != second.length_P_mod_J2 ||
      first.conormal_dimension != second.conormal_dimension ||
      first.h1_dimension != second.h1_dimension ||
      first.socle_dimension != second.socle_dimension ||
      first.maximum_individual_rank_lower_bound !=
          second.maximum_individual_rank_lower_bound ||
      first.leading_ideal_fingerprint != second.leading_ideal_fingerprint ||
      first.leading_ideal_signature != second.leading_ideal_signature) {
    result.detail =
        "proven modular screens disagree in dimensions, rank, or initial ideal";
    return result;
  }
  result.agreement = true;
  return result;
}

}  // namespace laughableengine

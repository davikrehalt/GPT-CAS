#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/candidate_executor.hpp"
#include "laughableengine/inverse_system.hpp"

namespace laughableengine {

struct InverseSystemDiscoveryOptions {
  ApolarityConvention convention =
      ApolarityConvention::OrdinaryDifferentiation;
  InverseSystemLimits inverse_system_limits{};
  PackedH1ScreenOptions h1_options{};
  // Compact search records retain dual generators and annihilator bases only
  // for hits unless explicitly requested for every candidate.
  bool retain_all_candidates = false;
};

struct InverseSystemDiscoveryRecord {
  DiscoveryScreenStatus status = DiscoveryScreenStatus::InvalidInput;
  std::size_t candidate_index = 0;
  std::uint32_t maximum_dual_degree = 0;
  std::size_t action_rank = 0;
  std::size_t kernel_dimension = 0;
  std::size_t quotient_length = 0;
  std::size_t h1_dimension = 0;
  std::size_t socle_dimension = 0;
  std::uint32_t modulus = 0;
  std::uint64_t leading_ideal_fingerprint = 0;
  std::string leading_ideal_signature;
  std::size_t maximum_individual_rank_lower_bound = 0;
  std::size_t maximum_individual_rank_upper_bound = 0;
  MatrixSpaceRankProof rank_proof = MatrixSpaceRankProof::ResourceLimit;
  // Compatibility alias for the best rank actually witnessed.
  std::size_t maximum_individual_rank = 0;
  bool full_rank_candidate = false;
  bool certified_faithful = false;
  std::vector<Polynomial<PrimeField>> retained_dual_generators;
  std::vector<Polynomial<PrimeField>> retained_annihilator_basis;
  std::optional<Polynomial<PrimeField>> witness;
  std::optional<std::string> detail;
};

inline InverseSystemDiscoveryRecord screen_inverse_system(
    const PolynomialRing<PrimeField>& operator_ring,
    std::span<const Polynomial<PrimeField>> dual_generators,
    std::size_t candidate_index,
    const InverseSystemDiscoveryOptions& options = {}) {
  InverseSystemDiscoveryRecord record;
  record.candidate_index = candidate_index;
  try {
    auto generated = macaulay_annihilator(
        operator_ring, dual_generators, options.convention,
        options.inverse_system_limits);
    record.maximum_dual_degree = generated.maximum_degree;
    record.action_rank = generated.action_rank;
    record.kernel_dimension = generated.kernel_dimension;
    record.quotient_length = generated.quotient_length;
    auto h1 = screen_full_h1(generated.annihilator, options.h1_options);
    record.status = h1.status;
    record.h1_dimension = h1.h1_dimension;
    record.socle_dimension = h1.socle_dimension;
    record.modulus = h1.modulus;
    record.leading_ideal_fingerprint = h1.leading_ideal_fingerprint;
    record.leading_ideal_signature = std::move(h1.leading_ideal_signature);
    record.maximum_individual_rank_lower_bound =
        h1.maximum_individual_rank_lower_bound;
    record.maximum_individual_rank_upper_bound =
        h1.maximum_individual_rank_upper_bound;
    record.rank_proof = h1.rank_proof;
    record.maximum_individual_rank =
        h1.maximum_individual_rank_lower_bound;
    record.full_rank_candidate = h1.full_socle_rank_candidate;
    record.certified_faithful = h1.certified_faithful;
    record.witness = std::move(h1.witness);
    record.detail = std::move(h1.detail);
    if (options.retain_all_candidates || record.full_rank_candidate) {
      record.retained_dual_generators = std::move(generated.dual_generators);
      record.retained_annihilator_basis =
          generated.annihilator.groebner_basis();
    }
    return record;
  } catch (const InverseSystemResourceLimit& error) {
    record.status = DiscoveryScreenStatus::ResourceLimit;
    record.detail = error.what();
    return record;
  } catch (const PackedDiscoveryResourceLimit& error) {
    record.status = DiscoveryScreenStatus::ResourceLimit;
    record.detail = error.what();
    return record;
  } catch (const ResourceLimitExceeded& error) {
    record.status = DiscoveryScreenStatus::ResourceLimit;
    record.detail = error.what();
    return record;
  } catch (const GroebnerResourceLimit& error) {
    record.status = DiscoveryScreenStatus::ResourceLimit;
    record.detail = error.what();
    return record;
  } catch (const std::overflow_error& error) {
    record.status = DiscoveryScreenStatus::ResourceLimit;
    record.detail = error.what();
    return record;
  } catch (const InverseSystemInputError& error) {
    record.status = DiscoveryScreenStatus::InvalidInput;
    record.detail = error.what();
    return record;
  } catch (const std::domain_error& error) {
    record.status = DiscoveryScreenStatus::InvalidInput;
    record.detail = error.what();
    return record;
  } catch (const std::invalid_argument& error) {
    record.status = DiscoveryScreenStatus::InvalidInput;
    record.detail = error.what();
    return record;
  }
}

inline std::vector<InverseSystemDiscoveryRecord>
search_inverse_systems_parallel(
    PolynomialRing<PrimeField> operator_ring,
    std::span<const std::vector<Polynomial<PrimeField>>> candidates,
    const InverseSystemDiscoveryOptions& discovery_options = {},
    const CandidateExecutorOptions& executor_options = {}) {
  return execute_candidates<
      std::vector<Polynomial<PrimeField>>,
      InverseSystemDiscoveryRecord>(
      candidates,
      [operator_ring](std::size_t) { return operator_ring; },
      [discovery_options](
          const PolynomialRing<PrimeField>& worker_ring,
          const std::vector<Polynomial<PrimeField>>& candidate,
          std::size_t index) {
        return screen_inverse_system(
            worker_ring,
            std::span<const Polynomial<PrimeField>>(
                candidate.data(), candidate.size()),
            index, discovery_options);
      },
      executor_options);
}

inline std::vector<InverseSystemDiscoveryRecord>
search_inverse_systems_parallel(
    PolynomialRing<PrimeField> operator_ring,
    const std::vector<std::vector<Polynomial<PrimeField>>>& candidates,
    const InverseSystemDiscoveryOptions& discovery_options = {},
    const CandidateExecutorOptions& executor_options = {}) {
  return search_inverse_systems_parallel(
      std::move(operator_ring),
      std::span<const std::vector<Polynomial<PrimeField>>>(
          candidates.data(), candidates.size()),
      discovery_options, executor_options);
}

}  // namespace laughableengine

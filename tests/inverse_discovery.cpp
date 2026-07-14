#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "laughableengine/inverse_discovery.hpp"

namespace {

using laughableengine::CandidateExecutorOptions;
using laughableengine::DiscoveryScreenStatus;
using laughableengine::GF;
using laughableengine::ApolarityConvention;
using laughableengine::InverseSystemDiscoveryOptions;
using laughableengine::MatrixSpaceRankProof;
using laughableengine::Order;
using laughableengine::Polynomial;
using laughableengine::PolynomialRing;
using laughableengine::PrimeField;
using laughableengine::search_inverse_systems_parallel;
using laughableengine::screen_inverse_system;

[[noreturn]] void fail(const std::string& message, int line) {
  throw std::runtime_error("line " + std::to_string(line) + ": " + message);
}

#define CHECK(expression)                                                     \
  do {                                                                        \
    if (!(expression)) {                                                      \
      fail("CHECK failed: " #expression, __LINE__);                          \
    }                                                                         \
  } while (false)

void test_parallel_inverse_system_search() {
  const PolynomialRing<PrimeField> ring(GF(5), {"x"}, Order::Grevlex);
  const auto x = ring.gen(0);
  std::vector<std::vector<Polynomial<PrimeField>>> candidates;
  for (std::size_t index = 0; index < 24; ++index) {
    candidates.push_back({index % 2 == 0 ? x.pow(4) : x.pow(3)});
  }
  CandidateExecutorOptions one;
  one.worker_count = 1;
  CandidateExecutorOptions many;
  many.worker_count = 4;
  const auto sequential = search_inverse_systems_parallel(
      ring, candidates, {}, one);
  const auto parallel = search_inverse_systems_parallel(
      ring, candidates, {}, many);
  CHECK(sequential.size() == candidates.size());
  CHECK(parallel.size() == candidates.size());
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    CHECK(sequential[index].candidate_index == index);
    CHECK(parallel[index].candidate_index == index);
    CHECK(sequential[index].status == DiscoveryScreenStatus::Complete);
    CHECK(parallel[index].status == sequential[index].status);
    CHECK(parallel[index].quotient_length ==
          sequential[index].quotient_length);
    CHECK(parallel[index].h1_dimension == sequential[index].h1_dimension);
    CHECK(parallel[index].socle_dimension ==
          sequential[index].socle_dimension);
    CHECK(parallel[index].maximum_individual_rank ==
          sequential[index].maximum_individual_rank);
    CHECK(parallel[index].certified_faithful ==
          sequential[index].certified_faithful);
    if (index % 2 == 0) {
      CHECK(sequential[index].certified_faithful);
      CHECK(!sequential[index].retained_dual_generators.empty());
      CHECK(!sequential[index].retained_annihilator_basis.empty());
    } else {
      CHECK(!sequential[index].certified_faithful);
      CHECK(sequential[index].retained_dual_generators.empty());
      CHECK(sequential[index].retained_annihilator_basis.empty());
    }
  }
}

void test_retention_and_resource_limit() {
  const PolynomialRing<PrimeField> ring(GF(7), {"x"}, Order::Grevlex);
  const auto x = ring.gen(0);
  const std::vector<std::vector<Polynomial<PrimeField>>> candidates{{x.pow(3)}};
  InverseSystemDiscoveryOptions retain;
  retain.retain_all_candidates = true;
  const auto retained = search_inverse_systems_parallel(
      ring, candidates, retain);
  CHECK(retained.size() == 1);
  CHECK(!retained.front().retained_dual_generators.empty());
  CHECK(!retained.front().retained_annihilator_basis.empty());

  InverseSystemDiscoveryOptions limited;
  limited.inverse_system_limits.max_basis_monomials = 0;
  const auto resource = search_inverse_systems_parallel(
      ring, candidates, limited);
  CHECK(resource.size() == 1);
  CHECK(resource.front().status == DiscoveryScreenStatus::ResourceLimit);
  CHECK(resource.front().detail.has_value());
}

void test_matrix_space_limit_is_inconclusive() {
  const PolynomialRing<PrimeField> operators(
      GF(2), {"x", "y"}, Order::Grevlex);
  const PolynomialRing<PrimeField> dual(
      GF(2), {"X", "Y"}, Order::Grevlex);
  const auto X = dual.gen(0);
  const auto Y = dual.gen(1);
  const std::vector<Polynomial<PrimeField>> generators{
      Y.pow(3) + X.pow(2) + Y.pow(2) + dual.one(), X.pow(3)};
  InverseSystemDiscoveryOptions options;
  options.convention = ApolarityConvention::DividedPowers;
  options.h1_options.certify_hits = false;
  options.h1_options.matrix_space_limits.max_minors = 0;
  const auto record = screen_inverse_system(
      operators, generators, 0, options);
  CHECK(record.status == DiscoveryScreenStatus::ResourceLimit);
  CHECK(record.maximum_individual_rank_lower_bound == 1);
  CHECK(record.maximum_individual_rank_upper_bound == 2);
  CHECK(record.maximum_individual_rank == 1);
  CHECK(record.rank_proof == MatrixSpaceRankProof::ResourceLimit);
  CHECK(record.detail.has_value());
}

}  // namespace

int main() {
  try {
    test_parallel_inverse_system_search();
    test_retention_and_resource_limit();
    test_matrix_space_limit_is_inconclusive();
    std::cout << "inverse discovery tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "inverse discovery test failure: " << error.what() << '\n';
    return 1;
  }
}

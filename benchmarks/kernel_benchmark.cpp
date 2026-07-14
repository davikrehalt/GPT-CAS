#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "laughableengine/laughableengine.hpp"

namespace {

using Clock = std::chrono::steady_clock;

using laughableengine::CycleAuditResult;
using laughableengine::CycleAuditStatus;
using laughableengine::CandidateExecutorOptions;
using laughableengine::DiscoveryScreenStatus;
using laughableengine::GF;
using laughableengine::H1ActionData;
using laughableengine::Ideal;
using laughableengine::InverseSystemDiscoveryOptions;
using laughableengine::InverseSystemDiscoveryRecord;
using laughableengine::Order;
using laughableengine::PackedCycleScreenOptions;
using laughableengine::PackedCycleScreenResult;
using laughableengine::PackedH1ScreenResult;
using laughableengine::Polynomial;
using laughableengine::PolynomialRing;
using laughableengine::PrimeField;
using laughableengine::QQ;
using laughableengine::RationalField;
using laughableengine::audit_cycle;
using laughableengine::full_h1_action;
using laughableengine::make_ring;
using laughableengine::screen_cycle;
using laughableengine::screen_cycles_parallel;
using laughableengine::screen_full_h1;
using laughableengine::search_inverse_systems_parallel;

struct Configuration {
  std::size_t warmup_samples = 1;
  std::size_t measured_samples = 5;
  std::size_t iterations = 1;
  std::size_t nf_batches_per_sample = 50;
  bool help = false;
};

struct Statistics {
  std::string name;
  std::string operation;
  std::size_t measured_samples;
  std::size_t logical_operations_per_sample;
  double median_milliseconds;
  double percentile95_milliseconds;
  double minimum_milliseconds;
  double median_operations_per_hour;
  std::optional<std::size_t> normalization_worker_count;
  std::optional<double> median_candidates_per_hour_per_core;
  std::optional<double> target_candidates_per_hour_per_core;
};

struct SetupRow {
  std::string name;
  double ideal_construction_milliseconds;
  std::size_t reduced_basis_size;
};

template <typename Value>
struct TimedSetup {
  Value value;
  double ideal_construction_milliseconds;
};

struct NormalFormFixture {
  PolynomialRing<PrimeField> ring;
  Ideal<PrimeField> ideal;
  std::vector<Polynomial<PrimeField>> batch;
};

template <typename Field>
struct CycleFixture {
  PolynomialRing<Field> ring;
  Polynomial<Field> distinguished_polynomial;
  Ideal<Field> ideal;
};

[[nodiscard]] std::uint64_t mix_checksum(
    std::uint64_t seed,
    std::uint64_t value) noexcept {
  constexpr std::uint64_t golden_ratio = 0x9e3779b97f4a7c15ULL;
  return seed ^ (value + golden_ratio + (seed << 6U) + (seed >> 2U));
}

[[nodiscard]] std::uint64_t checksum_normal_forms(
    const std::vector<Polynomial<PrimeField>>& values) {
  std::uint64_t checksum = values.size();
  for (const auto& polynomial : values) {
    checksum = mix_checksum(checksum, polynomial.term_count());
    checksum = mix_checksum(
        checksum, polynomial.total_degree().value_or(0));
    if (const auto* leading = polynomial.leading_term(); leading != nullptr) {
      checksum = mix_checksum(checksum, leading->coefficient.value());
      checksum = mix_checksum(
          checksum, leading->monomial.total_degree());
    }
  }
  return checksum;
}

template <typename Field>
[[nodiscard]] std::uint64_t checksum_audit(
    const CycleAuditResult<Field>& result) {
  std::uint64_t checksum =
      static_cast<std::uint64_t>(result.status());
  checksum = mix_checksum(checksum, result.finite_quotient());
  checksum = mix_checksum(checksum, result.supported_at_origin());
  checksum = mix_checksum(
      checksum, result.quotient_length().value_or(0));
  checksum = mix_checksum(
      checksum, result.cycle_valid().value_or(false));
  checksum = mix_checksum(checksum, result.faithful_cycle());
  if (result.colon_evidence().has_value()) {
    const auto& evidence = *result.colon_evidence();
    checksum = mix_checksum(checksum, evidence.annihilator_dimension());
    checksum = mix_checksum(checksum, evidence.colon_quotient_length());
    checksum = mix_checksum(
        checksum, evidence.colon_ideal().groebner_basis().size());
  }
  return checksum;
}

template <typename Field>
[[nodiscard]] std::uint64_t checksum_h1(const H1ActionData<Field>& result) {
  std::uint64_t checksum = result.length_Q;
  checksum = mix_checksum(checksum, result.length_P_mod_J2);
  checksum = mix_checksum(checksum, result.conormal_dimension);
  checksum = mix_checksum(checksum, result.h1_dimension);
  checksum = mix_checksum(checksum, result.socle_dimension);
  checksum = mix_checksum(checksum, result.action_matrices.size());
  checksum = mix_checksum(
      checksum, result.common_product_space_dimension);
  checksum = mix_checksum(
      checksum, result.individual_rank.lower_bound);
  checksum = mix_checksum(
      checksum, result.individual_rank.upper_bound);
  checksum = mix_checksum(
      checksum, result.best_h1_polynomial.has_value());
  checksum = mix_checksum(
      checksum, result.faithful_witness_audit.has_value());
  checksum = mix_checksum(
      checksum, result.common_annihilator_diagnostic.groebner_basis().size());
  return checksum;
}

[[nodiscard]] std::uint64_t checksum_prime_polynomial(
    const Polynomial<PrimeField>& polynomial) {
  std::uint64_t checksum = polynomial.term_count();
  checksum = mix_checksum(
      checksum, polynomial.total_degree().value_or(0));
  for (const auto& term : polynomial.terms()) {
    checksum = mix_checksum(checksum, term.coefficient.value());
    checksum = mix_checksum(checksum, term.monomial.total_degree());
    for (const auto exponent : term.monomial.exponents()) {
      checksum = mix_checksum(checksum, exponent);
    }
  }
  return checksum;
}

[[nodiscard]] std::uint64_t checksum_packed_cycle(
    const PackedCycleScreenResult& result) {
  std::uint64_t checksum = static_cast<std::uint64_t>(result.status);
  checksum = mix_checksum(checksum, result.length_Q);
  checksum = mix_checksum(checksum, result.length_P_mod_J2);
  checksum = mix_checksum(checksum, result.g_in_J);
  checksum = mix_checksum(checksum, result.derivatives_in_J);
  checksum = mix_checksum(checksum, result.cycle_valid);
  checksum = mix_checksum(checksum, result.multiplication_rank);
  checksum = mix_checksum(checksum, result.full_column_rank_candidate);
  checksum = mix_checksum(checksum, result.certified_faithful);
  if (result.certification.has_value()) {
    checksum = mix_checksum(
        checksum, checksum_audit(*result.certification));
  }
  return checksum;
}

[[nodiscard]] std::uint64_t checksum_packed_cycles(
    const std::vector<PackedCycleScreenResult>& results) {
  std::uint64_t checksum = results.size();
  for (const auto& result : results) {
    checksum = mix_checksum(checksum, checksum_packed_cycle(result));
  }
  return checksum;
}

[[nodiscard]] std::uint64_t checksum_packed_h1(
    const PackedH1ScreenResult& result) {
  std::uint64_t checksum = static_cast<std::uint64_t>(result.status);
  checksum = mix_checksum(checksum, result.length_Q);
  checksum = mix_checksum(checksum, result.length_P_mod_J2);
  checksum = mix_checksum(checksum, result.conormal_dimension);
  checksum = mix_checksum(checksum, result.h1_dimension);
  checksum = mix_checksum(checksum, result.socle_dimension);
  checksum = mix_checksum(
      checksum, result.maximum_individual_rank_lower_bound);
  checksum = mix_checksum(
      checksum, result.maximum_individual_rank_upper_bound);
  checksum = mix_checksum(
      checksum, static_cast<std::uint64_t>(result.rank_proof));
  checksum = mix_checksum(checksum, result.full_socle_rank_candidate);
  checksum = mix_checksum(checksum, result.certified_faithful);
  if (result.witness_coefficients.has_value()) {
    for (const auto coefficient : *result.witness_coefficients) {
      checksum = mix_checksum(checksum, coefficient);
    }
  }
  if (result.witness.has_value()) {
    checksum = mix_checksum(
        checksum, checksum_prime_polynomial(*result.witness));
  }
  if (result.certification.has_value()) {
    checksum = mix_checksum(
        checksum, checksum_audit(*result.certification));
  }
  return checksum;
}

[[nodiscard]] std::uint64_t checksum_inverse_records(
    const std::vector<InverseSystemDiscoveryRecord>& records) {
  std::uint64_t checksum = records.size();
  for (const auto& record : records) {
    checksum = mix_checksum(
        checksum, static_cast<std::uint64_t>(record.status));
    checksum = mix_checksum(checksum, record.candidate_index);
    checksum = mix_checksum(checksum, record.maximum_dual_degree);
    checksum = mix_checksum(checksum, record.action_rank);
    checksum = mix_checksum(checksum, record.kernel_dimension);
    checksum = mix_checksum(checksum, record.quotient_length);
    checksum = mix_checksum(checksum, record.h1_dimension);
    checksum = mix_checksum(checksum, record.socle_dimension);
    checksum = mix_checksum(checksum, record.maximum_individual_rank);
    checksum = mix_checksum(checksum, record.full_rank_candidate);
    checksum = mix_checksum(checksum, record.certified_faithful);
    if (record.witness.has_value()) {
      checksum = mix_checksum(
          checksum, checksum_prime_polynomial(*record.witness));
    }
    checksum = mix_checksum(
        checksum, record.retained_annihilator_basis.size());
  }
  return checksum;
}

[[nodiscard]] double elapsed_milliseconds(
    Clock::time_point start,
    Clock::time_point finish) {
  return std::chrono::duration<double, std::milli>(finish - start).count();
}

[[nodiscard]] std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    std::string_view description) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::invalid_argument(std::string(description) + " is too large");
  }
  return left * right;
}

[[nodiscard]] double percentile(
    const std::vector<double>& sorted_values,
    double probability) {
  if (sorted_values.empty() || probability < 0.0 || probability > 1.0) {
    throw std::invalid_argument("invalid percentile request");
  }
  if (sorted_values.size() == 1) {
    return sorted_values.front();
  }
  const auto position =
      probability * static_cast<double>(sorted_values.size() - 1);
  const auto lower = static_cast<std::size_t>(position);
  const auto upper = std::min(lower + 1, sorted_values.size() - 1);
  const auto fraction = position - static_cast<double>(lower);
  return sorted_values[lower] * (1.0 - fraction) +
         sorted_values[upper] * fraction;
}

template <typename Operation, typename Checksum>
[[nodiscard]] Statistics measure(
    std::string name,
    std::string operation_description,
    const Configuration& configuration,
    std::size_t invocations_per_sample,
    std::size_t logical_operations_per_invocation,
    Operation&& operation,
    Checksum&& checksum_of,
    std::uint64_t& consumed_checksum,
    std::optional<std::size_t> normalization_worker_count = std::nullopt,
    std::optional<double> target_candidates_per_hour_per_core =
        std::nullopt) {
  if (configuration.measured_samples == 0 || invocations_per_sample == 0 ||
      logical_operations_per_invocation == 0) {
    throw std::invalid_argument(
        "benchmark samples and operation counts must be positive");
  }
  const auto logical_operations_per_sample = checked_product(
      invocations_per_sample, logical_operations_per_invocation,
      "logical operation count");

  auto run_sample = [&]() {
    std::uint64_t sample_checksum = 0;
    const auto start = Clock::now();
    for (std::size_t iteration = 0; iteration < invocations_per_sample;
         ++iteration) {
      auto result = operation();
      sample_checksum =
          mix_checksum(sample_checksum, checksum_of(result));
    }
    const auto finish = Clock::now();
    consumed_checksum = mix_checksum(consumed_checksum, sample_checksum);
    return elapsed_milliseconds(start, finish) /
           static_cast<double>(logical_operations_per_sample);
  };

  for (std::size_t sample = 0; sample < configuration.warmup_samples;
       ++sample) {
    static_cast<void>(run_sample());
  }

  std::vector<double> milliseconds_per_operation;
  milliseconds_per_operation.reserve(configuration.measured_samples);
  for (std::size_t sample = 0; sample < configuration.measured_samples;
       ++sample) {
    milliseconds_per_operation.push_back(run_sample());
  }
  std::sort(
      milliseconds_per_operation.begin(),
      milliseconds_per_operation.end());

  const double median = percentile(milliseconds_per_operation, 0.50);
  const double percentile95 = percentile(milliseconds_per_operation, 0.95);
  const double operations_per_hour =
      median == 0.0
          ? std::numeric_limits<double>::infinity()
          : 3'600'000.0 / median;
  std::optional<double> candidates_per_hour_per_core;
  if (normalization_worker_count.has_value()) {
    if (*normalization_worker_count == 0) {
      throw std::invalid_argument(
          "candidate throughput worker count must be positive");
    }
    candidates_per_hour_per_core =
        operations_per_hour /
        static_cast<double>(*normalization_worker_count);
  }

  return Statistics{
      std::move(name),
      std::move(operation_description),
      configuration.measured_samples,
      logical_operations_per_sample,
      median,
      percentile95,
      milliseconds_per_operation.front(),
      operations_per_hour,
      normalization_worker_count,
      candidates_per_hour_per_core,
      target_candidates_per_hour_per_core};
}

[[nodiscard]] std::size_t parse_size(
    std::string_view text,
    std::string_view option,
    bool allow_zero = false) {
  std::size_t value = 0;
  const auto* first = text.data();
  const auto* last = first + text.size();
  const auto [end, error] = std::from_chars(first, last, value);
  if (error != std::errc() || end != last || (!allow_zero && value == 0)) {
    throw std::invalid_argument(
        std::string(option) + " requires " +
        (allow_zero ? "a nonnegative" : "a positive") + " integer");
  }
  return value;
}

[[nodiscard]] Configuration parse_configuration(int argc, char** argv) {
  Configuration configuration;
  for (int index = 1; index < argc; ++index) {
    if (std::string_view(argv[index]) == "--quick") {
      configuration.measured_samples = 3;
      configuration.nf_batches_per_sample = 10;
    }
  }

  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--quick") {
      continue;
    }
    if (argument == "--help" || argument == "-h") {
      configuration.help = true;
      continue;
    }
    auto following_value = [&](std::string_view option) -> std::string_view {
      if (index + 1 >= argc) {
        throw std::invalid_argument(
            std::string(option) + " requires a value");
      }
      return argv[++index];
    };
    if (argument == "--warmups") {
      configuration.warmup_samples =
          parse_size(following_value(argument), argument, true);
    } else if (argument == "--samples") {
      configuration.measured_samples =
          parse_size(following_value(argument), argument);
    } else if (argument == "--iterations") {
      configuration.iterations =
          parse_size(following_value(argument), argument);
    } else if (argument == "--nf-batches") {
      configuration.nf_batches_per_sample =
          parse_size(following_value(argument), argument);
    } else {
      throw std::invalid_argument(
          "unknown benchmark option: " + std::string(argument));
    }
  }
  return configuration;
}

void print_help(std::ostream& output) {
  output
      << "Usage: kernel_benchmark [options]\n"
      << "  --quick          use three samples and ten NF batches/sample\n"
      << "  --warmups N      warmup samples per benchmark (default: 1)\n"
      << "  --samples N      measured samples per benchmark (default: 5)\n"
      << "  --iterations N   API invocations per sample multiplier (default: 1)\n"
      << "  --nf-batches N   compiled NF batches per sample (default: 50)\n"
      << "  --help, -h       show this help\n";
}

[[nodiscard]] TimedSetup<NormalFormFixture> make_normal_form_fixture() {
  auto ring = make_ring(GF(101), {"x", "y", "z"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto z = ring.gen("z");
  std::vector<Polynomial<PrimeField>> generators{
      x.pow(4) - y.pow(2) - z,
      y.pow(3) - z.pow(2),
      z.pow(3)};

  std::vector<Polynomial<PrimeField>> batch;
  batch.reserve(64);
  for (std::size_t index = 0; index < 64; ++index) {
    const auto coefficient = [&](std::size_t salt) {
      return ring.integer(static_cast<long>(
          1 + ((index + 1) * (salt * 17 + 3)) % 100));
    };
    batch.push_back(
        coefficient(1) * x.pow(6 + index % 7) *
            y.pow(3 + (index * 3) % 7) * z.pow(2 + (index * 5) % 6) +
        coefficient(2) * x.pow(2 + (index * 5) % 9) *
            y.pow(7 + index % 5) * z.pow(1 + (index * 2) % 5) +
        coefficient(3) * x.pow(9 + index % 4) *
            y.pow(1 + (index * 7) % 6) * z.pow(5 + index % 4) +
        coefficient(4) * x.pow(index % 8) *
            y.pow(8 + (index * 2) % 5) * z.pow(6 + index % 3) +
        coefficient(5) * x.pow(5 + (index * 3) % 6) *
            y.pow(4 + index % 8) * z.pow(index % 7) +
        coefficient(6) * x.pow(3 + index % 10) *
            y.pow(2 + (index * 4) % 7) * z.pow(7 + index % 5));
  }

  const auto start = Clock::now();
  Ideal<PrimeField> ideal(ring, std::move(generators));
  const auto finish = Clock::now();
  return TimedSetup<NormalFormFixture>{
      NormalFormFixture{
          std::move(ring), std::move(ideal), std::move(batch)},
      elapsed_milliseconds(start, finish)};
}

[[nodiscard]] TimedSetup<CycleFixture<RationalField>> make_qq_seed_fixture() {
  auto ring = make_ring(QQ(), {"x", "y", "z"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto z = ring.gen("z");
  auto polynomial = y.pow(5) + z.pow(5) + y.pow(2) * z.pow(2);
  std::vector<Polynomial<RationalField>> generators{
      x.pow(2), x * y, x * z, polynomial,
      polynomial.derivative("y"), polynomial.derivative("z")};

  const auto start = Clock::now();
  Ideal<RationalField> ideal(ring, std::move(generators));
  const auto finish = Clock::now();
  return TimedSetup<CycleFixture<RationalField>>{
      CycleFixture<RationalField>{
          std::move(ring), std::move(polynomial), std::move(ideal)},
      elapsed_milliseconds(start, finish)};
}

[[nodiscard]] TimedSetup<CycleFixture<RationalField>>
make_tjurina_fixture() {
  auto ring = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  auto polynomial = x.pow(3) + y.pow(7) + x * y.pow(5);
  std::vector<Polynomial<RationalField>> generators{
      polynomial, polynomial.derivative("x"), polynomial.derivative("y")};

  const auto start = Clock::now();
  Ideal<RationalField> ideal(ring, std::move(generators));
  const auto finish = Clock::now();
  return TimedSetup<CycleFixture<RationalField>>{
      CycleFixture<RationalField>{
          std::move(ring), std::move(polynomial), std::move(ideal)},
      elapsed_milliseconds(start, finish)};
}

[[nodiscard]] TimedSetup<CycleFixture<PrimeField>> make_frobenius_fixture() {
  auto ring = make_ring(GF(5), {"x"}, Order::Grevlex);
  const auto x = ring.gen("x");
  auto polynomial = x.pow(5);
  std::vector<Polynomial<PrimeField>> generators{polynomial};

  const auto start = Clock::now();
  Ideal<PrimeField> ideal(ring, std::move(generators));
  const auto finish = Clock::now();
  return TimedSetup<CycleFixture<PrimeField>>{
      CycleFixture<PrimeField>{
          std::move(ring), std::move(polynomial), std::move(ideal)},
      elapsed_milliseconds(start, finish)};
}

[[nodiscard]] TimedSetup<CycleFixture<PrimeField>>
make_gf101_seed_fixture() {
  auto ring = make_ring(GF(101), {"x", "y", "z"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto z = ring.gen("z");
  auto polynomial = y.pow(5) + z.pow(5) + y.pow(2) * z.pow(2);
  std::vector<Polynomial<PrimeField>> generators{
      x.pow(2), x * y, x * z, polynomial,
      polynomial.derivative("y"), polynomial.derivative("z")};

  const auto start = Clock::now();
  Ideal<PrimeField> ideal(ring, std::move(generators));
  const auto finish = Clock::now();
  return TimedSetup<CycleFixture<PrimeField>>{
      CycleFixture<PrimeField>{
          std::move(ring), std::move(polynomial), std::move(ideal)},
      elapsed_milliseconds(start, finish)};
}

constexpr std::size_t changing_cycle_family_size = 16;
constexpr std::size_t inverse_system_family_size = 8;
constexpr std::size_t fixed_cycle_batch_size = 4096;

// This is deliberately end-to-end: each member changes the coefficient of
// the distinguished polynomial, rebuilds J and J^2, compiles both quotient
// plans, and then performs the packed screen.  Only result certification is
// disabled because this family has no full-column-rank candidates.
[[nodiscard]] std::vector<PackedCycleScreenResult>
screen_changing_cycle_family_end_to_end() {
  auto ring = make_ring(GF(101), {"x", "y", "z"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto z = ring.gen("z");
  PackedCycleScreenOptions options;
  options.certify_hits = false;
  std::vector<PackedCycleScreenResult> results;
  results.reserve(changing_cycle_family_size);
  for (std::size_t index = 0; index < changing_cycle_family_size; ++index) {
    const auto coefficient = ring.integer(
        static_cast<long>(index + 1));
    auto polynomial =
        y.pow(5) + z.pow(5) +
        coefficient * y.pow(2) * z.pow(2);
    Ideal<PrimeField> ideal(
        ring,
        {x.pow(2), x * y, x * z, polynomial,
         polynomial.derivative("y"), polynomial.derivative("z")});
    auto result = screen_cycle(std::move(ideal), polynomial, options);
    if (result.status != DiscoveryScreenStatus::Complete ||
        !result.cycle_valid || result.length_Q == 0) {
      throw std::runtime_error(
          "changing-candidate packed cycle invariant failed");
    }
    results.push_back(std::move(result));
  }
  return results;
}

// The initial ideal construction is intentionally inside this call.  This
// makes the row a candidate-level end-to-end measurement rather than a
// reusable-plan microbenchmark.
[[nodiscard]] PackedH1ScreenResult screen_gf101_seed_end_to_end() {
  auto ring = make_ring(GF(101), {"x", "y", "z"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto z = ring.gen("z");
  const auto polynomial =
      y.pow(5) + z.pow(5) + y.pow(2) * z.pow(2);
  Ideal<PrimeField> ideal(
      ring,
      {x.pow(2), x * y, x * z, polynomial,
       polynomial.derivative("y"), polynomial.derivative("z")});
  auto result = screen_full_h1(std::move(ideal));
  if (result.status != DiscoveryScreenStatus::Complete ||
      result.length_Q != 11 || result.h1_dimension != 19 ||
      result.socle_dimension != 3 ||
      result.maximum_individual_rank_lower_bound != 1 ||
      result.maximum_individual_rank_upper_bound != 1) {
    throw std::runtime_error("GF(101) packed full-H1 invariant failed");
  }
  return result;
}

// A family of generic binary degree-12 inverse systems (length 49). Candidate
// creation, the one
// sparse degree-(D+1) kernel, GB construction, and packed full-H1 screen are
// all inside the timed call.  The sequential executor makes per-core
// normalization unambiguous.
[[nodiscard]] std::vector<InverseSystemDiscoveryRecord>
screen_inverse_system_family_end_to_end() {
  auto ring = make_ring(GF(101), {"x", "y"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  std::vector<std::vector<Polynomial<PrimeField>>> candidates;
  candidates.reserve(inverse_system_family_size);
  for (std::size_t index = 0; index < inverse_system_family_size; ++index) {
    auto dual_generator = ring.zero();
    std::uint64_t state =
        0xd1b54a32d192ed03ULL ^
        (static_cast<std::uint64_t>(index + 1) *
         0x9e3779b97f4a7c15ULL);
    for (std::size_t x_exponent = 0; x_exponent <= 12; ++x_exponent) {
      state = state * 6364136223846793005ULL +
              1442695040888963407ULL;
      const auto coefficient = ring.integer(
          static_cast<long>(1 + (state >> 32U) % 100U));
      dual_generator =
          dual_generator + coefficient * x.pow(x_exponent) *
                               y.pow(12 - x_exponent);
    }
    candidates.push_back({std::move(dual_generator)});
  }
  CandidateExecutorOptions executor;
  executor.worker_count = 1;
  auto records = search_inverse_systems_parallel(
      ring, candidates, InverseSystemDiscoveryOptions{}, executor);
  for (const auto& record : records) {
    if (record.status != DiscoveryScreenStatus::Complete ||
        record.maximum_dual_degree != 12 ||
        record.quotient_length != 49) {
      throw std::runtime_error(
          "inverse-system generate-and-screen invariant failed");
    }
  }
  return records;
}

[[nodiscard]] std::vector<Polynomial<PrimeField>>
make_fixed_cycle_candidates(const CycleFixture<PrimeField>& fixture) {
  const auto x = fixture.ring.gen("x");
  const auto y = fixture.ring.gen("y");
  const auto z = fixture.ring.gen("z");
  std::vector<Polynomial<PrimeField>> candidates;
  candidates.reserve(fixed_cycle_batch_size);
  for (std::size_t index = 0; index < fixed_cycle_batch_size; ++index) {
    const auto a = fixture.ring.integer(
        static_cast<long>(index % 101));
    const auto b = fixture.ring.integer(
        static_cast<long>((index / 101) % 101));
    const auto c = fixture.ring.integer(
        static_cast<long>((index / (101 * 101)) % 101));
    const auto d = fixture.ring.integer(
        static_cast<long>((17 * index + 29) % 101));
    const auto multiplier =
        fixture.ring.one() + a * x + b * y + c * z +
        d * x * y;
    candidates.push_back(fixture.distinguished_polynomial * multiplier);
  }
  return candidates;
}

[[nodiscard]] PackedH1ScreenResult
screen_length81_homogeneous_end_to_end() {
  auto ring = make_ring(
      GF(101), {"x", "y", "z", "w"}, Order::Grevlex);
  const auto x = ring.gen("x");
  const auto y = ring.gen("y");
  const auto z = ring.gen("z");
  const auto w = ring.gen("w");
  Ideal<PrimeField> ideal(
      ring, {x.pow(3), y.pow(3), z.pow(3), w.pow(3)});
  auto result = screen_full_h1(std::move(ideal));
  if (result.status != DiscoveryScreenStatus::Complete ||
      result.length_Q != 81 || result.socle_dimension != 1) {
    throw std::runtime_error(
        "length-81 homogeneous packed full-H1 invariant failed");
  }
  return result;
}

void print_setup_rows(const std::vector<SetupRow>& rows) {
  std::cout << "\nOne-time setup used by repeated benchmarks\n"
            << "(time covers Ideal construction and reduced Groebner basis; "
               "ring/input construction is excluded)\n";
  std::cout << std::left << std::setw(38) << "fixture"
            << std::right << std::setw(17) << "single_ms"
            << std::setw(18) << "reduced_GB_size" << '\n';
  for (const auto& row : rows) {
    std::cout << std::left << std::setw(38) << row.name
              << std::right << std::setw(17) << std::fixed
              << std::setprecision(6)
              << row.ideal_construction_milliseconds
              << std::setw(18) << row.reduced_basis_size << '\n';
  }
}

void print_statistics(const std::vector<Statistics>& rows) {
  std::cout
      << "\nRepeated operation timings\n"
      << "Audit and full-H1 rows are complete API calls on a prebuilt Ideal; "
         "their internal squares, quotients, reductions, matrices, searches, "
         "and returned-result lifecycle are included unless the operation "
         "explicitly says end-to-end. End-to-end rows also include input, "
         "Ideal/GB, square, and rewrite-plan setup.\n"
      << "p50 and p95 use linear interpolation over independently timed "
         "samples. Candidate throughput is divided by the requested worker "
         "count; it is shown only for discovery rows.\n";
  std::cout << std::left << std::setw(38) << "benchmark"
            << std::setw(45) << "logical operation"
            << std::right << std::setw(10) << "samples"
            << std::setw(15) << "ops/sample"
            << std::setw(15) << "p50_ms/op"
            << std::setw(15) << "p95_ms/op"
            << std::setw(14) << "min_ms/op"
            << std::setw(21) << "aggregate_ops/hour"
            << std::setw(21) << "candidates/h/core" << '\n';
  for (const auto& row : rows) {
    std::cout << std::left << std::setw(38) << row.name
              << std::setw(45) << row.operation
              << std::right << std::setw(10) << row.measured_samples
              << std::setw(15) << row.logical_operations_per_sample
              << std::setw(15) << std::fixed << std::setprecision(6)
              << row.median_milliseconds
              << std::setw(15) << row.percentile95_milliseconds
              << std::setw(14) << row.minimum_milliseconds
              << std::setw(21) << std::fixed << std::setprecision(0)
              << row.median_operations_per_hour;
    if (row.median_candidates_per_hour_per_core.has_value()) {
      std::cout << std::setw(21) << std::fixed << std::setprecision(0)
                << *row.median_candidates_per_hour_per_core;
    } else {
      std::cout << std::setw(21) << "n/a";
    }
    std::cout << '\n';
  }

  std::cout << "\nDiscovery throughput targets (p50)\n";
  std::cout << std::left << std::setw(38) << "benchmark"
            << std::right << std::setw(21) << "candidates/h/core"
            << std::setw(15) << "target"
            << std::setw(13) << "ratio"
            << std::setw(10) << "result" << '\n';
  for (const auto& row : rows) {
    if (!row.target_candidates_per_hour_per_core.has_value() ||
        !row.median_candidates_per_hour_per_core.has_value()) {
      continue;
    }
    const auto achieved = *row.median_candidates_per_hour_per_core;
    const auto target = *row.target_candidates_per_hour_per_core;
    std::cout << std::left << std::setw(38) << row.name
              << std::right << std::setw(21) << std::fixed
              << std::setprecision(0) << achieved
              << std::setw(15) << target
              << std::setw(13) << std::setprecision(2)
              << achieved / target
              << std::setw(10) << (achieved >= target ? "met" : "gap")
              << '\n';
  }

  const Statistics* fixed_j_baseline = nullptr;
  for (const auto& row : rows) {
    if (std::string_view(row.name).starts_with("fixed_J_cycle_batch_") &&
        row.normalization_worker_count == 1) {
      fixed_j_baseline = &row;
      break;
    }
  }
  if (fixed_j_baseline != nullptr) {
    std::cout << "\nFixed-J parallel scaling (p50 aggregate throughput)\n";
    std::cout << std::left << std::setw(38) << "benchmark"
              << std::right << std::setw(10) << "workers"
              << std::setw(26) << "aggregate_candidates/hour"
              << std::setw(15) << "speedup"
              << std::setw(22) << "parallel_efficiency" << '\n';
    for (const auto& row : rows) {
      if (!std::string_view(row.name).starts_with(
              "fixed_J_cycle_batch_") ||
          !row.normalization_worker_count.has_value()) {
        continue;
      }
      const auto workers = *row.normalization_worker_count;
      const auto speedup =
          row.median_operations_per_hour /
          fixed_j_baseline->median_operations_per_hour;
      const auto efficiency =
          100.0 * speedup / static_cast<double>(workers);
      std::cout << std::left << std::setw(38) << row.name
                << std::right << std::setw(10) << workers
                << std::setw(26) << std::fixed << std::setprecision(0)
                << row.median_operations_per_hour
                << std::setw(15) << std::setprecision(2) << speedup
                << std::setw(21) << std::setprecision(1) << efficiency
                << "%\n";
      if (workers == 8 && speedup < 8.0) {
        std::cout << "8-worker scaling gap: observed " << std::fixed
                  << std::setprecision(2) << speedup
                  << "x aggregate speedup versus ideal 8.00x ("
                  << std::setprecision(1) << efficiency
                  << "% parallel efficiency).\n";
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto configuration = parse_configuration(argc, argv);
    if (configuration.help) {
      print_help(std::cout);
      return 0;
    }

    std::cout << "laughableengine native kernel benchmark\n"
              << "clock=std::chrono::steady_clock"
              << " warmups=" << configuration.warmup_samples
              << " samples=" << configuration.measured_samples
              << " iterations=" << configuration.iterations
              << " nf_batch_size=64"
              << " nf_batches/sample="
              << configuration.nf_batches_per_sample
              << " changing_cycle_family_size="
              << changing_cycle_family_size
              << " inverse_system_family_size="
              << inverse_system_family_size
              << " fixed_cycle_batch_size="
              << fixed_cycle_batch_size << '\n';
#ifdef NDEBUG
    std::cout << "build_assertions=disabled\n";
#else
    std::cout << "build_assertions=enabled\n";
#endif

    auto normal_forms = make_normal_form_fixture();
    auto qq_seed = make_qq_seed_fixture();
    auto tjurina = make_tjurina_fixture();
    auto frobenius = make_frobenius_fixture();
    auto gf101_seed = make_gf101_seed_fixture();
    const auto fixed_cycle_candidates =
        make_fixed_cycle_candidates(gf101_seed.value);

    std::uint64_t consumed_checksum = 0;
    consumed_checksum = mix_checksum(
        consumed_checksum, normal_forms.value.ideal.quotient_dimension());
    consumed_checksum = mix_checksum(
        consumed_checksum, qq_seed.value.ideal.quotient_dimension());
    consumed_checksum = mix_checksum(
        consumed_checksum, tjurina.value.ideal.quotient_dimension());
    consumed_checksum = mix_checksum(
        consumed_checksum, frobenius.value.ideal.quotient_dimension());
    consumed_checksum = mix_checksum(
        consumed_checksum, gf101_seed.value.ideal.quotient_dimension());
    for (const auto& candidate : fixed_cycle_candidates) {
      consumed_checksum = mix_checksum(
          consumed_checksum, checksum_prime_polynomial(candidate));
    }

    print_setup_rows({
        SetupRow{
            "GF(101) compiled-NF ideal",
            normal_forms.ideal_construction_milliseconds,
            normal_forms.value.ideal.groebner_basis().size()},
        SetupRow{
            "QQ three-variable seed ideal",
            qq_seed.ideal_construction_milliseconds,
            qq_seed.value.ideal.groebner_basis().size()},
        SetupRow{
            "QQ Tjurina G ideal",
            tjurina.ideal_construction_milliseconds,
            tjurina.value.ideal.groebner_basis().size()},
        SetupRow{
            "GF(5) (x^5) ideal",
            frobenius.ideal_construction_milliseconds,
            frobenius.value.ideal.groebner_basis().size()},
        SetupRow{
            "GF(101) fixed-J seed ideal",
            gf101_seed.ideal_construction_milliseconds,
            gf101_seed.value.ideal.groebner_basis().size()}});

    const auto nf_invocations = checked_product(
        configuration.nf_batches_per_sample,
        configuration.iterations,
        "NF batch invocation count");
    std::vector<Statistics> statistics;
    statistics.reserve(14);
    statistics.push_back(measure(
        "compiled_batch_nf_gf101",
        "one polynomial NF (batch size 64)",
        configuration,
        nf_invocations,
        normal_forms.value.batch.size(),
        [&]() {
          return normal_forms.value.ideal.reducer().normal_forms(
              normal_forms.value.batch);
        },
        checksum_normal_forms,
        consumed_checksum));

    statistics.push_back(measure(
        "packed_cycle_changing_J_e2e",
        "16 changing (J,g), including all setup",
        configuration,
        configuration.iterations,
        changing_cycle_family_size,
        screen_changing_cycle_family_end_to_end,
        checksum_packed_cycles,
        consumed_checksum,
        1,
        100'000.0));

    statistics.push_back(measure(
        "packed_full_h1_gf101_seed_e2e",
        "screen_full_h1, including all setup",
        configuration,
        configuration.iterations,
        1,
        screen_gf101_seed_end_to_end,
        checksum_packed_h1,
        consumed_checksum,
        1,
        10'000.0));

    statistics.push_back(measure(
        "inverse_binary_degree12_e2e",
        "annihilator + packed H1, 8 length-49 inputs",
        configuration,
        configuration.iterations,
        inverse_system_family_size,
        screen_inverse_system_family_end_to_end,
        checksum_inverse_records,
        consumed_checksum,
        1,
        10'000.0));

    PackedCycleScreenOptions fixed_cycle_options;
    fixed_cycle_options.certify_hits = false;
    for (const auto worker_count : {1U, 2U, 4U, 8U}) {
      CandidateExecutorOptions executor_options;
      executor_options.worker_count = worker_count;
      statistics.push_back(measure(
          "fixed_J_cycle_batch_gf101_w" +
              std::to_string(worker_count),
          "screen_cycles_parallel(prebuilt J, " +
              std::to_string(fixed_cycle_candidates.size()) + " inputs)",
          configuration,
          configuration.iterations,
          fixed_cycle_candidates.size(),
          [&]() {
            auto results = screen_cycles_parallel(
                gf101_seed.value.ideal, fixed_cycle_candidates,
                fixed_cycle_options, executor_options);
            for (const auto& result : results) {
              if (result.status != DiscoveryScreenStatus::Complete ||
                  !result.cycle_valid) {
                throw std::runtime_error(
                    "fixed-J parallel cycle invariant failed");
              }
            }
            return results;
          },
          checksum_packed_cycles,
          consumed_checksum,
          worker_count,
          100'000.0));
    }

    statistics.push_back(measure(
        "distinguished_audit_qq_seed",
        "audit_cycle(prebuilt Ideal)",
        configuration,
        configuration.iterations,
        1,
        [&]() {
          auto result = audit_cycle(
              qq_seed.value.ideal,
              qq_seed.value.distinguished_polynomial);
          if (result.status() != CycleAuditStatus::Complete ||
              result.quotient_length() != 11 ||
              result.cycle_valid() != true) {
            throw std::runtime_error("QQ seed audit invariant failed");
          }
          return result;
        },
        checksum_audit<RationalField>,
        consumed_checksum));

    statistics.push_back(measure(
        "distinguished_audit_gf5_x5",
        "audit_cycle(prebuilt Ideal)",
        configuration,
        configuration.iterations,
        1,
        [&]() {
          auto result = audit_cycle(
              frobenius.value.ideal,
              frobenius.value.distinguished_polynomial);
          if (result.status() != CycleAuditStatus::Complete ||
              result.quotient_length() != 5 || !result.faithful_cycle()) {
            throw std::runtime_error("GF(5) x^5 audit invariant failed");
          }
          return result;
        },
        checksum_audit<PrimeField>,
        consumed_checksum));

    statistics.push_back(measure(
        "full_h1_qq_three_variable_seed",
        "full_h1_action(prebuilt Ideal)",
        configuration,
        configuration.iterations,
        1,
        [&]() {
          auto result = full_h1_action(qq_seed.value.ideal);
          if (result.length_Q != 11 || result.h1_dimension != 19 ||
              result.socle_dimension != 3) {
            throw std::runtime_error("QQ seed full-H1 invariant failed");
          }
          return result;
        },
        checksum_h1<RationalField>,
        consumed_checksum));

    statistics.push_back(measure(
        "full_h1_qq_tjurina_G",
        "full_h1_action(prebuilt Ideal)",
        configuration,
        configuration.iterations,
        1,
        [&]() {
          auto result = full_h1_action(tjurina.value.ideal);
          if (result.length_Q != 11 || result.h1_dimension != 15 ||
              result.socle_dimension != 2) {
            throw std::runtime_error("Tjurina full-H1 invariant failed");
          }
          return result;
        },
        checksum_h1<RationalField>,
        consumed_checksum));

    statistics.push_back(measure(
        "full_h1_gf5_x5",
        "full_h1_action(prebuilt Ideal)",
        configuration,
        configuration.iterations,
        1,
        [&]() {
          auto result = full_h1_action(frobenius.value.ideal);
          if (result.length_Q != 5 || result.h1_dimension != 5 ||
              result.socle_dimension != 1 ||
              !result.faithful_witness_audit.has_value()) {
            throw std::runtime_error("GF(5) x^5 full-H1 invariant failed");
          }
          return result;
        },
        checksum_h1<PrimeField>,
        consumed_checksum));

    Configuration capability_configuration = configuration;
    capability_configuration.warmup_samples = 0;
    capability_configuration.measured_samples =
        std::min<std::size_t>(configuration.measured_samples, 3);
    statistics.push_back(measure(
        "capability_h1_gf101_length81_e2e",
        "4-var homogeneous full-H1, including setup",
        capability_configuration,
        1,
        1,
        screen_length81_homogeneous_end_to_end,
        checksum_packed_h1,
        consumed_checksum,
        1,
        10'000.0));

    print_statistics(statistics);
    std::cout << "\nconsumed_checksum=" << consumed_checksum << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "kernel benchmark error: " << error.what() << '\n';
    return 1;
  }
}

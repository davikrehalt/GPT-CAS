#include <atomic>
#include <cstddef>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/candidate_executor.hpp"

namespace {

using laughableengine::CandidateExecutorOptions;
using laughableengine::DiscoveryScreenStatus;
using laughableengine::GF;
using laughableengine::Ideal;
using laughableengine::Order;
using laughableengine::PackedCycleScreenOptions;
using laughableengine::Polynomial;
using laughableengine::PolynomialRing;
using laughableengine::PrimeField;
using laughableengine::execute_candidates;
using laughableengine::resolved_worker_count;
using laughableengine::screen_cycles_parallel;

[[noreturn]] void fail(const std::string& message, int line) {
  throw std::runtime_error("line " + std::to_string(line) + ": " + message);
}

#define CHECK(expression)                                                     \
  do {                                                                        \
    if (!(expression)) {                                                      \
      fail("CHECK failed: " #expression, __LINE__);                          \
    }                                                                         \
  } while (false)

template <typename Exception, typename Function>
void expect_throw(Function&& function, int line) {
  try {
    std::invoke(std::forward<Function>(function));
  } catch (const Exception&) {
    return;
  } catch (const std::exception& error) {
    fail("unexpected exception type: " + std::string(error.what()), line);
  }
  fail("expected exception was not thrown", line);
}

#define EXPECT_THROW(exception, expression)                                   \
  expect_throw<exception>([&] { static_cast<void>(expression); }, __LINE__)

void test_deterministic_generic_executor() {
  std::vector<int> candidates;
  for (int value = 0; value < 1000; ++value) {
    candidates.push_back(value);
  }
  struct Session {
    int worker;
  };
  std::atomic<std::size_t> sessions{0};
  CandidateExecutorOptions options;
  options.worker_count = 8;
  const auto result = execute_candidates<int, int>(
      candidates,
      [&](std::size_t worker) {
        ++sessions;
        return Session{static_cast<int>(worker)};
      },
      [](Session&, int candidate, std::size_t index) {
        return candidate * candidate + static_cast<int>(index);
      },
      options);
  CHECK(sessions == 8);
  CHECK(result.size() == candidates.size());
  for (std::size_t index = 0; index < result.size(); ++index) {
    CHECK(result[index] ==
          candidates[index] * candidates[index] + static_cast<int>(index));
  }
  CHECK(resolved_worker_count(0, options) == 0);
  CHECK(resolved_worker_count(3, options) == 3);
}

void test_failure_propagation() {
  const std::vector<int> candidates{0, 1, 2, 3, 4};
  CandidateExecutorOptions options;
  options.worker_count = 3;
  EXPECT_THROW(
      std::runtime_error,
      (execute_candidates<int, int>(
          candidates,
          [](std::size_t) { return 0; },
          [](int&, int candidate, std::size_t) {
            if (candidate == 2) {
              throw std::runtime_error("candidate failure");
            }
            return candidate;
          },
          options)));
}

void test_parallel_cycle_screen_matches_single_worker() {
  const PolynomialRing<PrimeField> ring(GF(5), {"x"}, Order::Grevlex);
  const auto x = ring.gen(0);
  const Ideal<PrimeField> ideal(ring, {x.pow(5)});
  std::vector<Polynomial<PrimeField>> candidates;
  for (std::size_t index = 0; index < 64; ++index) {
    candidates.push_back(
        index % 3 == 0 ? x.pow(5)
                       : (index % 3 == 1 ? ring.zero() : x.pow(6)));
  }
  PackedCycleScreenOptions screens;
  screens.certify_hits = false;
  CandidateExecutorOptions one;
  one.worker_count = 1;
  CandidateExecutorOptions many;
  many.worker_count = 8;
  const auto sequential =
      screen_cycles_parallel(ideal, candidates, screens, one);
  const auto parallel =
      screen_cycles_parallel(ideal, candidates, screens, many);
  CHECK(sequential.size() == parallel.size());
  for (std::size_t index = 0; index < sequential.size(); ++index) {
    CHECK(sequential[index].status == DiscoveryScreenStatus::Complete);
    CHECK(parallel[index].status == sequential[index].status);
    CHECK(parallel[index].length_Q == sequential[index].length_Q);
    CHECK(parallel[index].length_P_mod_J2 ==
          sequential[index].length_P_mod_J2);
    CHECK(parallel[index].cycle_valid == sequential[index].cycle_valid);
    CHECK(parallel[index].multiplication_rank ==
          sequential[index].multiplication_rank);
    CHECK(parallel[index].full_column_rank_candidate ==
          sequential[index].full_column_rank_candidate);
  }
}

}  // namespace

int main() {
  try {
    test_deterministic_generic_executor();
    test_failure_propagation();
    test_parallel_cycle_screen_matches_single_worker();
    std::cout << "candidate executor tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "candidate executor test failure: " << error.what() << '\n';
    return 1;
  }
}

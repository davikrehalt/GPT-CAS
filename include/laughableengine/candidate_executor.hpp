#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "laughableengine/discovery.hpp"

namespace laughableengine {

struct CandidateExecutorOptions {
  // Zero selects min(hardware_concurrency, candidate_count), with a fallback
  // of one worker when the platform does not report its hardware count.
  std::size_t worker_count = 0;
};

inline std::size_t resolved_worker_count(
    std::size_t candidate_count,
    const CandidateExecutorOptions& options = {}) {
  if (candidate_count == 0) {
    return 0;
  }
  auto workers = options.worker_count;
  if (workers == 0) {
    workers = std::thread::hardware_concurrency();
    if (workers == 0) {
      workers = 1;
    }
  }
  return std::min(workers, candidate_count);
}

// Deterministic indexed candidate executor.  Each worker receives its own
// session from session_factory; no Groebner scratch/cache is shared between
// workers. Results are returned in input order regardless of scheduling.
template <typename Candidate,
          typename Result,
          typename SessionFactory,
          typename Evaluator>
[[nodiscard]] std::vector<Result> execute_candidates(
    std::span<const Candidate> candidates,
    SessionFactory session_factory,
    Evaluator evaluator,
    const CandidateExecutorOptions& options = {}) {
  const auto worker_count = resolved_worker_count(candidates.size(), options);
  if (worker_count == 0) {
    return {};
  }

  std::vector<std::optional<Result>> slots(candidates.size());
  std::atomic<std::size_t> next{0};
  std::atomic<bool> stop{false};
  std::mutex failure_mutex;
  std::exception_ptr failure;
  std::vector<std::thread> workers;
  workers.reserve(worker_count);

  static_assert(
      std::is_copy_constructible_v<SessionFactory> &&
          std::is_copy_constructible_v<Evaluator>,
      "parallel candidate callables must be copy constructible so every worker owns a private copy");

  // Capture callable state by value. std::thread copies this closure once per
  // worker, so mutable user functors are never invoked concurrently through
  // one shared instance.
  auto run = [&, session_factory, evaluator](
                 std::size_t worker_index) mutable {
    try {
      auto session = session_factory(worker_index);
      while (!stop.load(std::memory_order_relaxed)) {
        const auto index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= candidates.size()) {
          break;
        }
        slots[index].emplace(evaluator(session, candidates[index], index));
      }
    } catch (...) {
      stop.store(true, std::memory_order_relaxed);
      std::lock_guard lock(failure_mutex);
      if (!failure) {
        failure = std::current_exception();
      }
    }
  };

  try {
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back(run, worker);
    }
  } catch (...) {
    stop.store(true, std::memory_order_relaxed);
    for (auto& worker : workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    throw;
  }
  for (auto& worker : workers) {
    worker.join();
  }
  if (failure) {
    std::rethrow_exception(failure);
  }

  std::vector<Result> result;
  result.reserve(slots.size());
  for (auto& slot : slots) {
    if (!slot.has_value()) {
      throw std::logic_error(
          "candidate executor completed without producing every result");
    }
    result.push_back(std::move(*slot));
  }
  return result;
}

inline std::vector<PackedCycleScreenResult> screen_cycles_parallel(
    Ideal<PrimeField> ideal,
    std::span<const Polynomial<PrimeField>> candidates,
    const PackedCycleScreenOptions& screen_options = {},
    const CandidateExecutorOptions& executor_options = {}) {
  for (const auto& candidate : candidates) {
    if (!ideal.ring().zero().same_ring(candidate)) {
      throw std::invalid_argument(
          "parallel cycle candidates require the ideal's exact ring");
    }
  }
  return execute_candidates<
      Polynomial<PrimeField>,
      PackedCycleScreenResult>(
      candidates,
      [ideal, screen_options](std::size_t) {
        return PackedCycleDiscoverySession(ideal, screen_options);
      },
      [](const PackedCycleDiscoverySession& session,
         const Polynomial<PrimeField>& candidate,
         std::size_t) { return session.screen(candidate); },
      executor_options);
}

inline std::vector<PackedCycleScreenResult> screen_cycles_parallel(
    Ideal<PrimeField> ideal,
    const std::vector<Polynomial<PrimeField>>& candidates,
    const PackedCycleScreenOptions& screen_options = {},
    const CandidateExecutorOptions& executor_options = {}) {
  return screen_cycles_parallel(
      std::move(ideal),
      std::span<const Polynomial<PrimeField>>(
          candidates.data(), candidates.size()),
      screen_options, executor_options);
}

}  // namespace laughableengine

# laughableengine kernel roadmap

## Goal

Deliver the narrow native kernel in the faithful cotangent-`H1`
specification. The production engine audits a distinguished cycle, computes
the full `H1`/socle action, and generates Artin ideals from Macaulay inverse
systems without calling Sage or Macaulay2. Sage remains an independent test
oracle.

This is deliberately not a general computer-algebra system.

## 0.3.0 completed scope

1. **Exact polynomial and ideal kernel**
   - `QQ` and `GF(p)` for prime `p < 2^31`, one to sixteen exact variables, and
     lex/grevlex orders.
   - Sparse polynomials, deterministic reduced Buchberger bases, batched
     normal forms, standard monomials, quotient coordinates, `J^2`, ideal
     equality, zero-dimensional principal colons, and small elimination jobs.

2. **Required cotangent-`H1` contracts**
   - `audit_cycle(J,g)` checks `g in J`, all derivative memberships, and
     computes `J^2:g`; a hit is exactly `J^2:g == J`.
   - `cotangent_h1({ring,generators,maximal_power=N})` specializes to
     `J=(generators)+m^N`. Sparse row spaces construct `P/J` and `P/J^2`
     exactly inside degrees `<N` and `<2N`; the complete `H1` presentation is
     the kernel of `f |-> (f,df)` from `P/J^2` to `P/J + (P/J)^e`.
   - The structured presentation reports exact dimensions and
     `h1_relation_matrix` without materializing every kernel vector.
     `h1_kernel_coordinates(limits)` and `h1_basis(limits)` expose the full
     exact kernel on request. Class proofs expose `annihilator_basis` and
     explicit `colon_generators`. Materializers return a complete answer or a
     resource-limit error, never a dense/numeric/partial fallback.
   - `full_h1_action(J)` constructs `J/J^2`, the differential kernel, the
     socle, the bilinear action, and exact-arithmetic lower/upper rank bounds
     with an explicit proof state. Every proposed faithful witness is replayed
     through the principal-colon audit.
   - Colon closure retains every intermediate ideal, quotient length, and
     stop reason.

3. **Macaulay inverse systems**
   - One sparse exact action matrix through degree `D+1` produces
     `Ann(F_1,...,F_t)` over `QQ` or `GF(p)`.
   - Ordinary-differentiation and divided-power conventions are explicit.
   - The returned compact ideal is checked against the action kernel and its
     quotient length.

4. **Finite-field discovery path**
   - Six-lane packed monomials, structure-of-arrays packed polynomials, and
     one-modulus-per-container `uint32_t` finite-field matrices.
   - Fixed quotients compile border rewrite tables once. Whole polynomial
     batches and matrices reduce through packed variable actions with no
     scalar normal-form calls.
   - Sparse packed multiplication and differential maps avoid dense work in
     the common finite-quotient paths.

5. **Exact linear algebra**
   - Dense exact matrices plus immutable deterministic sparse matrices.
   - Sparse rank, right kernel, image, and solve use native sparse elimination;
     they do not materialize a dense fallback.

6. **Search and certification workflow**
   - Deterministic candidate-parallel execution with a private compiled
     session for every worker and results returned in input order.
   - Compact inverse-system search records retain large bases for full-rank
     candidates, including certified hits, unless requested for every case.
   - Two-prime structural signatures and a deliberately conservative
     same-support small-integer lift support the finite-field-to-`QQ` handoff.
     Every lifted candidate still requires an exact `QQ` audit.

7. **Independent verification and interfaces**
   - A standalone, standard-library-only `laughable-jg-verify` recomputes a
     proposed `(J,g)` result from raw JSON input. It does not import or invoke
     laughableengine, Sage, Singular, or Macaulay2.
   - Header-only C++, CLI, Python bindings, CMake package, wheel build, tests,
     and a benchmark harness expose the same narrow contracts.

## Mandatory regression gates

- Characteristic-zero homogeneous cases have zero socle action.
- The supplied three-variable seed has quotient lengths `11/48`, `H1`
  dimension `19`, socle dimension `3`, and maximum individual rank `1`.
- `G=x^3+y^7+x*y^5` has quotient lengths `11/34`, `H1` dimension `15`,
  socle dimension `2`, maximum rank `1`, and a nonfaithful distinguished
  class.
- In characteristic `p`, `J=(x^p)` detects `[x^p]` as faithful.
- Lifted integer input agrees at `GF(101)` and `GF(103)` before exact `QQ`
  certification.
- Resource exhaustion is inconclusive, never a mathematical negative.
- Finite-field matrix-space search never mistakes generic extension-field
  rank for a base-field witness.
- The independent verifier accepts the golden faithful certificate and
  rejects malformed, tampered, non-cycle, and nonfaithful inputs.
- The completed ten-variable E10 structured case over exact `QQ` has
  `length(Q)=176`, `length(P/J^2)=2728`, conormal dimension `2552`, and `H1`
  dimension `1873`. Its distinguished class has multiplication rank `176`,
  zero annihilator, and `(J^2:g)=J`; the proof exposes `725` colon generators.
  The fully materialized kernel has `1873` polynomials and `2092` nonzero
  terms. The regression uses the generic sparse path with no special case or
  fallback.

## 0.3 performance baseline

The Release benchmark on an Apple M2 Ultra reports roughly:

| End-to-end workload | Median | Throughput |
|---|---:|---:|
| Changing small `(J,g)` cycle screen | `0.60 ms` | `5.95M/hour/core` |
| Supplied GF(101) full-`H1` seed | `0.91 ms` | `3.96M/hour/core` |
| Binary inverse system, degree 12, length 49, generate + full `H1` | `12.6 ms` | `285k/hour/core` |
| Four-variable homogeneous quotient, length 81, full `H1` | `117 ms` | `30.7k/hour/core` |

These are local baselines, not universal claims. They exceed the supplied raw
throughput targets. In the final quick run, a fixed-ideal 4,096-candidate
cycle batch scaled by about `1.00x/2.21x/4.12x/6.71x` at `1/2/4/8` workers.
Making high-core performance less sensitive to scheduling and memory locality
is still open.

## Next engineering work

1. Add a 60-bit or SIMD finite-field backend; the current packed discovery
   path is limited to primes below `2^31`.
2. Reuse compiled templates across changing ideals with the same initial
   monomial stratum, not only across candidates for one fixed ideal.
3. Improve parallel scheduling and memory locality beyond four workers;
   consider process workers for long Python-driven searches.
4. Implement general CRT/rational reconstruction. The present two-prime lift
   accepts only matching, bounded, centered integer coefficients and never
   combines unrelated modular `H1` basis coordinates.
5. Profile Groebner-heavy families before adding F4/FGLM or modular `QQ`
   bases. The measured finite-quotient linear-algebra paths no longer justify
   replacing Buchberger merely for fashion.
6. Bundle GMP when producing redistributable macOS wheels.

## Secondary additions

- First syzygies/Hilbert--Burch matrices for small height-two ideals.
- FGLM conversion for small zero-dimensional quotients.
- Combination certificates against the original, unreduced generators.
- Additional candidate-family plugins and streamed search persistence.

## Deliberate exclusions

No schemes, cotangent-complex package, derived categories, general
resolutions, arbitrary coefficient rings, symbolic transcendental functions,
notebook UI, or compatibility layer for Sage/Macaulay2 syntax.

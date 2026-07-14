# laughableengine exact-algebra roadmap

## Goal

Grow a compact native exact-algebra engine from a trustworthy core. Version
0.3 supplies exact fields, sparse polynomials, ideals, finite quotients, exact
linear algebra, and a first deep vertical: structured cotangent `H1` and
annihilators. Optional search and inverse-system workflows build on that core
without calling Sage or Macaulay2. Sage remains an independent test oracle.

This is not yet a general computer-algebra system. The architecture is meant
to grow without pretending that unimplemented breadth already exists.

## 0.3.0 completed scope

1. **Exact polynomial and ideal kernel**
   - `QQ` and `GF(p)` for prime `p < 2^31`, one to sixteen exact variables, and
     lex/grevlex orders.
   - Sparse polynomials, deterministic reduced Buchberger bases, batched
     normal forms, standard monomials, quotient coordinates, `J^2`, ideal
     equality, zero-dimensional principal colons, and small elimination jobs.

2. **Structured local-quotient and cotangent-`H1` vertical**
   - `local_ideal`, `quotient`, `conormal_module`, `derivative_map`, and
     `kernel` expose the stages as separate algebraic objects. The kernel's
     exact constraint matrix gives a complete presentation without
     materializing every vector; `kernel_coordinates(max_coordinate_entries=...)`
     and `basis(max_coordinate_entries=...)` materialize the full exact kernel
     on request.
   - Those public steps are genuine cached computation boundaries: quotient
     arithmetic does not construct `J^2`, conormal construction does not
     differentiate, derivative construction does not rank the stacked kernel,
     and class construction does not compute an annihilator.
   - `class_of(g)` returns a kernel element, `annihilator()` returns an ideal
     in the quotient, and `preimage()` returns `J^2:g` in `P`. Examples express
     theorem conclusions by comparing those ideals; the core does not replace
     them with a `faithful` flag. Materializers return a complete answer or a
     resource-limit error, never a dense/numeric/partial fallback.
   - The compatibility `cotangent_h1` presentation is backed by the same
     sparse row spaces. They construct `P/J` and `P/J^2` exactly inside degrees
     `<N` and `<2N`; the complete `H1` presentation is the kernel of
     `f |-> (f,df)` from `P/J^2` to `P/J + (P/J)^e`.

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

6. **Optional search and certification workflows**
   - `audit_cycle(J,g)` checks membership and derivatives and computes
     `J^2:g`. `full_h1_action(J)` constructs the differential kernel, socle,
     bilinear action, and exact rank bounds. Colon closure retains every
     intermediate ideal, quotient length, and stop reason.
   - Deterministic candidate-parallel execution with a private compiled
     session for every worker and results returned in input order.
   - Compact inverse-system search records retain large bases for full-rank
     candidates, including certified hits, unless requested for every case.
   - Two-prime structural signatures and a deliberately conservative
     same-support small-integer lift support the finite-field-to-`QQ` handoff.
     Every lifted candidate still requires an exact `QQ` audit.

7. **Interfaces and optional independent verification**
   - A standalone, standard-library-only `laughable-jg-verify` recomputes a
     proposed `(J,g)` result from raw JSON input. It does not import or invoke
     laughableengine, Sage, Singular, or Macaulay2.
   - Header-only C++, CLI, Python bindings, CMake package, wheel build, tests,
     and a benchmark harness expose the implemented contracts.

## Regression gates

- Characteristic-zero homogeneous cases have zero socle action.
- The supplied three-variable seed has quotient lengths `11/48`, `H1`
  dimension `19`, socle dimension `3`, and maximum individual rank `1`.
- `G=x^3+y^7+x*y^5` has quotient lengths `11/34`, `H1` dimension `15`,
  socle dimension `2`, maximum rank `1`, and a distinguished class with
  nonzero annihilator.
- In characteristic `p`, the annihilator of `[x^p]` in `P/(x^p)` is the zero
  ideal, and its ambient preimage is `(x^p)`.
- Lifted integer input agrees at `GF(101)` and `GF(103)` before exact `QQ`
  certification.
- Resource exhaustion is inconclusive, never a mathematical negative.
- Finite-field matrix-space search never mistakes generic extension-field
  rank for a base-field witness.
- The optional independent verifier accepts the golden faithful certificate
  and rejects malformed, tampered, non-cycle, and nonfaithful inputs.
- The completed ten-variable E10 structured case over exact `QQ` has
  `length(Q)=176`, `length(P/J^2)=2728`, conormal dimension `2552`, and `H1`
  dimension `1873`. Its distinguished class has multiplication rank `176`,
  annihilator equal to the quotient zero ideal, and annihilator preimage equal
  to `J`; that preimage exposes `725` generators.
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

1. Extend quotient, ideal-in-quotient, module, map, and element abstractions
   beyond the structured local case only as implemented algorithms can honor
   their contracts; do not add a broad but hollow façade.
2. Add a 60-bit or SIMD finite-field backend; the current packed discovery
   path is limited to primes below `2^31`.
3. Reuse compiled templates across changing ideals with the same initial
   monomial stratum, not only across candidates for one fixed ideal.
4. Improve parallel scheduling and memory locality beyond four workers;
   consider process workers for long Python-driven searches.
5. Implement general CRT/rational reconstruction. The present two-prime lift
   accepts only matching, bounded, centered integer coefficients and never
   combines unrelated modular `H1` basis coordinates.
6. Profile Groebner-heavy families before adding F4/FGLM or modular `QQ`
   bases. The measured finite-quotient linear-algebra paths no longer justify
   replacing Buchberger merely for fashion.
7. Bundle GMP when producing redistributable macOS wheels.

## Secondary additions

- First syzygies/Hilbert--Burch matrices for small height-two ideals.
- FGLM conversion for small zero-dimensional quotients.
- Combination certificates against the original, unreduced generators.
- Additional candidate-family plugins and streamed search persistence.

## Deliberate exclusions

No schemes, cotangent-complex package, derived categories, general
resolutions, arbitrary coefficient rings, symbolic transcendental functions,
notebook UI, or compatibility layer for Sage/Macaulay2 syntax.

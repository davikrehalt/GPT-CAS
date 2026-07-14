# laughableengine 0.3 handoff

## Status

The narrow replacement contract is implemented and integrated. The next
agent can build search families or continue optimization on top of stable
C++, Python, and CLI entry points; it should not need Sage or Macaulay2 in a
production computation.

“Ready” here means ready for the faithful cotangent-`H1` workflow. It does not
mean a broad Sage/Macaulay2 replacement.

New public APIs should follow `DESIGN.md`: named operations, explicit input
records, immutable evidence objects, stable coordinate conventions, and no
Macaulay2-style type-directed reinterpretation of a command.

## Stable contracts

1. `audit_cycle(J,g)` verifies `g in J`, verifies every partial derivative is
   in `J`, computes `J^2:g`, and reports a hit exactly when `J^2:g == J`.
2. `cotangent_h1({ring,generators,maximal_power=N})` has one deliberately
   narrow meaning: `J=(generators)+m^N`. It constructs exact sparse matrices
   for `P/J^2 -> P/J`, `f |-> df`, and `f |-> (f,df)`, so both `J/J^2` and the
   complete cotangent `H1` are kernels. It does not construct a Groebner basis
   of `J^2`.
3. `full_h1_action(J)` computes `J/J^2`, the derivative kernel, `Soc(P/J)`,
   the bilinear action, and exact-arithmetic rank bounds with an explicit proof
   state. A full-rank witness is always sent through `audit_cycle` again.
4. `macaulay_annihilator(Fs)` builds one sparse action matrix through degree
   `D+1`, takes its exact kernel, and returns the compact annihilator ideal.
5. Packed discovery is finite-field-only. Exact `QQ` is the final
   reconstruction/certification layer.
6. Resource exhaustion is reported as inconclusive and is never silently
   converted into a negative mathematical answer.

## Workstream map

### A. Polynomial and Groebner kernel

Primary files:

- `include/laughableengine/field.hpp`
- `include/laughableengine/polynomial.hpp`
- `include/laughableengine/groebner.hpp`
- `include/laughableengine/ideal.hpp`
- `include/laughableengine/quotient_context.hpp`
- `include/laughableengine/sparse_matrix.hpp`

This layer owns exact `QQ`/`GF(p)`, canonical ring identity, sparse
polynomials, deterministic reduced bases, batched normal forms, standard
monomials, ideals, `J^2`, colons, equality, and native sparse elimination.
The exact monomial representation accepts one to sixteen variables. The
packed discovery representation is a separate six-lane type; do not let its
six-variable limit leak into exact APIs, and do not route the ten-variable
E10 computation through it.

### B. Finite quotient and `H1` action

Primary files:

- `include/laughableengine/cycle_audit.hpp`
- `include/laughableengine/cotangent_h1.hpp`
- `include/laughableengine/h1.hpp`
- `include/laughableengine/truncated_monomial_space.hpp`
- `include/laughableengine/incremental_sparse_row_space.hpp`
- `include/laughableengine/matrix_space.hpp`
- `include/laughableengine/packed_prime.hpp`
- `include/laughableengine/packed_polynomial.hpp`
- `include/laughableengine/compiled_quotient.hpp`
- `include/laughableengine/discovery.hpp`

`h1.hpp` is the exact reference/certification path. `discovery.hpp` is the
packed `GF(p)` path. `CompiledPrimeQuotientPlan` compiles border actions once
and reduces batches without scalar normal-form calls. Keep the exact path as a
differential oracle when optimizing packed code.

`cotangent_h1.hpp` is the scalable exact path for the structured local input
`J=(G)+m^N`. It works in `P/m^(2N)`, which is exact because
`m^(2N) subset J^2`. It returns a `CotangentH1Presentation`; the presentation
is already the complete answer even when no explicit kernel basis is
materialized. Do not replace it with a Groebner basis of `J^2` merely to make
the output resemble the older general-ideal path.

### C. Inverse systems and search drivers

Primary files:

- `include/laughableengine/inverse_system.hpp`
- `include/laughableengine/inverse_discovery.hpp`
- `include/laughableengine/candidate_executor.hpp`
- `include/laughableengine/reconstruction.hpp`

The inverse-system convention is explicit:
`OrdinaryDifferentiation` and `DividedPowers` must never be silently mixed.
Parallel results are deterministic and retain input order. Each worker owns a
private compiled session. Compact search records retain large bases for
full-rank candidates, including certified hits, unless
`retain_all_candidates` is set.

The reconstruction helper only accepts identical support and matching
bounded centered coefficients at two distinct primes. It is intentionally not
general CRT/rational reconstruction, and a lifted polynomial is not a hit
until an exact `QQ` audit succeeds.

### D. Independent verifier

Primary files:

- `include/laughableengine/certificate.hpp`
- `tools/laughable_jg_verify.py`
- `tests/independent_verifier_test.py`
- `tests/certificates/gf5_x5_faithful.json`

The strict `laughable-jg-v1` document contains only raw inputs. The verifier
uses the Python standard library and must remain independent: do not import
the engine or invoke Sage, Singular, or Macaulay2 from it. It recomputes the
Groebner bases, membership conditions, standard monomials, and multiplication
rank needed to certify `J^2:g == J`.

## Public entry points

### C++

Include `laughableengine/laughableengine.hpp`. Important functions/types are:

- `audit_cycle`, `colon_closure`, `full_h1_action`
- `cotangent_h1`, `CotangentH1Spec`, `CotangentH1Presentation`,
  `CotangentClassProof`
- `macaulay_annihilator`
- `screen_cycle`, `screen_full_h1`, `PackedCycleDiscoverySession`
- `screen_cycles_parallel`, `search_inverse_systems_parallel`
- `reduce_mod_prime`, `screen_signature`,
  `lift_matching_small_integer_polynomial`
- `make_jg_certificate_json`

### Python

Construct rings with `laughableengine.QQ(...)` or `laughableengine.GF(...)`.
The corresponding `Ring` methods are:

- `audit_cycle`, `colon_closure`, `full_h1_action`
- `cotangent_h1(generators=..., maximal_power=N)`; the returned presentation
  has `verify_class(g)`, `h1_kernel_coordinates(limits)`, and
  `h1_basis(limits)`
- `macaulay_annihilator`
- `screen_cycle`, `screen_full_h1`, `screen_cycles_parallel`
- `search_inverse_systems`
- `make_jg_certificate`

The packed screen/search methods reject `QQ` deliberately. Parallel methods
release the GIL around the complete native job.

### CLI

The relevant commands are `audit`, `closure`, `h1`, `cotangent-h1`,
`verify-h1-class`, `screen-audit`, `screen-h1`, `inverse-system`, and
`certificate`. Both structured cotangent commands require
`--maximal-power N`; positional polynomials are the lower generators `G` in
`J=(G)+m^N`. `--apolarity ordinary` and `--apolarity divided` select the
inverse-system convention.

Exit status is stable: `0` completed, `2` invalid input, `3`
resource-limited/inconclusive, and `1` internal/I/O failure. The independent
verifier also uses `2` for a conclusive nonfaithful certificate.

## Structured cotangent coordinate contract

For `J=(G)+m^N`, the presentation works in `B=P/J^2` using the exact
truncation `P/m^(2N)`. Its public matrices all act on column vectors:

- `reduction_matrix : B -> Q` represents `f |-> f mod J`;
- `derivative_matrix : B -> Q^e` represents `f |-> df mod J`;
- `h1_relation_matrix : B -> Q + Q^e` is the vertical stack of reduction
  followed by the derivative blocks.

Consequently `conormal_dimension = nullity(reduction_matrix)` and
`h1_dimension = nullity(h1_relation_matrix)`. This is the complete kernel
presentation. `h1_kernel_coordinates(limits)` exposes the exact coordinate
kernel and `h1_basis(limits)` exposes its polynomial representatives. These
are optional materialization requests, not prerequisites for a complete
result. They return a complete exact basis or a resource-limit error; there is
no dense, numeric, sampled, or partial fallback.

Ambient monomials are ordered by increasing total degree, then descending
lexicographic exponent tuple. The quotient bases are the nonpivot monomials
in that order. Reduction rows use the `Q` basis. Derivative rows use one
`Q`-coordinate block per variable in ring-variable order. Class
multiplication matrices have `P/J^2` rows and `Q` columns.

For a valid representative `g`, `verify_class(g)` computes
`mu_g : Q -> J/J^2`. The exact identity

```text
kernel(mu_g) = Ann_Q([g]) = (J^2:g)/J
```

is the reason the proof record may report all of the following equivalent
facts: full column rank, zero annihilator dimension, faithfulness, and
`colon_equals_ideal`. The last flag is a rank-nullity certificate of colon
equality; it is not a second hidden Groebner computation. The proof also
exposes exact `annihilator_basis` representatives and explicit
`colon_generators` for `J^2:g`.

## Invariants that must remain true

- The quotient dimension equals the number of standard monomials.
- If `g in J`, then `J` is contained in `J^2:g`.
- Every returned `H1` representative lies in `J` and all of its derivatives
  lie in `J`.
- In a structured presentation, `h1_relation_matrix` is exactly the vertical
  stack of `reduction_matrix` and `derivative_matrix`, and its nullity equals
  the reported `h1_dimension`.
- Materialized `h1_kernel_coordinates` span exactly the kernel of
  `h1_relation_matrix`, and `h1_basis` uses the documented `P/J^2` basis.
- Every returned annihilator basis element kills the class, and the returned
  `colon_generators` generate `J^2:g`.
- `d(J^2)` reduces to zero modulo `J`; the truncated square builder checks
  this invariant by default.
- Every socle representative is killed by every variable modulo `J`.
- Every action product lies in `J/J^2`.
- A full-socle-rank witness passes independent principal-colon equality.
- Modular integer inputs agree at two primes before any lift.
- Generic rank over an extension is not mistaken for a base-field witness.

Mandatory numerical regressions are documented in `README.md` and tested in
`tests/h1.cpp`, `tests/cotangent_h1.cpp`, `tests/discovery.cpp`, and the Sage
oracle.

The completed ten-variable E10 structured regression over exact `QQ` reports:

```text
length(Q)                    176
length(P/J^2)                2728
dimension(J/J^2)             2552
dimension(H1)                1873
distinguished-class rank     176
annihilator dimension        0
faithful / colon equals J    true / true
colon generator count        725
materialized H1 basis        1873 polynomials / 2092 terms
```

This result used the generic `cotangent_h1` implementation with exact sparse
elimination. There is no E10-specific branch, dense or numeric fallback, or
external Sage/Macaulay2 computation. Indicative Release timings on the
development Mac Studio were `201.5 s` to build the exact `QQ` presentation,
`17.8 s` to verify the class, and `0.24 s` to materialize the already-computed
kernel as an explicit basis. The same generic computation over `GF(101)` took
`6.66 s` to build and `0.36 s` to verify, with the same four presentation
dimensions and rank `176`. These numbers are machine-local; the finite-field
timings are not the characteristic-zero certification.

## Build and validation

```sh
PYTHON=python3
"$PYTHON" -m pip install build pybind11 pytest
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DLAUGHABLEENGINE_BUILD_PYTHON=ON \
  -DLAUGHABLEENGINE_BUILD_BENCHMARKS=ON \
  -Dpybind11_DIR="$("$PYTHON" -m pybind11 --cmakedir)" \
  -DPython_EXECUTABLE="$("$PYTHON" -c 'import sys; print(sys.executable)')"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/laughableengine_kernel_benchmark --quick
sage tests/oracles/cotangent_h1_reference.sage
```

For a wheel:

```sh
"$PYTHON" -m build --wheel --outdir dist-0.3
```

The wheel installs the Python API and a PATH-visible
`laughable-jg-verify` console command. It deliberately does not build the
native `laughable` executable; use the CMake build above for that CLI.

Before handing out a binary macOS wheel, inspect its GMP linkage. The current
project intentionally does not claim that GMP is bundled.

## Validated handoff artifact

The unambiguous local wheel is
`dist-0.3/laughableengine-0.3.0-cp313-cp313-macosx_15_0_arm64.whl`
(683,016 bytes), with SHA-256
`d9ba0c53a04cf5ce274b550f572f77926e0795f865997ba19d68b130c1df224e`.
It targets CPython 3.13 on arm64 macOS 15 or newer and dynamically links the
Homebrew GMP libraries. The older wheel under `dist/` is a retained 0.2
artifact; do not select wheels with an ambiguous `dist/*.whl` glob.

Final validation completed on the development Mac Studio:

- native Release CTest: 21/21 passed;
- CPython 3.9 and CPython 3.13 builds: 22/22 CTest cases passed on each;
- clean installed-wheel Python suite: 20/20 passed;
- ASan plus UBSan suite: 21/21 passed with no findings;
- independent Sage oracle passed;
- clean installed CMake package was consumed by an external project;
- installed native and wheel verifier paths both accepted the faithful
  `GF(5), J=(x^5), g=x^5` certificate, and the resource-limit path exited 3;
- Clang static analysis of the Python binding reported no diagnostics.

## Current performance and remaining gaps

On the development M2 Ultra, the packed finite-field quick benchmark exceeds
the requested raw rates for distinguished cycles, full `H1` at length at most
100, and sequential inverse-system generation plus packed audit at length at
most 50. The length-81 four-variable packed full-`H1` case is about
`31k/hour/core`. Exact `QQ` rows are certification baselines, not discovery
throughput claims.

The honest remaining performance gaps are:

- the final quick run reached `6.71x` at eight workers, but high-core results
  remain sensitive to scheduling and memory locality;
- packed primes are below `2^31`, with no 60-bit/SIMD backend;
- compiled state is reused for one fixed ideal, not yet across changing ideals
  sharing an initial monomial stratum;
- there is no general modular `QQ` reconstruction;
- Groebner bases remain correctness-first Buchberger rather than F4/FGLM.

These are suitable next-agent tasks. They are not blockers for the supplied
replacement contract.

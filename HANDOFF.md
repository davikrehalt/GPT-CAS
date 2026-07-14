# laughableengine 0.3 handoff

## Status

The exact polynomial/ideal kernel and the first structured local-quotient
vertical are implemented and integrated. The next agent can extend reusable
algebraic objects, build search families, or continue optimization on top of
stable C++, Python, and CLI entry points. Production computations do not need
Sage or Macaulay2.

“Ready” means the documented operations and the structured cotangent-`H1`
case study are executable and exact. It does not mean the project is already
a broad Sage/Macaulay2 replacement.

New public APIs should follow `DESIGN.md`: named operations, explicit input
records, immutable evidence objects, stable coordinate conventions, and no
Macaulay2-style type-directed reinterpretation of a command.

## Stable contracts

1. Exact rings, polynomials, ideals, Groebner bases, batched normal forms,
   standard monomials, ideal products, principal colons, equality, and exact
   linear algebra are independent reusable operations.
2. `local_ideal({ring,generators,maximal_power=N})` has one deliberately narrow
   meaning: `J=(generators)+m^N`. The chain `J.quotient()`,
   `R.conormal_module()`, `C.derivative_map()`, and `d.kernel()` exposes the
   quotient, conormal module, derivative map, and complete cotangent `H1` as
   separate objects. It does not construct a Groebner basis of `J^2`.
3. `H1.class_of(g)` validates a class; `xi.annihilator()` returns its actual
   quotient ideal; and `ann.preimage()` returns `J^2:g` in the polynomial
   ring. The example, not the core, decides whether equality with the zero
   ideal or with `J` proves a theorem.
4. `macaulay_annihilator(Fs)` builds one sparse action matrix through degree
   `D+1`, takes its exact kernel, and returns the compact annihilator ideal.
5. `audit_cycle`, `full_h1_action`, colon closure, packed screening, and
   certificate replay are optional compatibility/research workflows assembled
   from the core. Packed discovery is finite-field-only; exact `QQ` remains the
   final reconstruction/certification layer.
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
six-variable limit leak into exact APIs or route higher-variable exact jobs
through it.

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

`cotangent_h1.hpp` is the scalable exact implementation for the structured
local input `J=(G)+m^N`. It works in `P/m^(2N)`, which is exact because
`m^(2N) subset J^2`. The public façade exposes a structured ideal, quotient,
conormal module, derivative map, kernel, kernel element, annihilator ideal, and
ambient preimage. The underlying presentation is already the complete answer
even when no explicit kernel basis is materialized. Do not replace it with a
Groebner basis of `J^2` merely to make the output resemble the older
general-ideal path.

The implementation boundary is real: quotient data, conormal/`J^2` data, the
ambient derivative matrix, and the stacked `H1` kernel presentation are four
immutable stages cached with `call_once` inside one exact quotient context.
Annihilator multiplication is a fifth request-only calculation. Preserve
these boundaries when optimizing: a resource limit in a later stage must not
make an earlier object fail before that stage is requested.

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

### D. Optional independent verifier

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

- `PolynomialRing`, `Polynomial`, `Ideal`, `QuotientContext`, exact dense and
  sparse matrices, Groebner bases, normal forms, kernels, and solves
- `origin_power_ideal`, `quotient`, `conormal_module`, `derivative_map`, and
  `kernel`; their returned objects provide `class_of`, `annihilator`, and
  `preimage`
- `macaulay_annihilator`
- compatibility/workflow entry points `audit_cycle`, `colon_closure`,
  `full_h1_action`, `cotangent_h1`, `CotangentH1Spec`,
  `CotangentH1Presentation`, and `CotangentClassProof`
- `screen_cycle`, `screen_full_h1`, `PackedCycleDiscoverySession`
- `screen_cycles_parallel`, `search_inverse_systems_parallel`
- `reduce_mod_prime`, `screen_signature`,
  `lift_matching_small_integer_polynomial`
- `make_jg_certificate_json`

### Python

Construct rings with `laughableengine.QQ(...)` or `laughableengine.GF(...)`.
The corresponding `Ring` methods are:

- foundational polynomial and ideal operations such as `ideal`,
  `groebner_basis`, `normal_form`, `normal_forms`, `square`, `colon`, and
  `standard_monomials`
- `local_ideal(generators=..., maximal_power=N)`, followed by `quotient`,
  `conormal_module`, `derivative_map`, `kernel`, `class_of`, `annihilator`, and
  `preimage`
- `macaulay_annihilator`
- compatibility/workflow methods `audit_cycle`, `colon_closure`,
  `full_h1_action`, and `cotangent_h1(...).verify_class(g)`
- `screen_cycle`, `screen_full_h1`, `screen_cycles_parallel`
- `search_inverse_systems`
- `make_jg_certificate`

The packed screen/search methods reject `QQ` deliberately. Parallel methods
release the GIL around the complete native job.

### CLI

The CLI exposes ordinary algebra commands (`print`, `diff`, `nf`, `divide`,
`gb`, `dim`, `std`, `colon`, and `eliminate`) alongside compatibility/research
commands (`audit`, `closure`, `h1`, `cotangent-h1`, `verify-h1-class`,
`screen-audit`, `screen-h1`, `inverse-system`, and `certificate`). Both
structured cotangent commands require `--maximal-power N`; positional
polynomials are the lower generators `G` in `J=(G)+m^N`.
`--apolarity ordinary` and `--apolarity divided` select the inverse-system
convention.

Exit status is stable: `0` completed, `2` invalid input, `3`
resource-limited/inconclusive, and `1` internal/I/O failure. The independent
verifier also uses `2` for a conclusive nonfaithful certificate.

## Structured cotangent coordinate contract

For `J=(G)+m^N`, the implementation works in `B=P/J^2` using the exact
truncation `P/m^(2N)`. Put `R=P/J`, `C=R.conormal_module()`,
`d=C.derivative_map()`, and `H1=d.kernel()`. Their public matrices all act on
column vectors:

- `C.constraint_matrix : B -> R` represents `f |-> f mod J`;
- `d.ambient_matrix : B -> R^e` represents `f |-> df mod J`;
- `H1.constraint_matrix : B -> R + R^e` is the vertical stack of reduction
  followed by the derivative blocks.

Consequently `C.dimension = nullity(C.constraint_matrix)` and
`H1.dimension = nullity(H1.constraint_matrix)`. This is the complete kernel
presentation. `H1.kernel_coordinates(max_coordinate_entries=...)` exposes the
exact coordinate kernel and `H1.basis(max_coordinate_entries=...)` exposes its
polynomial representatives. These are
optional materialization requests, not prerequisites for a complete result.
They return a complete exact basis or a resource-limit error; there is no
dense, numeric, sampled, or partial fallback.

Ambient monomials are ordered by increasing total degree, then descending
lexicographic exponent tuple. The quotient bases are the nonpivot monomials
in that order. Reduction rows use the `R` basis. Derivative rows use one
`R`-coordinate block per variable in ring-variable order. Class
multiplication matrices have `P/J^2` rows and `R` columns.

For a valid `xi=H1.class_of(g)`, `xi.annihilator()` computes the quotient ideal
which is the kernel of `mu_g : R -> J/J^2`. The exact identity

```text
kernel(mu_g) = Ann_R([g]) = (J^2:g)/J
```

is why `ann.preimage()` returns the ambient ideal `J^2:g`. The objects retain
exact generators and coordinates. Calling code can compare `ann` with
`R.zero_ideal()` or its preimage with `J`; those theorem-level comparisons are
not baked into the element or ideal.

The legacy `CotangentH1Presentation` accessors `reduction_matrix`,
`derivative_matrix`, and `h1_relation_matrix`, together with
`verify_class(g)`, expose the same calculation for compatibility. Do not use
their theorem-oriented summary flags as the model for new APIs.

## Invariants that must remain true

- The quotient dimension equals the number of standard monomials.
- If `g in J`, then `J` is contained in `J^2:g`.
- Every returned `H1` representative lies in `J` and all of its derivatives
  lie in `J`.
- In a structured presentation, `H1.constraint_matrix` is exactly the vertical
  stack of `C.constraint_matrix` and `d.ambient_matrix`, and its nullity equals
  `H1.dimension`.
- Materialized `H1.kernel_coordinates()` span exactly the kernel of
  `H1.constraint_matrix`, and `H1.basis()` uses the documented `P/J^2` basis.
- Every generator of `xi.annihilator()` kills the class, and the generators of
  `xi.annihilator().preimage()` generate `J^2:g`.
- `d(J^2)` reduces to zero modulo `J`; the truncated square builder checks
  this invariant by default.
- Every socle representative is killed by every variable modulo `J`.
- Every action product lies in `J/J^2`.
- A full-socle-rank witness passes independent principal-colon equality.
- Modular integer inputs agree at two primes before any lift.
- Generic rank over an extension is not mistaken for a base-field witness.

The numerical regression suite is documented in `README.md` and tested in
`tests/finite_algebra.cpp`, `tests/h1.cpp`, `tests/cotangent_h1.cpp`,
`tests/discovery.cpp`, and the Sage oracle.

The completed ten-variable E10 structured regression over exact `QQ` reports:

```text
length(Q)                    176
length(P/J^2)                2728
dimension(J/J^2)             2552
dimension(H1)                1873
distinguished-class rank     176
annihilator dimension        0
annihilator equals zero      true
annihilator preimage equals J true
colon generator count        725
materialized H1 basis        1873 polynomials / 2092 terms
```

This result used the general structured-object façade backed by exact sparse
elimination. There is no E10-specific branch, dense or numeric fallback, or
external Sage/Macaulay2 computation. Indicative Release timings on the
development Mac Studio were `201.5 s` to build the exact `QQ` presentation,
`17.8 s` to construct the class and annihilator, and `0.24 s` to materialize
the already-computed kernel as an explicit basis. The same generic computation
over `GF(101)` took `6.66 s` to build and `0.36 s` for the class and
annihilator, with the same four presentation dimensions and rank `176`. These
numbers are machine-local; the finite-field timings are not the
characteristic-zero certification.

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
"$PYTHON" -m build --wheel --outdir dist-current
```

The wheel installs the Python API and a PATH-visible
`laughable-jg-verify` console command. It deliberately does not build the
native `laughable` executable; use the CMake build above for that CLI.

Before handing out a binary macOS wheel, inspect its GMP linkage. The current
project intentionally does not claim that GMP is bundled.

## Binary artifact and validation note

Do not hand out an older wheel from `dist/` or `dist-0.3`: a wheel built before
the structured object façade does not contain the current Python API. Build a
fresh wheel from the source being handed off, run the Python suite against the
installed wheel, and inspect its GMP linkage. The project does not currently
claim that GMP is bundled.

Source validation covers the native Release suite, Python bindings, the
structured finite-algebra object chain, the independent verifier, the Sage
oracle, sanitizer builds, and external CMake-package consumption. The exact
ten-variable Python example is an explicit slow check rather than an ordinary
fast test.

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

These are suitable next-agent tasks. They are not blockers for the implemented
0.3 exact-algebra surface or the structured cotangent case study.

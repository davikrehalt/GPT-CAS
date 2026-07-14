> **LLM-output notice:** Everything that follows in this README, and the engine
> implementation it describes, was produced by OpenAI Codex during the session
> identified by Andy Jiang as **“5.6 Sol Ultra”**, under Andy Jiang's direction.

# laughableengine

> **Status:** Research prototype v0.3.0. Deliberately narrow: this is not a
> general computer-algebra system and not a drop-in replacement for Sage or
> Macaulay2.

`laughableengine` 0.3.0 is a small native exact-algebra engine for one
research workflow: finding and certifying faithful elements in cotangent
`H1` of a zero-dimensional quotient supported at the origin.

It is ready to serve as the computational kernel for that workflow.

## Implementation provenance

The source implementation and initial repository documentation were produced
entirely by **OpenAI Codex** during the session identified by Andy Jiang as
**“5.6 Sol Ultra”**, under Andy Jiang's direction, mathematical requirements,
and review criteria. No human-written engine implementation was incorporated.

This provenance statement does not claim Codex authorship of the underlying
mathematics or of third-party dependencies such as GMP, pybind11, CMake, or
the C++ and Python standard libraries. Sage is used only as an independent
test oracle; Sage and Macaulay2 are not runtime dependencies, and no Sage or
Macaulay2 source code is included.

For `P = k[x_1,...,x_e]`, `Q = P/J`, and a proposed representative `g`, the
primary contract is

```text
g in J,  every partial_i(g) in J,  and  (J^2 : g) = J.
```

The engine also constructs `J/J^2`, the derivative kernel, `Soc(Q)`, and the
full bilinear socle action. It searches the resulting exact matrix space for
one full-socle-rank element and confirms every hit again by the principal
colon test.

## What 0.3 implements

- Exact `QQ` and prime fields `GF(p)` for `p < 2^31`.
- Sparse exact polynomials in one to sixteen variables with lex or grevlex
  order; the optional packed finite-field discovery path uses six variables.
- Deterministic reduced Groebner bases, batched normal forms, standard
  monomials, quotient coordinates, ideal products and squares, principal
  colons, ideal equality, and small elimination jobs.
- Exact dense and genuinely sparse rank, kernel, image, and solve.
- `audit_cycle(J,g)`, colon closure, and `full_h1_action(J)` with replayable
  evidence and explicit resource-limit states.
- A sparse truncated `cotangent_h1(generators, maximal_power=N)` path for the
  structured local ideal `J = (generators) + m^N`. It computes the complete
  conormal/kernel presentation without constructing a Groebner basis of
  `J^2`; explicit conormal and `H1` basis vectors remain opt-in.
- `macaulay_annihilator(F_1,...,F_t)` through one sparse kernel computation in
  degrees at most `D+1`, with ordinary and divided-power conventions.
- A packed `GF(p)` discovery path using packed monomials, compiled border
  rewrite tables, batched reductions, and sparse packed matrices.
- Deterministic candidate-parallel cycle and inverse-system searches.
- A conservative two-prime small-integer lifting helper followed by required
  exact `QQ` certification.
- Header-only C++, a native CLI, Python bindings, CMake packaging, tests, and
  an independent standard-library-only certificate verifier.

The source tree's [DESIGN.md](DESIGN.md) records the intentionally explicit
API philosophy. [HANDOFF.md](HANDOFF.md) lists the other agent's entry points,
while [ROADMAP.md](ROADMAP.md) records measured limits and future work.

## Direct cotangent kernel for `J = (G) + m^N`

The specialized cotangent path takes exactly three mathematical inputs: a
polynomial ring `P`, a list `G` of polynomials vanishing at the origin, and a
positive integer `N`. Its ideal is always

```text
J = ideal(G) + maximal_ideal^N.
```

There is no type-dependent reinterpretation of `G` and no implicit choice of
truncation. In particular, `maximal_power=N` means the exponent `N` in this
formula; it is not a matrix-size limit or a search heuristic.

Put `Q=P/J` and `B=P/J^2`. Because `m^(2N)` is contained in `J^2`, both
quotients are computed exactly inside the finite monomial space of degrees
less than `2N`. The returned presentation stores the sparse maps

```text
reduction:   B -> Q             f |-> f
derivative:  B -> Q^e           f |-> df
h1 relation: B -> Q + Q^e       f |-> (f, df)
```

and therefore

```text
J/J^2 = kernel(reduction),       H1(L_(Q/k)) = kernel(h1 relation).
```

Equivalently, the usual conormal map
`J/J^2 -> Omega_(P/k) tensor_P Q` is `derivative_matrix` restricted to
`kernel(reduction_matrix)`. The stacked matrix computes that restricted
kernel without first expanding a large conormal basis.

This is a complete computation of `H1`: `h1_relation_matrix`, its exact rank,
and `h1_dimension` determine the kernel. Explicit vectors are still
available. `h1_kernel_coordinates(limits)` returns exact coordinate vectors
and `h1_basis(limits)` returns the corresponding polynomials. These are
separate materialization requests because they can use much more memory. They
either return the complete exact basis or raise a resource-limit error; there
is no dense, numeric, sampled, or partial fallback. The same complete-versus-
materialized distinction applies to the conormal basis.

Coordinates are stable and inspectable. Ambient monomials are ordered first
by increasing total degree and then by descending lexicographic exponent
tuple. Quotient bases are the nonpivot monomials in that order. Every sparse
matrix acts on column vectors: columns of `reduction_matrix`,
`derivative_matrix`, and `h1_relation_matrix` use the standard-monomial basis
of `P/J^2`; reduction rows use the basis of `Q`; derivative rows are grouped
by variable, with one `Q` block per variable; and `h1_relation_matrix` stacks
the reduction block before those derivative blocks.

The C++ entry point is deliberately data-first:

```cpp
auto H = cotangent_h1(CotangentH1Spec{
    ring,
    std::vector{lower_generator_1, lower_generator_2},
    std::size_t{4},
});
auto proof = H.verify_class(candidate);
```

For a completed class check, `verify_class(g)` constructs the exact linear
map `Q -> J/J^2`, `q |-> qg`. Its kernel is
`Ann_Q([g]) = (J^2:g)/J`, so rank-nullity gives equivalent certificates:

```text
rank = length(Q)  <=>  annihilator_dimension = 0
                  <=>  (J^2:g) = J.
```

Thus `faithful` and `colon_equals_ideal` are exact consequences of the stored
multiplication matrix. The proof exposes `annihilator_basis` in the documented
`Q` basis and explicit `colon_generators` for `J^2:g`; these are evidence, not
hidden fallback calculations. `audit_cycle(J,g)` remains available when a
separate Groebner/colon replay is wanted for a small general ideal.

### E10 complete regression

The characteristic-zero E10 example fits the specialized interface exactly:

```python
import laughableengine as le

P = le.QQ("x1 x2 x3 x4 x5 x6 x7 x8 x9 x10")
xs = P.gens()

F = sum((x**3 for x in xs), P.zero()) + le.elementary_symmetric(xs, 4)
partials = [F.derivative(x) for x in xs]

H = P.cotangent_h1(generators=partials, maximal_power=4)
proof = H.verify_class(F)

assert H.length_Q == 176
assert H.length_P_mod_J2 == 2728
assert H.conormal_dimension == 2552
assert H.h1_dimension == 1873
assert proof.multiplication_rank == 176
assert proof.annihilator_dimension == 0
assert proof.faithful
assert proof.colon_equals_ideal
assert len(proof.colon_generators) == 725

explicit_h1 = H.h1_basis()
assert len(explicit_h1) == 1873
assert sum(f.term_count() for f in explicit_h1) == 2092
```

The completed exact `QQ` run gives

```text
length(Q)             = 176
length(P/J^2)         = 2728
dimension(J/J^2)      = 2552
dimension(H1)         = 1873
rank(Q -> J/J^2)      = 176
dimension(annihilator)= 0
colon generator count = 725
explicit H1 basis     = 1873 polynomials, 2092 nonzero terms
```

This was the ordinary structured algorithm over `QQ`: there is no E10
special case, dense or numeric fallback, Sage call, or Macaulay2 call.

Indicative Release timings on the development Mac Studio were:

| Coefficient field | Presentation build | Class proof | Explicit `H1` basis |
|---|---:|---:|---:|
| exact `QQ` | `201.5 s` | `17.8 s` | `0.24 s` |
| `GF(101)` comparison | `6.66 s` | `0.36 s` | not separately timed |

These are machine-local timings, not universal throughput claims. The
`GF(101)` run returned the same four presentation dimensions and rank `176`,
but it is only a finite-field comparison; the `QQ` row is the
characteristic-zero computation and certification.

## Python quick start

Build requirements are CMake, a C++20 compiler, GMP/GMPXX, Python 3.9 or
newer, and ordinary Python build tooling:

```sh
python3 -m pip install .
```

This installs the Python API and the independent `laughable-jg-verify`
command. The native `laughable` executable is built and installed through
CMake, not included in the wheel.

Use the exact API over either `QQ` or `GF(p)`:

```python
import laughableengine as le

R = le.GF(5, "x")
(x,) = R.gens()
J = R.ideal([x**5])

audit = R.audit_cycle(J, x**5)
assert audit.cycle_valid
assert audit.colon_ideal == J
assert audit.faithful

h1 = R.full_h1_action(J)
assert h1.h1_dimension == 5
assert h1.socle_dimension == 1
assert h1.has_faithful_witness
assert h1.faithful_witness_audit.faithful
```

Use the packed finite-field API for discovery. A hit is still colon-certified
unless `certify_hits=False` is explicitly requested:

```python
screen = R.screen_full_h1(J)
assert screen.status == "complete"
assert screen.maximum_individual_rank == 1
assert screen.hit

candidates = [x**5, R.zero(), x**6] * 100
records = R.screen_cycles_parallel(
    J, candidates, workers=4, certify_hits=False
)
```

Generate an annihilator from a Macaulay inverse system. The operator and dual
rings share a field and variable count but remain distinct ring contexts:

```python
P = le.QQ("x y")
x, y = P.gens()
D = le.QQ("X Y", order="lex")
X, Y = D.gens()

data = P.macaulay_annihilator([X**2 + Y])
assert data.annihilator == P.ideal([x**2 - 2*y, x*y, y**2])
assert data.action_rank == data.quotient_length == 3
```

For high-volume inverse-system discovery, `GF(p)` rings expose
`search_inverse_systems(candidates, workers=...)`. Results remain in input
order, and by default large dual/annihilator bases are retained for full-rank
candidates, including certified hits.

`QQ` witness coefficients are Python `fractions.Fraction` values; finite-field
coefficients are integers. Objects from separately constructed rings are not
interchangeable, even if their printed definitions match.

## Command line

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DLAUGHABLEENGINE_BUILD_BENCHMARKS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The ordinary E10 regression runs over `GF(101)` in CTest. Run the complete
exact characteristic-zero regression explicitly with:

```sh
cmake --build build --target laughableengine_check_e10_qq
```

The same check through the public Python API is
[`examples/e10_direct.py`](examples/e10_direct.py).

Representative commands:

```sh
./build/laughable --field 'GF(5)' --vars x \
  audit --g 'x^5' 'x^5'

./build/laughable --field 'GF(5)' --vars x \
  cotangent-h1 --maximal-power 5

./build/laughable --field 'GF(5)' --vars x \
  verify-h1-class --maximal-power 5 --g 'x^5'

./build/laughable --field 'GF(5)' --vars x \
  screen-h1 'x^5'

./build/laughable --field QQ --vars x,y \
  inverse-system 'x^2+y'

./build/laughable --field QQ --vars x,y,z \
  eliminate --elim x 'x*y-1' 'x-z'
```

Run `laughable --help` for the complete syntax. Quote expressions containing
`*`; implicit multiplication is intentionally rejected. Exit `0` means a
completed computation, including a conclusive negative. Exit `2` is invalid
input, `3` is resource-limited/inconclusive, and `1` is an internal or I/O
failure.

## Independent verification

The `certificate` command emits only the raw field, variables, order, original
generators of `J`, and `g`. It deliberately does not serialize the engine’s
Groebner bases, ranks, or claimed answer. The separate verifier recomputes the
answer using only Python’s standard library:

```sh
./build/laughable --field 'GF(5)' --vars x \
  certificate --g 'x^5' 'x^5' \
  | python3 tools/laughable_jg_verify.py -
```

A verified faithful certificate exits `0`; a conclusive nonfaithful result
exits `2`; a verifier resource limit exits `3`; malformed input or an internal
failure exits `1`. A CMake install names the verifier
`laughable-jg-verify`.

Python can emit the same strict JSON with
`R.make_jg_certificate(generators, g)`.

## C++ and CMake

```cpp
#include <laughableengine/laughableengine.hpp>

using namespace laughableengine;

int main() {
  auto ring = make_ring(GF(5), {"x"}, Order::Grevlex);
  auto x = ring.gen("x");
  Ideal<PrimeField> ideal(ring, {x.pow(5)});
  auto result = audit_cycle(ideal, x.pow(5));
  return result.faithful_cycle() ? 0 : 1;
}
```

Installed consumers use:

```cmake
find_package(laughableengine CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE laughableengine::laughableengine)
```

The umbrella header exposes the exact kernel, packed discovery types,
candidate executor, inverse-system driver, reconstruction helpers, and JSON
certificate writer.

## Correctness gates

The native suite checks all mandatory examples:

| Case | Exact result |
|---|---|
| Supplied three-variable seed | `length(Q)=11`, `length(P/J^2)=48`, `dim(H1)=19`, socle `3`, maximum individual rank `1` |
| `G=x^3+y^7+x*y^5` Tjurina ideal | lengths `11/34`, `dim(H1)=15`, socle `2`, maximum rank `1`, distinguished class nonfaithful |
| `GF(5)`, `J=(x^5)` | `[x^5]` is a faithful primitive class |
| Characteristic-zero homogeneous example | socle action is zero |
| Non-origin quotient `(x^2-x)` | rejected by the full-`H1` origin-support gate |

The supplied seed is also compared at `GF(101)` and `GF(103)` before exact
`QQ` certification. Matrix-space results distinguish proven rank, proven
full-column-rank witnesses, generic-only bounds, and resource exhaustion.
`Ann_Q(H1)` is diagnostic only and is never substituted for the required
individual-witness test.

## Apple Silicon performance

Release medians on an Apple M2 Ultra Mac Studio are approximately:

| End-to-end workload | Median | Throughput |
|---|---:|---:|
| Changing small `(J,g)` screen | `0.60 ms` | `5.95M/hour/core` |
| Supplied GF(101) full-`H1` seed | `0.91 ms` | `3.96M/hour/core` |
| Degree-12 binary inverse system, length 49, generate + full `H1` | `12.6 ms` | `285k/hour/core` |
| Four-variable homogeneous quotient, length 81, full `H1` | `117 ms` | `30.7k/hour/core` |

Run `./build/laughableengine_kernel_benchmark --quick` to reproduce the local
comparison. These are baselines from the final quick validation run, not
universal claims. One-, two-, four-, and eight-worker fixed-ideal batches
scaled by about `1.00x`, `2.21x`, `4.12x`, and `6.71x` in that run; high-core
scaling is useful, but remains sensitive to scheduling and memory locality.

## Deliberate limits

- This is a narrow research kernel, not a general-purpose CAS.
- The structured cotangent path is finite and exact, but its ambient monomial
  count grows combinatorially with the variable count and `maximal_power`.
  Explicit resource limits fail loudly instead of changing the algorithm or
  returning a partial kernel.
- The exact Groebner engine is deterministic Buchberger, not F4/F5. The
  measured target workloads are currently dominated by compiled quotient and
  linear-algebra paths.
- Packed discovery uses primes below `2^31` and 8-bit exponent lanes
  (`0..255`). The exact polynomial layer supports exponents through `65,535`.
- The two-prime helper is a bounded small-integer lift, not general CRT or
  rational reconstruction.
- Candidate workers are native threads with private engine sessions. More
  stable high-core scaling remains an optimization target.
- Nontrivial positive-dimensional principal colon is rejected.
- Binary wheels link the build machine’s GMP dynamically; bundle GMP before
  distributing a supposedly self-contained macOS wheel.

## License

`laughableengine` is available under the [MIT License](LICENSE).

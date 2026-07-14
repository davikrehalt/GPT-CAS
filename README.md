> **LLM-output notice:** Everything that follows in this README, and the engine
> implementation it describes, was produced by OpenAI Codex during the session
> identified by Andy Jiang as **“5.6 Sol Ultra”**, under Andy Jiang's direction.

# laughableengine

> **Status:** Research prototype v0.3.0. This is a small exact-algebra core
> designed to grow, not yet a broad computer-algebra system or a drop-in
> replacement for Sage or Macaulay2.

`laughableengine` starts with exact fields, sparse polynomials, ideals,
finite quotients, exact matrices, kernels, and annihilators. Its first deep
vertical is cotangent `H1` for structured zero-dimensional local quotients.
That calculation is an application assembled from reusable algebraic objects,
not the identity of the engine.

The current surface is deliberately modest. The intent is to grow by adding
clear objects and operations with stable meanings, while keeping the exact
arithmetic and performance-oriented representations underneath them.

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

## Exact algebra first

The ordinary public objects are rings, polynomials, ideals, quotients, finite
presentations, linear maps, and elements. Substantial operations have one
descriptive meaning:

```python
import laughableengine as le

P = le.QQ("x y", order="grevlex")
x, y = P.gens()
I = P.ideal([x**2 - y, y**3])

basis = I.groebner_basis
remainders = I.normal_forms([x**4, x*y])
standard = I.standard_monomials()
K = I.square().colon(x)
```

This is the level on which new functionality should compose. Specialized
algorithms may exploit a structured representation internally, but they must
still return inspectable algebraic objects and must state their supported
domain explicitly.

## What 0.3 implements

- Exact `QQ` and prime fields `GF(p)` for `p < 2^31`.
- Sparse exact polynomials in one to sixteen variables with lex or grevlex
  order; the optional packed finite-field discovery path uses six variables.
- Deterministic reduced Groebner bases, batched normal forms, standard
  monomials, quotient coordinates, ideal products and squares, principal
  colons, ideal equality, and small elimination jobs.
- Exact dense and genuinely sparse rank, kernel, image, and solve.
- Structured local ideals `J=(G)+m^N`, finite quotients `P/J`, conormal
  modules, derivative maps, their exact kernels, and elements of those
  kernels. The sparse implementation avoids constructing a Groebner basis of
  `J^2`; explicit conormal and `H1` bases remain opt-in.
- `macaulay_annihilator(F_1,...,F_t)` through one sparse kernel computation in
  degrees at most `D+1`, with ordinary and divided-power conventions.
- Optional research workflows: `audit_cycle(J,g)`, colon closure,
  `full_h1_action(J)`, packed finite-field screening, and independent
  certificate replay. These compose the core operations for the original
  search problem; they are not the primary algebraic abstraction.
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

## Structured local quotients and cotangent `H1`

The structured local-ideal constructor takes exactly three mathematical
inputs: a polynomial ring `P`, a list `G` of polynomials vanishing at the
origin, and a positive integer `N`. It constructs

```text
J = ideal(G) + maximal_ideal^N.
```

There is no type-dependent reinterpretation of `G` and no implicit choice of
truncation. In particular, `maximal_power=N` means the exponent `N` in this
formula; it is not a matrix-size limit or a search heuristic. The calculation
then follows ordinary objects:

```python
J = P.local_ideal(generators=G, maximal_power=N)
R = J.quotient()
C = R.conormal_module()
d = C.derivative_map()
H1 = d.kernel()
```

This chain is also the internal computation boundary, not just a pleasant
facade. `J.quotient()` builds only `P/J`; `R.conormal_module()` then builds
`P/J^2` and the reduction map; `C.derivative_map()` builds the derivative
matrix; `d.kernel()` builds and ranks the stacked constraint; and
`xi.annihilator()` builds the class-multiplication matrix and its kernel. Each
immutable handle retains its parent data, and repeated traversal of one
quotient context reuses the completed stages. A limit for a later stage cannot
make an earlier operation fail before that stage is requested.

Put `R=P/J` and `B=P/J^2`. The quotient `R` is computed exactly in degrees
less than `N`. When conormal data is requested, the containment
`m^(2N) in J^2` makes `B` an exact finite computation in degrees less than
`2N`. The objects retain the sparse maps

```text
reduction:          B -> R       f |-> f
derivative:         B -> R^e     f |-> df
stacked constraint: B -> R+R^e   f |-> (f, df)
```

and therefore

```text
J/J^2 = kernel(reduction),       H1(L_(R/k)) = kernel(stacked constraint).
```

Equivalently, the usual conormal map
`J/J^2 -> Omega_(P/k) tensor_P R` is the derivative map restricted to the
conormal module. The stacked constraint matrix computes that restricted
kernel without first expanding a large conormal basis.

This is a complete computation of `H1`: `H1.constraint_matrix`, its exact
rank, and `H1.dimension` determine the kernel. Explicit vectors are still
available. `H1.kernel_coordinates(max_coordinate_entries=...)` returns exact
coordinate vectors and `H1.basis(max_coordinate_entries=...)` returns the
corresponding polynomials. These are
separate materialization requests because they can use much more memory. They
either return the complete exact basis or raise a resource-limit error; there
is no dense, numeric, sampled, or partial fallback. The same complete-versus-
materialized distinction applies to `C.basis(max_coordinate_entries=...)`.

Coordinates are stable and inspectable. Ambient monomials are ordered first
by increasing total degree and then by descending lexicographic exponent
tuple. Quotient bases are the nonpivot monomials in that order. Every sparse
matrix acts on column vectors: columns of `C.constraint_matrix`,
`d.ambient_matrix`, and `H1.constraint_matrix` use the standard-monomial basis
of `P/J^2`; reduction rows use the basis of `R`; derivative rows are grouped
by variable, with one `R` block per variable; and `H1.constraint_matrix` stacks
the reduction block before those derivative blocks.

An element and its annihilator are ordinary objects as well:

```python
xi = H1.class_of(g)
ann = xi.annihilator()
colon = ann.preimage()
```

`class_of(g)` requires `g in J` and every partial derivative of `g` to lie in
`J`. Multiplication by the resulting class is the exact linear map
`R -> J/J^2`, `q |-> qg`. Its kernel is
`Ann_R([g]) = (J^2:g)/J`, and `ann.preimage()` is the ideal `J^2:g` in `P`.
Thus the mathematical equivalences are

```text
rank = length(R)  <=>  ann = R.zero_ideal()
                  <=>  (J^2:g) = J.
```

The engine returns those algebraic objects; the calling example decides which
equality expresses its theorem. Version 0.3 retains
`P.cotangent_h1(...).verify_class(g)`, `CotangentClassProof`, and their
`faithful`/`colon_equals_ideal` summary flags as compatibility workflow APIs.
New code should use the object chain above. `audit_cycle(J,g)` likewise remains
available when a separate Groebner/colon replay is wanted for a small general
ideal.

### E10 complete regression

The characteristic-zero E10 example assembles the general structured objects
directly:

```python
import laughableengine as le

P = le.QQ("x1 x2 x3 x4 x5 x6 x7 x8 x9 x10")
xs = P.gens()

F = sum((x**3 for x in xs), P.zero()) + le.elementary_symmetric(xs, 4)
partials = [F.derivative(x) for x in xs]

J = P.local_ideal(generators=partials, maximal_power=4)
R = J.quotient()
C = R.conormal_module()
H1 = C.derivative_map().kernel()

assert R.remainder(F).is_zero()
assert all(R.remainder(partial).is_zero() for partial in partials)
xi = H1.class_of(F)

ann = xi.annihilator()
colon = ann.preimage()

assert R.length == 176
assert R.square_quotient_length == 2728
assert H1.constraint_matrix.shape[1] == R.square_quotient_length
assert C.dimension == 2552
assert H1.dimension == 1873
assert ann.dimension == 0
assert ann.generators == []
assert ann == R.zero_ideal()
assert colon == J
assert len(colon.generators()) == 725

explicit_h1 = H1.basis()
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

| Coefficient field | Presentation build | Class + annihilator | Explicit `H1` basis |
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

Use the same exact-algebra objects over either `QQ` or `GF(p)`:

```python
import laughableengine as le

P = le.GF(5, "x y")
x, y = P.gens()
I = P.ideal([x**2, y**3])

assert I.is_zero_dimensional()
assert I.quotient_dimension() == 6
assert I.normal_form(x**3 + y) == y

K = I.square().colon(x)
assert x**3 in K
```

### Optional research workflows

The packed finite-field screen and candidate-parallel drivers remain available
for the original search workload. A reported hit is colon-certified unless
`certify_hits=False` is explicitly requested:

```python
P = le.GF(5, "x")
(x,) = P.gens()
J = P.ideal([x**5])

screen = P.screen_full_h1(J)
assert screen.status == "complete"
assert screen.maximum_individual_rank == 1
assert screen.hit

candidates = [x**5, P.zero(), x**6] * 100
records = P.screen_cycles_parallel(
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

The CLI exposes the algebraic primitives first and retains the original
research commands for compatibility. Representative commands are:

```sh
./build/laughable --field QQ --vars x,y --order lex \
  gb 'x^2-y' 'x*y-1'

./build/laughable --field QQ --vars x,y \
  nf 'x^3+y' --by 'x^2' --by 'y^3'

./build/laughable --field QQ --vars x,y \
  colon --g x 'x^2' 'x*y' 'y^3'

./build/laughable --field 'GF(5)' --vars x \
  cotangent-h1 --maximal-power 5

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

## Optional workflow certificate verifier

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
`P.make_jg_certificate(generators, g)`.

## C++ and CMake

```cpp
#include <laughableengine/laughableengine.hpp>

using namespace laughableengine;

int main() {
  auto ring = make_ring(QQ(), {"x", "y"}, Order::Grevlex);
  auto x = ring.gen("x");
  auto y = ring.gen("y");
  Ideal<RationalField> ideal(ring, {x.pow(2), y.pow(3)});
  return ideal.quotient_dimension() == 6 ? 0 : 1;
}
```

Installed consumers use:

```cmake
find_package(laughableengine CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE laughableengine::laughableengine)
```

The umbrella header exposes exact fields, polynomials, ideals, quotients,
matrices, structured local-quotient objects, and their operations. It also
includes the optional packed discovery types, candidate executor,
inverse-system driver, reconstruction helpers, and JSON certificate writer.

## Regression coverage

The native suite checks the general polynomial/ideal kernel, exact linear
algebra, structured local-quotient objects, and the current research
workflows. The larger vertical regressions include:

| Case | Exact result |
|---|---|
| Supplied three-variable seed | `length(Q)=11`, `length(P/J^2)=48`, `dim(H1)=19`, socle `3`, maximum individual rank `1` |
| `G=x^3+y^7+x*y^5` Tjurina ideal | lengths `11/34`, `dim(H1)=15`, socle `2`, maximum rank `1`, distinguished class has nonzero annihilator |
| `GF(5)`, `J=(x^5)` | the annihilator of `[x^5]` is the zero ideal |
| Characteristic-zero homogeneous example | socle action is zero |
| Non-origin quotient `(x^2-x)` | rejected by the full-`H1` origin-support gate |

The supplied seed is also compared at `GF(101)` and `GF(103)` before exact
`QQ` certification. Matrix-space results distinguish proven rank, proven
full-column-rank witnesses, generic-only bounds, and resource exhaustion.
`Ann_Q(H1)` is diagnostic only and is never substituted for the required
individual-witness test.

## Current research-workload performance on Apple Silicon

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

- This is a growing exact-algebra core, not yet a general-purpose CAS.
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

# laughableengine design

## The short version

Make simple things obvious and hard things possible.

The surface should feel Pythonic: short names for ordinary mathematical
actions, keyword arguments where a bare number would be ambiguous, useful
value objects, and errors that say what input was rejected. The core should
feel Haskell-esque: explicit data flow, deterministic pure-looking
computations, immutable results, and types that prevent unrelated contexts
from being mixed.

This is a direction for public API design, not an excuse to imitate either
language mechanically in C++.

## One name, one job

An algorithmic name has one mathematical meaning. We do not copy the
Macaulay2 pattern in which the same short command can mean unrelated things
depending on a complicated dispatch lattice.

```python
H = P.cotangent_h1(generators=partials, maximal_power=4)
proof = H.verify_class(F)
```

Here `maximal_power=4` always defines
`J = ideal(generators) + maximal_ideal**4`. It is never reinterpreted as a
degree bound, search depth, or performance option. Resource limits have
separate, descriptive names.

Conventional arithmetic may use `+`, `-`, `*`, `/`, and `**`. Substantial
algorithms use named functions or methods. A Python binding may dispatch once
on `QQ` versus `GF(p)` at its boundary, but the public operation and return
contract do not change with the field.

## Context is real data

A polynomial belongs to one exact ring object. Equal-looking polynomials from
separately constructed rings are not silently mixed. Coercion is explicit,
and there is no global current ring, current monomial order, or current
characteristic.

Input records say what a computation needs. Result records retain the evidence
needed to inspect or replay the answer. Negative states and resource limits
are explicit; they are not encoded as an empty list, a magic rank, or a
plausible `False`.

## Complete does not mean eagerly expanded

Large algebra is often most naturally represented by a finite matrix, not by
printing thousands of polynomials. For the structured cotangent calculation,
`h1_relation_matrix` plus its exact rank is a complete presentation of the
kernel and determines `h1_dimension`.

When explicit vectors are needed, ask for them:

```python
coordinates = H.h1_kernel_coordinates()
basis = H.h1_basis()
```

These calls are resource-metered. They either return exact coordinates and
polynomials or raise a clear resource-limit error. They never switch to a
dense fallback, floating-point approximation, random sample, or partial basis.

The same rule applies to class verification. The proof exposes the exact
multiplication matrix, `annihilator_basis`, and `colon_generators`; a Boolean
`faithful` is a summary of that evidence, not a replacement for it.

## Coordinates are part of the contract

Every matrix acts on column vectors. Every basis order is documented and
stable. For truncated local calculations, monomials are ordered by increasing
total degree and then descending lexicographic exponent tuple. Quotient bases
are the nonpivot monomials in that order.

We do not hide transposes, silently reorder a basis for presentation, or call
both rows and columns “generators.” A coordinate vector always names its basis
and a matrix accessor documents its source and target.

## Exact means exact

Certification uses `QQ` or `GF(p)` arithmetic, exact sparse elimination, and
deterministic pivot choices. A resource limit produces an inconclusive error,
not a guessed rank. Packed finite-field discovery is allowed to be specialized
and fast, but every claimed characteristic-zero hit returns to the exact
certification path.

The exact polynomial representation supports up to sixteen variables. The
packed discovery representation has six exponent lanes. They are separate
types with separate limits; an exact ten-variable job must not fail because a
packed six-variable optimization exists elsewhere.

## Optimize the representation before the abstraction

This engine is intentionally narrow. Fixed quotients should compile rewrite
tables and reduce batches. Structured ideals such as `J=(G)+m^N` should use
finite truncated row spaces. Candidate searches should reuse fixed state and
parallelize at the candidate level.

We add a general algebraic abstraction only when a real supported workflow
needs it. There is no scheme layer, derived-category facade, general
resolution engine, or compatibility language pretending to be Sage or
Macaulay2.

## Low-level access stays available

The pleasant high-level path must not trap an expert. Presentations expose
bases, sparse matrices, kernel coordinates, normal forms, annihilator
generators, and resource controls. Simple code stays short; difficult audits
can descend all the way to exact coordinates without leaving the engine.

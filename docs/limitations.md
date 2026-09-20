# Scope and limitations

This package is a research implementation of TT compression and feasibility repair for discrete obstacle problems. It is **not a complete PDE solver**. In particular, it does not provide a scalable obstacle iteration, adaptive meshes, a general nonlinear PDE interface, or a proof that an arbitrary PDE solution has small TT ranks.

## Discrete feasibility and floating-point arithmetic

The mathematical envelopes dominate the residual in exact arithmetic. The code uses ordinary NumPy floating-point operations, without interval arithmetic or directed rounding. Tests and diagnostics can reveal failures; they do not certify all tensor entries in a large problem.

`repair_feasible` and `feasible_compress` assume the original tensor is feasible. They do not search all entries to establish that assumption. The output dominates the original under the selected method's mathematical hypotheses. Rounding or modifying that output again can destroy feasibility unless a new repair is performed.

The constraint is imposed on interior nodal coefficients. With a Q1 obstacle represented in the same positive nodal basis and compatible boundary data, nodal dominance implies dominance over that interpolated obstacle. It does not by itself control a different continuous obstacle or the discretization error of the PDE.

## Tolerances, ranks, and energy are different controls

- `TensorTrain.round(rtol=...)` controls Frobenius error; the initial rounding in `feasible_compress` controls mass error.
- A binding `max_rank` takes precedence over the rounding tolerance. Compression diagnostics report whether the initial mass tolerance was met.
- Projection and repair can increase ranks beyond that cap. The final tensor can use more storage than the original.
- A small mass or Frobenius error need not be a small energy error. `RepairResult.error_bound` uses measured energy norms and bounds error relative to the supplied original tensor.

The quadratic correction can have bond rank $1+r(r+1)/2$ when the residual rank is $r$. Residual ranks themselves can include both original and candidate ranks. A feasibility construction is therefore not a guarantee of successful compression.

The finite local matrices in `Q1Operator` are dense NumPy arrays. Core-only tensor arithmetic avoids the full tensor, but does not make arbitrarily large one-dimensional eigendecompositions inexpensive.

## Structural assumptions must actually hold

Small TT rank does not imply small interaction degree, and degree truncation generally does not preserve nonnegativity. A numerical degree check with tolerance is not an exact membership certificate.

The dimension-independent quadratic energy estimate requires a bound on the **entire** centered local function space, a bounded interaction degree, controlled mass comparison, and the graph sign condition on $K_k-\lambda_{k,1}M_k$. `Q1Operator` accepts arbitrary SPD local factors for energy computations; this acceptance does not verify those additional assumptions. See [the precise hypotheses](mathematics.md#conditional-energy-bound-independent-of-dimension).

Likewise, `structure_certified=True` is a caller declaration for the controlled rank-one envelope. The code cannot infer the intended local spaces from arbitrary TT cores. Supplying an unjustified $\kappa$ can invalidate domination itself.

Low-degree data also need not give a low-degree solution with uniformly small error **relative to the solution**. The archive contains a Q1 family where almost all solution energy lies in growing interaction orders, although the solution itself has TT rank one. Data-relative approximation estimates remain compatible with that example. See [the explicit solution obstruction](../theory/solution_obstructions_research.md) and [the positive approximation results](../theory/SOLUTION_APPROXIMATION_RU.md).

## Why a positive repair can be expensive

A concrete parity example shows a limitation of the nonnegative corrections used here. Take two interior nodes per coordinate, with

$$
K=3\begin{pmatrix}2&-1\\-1&2\end{pmatrix},\qquad
M=\frac1{18}\begin{pmatrix}4&1\\1&4\end{pmatrix}.
$$

Let $B_0=(1,1)^{\otimes d}$, $Z=(1,-1)^{\otimes d}$, and $U=B_0+\varepsilon Z$, where $0<\varepsilon<1$. Rank-one TT-SVD returns $V=B_0$. Set the obstacle to $\psi=U$ and the load to $f=AU-B_0$; then $U$ is the exact discrete obstacle solution with a strictly positive contact multiplier.

For $d\ge4$, every **nonnegative correction** $D\ge0$ making $W=V+D\ge U$ obeys

$$
\frac{\|W-U\|_A}{\|V-U\|_A}
\ge\frac12\sqrt{1+\frac13(5/3)^{d-1}}.
$$

The factor grows exponentially even when $\varepsilon$ is small. This is a lower bound for this correction class, **not for all TT methods**: the signed rank-one correction $D=\varepsilon Z$ recovers $U$ exactly, and $U$ has TT rank two. The obstacle is deliberately constructed for the counterexample; no claim is made that this behavior is typical of applications. See [the parity proof, §7](../theory/rounding_geometry.md).

The unconditional core envelope therefore has no general uniform energy guarantee. The quadratic envelope also needs its stated structural hypotheses for a uniform energy bound. A tiny high-order tail in an ordinary norm is insufficient on its own; see [the tail obstruction](../theory/rounding_limits_literature.md).

## Dense reference tools and certificates

`solve_obstacle` enumerates active sets and is only a small validation reference. Its unknown-count guard limits exponential work. KKT acceptance uses a tolerance, so its raw returned arrays may contain small feasibility violations; they are not clipped.

`energy_error_bound` requires actual componentwise primal and dual feasibility and rejects even small negative slacks or multipliers. Its output is a floating-point evaluation of a valid mathematical estimate, not an interval-certified upper bound. It concerns the fixed discrete problem and excludes PDE discretization error.

Explicit TT materialization is guarded by a tensor-entry count. `Q1Operator.to_dense` and `mass_dense` are guarded by a **matrix-entry count**, the square of the number of unknowns. Dense reference and certificate functions accept already materialized matrices; they are not tensor-scalable algorithms.

The [archived theory](../theory/RESEARCH_INDEX_RU.md) contains additional conditional results that are not all implemented as solvers. Internal audits and successful tests are not external peer review, and scientific priority has not been established.

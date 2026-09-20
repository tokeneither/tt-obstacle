# API reference

All public names below are importable from `tt_obstacle`, unless stated otherwise. Arrays are real, finite `float64` values; coordinate indices are zero based. This package provides TT approximation and repair operations for a fixed discrete problem. It does not provide a scalable obstacle PDE solver or a discretization-error estimator.

## TensorTrain

```python
import numpy as np
from tt_obstacle import TensorTrain

x = TensorTrain.rank_one([np.array([1.0, 2.0]), np.array([-1.0, 3.0])])
y = x.add(x.scaled(0.5)).round(rtol=1e-12)
assert abs(y.evaluate((1, 0)) + 3.0) < 1e-12
```

`TensorTrain(cores)` copies a nonempty sequence of three-dimensional cores with shapes `(r_left, mode_size, r_right)`. Adjacent ranks must agree; exterior ranks are one. Public core arrays are read-only. Arithmetic returns new objects.

| Member | Meaning |
| --- | --- |
| `rank_one(factors)` | Class method constructing the outer product of one vector per coordinate. |
| `zeros(shape)` | Class method constructing a zero tensor with positive core dimensions. |
| `from_dense(array, *, rtol=0.0, max_rank=None)` | Class method using TT-SVD on an already dense input. |
| `shape`, `ranks`, `storage_size`, `cores` | Mode sizes, **internal stored bond dimensions**, number of stored scalars, and core tuple. |
| `evaluate(indices)` | Evaluate one entry without materializing the tensor. |
| `norm()` | Frobenius norm computed through QR orthogonalization. |
| `copy()`, `scaled(scalar)` | Copy or scalar multiplication. |
| `add(other)`, `subtract(other)` | Exact-arithmetic TT addition/subtraction; `+` and `-` are also supported. |
| `hadamard(other)` | Entrywise product; bond dimensions multiply. |
| `square()` | Entrywise square; each stored bond `r` becomes `r*(r+1)//2`. |
| `mode_product(matrices)` | Apply one matrix of shape `(new_size, old_size)` per coordinate; `None` means identity. |
| `scale_modes(factors)` | Entrywise multiplication by an outer product of local vectors. |
| `round(*, rtol=0.0, max_rank=None)` | QR/SVD rounding on cores, with a relative **Frobenius-norm** target. |
| `to_dense(max_entries=1_000_000)` | Explicit conversion, rejected when the tensor's number of entries exceeds the guard. |

For `from_dense` and `round`, `max_rank` is either a positive integer or a sequence of positive caps, one per internal bond. A cap takes precedence over `rtol`, so the requested accuracy may not be achieved. Without a binding cap, the usual TT-SVD/rounding Frobenius estimate applies up to roundoff.

Stored bond dimensions need not be minimal tensor ranks. Addition, squaring, and interaction projection can leave redundant directions. Rounding may remove them, but ordinary rounding does not preserve an obstacle constraint.

## Q1Operator and local assembly

```python
from tt_obstacle import Q1Operator, assemble_p1

op = Q1Operator.uniform((3, 4), lengths=(1.0, 2.0))
b = op.ground_tensor
assert abs(op.mass_norm(b) - 1.0) < 1e-12

K, M = assemble_p1([0.0, 0.15, 0.6, 1.0])
nonuniform = Q1Operator((K,), (M,))
```

`Q1Operator(stiffness, mass)` represents

$$
A=\sum_k K_k\otimes M_{-k},\qquad \mathcal M=\bigotimes_k M_k.
$$

Local factors are dense symmetric positive-definite matrices, with matching sizes in each coordinate. Small asymmetry within the validation tolerance is symmetrized. General SPD factors are accepted for mass and energy operations; accepting them does **not** verify every hypothesis of the Q1 repair theory.

| Function/member | Meaning |
| --- | --- |
| `assemble_p1(nodes)` | Return `(K, M)` for a strictly increasing one-dimensional mesh, including both endpoints and at least one interior node. Homogeneous Dirichlet boundary nodes are eliminated; mass is consistent. |
| `uniform_p1(n, length=1.0)` | Return local matrices for `n` interior nodes. |
| `generalized_eigh(K, M)` | Return increasing generalized eigenvalues and **M-orthonormal columns**. |
| `Q1Operator.uniform(shape, lengths=None)` | Build one uniform local mesh per coordinate; default interval lengths are one. |
| `shape`, `dimension`, `stiffness`, `mass` | Tensor shape, number of coordinates, and tuples of local matrices. |
| `eigenvalues`, `eigenvectors` | Per-coordinate eigensystems; column zero is the first eigenvector. |
| `ground_factors`, `ground_eigenvalues`, `ground_tensor` | Local normalized positive ground states, their eigenvalues, and their TT product `B`. |
| `projectors` | Local ground-state projectors `np.outer(b, M @ b)`. |
| `diagonal_mass_factors` | Vectors `d_k = (M_k @ b_k) / b_k`. |
| `comparison_constant` | Computed common `c` satisfying `c*diag(d_k) <= M_k` in the positive-semidefinite order. |
| `mass_inner(x, y)`, `energy_inner(x, y)` | TT contractions for the bilinear forms. |
| `mass_norm(x)`, `energy_norm(x)`, `diagonal_mass_norm(x)` | Stable whitening/QR norms; no global matrix is assembled. |
| `mass_dense(max_entries=1_000_000)`, `to_dense(max_entries=1_000_000)` | Explicit global mass/stiffness matrices. The guard counts **matrix entries**, i.e. `prod(shape)**2`. |

Ground-state methods reject factors without strictly positive first eigenvectors or with negative mass entries. Computed eigenvalues, comparison constants, and norms are floating-point quantities, not interval certificates.

## Interaction and spectral projections

| Function | Operation |
| --- | --- |
| `project_degree(tensor, projectors, degree)` | Retain interaction components with at most `degree` non-ground coordinates. The counting MPO increases each bond by at most `degree + 1`; `degree >= dimension` returns a copy. |
| `project_coordinates(tensor, projectors, keep)` | Retain the listed coordinate indices and apply local ground projectors elsewhere. |
| `project_spectral(tensor, operator, num_modes)` | Retain up to `num_modes` local generalized eigenvectors; **the ground state counts as one**. |
| `spectral_feature_bound(operator, num_modes)` | Compute the whole-space local ratio bound `kappa` for the specified spectral spaces. It does not check that an arbitrary tensor lies in those spaces. |
| `interaction_amplification_bound(degree, kappa, comparison)` | Evaluate the conditional quadratic-envelope energy constant for `degree >= 1`. It does not certify its structural assumptions. |

With `operator.projectors`, interaction projections are orthogonal in the Q1 energy and mass norms. Degree projection and spectral projection generally **do not preserve nonnegativity**. Coordinate projection preserves nonnegativity with the positive ground projectors, but does not preserve an arbitrary nonconstant obstacle: apply it to a nonnegative gap and add the obstacle back when appropriate.

## Envelopes and repair

`quadratic_envelope(residual, operator)` returns

$$
E=\frac{R^2/B+\sigma^2 B}{2\sigma},\qquad
\sigma=\|R\|_{\otimes\operatorname{diag}(d_k)},
$$

with zero returned for a zero residual. Its exact-arithmetic domination `E >= abs(R)` needs no low-interaction assumption. Dimension-independent **energy estimates** require the additional hypotheses in the theory.

`core_envelope(residual)` gives an unconditional rank-one envelope. It first left-orthogonalizes the TT, then multiplies spectral norms of local matrix slices. This construction has no general dimension-independent energy bound. `tt_obstacle.repair.rank_one_envelope` is a module-level alias; it is not exported at package level.

```text
controlled_rank_one_envelope(
    residual, operator, degree, kappa, *, structure_certified=False
)
```

This returns `sqrt(K_p)*sigma*B`, where `K_p = sum(comb(d, j)*kappa**(2*j), j=0..min(p,d))`. It requires `structure_certified=True`, meaning the **caller asserts** both the interaction bound and membership of `R/B` in centered local spaces with the whole-space `L-infinity/L2(mu_k)` bound `kappa`. The local probabilities are `mu_k = b_k*(M_k @ b_k)`. A relative `1e-10` mass-norm degree check is performed; local-space membership is not checked. Neither the flag nor the numerical check is a rigorous certificate.

```text
repair_feasible(
    original, candidate, operator, method="quadratic",
    degree=None, kappa=None, *, structure_certified=False
)
```

Methods are `"quadratic"`, `"core"` (`"rank_one"` is an alias), and `"controlled_rank_one"`. With `R = original - candidate`, the result is `candidate + E(R)`. It dominates `original` in exact arithmetic under the chosen method's assumptions. If the caller knows `original >= obstacle`, this preserves feasibility; the function does not check the original against an obstacle.

`RepairResult` contains:

- `tensor`, `candidate`, `correction`, `residual`;
- `raw_error = ||residual||_A`, `correction_energy = ||correction||_A`, and `error_bound = raw_error + correction_energy`;
- `original_storage`, `candidate_storage`, `result_storage`, `method`, and `assumptions`;
- optional rounding fields described below.

The reported energy bound is a directly evaluated triangle-inequality estimate, not an automatic application of the structural theorem. There is no guarantee that repair reduces storage.

## Feasible compression

```python
from tt_obstacle import feasible_compress

op = Q1Operator.uniform((3, 3, 3))
original = op.ground_tensor.add(
    TensorTrain.rank_one([np.ones(3)] * 3).scaled(0.01)
)
result = feasible_compress(original, op, rtol=1e-6, max_rank=1, method="core")
print(result.rounding_mass_error, result.rounding_tolerance_met)
print(result.original_storage, result.result_storage)
```

```text
feasible_compress(
    original, operator, rtol=1e-8, max_rank=None, degree=None,
    method="quadratic", kappa=None, *, structure_certified=False
)
```

The sequence is local mass whitening by `chol(M_k).T`, TT rounding, inverse whitening, optional degree projection, then repair. `rtol` targets the initial **mass-norm rounding error**, not energy error or the final repaired error. `max_rank` caps **only the initial rounded candidate**. Projection and repair may increase its ranks.

When `degree` is provided, the input is first checked for that degree to relative mass tolerance `1e-10`; an input that fails raises `ValueError`. It is not silently projected into the assumed class.

The returned `RepairResult` additionally records:

| Field | Meaning |
| --- | --- |
| `rounding_mass_error` | Actual mass error before degree projection and repair. |
| `rounding_tolerance` | `rtol * operator.mass_norm(original)`. |
| `rounding_tolerance_met` | Whether that target was reached, allowing a small floating-point floor. A binding rank cap can make it false. |
| `rounding_ranks` | Internal stored ranks immediately after rounding. |

For controlled rank-one repair, all structural assumptions still apply to the residual. This example constructs membership in a known spectral class:

```python
from tt_obstacle import spectral_feature_bound

op = Q1Operator.uniform((3, 3, 3))
original = op.ground_tensor
for k in range(op.dimension):
    factors = list(op.ground_factors)
    factors[k] = op.eigenvectors[k][:, 1]
    original = original.add(TensorTrain.rank_one(factors).scaled(0.05))

result = feasible_compress(
    original, op, rtol=1e-6, max_rank=1, degree=1,
    method="controlled_rank_one",
    kappa=spectral_feature_bound(op, num_modes=2),
    structure_certified=True,
)
```

None of the TT arithmetic, projections, envelopes, or compression functions materializes a full physical tensor. They still use dense local matrices and TT cores, whose sizes may grow substantially.

## Small dense reference and a posteriori estimates

`solve_obstacle(A, f, psi, max_unknowns=16, tol=1e-10)` exhaustively enumerates active sets for

$$
\min_{u\ge\psi}\tfrac12u^TAu-f^Tu.
$$

It accepts a dense SPD matrix and one-dimensional vectors and returns `ReferenceSolution(solution, multiplier)`, with the raw multiplier `A @ solution - f`. Complexity is exponential; the unknown-count guard is checked before factorization/enumeration. KKT acceptance uses the supplied absolute tolerance, so returned arrays can contain small feasibility violations. Arrays are not clipped.

`energy_error_bound(A, f, psi, candidate, multiplier, tol=1e-12)` returns `ErrorBound(residual_norm, complementarity, sharp_bound, gap_bound)`. For a primal/dual feasible pair,

$$
r=\|Aw-f-\lambda\|_{A^{-1}},\quad c=\lambda^T(w-\psi),\qquad
\|w-u_*\|_A\le\frac{r+\sqrt{r^2+4c}}2,\quad
\|w-u_*\|_A\le\sqrt{r^2+2c}.
$$

Actual array entries must satisfy `candidate >= psi` and `multiplier >= 0`; even a tiny negative value raises `ValueError`. Here `tol` controls matrix-symmetry validation only. The routine receives already dense arrays, performs a dense Cholesky solve, and has no separate size guard.

```python
from tt_obstacle import solve_obstacle, energy_error_bound

op = Q1Operator.uniform((2, 2))
A = op.to_dense(max_entries=16)
psi = np.zeros(4)
f = -np.ones(4)
reference = solve_obstacle(A, f, psi)
candidate = reference.solution + 0.05
bound = energy_error_bound(A, f, psi, candidate, reference.multiplier)
print(bound.sharp_bound, bound.gap_bound)
```

These formulas concern the fixed discrete obstacle problem. Their implementation uses ordinary floating-point arithmetic: the output is a diagnostic mathematical-bound evaluation, not an interval-certified guarantee or a bound on continuum PDE error.

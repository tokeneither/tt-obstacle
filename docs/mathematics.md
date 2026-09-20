# Mathematical model and guarantees

`tt-obstacle` implements tensor-train (TT) approximation and feasibility repairs for a **fixed discrete obstacle problem**. The statements below hold in exact arithmetic. The implementation evaluates them in floating point; see [limitations](limitations.md).

## Discrete obstacle problem

For tensor-product Q1 elements with homogeneous Dirichlet boundary conditions, let

$$
A=\sum_{k=1}^d K_k\otimes M_{-k},\qquad
\mathcal M=\bigotimes_{k=1}^d M_k,
$$

where $K_k,M_k$ are the interior one-dimensional P1 stiffness and consistent mass matrices. The Kronecker factors retain their coordinate order. The nodal problem is

$$
u_* = \arg\min_{u\ge\psi}
\left(\tfrac12u^TAu-f^Tu\right).
$$

Here $f$ is the assembled load vector and the inequality is componentwise at interior nodes. If a coefficient vector $F$ represents the load through the mass matrix, then $f=\mathcal M F$. These are different inputs.

Positive definiteness gives a unique minimizer, characterized by

$$
u_*\ge\psi,\qquad \lambda=Au_*-f\ge0,\qquad
\lambda\odot(u_*-\psi)=0.
$$

The norms used by the package are $\|x\|_A=(x^TAx)^{1/2}$, $\|x\|_{\mathcal M}$, and the Euclidean/Frobenius norm. They are not interchangeable. `TensorTrain.round` controls the Frobenius norm; `feasible_compress` first whitens by local mass Cholesky factors, so its initial rounding tolerance is in the **mass norm**.

## Ground state and positive envelopes

Choose positive first generalized eigenvectors

$$
K_kb_k=\lambda_{k,1}M_kb_k,\qquad b_k^TM_kb_k=1,
\quad B=\bigotimes_k b_k,
$$

and define

$$
\widehat M_k=\operatorname{diag}\!\left(\frac{M_kb_k}{b_k}\right),
\quad \widehat{\mathcal M}=\bigotimes_k\widehat M_k,
\quad \mu_k(i)=b_k(i)(M_kb_k)(i).
$$

For standard P1 matrices, $\widehat M_k$ is positive, $\sum_i\mu_k(i)=1$, and $\widehat M_kb_k=M_kb_k$. `Q1Operator` exposes the ground factors, projectors and diagonal mass factors.

Given residual $R=U-V$, put $\sigma=\|R\|_{\widehat{\mathcal M}}$. For $R\ne0$, the quadratic envelope is

$$
E=\frac{1}{2\sigma}\left(\frac{R\odot R}{B}+\sigma^2B\right).
$$

All divisions are entrywise. The scalar inequality $a^2+s^2\ge2s|a|$ gives $E\ge|R|$. For a zero residual, take $E=0$. **This domination needs no low-interaction or local spectral assumption.**

`quadratic_envelope` evaluates this formula on TT cores. Symmetric-square bond bases give

$$
r_k(E)\le 1+\frac{r_k(R)(r_k(R)+1)}2.
$$

The bound is on the correction; adding it to the candidate can increase ranks further. The construction and its energy analysis are proved in [the interaction theory, §§2–5](../theory/INTERACTION_THEORY_RU.md).

## Repair error versus error of the obstacle solution

`repair_feasible(U, V, operator)` returns $W=V+E$. Since $E\ge|U-V|$,

$$
W\ge U,\qquad
\|W-U\|_A\le\|U-V\|_A+\|E\|_A.
$$

Thus an input known to satisfy $U\ge\psi$ remains feasible after repair. `RepairResult.raw_error`, `correction_energy`, and `error_bound` report the two terms and their sum. This **direct triangle estimate** uses computed norms and does not invoke the conditional theorem below. It bounds error relative to the supplied original $U$, not automatically relative to $u_*$.

For a separate estimate relative to $u_*$, `energy_error_bound` takes a feasible candidate $w$ and any nonnegative multiplier $\lambda$. Define

$$
r=\|Aw-f-\lambda\|_{A^{-1}},\qquad
c=\lambda^T(w-\psi).
$$

Then

$$
\|w-u_*\|_A\le\frac{r+\sqrt{r^2+4c}}2
\le\sqrt{r^2+2c}.
$$

This small, dense diagnostic requires actual primal and dual feasibility; its tolerance does not permit negative slacks or multipliers. See [the certificate proof](../theory/certificates_convergence.md).

## Conditional energy bound independent of dimension

Define $P_k=b_kb_k^TM_k$, $Q_k=I-P_k$. A tensor has interaction degree at most $p$ when its expansion using $P_k,Q_k$ contains at most $p$ active $Q_k$ factors per term. This condition is different from having small TT ranks.

For the quadratic envelope, the archived theorem assumes all of the following:

- $R/B$ belongs to a product of local spaces $\operatorname{span}\{1\}\oplus\mathscr V_k$, where $\mathscr V_k$ is centered under $\mu_k$.
- The residual has interaction degree at most $p\ge1$.
- Every function in each local space satisfies $\|z\|_\infty\le\kappa\|z\|_{L^2(\mu_k)}$, with $\kappa\ge1$. Checking individual basis vectors with the same constant is insufficient.
- A common $c>0$ satisfies $c\widehat M_k\preceq M_k\preceq\widehat M_k$.
- The shifted matrices $L_k=K_k-\lambda_{k,1}M_k$ have nonpositive off-diagonal entries. This graph condition holds for standard P1 stiffness and mass matrices; arbitrary SPD factors accepted by `Q1Operator` need not satisfy it.

Under these hypotheses,

$$
\boxed{\quad
\|E\|_A\le C_{p,\kappa,c}\|R\|_A,
\qquad C_{p,\kappa,c}
=\sqrt{1+\kappa^2}\,(\kappa+3)^{p-1/2}c^{-p/2}.
\quad}
$$

The constant has no explicit $d$, but uniformity requires fixed $p,\kappa$ and a uniform positive lower bound on $c$. The proof combines a comparison of consistent and diagonal mass on active coordinates with a product-space fourth-moment estimate. See [the full proof](../theory/INTERACTION_THEORY_RU.md).

`interaction_amplification_bound` only evaluates this constant. It does not certify the hypotheses. `spectral_feature_bound` computes a whole-space $\kappa$ for the specified local eigenmode spaces; it does not prove that a supplied residual belongs to those spaces.

## Rank-one alternatives

`core_envelope` first left-orthogonalizes the TT. For its resulting cores $G_k$, it forms

$$
E(i_1,\ldots,i_d)=\prod_{k=1}^d\|G_k(:,i_k,:)\|_2.
$$

Matrix-norm submultiplicativity proves $E\ge|R|$. This is an unconditional rank-one envelope, but there is **no general dimension-independent energy bound** for it. See [the core-envelope proof](../theory/envelopes_rank.md).

With the asserted local-space and interaction hypotheses, `controlled_rank_one_envelope` instead uses

$$
E_1=\sqrt{K_p}\,\sigma B,\qquad
K_p=\sum_{j=0}^{\min(p,d)}\binom dj\kappa^{2j}.
$$

The evaluation-kernel bound proves domination. Its energy price is at most $\sqrt{K_p}c^{-p/2}$, which can grow as $d^{p/2}$ for fixed $p$. `structure_certified=True` records a caller assertion, not a certificate produced by the package. The degree check is numerical. See [the rank-one analysis, §6.2](../theory/INTERACTION_THEORY_RU.md).

These are research constructions. Scientific priority of the combined results has not been established; the archive records proofs, audits, and related work rather than a peer-reviewed claim of novelty.

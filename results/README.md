# Recorded examples

These small outputs were generated from this source tree with NumPy 2.4.6.
They record correctness and tradeoff diagnostics, not performance claims.
Use the [reproduction commands](../docs/getting_started.md) to regenerate them.

- `compression.json` and `compression.png`: mixed-contact 3 × 3 nodal obstacle problem.
- `rank_energy_tradeoff.json`: an eight-coordinate constructed-input experiment.
- `dimension_sweep.json`: the same construction in 4, 8, and 16 coordinates.
- `failure_cases.json`: parity obstruction for nonnegative repairs, up to 32 coordinates.
- `validation.json`: local packaging and test checks for this release.

The dense obstacle reference is a numerical solution checked to a tolerance.
The high-dimensional examples do not solve a PDE; they start with explicitly
constructed TT inputs. Sampled inequalities are not global certificates.

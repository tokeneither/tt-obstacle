# Getting started and reproducing results

Use Python 3.10 or newer. NumPy is the only runtime dependency; Matplotlib is
optional. Run the following commands from the extracted repository root.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Windows PowerShell activation is `.venv\Scripts\Activate.ps1`.
For development, replace the last command with
`python -m pip install -e ".[plots]"`. For a regular installation with plotting,
use `python -m pip install ".[plots]"`. No SciPy or pytest is required.

Check the installed package:

```bash
python -c "import tt_obstacle; print(tt_obstacle.__version__)"
python -m unittest discover -s tests -v
```

## Examples

```bash
python examples/compression_and_repair.py --output results/compression.json
python examples/rank_energy_tradeoff.py --dimension 8 --output results/rank_energy_tradeoff.json
python examples/failure_cases.py --output results/failure_cases.json
python experiments/dimension_sweep.py --output results/dimension_sweep.json
```

The first example independently solves a small obstacle quadratic program,
then evaluates raw rounding and repaired approximations. The second constructs
a sum of positive tensor products and works directly with TT cores. Its
sampled inequalities are diagnostics, not global numerical certificates.
The third checks a parity-family obstruction to dimension-uniform energy
control for **nonnegative corrections**. Signed corrections can restore that
input exactly, so it is not a lower bound for all TT methods.

With the plotting extra installed:

```bash
python examples/compression_and_repair.py --output results/compression.json --plot results/compression.png
```

The scripts overwrite the requested output files. JSON records include the
package and NumPy versions, structural parameters, and deterministic seeds
where random construction is used. Floating-point values can vary slightly
across NumPy/BLAS versions. Stored ranks can also vary near numerical rank
thresholds; no performance benchmark is claimed.

## Interpreting the diagnostics

`rounding_tolerance_met` concerns the initial mass-norm rounding target. A
binding `max_rank` may make it false even when the final tensor satisfies the
obstacle. `error_bound` is the computed triangle bound for energy error against
the input tensor; it does not include PDE discretization error.

`stored_floats` counts all core coefficients. It is not the number of degrees
of freedom in a minimal representation. Repairs can increase ranks and
storage. Applying ordinary rounding again to the repaired result can destroy
its inequality, which is why the package does not silently do so.

The [API reference](api.md) explains structural assertions, shape guards, and
norms. The [mathematical overview](mathematics.md) gives the relevant hypotheses.
Independent scripts in [research/audits](../research/audits/README.md) check
additional theory that is not all exposed as package algorithms.

# Independent numerical audits

These scripts were developed alongside the research notes and intentionally
use independent reference calculations. They are distinct from the package
unit tests. They need Python and NumPy and do not require an installed
`tt_obstacle` package.

From the repository root:

```bash
python research/audits/verify_theory.py
python research/audits/verify_spectral.py
python research/audits/verify_multidim.py
python research/audits/verify_interaction.py
python research/audits/verify_solution_structure.py
python research/audits/verify_contact_approximation.py
```

The first five scripts overwrite the adjacent result JSON files. The contact
approximation script prints its JSON to standard output; to save it, use:

```bash
python research/audits/verify_contact_approximation.py > research/audits/contact_approximation_results.json
```

The checks cover envelope domination, energy identities, interaction
projections, actual rounding residuals, primal–dual estimates, contact
stability, and counterexamples. Most use small dense references; some
structured cases avoid dense tensor assembly.

Recorded values are floating-point diagnostics. They do not certify all
inputs, establish scientific priority, or benchmark a scalable PDE solver.
See the [research guide](../../theory/README.md) for the associated proofs and
the [package examples](../../docs/getting_started.md) for the public API.

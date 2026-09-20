# Guide to the research notes

These working notes accompany the v0.1 implementation. Most of the detailed
derivations are in Russian. They contain exact-arithmetic arguments and
internal reviews; they have not been externally peer reviewed, and novelty
has not been established.

Start with the [package mathematical overview](../docs/mathematics.md) for
the implemented constructions and their assumptions. The
[Russian research index](RESEARCH_INDEX_RU.md) lists all individual files.

| Topic | Main note | Supporting material |
| --- | --- | --- |
| Basic obstacle setting, envelopes, error estimates | [Initial note](RESEARCH_NOTE_RU.md) | [Envelopes and rank](envelopes_rank.md), [primal–dual estimates](certificates_convergence.md) |
| Multidimensional quadratic correction | [Multidimensional theory](MULTIDIM_THEORY_RU.md) | [Core identities](multidim_core_research.md), [polynomial constructions](multidim_polynomial_research.md) |
| Fixed interaction order and actual TT rounding | [Interaction theory](INTERACTION_THEORY_RU.md) | [Mass comparison](rounding_structure.md), [rounding geometry](rounding_geometry.md) |
| Approximation of the solution from assumptions on the data | [Solution approximation](SOLUTION_APPROXIMATION_RU.md) | [Contact structure](solution_approximation_research.md), [obstructions](solution_obstructions_research.md) |
| Spectral envelopes | [Spectral construction](spectral_envelope.md) | [Independent audit script](../research/audits/verify_spectral.py) |
| Related work and limits of novelty claims | [Initial literature notes](related_work.md) | [Rounding literature](rounding_limits_literature.md), [approximation literature](approximation_related_work.md) |

The files named `REVIEW` record internal checks and corrections, not journal
referee reports. [Independent numerical audits](../research/audits/README.md)
test formulas on finite examples; passing them does not prove a theorem.

The archive is broader than the Python API. In particular, data-to-solution
approximation estimates, contact stability arguments, and iterative
convergence results are research material rather than complete implemented
solvers. The package's current scope is documented in
[limitations](../docs/limitations.md).

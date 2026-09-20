# Contributing

Install with `python -m pip install -e .` and run
`python -m unittest discover -s tests -v` from the repository root.

For numerical changes, include a small independent reference calculation or
an invariant that would detect an incorrect implementation. Keep core TT
operations free of full-tensor assembly. Dense reference utilities must retain
explicit size guards.

For mathematical changes, state the domain, norm, structural assumptions,
and finite-precision limitations. Distinguish a proof from a numerical audit,
and a data-relative estimate from a solution-relative estimate. Cite primary
sources and do not claim novelty based only on the absence of a search result.

Examples use deterministic seeds. Commit small result files with enough
metadata to reproduce them. Avoid generated caches and large dense arrays.

"""Core-only compression of a positive mixture in eight dimensions.

This is a constructed feasible input, not a solved high-dimensional PDE.
Feasibility after repair follows from the envelope construction in exact
arithmetic; point samples below are diagnostics, not global certificates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tt_obstacle import Q1Operator, TensorTrain, feasible_compress


def run(dimension=8):
    if dimension < 2 or dimension > 32:
        raise ValueError("This bounded example accepts dimensions from 2 to 32.")
    rng = np.random.default_rng(20260920)
    op = Q1Operator.uniform((3,) * dimension)
    original = TensorTrain.zeros(op.shape)
    for _ in range(5):
        factors = []
        for b in op.ground_factors:
            factors.append(b * (1 + rng.uniform(-.15, .15) * np.array([-1., 0., 1.])))
        original = original.add(TensorTrain.rank_one(factors).scaled(.2))
    # Zero approximation budget: changes of gauge and floating-point effects
    # only. No claim is made about exact minimal ranks of the stored input.
    original = original.round(rtol=0.0)
    rows = []
    for method in ("quadratic", "core"):
        report = feasible_compress(original, op, rtol=1e-3, max_rank=1, method=method)
        sampled_increment = min(
            report.tensor.evaluate(indices) - original.evaluate(indices)
            for indices in rng.integers(0, 3, size=(100, dimension))
        )
        rows.append({
            "method": method,
            "stored_floats": report.result_storage,
            "maximum_stored_bond": max(report.tensor.ranks),
            "relative_energy_error": op.energy_norm(report.tensor.subtract(original)) / op.energy_norm(original),
            "relative_triangle_bound": report.error_bound / op.energy_norm(original),
            "sampled_minimum_increment": float(sampled_increment),
            "sample_count": 100,
            "rounding_mass_tolerance_met": report.rounding_tolerance_met,
        })
        if sampled_increment < -1e-9:
            raise RuntimeError("A sampled dominance check failed.")
    return {
        "package_version": "0.1.0", "numpy_version": np.__version__, "seed": 20260920,
        "dimension": dimension, "mode_size": 3,
        "dense_tensor_entries_not_assembled": str(3 ** dimension),
        "original_stored_floats": original.storage_size,
        "original_maximum_stored_bond": max(original.ranks),
        "scope": "Core-only constructed-input experiment; sampling is not a global numerical certificate",
        "repairs": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(run(args.dimension), indent=2, allow_nan=False)
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n")


if __name__ == "__main__":
    main()

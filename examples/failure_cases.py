"""Actual TT rounding and the growing cost of a nonnegative repair.

The candidate and correction are constructed as TT cores. The minimum-energy
nonnegative correction used here is proved for the two-node Q1 family, d>=4.
This is not a lower bound for arbitrary signed corrections or for all TT methods.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tt_obstacle import Q1Operator, TensorTrain


def run():
    rows = []
    epsilon = .25
    for d in (4, 8, 16, 32):
        op = Q1Operator.uniform((2,) * d)
        background = TensorTrain.rank_one([np.ones(2)] * d)
        parity = TensorTrain.rank_one([np.array([1., -1.])] * d)
        original = background.add(parity.scaled(epsilon))
        rounded = original.round(max_rank=1)
        if rounded.subtract(background).norm() > 1e-10 * background.norm():
            raise RuntimeError("The rank-one rounding did not reproduce the theoretical candidate.")
        correction = background.add(parity).scaled(epsilon / 2)
        repaired = rounded.add(correction)
        raw_error = op.energy_norm(original.subtract(rounded))
        corrected_error = op.energy_norm(repaired.subtract(original))
        predicted = .5 * np.sqrt(1 + (5 / 3) ** (d - 1) / 3)
        ratio = corrected_error / raw_error
        if not np.isclose(ratio, predicted, rtol=1e-8):
            raise RuntimeError("Energy ratio disagrees with the proved formula.")
        rows.append({"dimension": d, "raw_maximum_bond": max(rounded.ranks),
                     "repair_error_over_rounding_error": ratio,
                     "predicted_ratio": float(predicted)})
    return {"package_version": "0.1.0", "numpy_version": np.__version__,
            "scope": "psi=original; nonnegative-correction obstruction; signed epsilon*parity restores the input exactly",
            "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(run(), indent=2, allow_nan=False)
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n")


if __name__ == "__main__":
    main()

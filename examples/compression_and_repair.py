"""Solve a tiny mixed-contact obstacle problem, compress it, and repair it.

Run after installing the package: python examples/compression_and_repair.py
Optional: --output results/compression.json --plot results/compression.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tt_obstacle import Q1Operator, TensorTrain, feasible_compress, solve_obstacle


def run():
    op = Q1Operator.uniform((3, 3))
    a = op.to_dense()
    ground = op.ground_tensor.to_dense()
    psi = .3 * ground
    x = np.linspace(-1, 1, 3)
    forcing = ground * (2 * x[:, None] + x[None, :])
    f = a @ psi.ravel() + op.mass_dense() @ forcing.ravel()
    reference = solve_obstacle(a, f, psi.ravel())
    solution = reference.solution.reshape(op.shape)
    original = TensorTrain.from_dense(solution)
    rows = []
    tensors = {}
    for method in ("quadratic", "core"):
        report = feasible_compress(original, op, rtol=1e-3, max_rank=1, method=method)
        repaired = report.tensor.to_dense()
        raw = report.candidate.to_dense()
        error = op.energy_norm(report.tensor.subtract(original))
        row = {
            "method": method,
            "stored_core_ranks": list(report.tensor.ranks),
            "stored_floats": report.result_storage,
            "minimum_slack": float(np.min(repaired - psi)),
            "minimum_increment_over_original": float(np.min(repaired - solution)),
            "energy_error": error,
            "computed_triangle_bound": report.error_bound,
            "relative_energy_error": error / op.energy_norm(original),
            "rounding_mass_tolerance_met": report.rounding_tolerance_met,
        }
        if row["minimum_slack"] < -1e-10 or error > report.error_bound + 1e-9:
            raise RuntimeError("Repair diagnostics failed.")
        rows.append(row)
        tensors[method] = repaired
    summary = {
        "package_version": "0.1.0", "numpy_version": np.__version__,
        "scope": "3x3 nodal reference QP; floating-point diagnostics, not a performance benchmark",
        "shape": list(op.shape),
        "contact_nodes": int(np.count_nonzero(np.abs(solution - psi) < 1e-9)),
        "original": {"stored_core_ranks": list(original.ranks), "stored_floats": original.storage_size},
        "raw": {"stored_core_ranks": list(report.candidate.ranks),
                "stored_floats": report.candidate_storage,
                "minimum_slack": float(np.min(raw - psi)),
                "relative_energy_error": report.raw_error / op.energy_norm(original)},
        "repairs": rows,
    }
    if summary["raw"]["minimum_slack"] >= 0 or not 0 < summary["contact_nodes"] < 9:
        raise RuntimeError("The example must have mixed contact and a violating raw compression.")
    return summary, (solution, psi, raw, tensors)


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Original", "Raw TT", "Quadratic repair", "Core repair"]
    rows = summary["repairs"]
    storage = [summary["original"]["stored_floats"], summary["raw"]["stored_floats"]] + [r["stored_floats"] for r in rows]
    errors = [0, summary["raw"]["relative_energy_error"]] + [r["relative_energy_error"] for r in rows]
    colors = ["#64748b", "#e76f51", "#5379b8", "#2a9d8f"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, values, title in zip(axes, (storage, errors),
                                 ("Stored TT coefficients", "Relative Q1 energy error")):
        bars = ax.bar(labels, values, color=colors, width=.65)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=18)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.3g}", (bar.get_x() + bar.get_width()/2, value),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
        ax.margins(y=.2)
    fig.suptitle("A 3 × 3 obstacle example: compression, feasibility, and storage", fontsize=12)
    fig.text(.5, -.025, "Raw TT violates the obstacle; both repairs satisfy it in this numerical check. Stored ranks are not minimal ranks.",
             ha="center", fontsize=8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    result, _ = run()
    text = json.dumps(result, indent=2, allow_nan=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if args.plot:
        plot(result, args.plot)


if __name__ == "__main__":
    main()

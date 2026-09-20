"""Run the constructed-mixture experiment across selected dimensions."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--output", type=Path, default=Path("results/dimension_sweep.json"))
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "examples" / "rank_energy_tradeoff.py"
    spec = importlib.util.spec_from_file_location("tradeoff_example", source)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    result = {"scope": "Constructed-input scaling experiment; no timing or PDE-solver claims",
              "cases": [example.run(d) for d in args.dimensions]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(result['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()

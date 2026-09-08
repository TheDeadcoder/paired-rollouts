"""Per-run registers (curves, areas, final sets, luck share per checkpoint, training groups, timings).

    python scripts/build_run_register.py --runs outputs/runs/gate1-c2-paired-s0 [--threshold 0.617]

Writes registers/runs/<run_id>.json for every run directory given and prints a summary of each.
"""

import argparse
import json
import pathlib

from pairedrl.analysis.curves import format_run, run_register


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="run directories under outputs/runs")
    parser.add_argument("--out-dir", default="registers/runs")
    parser.add_argument("--threshold", type=float, default=None, help="steps-to-threshold target on the noisy validation set")
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for run_dir in args.runs:
        entry = run_register(run_dir, threshold=args.threshold)
        path = out_dir / f"{entry['run_id']}.json"
        path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(format_run(entry))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

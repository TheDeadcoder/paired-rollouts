"""Accept or reject a collected run directory: job status, evidence completeness along the attempt lineage, and
(when the weights were collected) the trainer checkpoints and final adapter.

    python scripts/verify_run.py outputs/runs/<run_id> [more run dirs] [--weights] [--threshold 0.617]

Exit code 0 when every run is COMPLETE and complete (and, with --weights, has every expected checkpoint complete
plus adapter_final); 1 otherwise. Run it before destroying a droplet or trusting a number.
"""

import argparse
import sys

from pairedrl.analysis.curves import format_run, run_register


def verdict(entry: dict, weights: bool) -> tuple[bool, list[str]]:
    reasons = []
    if entry["status"] != "COMPLETE":
        reasons.append(f"status {entry['status']}")
    comp = entry["completeness"]
    if not comp["complete"]:
        reasons.append(
            f"evidence incomplete (missing sets {len(comp['periodic_and_final_missing'])}, diagnostic defects "
            f"{len(comp['diagnostic_missing'])}, groups {comp['training_groups_found']}/{comp['training_groups_expected']}, "
            f"rollouts {comp['training_rollouts_found']}/{comp['training_rollouts_expected']})"
        )
    if weights:
        ck = entry["checkpoints"]
        if ck is None:
            reasons.append("weights not collected (no trainer/ directory)")
        else:
            if ck["missing"] or ck["incomplete"]:
                reasons.append(f"checkpoints missing {ck['missing']}, incomplete {ck['incomplete']}")
            if not ck["adapter_final"]:
                reasons.append("adapter_final missing")
    return not reasons, reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--weights", action="store_true", help="also require every expected checkpoint and the final adapter")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    all_ok = True
    for run_dir in args.run_dirs:
        entry = run_register(run_dir, threshold=args.threshold)
        ok, reasons = verdict(entry, args.weights)
        all_ok = all_ok and ok
        print(format_run(entry))
        print(f"  verdict: {'ACCEPT' if ok else 'REJECT'}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

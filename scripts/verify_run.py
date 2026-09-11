"""Accept or reject a collected run directory: job status, evidence identity and completeness along the attempt
lineage against the frozen task pools, attempt consistency (code commit, model snapshot, pools) and, when the
weights were collected, the trainer checkpoints and final adapter.

    python scripts/verify_run.py outputs/runs/<run_id> [more run dirs] [--weights] [--threshold 0.617] [--data-dir data/tasks]

Exit code 0 when every run is COMPLETE, complete (every expected evaluation, diagnostic and training record present
with its task and schedule, nothing unexpected, exact totals, pool hashes matching the manifest), consistent across
attempts and, with --weights, has every expected checkpoint complete plus adapter_final; 1 otherwise. Run it before
destroying a droplet or trusting a number.
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
            f"evidence incomplete (missing sets {len(comp['periodic_and_final_missing'])}, identity defects "
            f"{len(comp['identity_defects'])}, unexpected phases {len(comp['unexpected_phases'])}, diagnostic defects "
            f"{len(comp['diagnostic_missing'])}, group defects {len(comp['group_defects'])}, groups "
            f"{comp['training_groups_found']}/{comp['training_groups_expected']}, rollouts "
            f"{comp['training_rollouts_found']}/{comp['training_rollouts_expected']}, episodes "
            f"{comp['episodes_found']}/{comp['episodes_expected']}, pools {'match' if comp['task_pools_match'] else 'mismatch'})"
        )
    consistency = entry["attempt_consistency"]
    if not consistency["commits_consistent"]:
        reasons.append(f"attempt commits changed without acknowledgement {consistency['commit_changes']}")
    if not consistency["model_consistent"]:
        reasons.append(f"model snapshot changed across attempts {consistency['model_commit_hashes']}")
    if not consistency["task_pools_consistent"]:
        reasons.append("task pools changed across attempts")
    if weights:
        ck = entry["checkpoints"]
        if ck is None:
            reasons.append("weights not collected (no trainer/ directory)")
        else:
            if ck["missing"] or ck["incomplete"]:
                reasons.append(f"checkpoints missing {ck['missing']}, incomplete {ck['incomplete']} {ck['defects']}")
            if not ck["adapter_final"]:
                reasons.append("adapter_final missing or without adapter_config.json")
    return not reasons, reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--weights", action="store_true", help="also require every expected checkpoint and the final adapter")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--data-dir", default="data/tasks", help="the frozen task pools the run must have used")
    args = parser.parse_args()
    all_ok = True
    for run_dir in args.run_dirs:
        entry = run_register(run_dir, threshold=args.threshold, data_dir=args.data_dir)
        ok, reasons = verdict(entry, args.weights)
        all_ok = all_ok and ok
        print(format_run(entry))
        print(f"  verdict: {'ACCEPT' if ok else 'REJECT'}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Accept or reject a collected run directory: job status, evidence identity and completeness along the attempt
lineage against the frozen task pools, attempt consistency (code commit, model snapshot, pools) and, when the
weights were collected, the trainer checkpoints and final adapter.

    python scripts/verify_run.py outputs/runs/<run_id> [more run dirs] [--weights] [--threshold 0.617] [--data-dir data/tasks]
        [--accept-missing-checkpoints 20,40,60]

Exit code 0 when every run is COMPLETE, complete (every expected evaluation, diagnostic and training record present
with its task and schedule, nothing unexpected, exact totals, pool hashes matching the manifest), consistent across
attempts and, with --weights, has every expected checkpoint complete plus adapter_final; 1 otherwise. Run it before
destroying a droplet or trusting a number. --accept-missing-checkpoints names expected checkpoints known to be absent
(gate 1 kept only 80 and 100); they are reported as accepted, and naming a checkpoint that is not missing rejects the run.
"""

import argparse
import sys

from pairedrl.analysis.curves import format_run, run_register


def parse_steps(text: str | None) -> list[int]:
    """Comma-separated checkpoint steps, for instance "20,40,60"."""
    return sorted({int(s) for s in (text or "").split(",") if s.strip()})


def verdict(entry: dict, weights: bool, accept_missing=()) -> tuple[bool, list[str]]:
    """ACCEPT or REJECT with the reasons. `accept_missing` lists expected checkpoint steps that may be absent; it only
    relaxes the missing-checkpoint test for exactly those steps and rejects a step that is not missing, so an accepted
    gap is always explicit and never covers an incomplete checkpoint or a missing final adapter."""
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
            accepted = set(accept_missing)
            not_missing = sorted(accepted - set(ck["missing"]))
            if not_missing:
                reasons.append(f"accepted as missing but not missing: checkpoints {not_missing}")
            missing = [s for s in ck["missing"] if s not in accepted]
            if missing or ck["incomplete"]:
                reasons.append(f"checkpoints missing {missing}, incomplete {ck['incomplete']} {ck['defects']}")
            if not ck["adapter_final"]:
                reasons.append("adapter_final missing or without adapter_config.json")
    return not reasons, reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--weights", action="store_true", help="also require every expected checkpoint and the final adapter")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--data-dir", default="data/tasks", help="the frozen task pools the run must have used")
    parser.add_argument("--accept-missing-checkpoints", default=None,
                        help="with --weights: comma-separated checkpoint steps known to be absent, for instance 20,40,60")
    args = parser.parse_args()
    accept = parse_steps(args.accept_missing_checkpoints)
    if accept and not args.weights:
        parser.error("--accept-missing-checkpoints applies only with --weights")
    all_ok = True
    for run_dir in args.run_dirs:
        entry = run_register(run_dir, threshold=args.threshold, data_dir=args.data_dir)
        ok, reasons = verdict(entry, args.weights, accept_missing=accept)
        all_ok = all_ok and ok
        print(format_run(entry))
        if accept:
            print(f"  accepted missing checkpoints: {accept}")
        print(f"  verdict: {'ACCEPT' if ok else 'REJECT'}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

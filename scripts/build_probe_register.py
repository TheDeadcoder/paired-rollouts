"""Merge a trajectory's gradient-probe step files into one register and apply the H1(b) decision (amendment A1).

    python scripts/build_probe_register.py --probes outputs/runs/probe-t1-c2-paired-s1-a outputs/runs/probe-t1-c2-paired-s1-b \
        --trajectory t1-c2-paired-s1 --base-from outputs/runs/probe-t1-c2-paired-s1-a --out registers/probe/t1-c2-paired-s1.json

Refuses a smoke probe, a manifest that is not COMPLETE, or an expected step whose step file is absent.
"""

import argparse
import json
import pathlib

from pairedrl.analysis.probe import decide_trajectory


def load_probe(probe_dir) -> tuple[dict, dict]:
    probe_dir = pathlib.Path(probe_dir)
    manifest = json.loads((probe_dir / "run_manifest.json").read_text(encoding="utf-8"))
    steps = {}
    for path in sorted(probe_dir.glob("step*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        steps[int(payload["step"])] = payload
    return manifest, steps


def check(manifest: dict, steps: dict, probe_dir: str) -> None:
    if manifest.get("spec", {}).get("smoke"):
        raise SystemExit(f"{probe_dir}: a smoke probe cannot enter a decision register")
    if manifest.get("status") != "COMPLETE":
        raise SystemExit(f"{probe_dir}: manifest status {manifest.get('status')!r}, expected COMPLETE")
    for step in manifest.get("steps", []):
        if int(step) not in steps:
            raise SystemExit(f"{probe_dir}: expected step {step} has no step{step}.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", nargs="+", required=True, help="probe run directories under outputs/runs")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--base-from", help="probe directory to take the base-model step 0 from")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    checkpoints: dict[int, dict] = {}
    commits = set()
    for probe_dir in args.probes:
        manifest, steps = load_probe(probe_dir)
        check(manifest, steps, probe_dir)
        commits.add(manifest.get("git_commit"))
        checkpoints.update(steps)
    if args.base_from:
        manifest, steps = load_probe(args.base_from)
        check(manifest, steps, args.base_from)
        commits.add(manifest.get("git_commit"))
        if 0 not in steps:
            raise SystemExit(f"{args.base_from}: no step0.json to take the base model from")
        checkpoints[0] = steps[0]

    summaries = {step: payload["summary"] for step, payload in checkpoints.items()}
    verdict = decide_trajectory(summaries)
    register = {
        "trajectory": args.trajectory,
        "steps": sorted(checkpoints),
        "commits": sorted(c for c in commits if c),
        "checkpoints": {str(step): checkpoints[step] for step in sorted(checkpoints)},
        "verdict": verdict,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(register, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"trajectory {args.trajectory}  steps {sorted(checkpoints)}")
    print("step  lhs_mean  rhs_mean  ratio_out_mc  ratio_out_im  ratio_trans_res  ratio_trans_strat")
    for step in sorted(checkpoints):
        oc = summaries[step]["outcome"]
        tr = summaries[step]["transition"]
        t = tr["mean_centered"]["trace_var"]
        ratio_strat = t["paired"] / t["stratified"] if t["stratified"] != 0 else float("inf")
        print(f"{step:>4}  {oc['lhs_mean']:.6f}  {oc['rhs_mean']:.6f}  "
              f"{oc['mean_centered']['ratio_paired_over_independent']:.4f}  "
              f"{oc['implemented']['ratio_paired_over_independent']:.4f}  "
              f"{tr['mean_centered']['ratio_paired_over_independent']:.4f}  {ratio_strat:.4f}")
    print(f"outcome_noise_every_checkpoint {verdict['outcome_noise_every_checkpoint']}")
    print(f"transition_noise_majority {verdict['transition_noise_majority']}")
    print(f"H1(b) passes {verdict['passes']}")
    print(f"P13 {verdict['p13']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

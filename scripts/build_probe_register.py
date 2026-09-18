"""Merge a trajectory's gradient-probe step files into one register and apply the H1(b) decision (amendment A1).

    python scripts/build_probe_register.py --probes outputs/runs/probe-t1-c2-paired-s1-a outputs/runs/probe-t1-c2-paired-s1-b \
        --trajectory t1-c2-paired-s1 --base-from outputs/runs/probe-t1-c2-paired-s1-a --out registers/probe/t1-c2-paired-s1.json

Refuses a smoke probe, a manifest that is not COMPLETE, a missing or conflicting step, a step not at the full design,
a non-finite or negative pooled/within trace, a coordinate set or trajectory that differs across the merged steps, or
a merged step set that is not the trajectory's declared checkpoints.
"""

import argparse
import json
import math
import pathlib

from pairedrl.analysis.probe import decide_trajectory

DECLARED_STEPS = {"gate1-c2-paired-s0": {0, 80, 100}, "t1-c2-paired-s1": {0, 20, 40, 60, 80, 100}}
FULL_DESIGN = {"diagnostic_tasks": 16, "schedules": 8, "samples": 8, "clean_tasks": 64, "clean_rollouts": 8, "resamples": 64}
TOL = 1e-9


def load_probe(probe_dir) -> tuple[dict, dict]:
    probe_dir = pathlib.Path(probe_dir)
    manifest = json.loads((probe_dir / "run_manifest.json").read_text(encoding="utf-8"))
    steps = {int(json.loads(p.read_text(encoding="utf-8"))["step"]): json.loads(p.read_text(encoding="utf-8"))
             for p in sorted(probe_dir.glob("step*.json")) if not p.name.endswith("_evidence.json.gz")}
    if manifest.get("spec", {}).get("smoke"):
        raise SystemExit(f"{probe_dir}: a smoke probe cannot enter a decision register")
    if manifest.get("status") != "COMPLETE":
        raise SystemExit(f"{probe_dir}: manifest status {manifest.get('status')!r}, expected COMPLETE")
    for step in manifest.get("steps", []):
        if int(step) not in steps:
            raise SystemExit(f"{probe_dir}: expected step {step} has no step{step}.json")
    return manifest, steps


def _finite(obj, path="") -> list:
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad += _finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += _finite(v, f"{path}[{i}]")
    elif isinstance(obj, float) and not math.isfinite(obj):
        bad.append(f"{path}={obj}")
    return bad


def check_step(payload, trajectory, coord_sha, git_commit) -> None:
    step = payload["step"]
    if payload.get("trajectory") != trajectory:
        raise SystemExit(f"step {step}: trajectory {payload.get('trajectory')!r} != {trajectory!r}")
    design = payload.get("provenance", {}).get("design_counts")
    if design != FULL_DESIGN:
        raise SystemExit(f"step {step}: design {design} is not the full design {FULL_DESIGN}")
    if payload.get("coordinate_names_sha256") != coord_sha:
        raise SystemExit(f"step {step}: coordinate names differ across the merged steps")
    if payload.get("git_commit") != git_commit:
        raise SystemExit(f"step {step}: git_commit differs across the merged steps")
    bad = _finite(payload["summary"])
    if bad:
        raise SystemExit(f"step {step}: non-finite statistics {bad[:5]}")
    for est in ("mean_centered", "implemented"):
        for noise, designs in (("outcome", ("paired", "independent")),
                               ("transition", ("paired", "independent_resampled", "stratified"))):
            block = payload["summary"][noise][est]
            for key in ("trace_var", "trace_var_within"):
                for d in designs:
                    v = block[key][d]
                    if v < -TOL * (abs(v) + 1.0):
                        raise SystemExit(f"step {step}: {noise}.{est}.{key}.{d} negative ({v})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", nargs="+", required=True, help="probe run directories under outputs/runs")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--base-from", help="probe directory to take the base-model step 0 from")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.trajectory not in DECLARED_STEPS:
        raise SystemExit(f"unknown trajectory {args.trajectory!r}; declared: {sorted(DECLARED_STEPS)}")
    checkpoints: dict[int, dict] = {}
    for probe_dir in args.probes:
        _, steps = load_probe(probe_dir)
        for step, payload in steps.items():
            if step in checkpoints and checkpoints[step] != payload:
                raise SystemExit(f"conflicting duplicate step {step} across {args.probes}")
            checkpoints[step] = payload
    base_fingerprint = None
    if args.base_from:
        _, steps = load_probe(args.base_from)
        if 0 not in steps:
            raise SystemExit(f"{args.base_from}: no step0.json to take the base model from")
        checkpoints[0] = steps[0]
        base_fingerprint = steps[0].get("provenance", {}).get("base_fingerprint")

    if set(checkpoints) != DECLARED_STEPS[args.trajectory]:
        raise SystemExit(f"merged steps {sorted(checkpoints)} != declared {sorted(DECLARED_STEPS[args.trajectory])}")
    first = checkpoints[min(checkpoints)]
    coord_sha = first.get("coordinate_names_sha256")
    git_commit = first.get("git_commit")
    for payload in checkpoints.values():
        check_step(payload, args.trajectory, coord_sha, git_commit)

    summaries = {step: payload["summary"] for step, payload in checkpoints.items()}
    verdict = decide_trajectory(summaries)
    register = {
        "trajectory": args.trajectory, "steps": sorted(checkpoints), "git_commit": git_commit,
        "coordinate_names_sha256": coord_sha, "base_fingerprint": base_fingerprint,
        "spec_sha256_by_step": {str(s): checkpoints[s].get("provenance", {}).get("spec_sha256") for s in sorted(checkpoints)},
        "checkpoints": {str(step): checkpoints[step] for step in sorted(checkpoints)}, "verdict": verdict,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(register, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"trajectory {args.trajectory}  steps {sorted(checkpoints)}")
    print("step  lhs_mean  rhs_mean  lhs/rhs  out_mc_pool  out_im_pool  out_mc_within  out_im_within  "
          "trans_resamp  trans_strat  rho_zero")
    for step in sorted(checkpoints):
        s = summaries[step]
        oc, tr = s["outcome"], s["transition"]
        tv = tr["mean_centered"]["trace_var"]
        strat = tv["paired"] / tv["stratified"] if tv["stratified"] != 0 else float("inf")
        rho_zero = checkpoints[step].get("counts", {}).get("rho_zero_fraction", 0.0)
        print(f"{step:>4}  {oc['lhs_mean']:.6f}  {oc['rhs_mean']:.6f}  {oc['lhs_over_rhs']:.4f}  "
              f"{oc['mean_centered']['ratio_paired_over_independent']:.4f}  "
              f"{oc['implemented']['ratio_paired_over_independent']:.4f}  "
              f"{oc['mean_centered']['ratio_within_paired_over_independent']:.4f}  "
              f"{oc['implemented']['ratio_within_paired_over_independent']:.4f}  "
              f"{tr['mean_centered']['ratio_paired_over_independent']:.4f}  {strat:.4f}  {rho_zero:.3f}")
    print(f"outcome_noise_every_checkpoint {verdict['outcome_noise_every_checkpoint']}")
    print(f"transition_noise_majority {verdict['transition_noise_majority']}")
    print(f"H1(b) passes {verdict['passes']}")
    print(f"P13 {verdict['p13']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

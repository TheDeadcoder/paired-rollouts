"""Spawn run specs (or gradient-probe specs) onto the deployed pairedrl-train app and record the launch.

Returns within seconds. The spawned runs live on Modal, not on this machine: closing the laptop or losing the network
afterwards does not affect them. Refuses to launch from a tree with uncommitted changes, because the deployed image
and the run manifest both record the commit.
"""

import argparse
import json
import pathlib

import modal

from pairedrl.ops.ledger import (
    append_rows,
    git_state,
    launch_register_path,
    ledger_row,
    preregistration_note,
    refuse_duplicate_launches,
    utc_now,
    write_launch_register,
)
from pairedrl.train.runner import RunSpec

APP_NAME = "pairedrl-train"
FUNCTION_NAME = "run"


class _IdShim:
    """A stand-in with the two attributes refuse_duplicate_launches reads, so probe_id can be checked the same way."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.extend_previous = False


def trajectory_meta(trajectory: str) -> tuple[str | None, str, str]:
    """The model, condition and arm of a probe's trajectory, from its config (gate1's config is present) or, as a
    fallback, its run register."""
    cfg = pathlib.Path("configs") / f"{trajectory}.json"
    if cfg.exists():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        return data["model"], data["condition"], data["arm"]
    reg = json.loads((pathlib.Path("registers/runs") / f"{trajectory}.json").read_text(encoding="utf-8"))
    return reg.get("model"), reg["condition"], reg["arm"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", choices=["run", "probe"], default="run", help="the deployed function to spawn")
    parser.add_argument("--specs", required=True, help="comma-separated run or probe spec JSON files")
    parser.add_argument("--label", required=True, help="short name for this launch, used in the register file name")
    parser.add_argument("--ledger", default="docs/RUN_LEDGER.md")
    parser.add_argument("--register-dir", default="registers/launches")
    parser.add_argument("--allow-dirty", action="store_true", help="launch even with uncommitted changes (the run will be refused by the provenance check)")
    parser.add_argument("--preregistration", default=None, help="label of the frozen pre-registration these specs belong to (for instance v1); every spec's notes must carry 'pre-registered <label>'; omit for exploratory runs")
    parser.add_argument("--relaunch", action="store_true", help="launch a run id whose newest ledger row is still LAUNCHED or RUNNING (only when that job is known to be dead)")
    args = parser.parse_args()

    head, dirty = git_state()
    if dirty and not args.allow_dirty:
        raise SystemExit(f"refusing to launch: the working tree has uncommitted changes in {dirty}; commit and deploy first")
    commit = head + ("-dirty" if dirty else "")
    paths = [pathlib.Path(p.strip()) for p in args.specs.split(",") if p.strip()]

    if args.function == "probe":
        from pairedrl.train.probe import ProbeSpec, probe_ledger_row

        specs = [ProbeSpec.from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in paths]
        for spec in specs:
            preregistration_note(spec, args.preregistration)
        refuse_duplicate_launches(args.ledger, [_IdShim(s.probe_id) for s in specs], relaunch=args.relaunch)
    else:
        specs = [RunSpec.from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in paths]
        for spec in specs:
            preregistration_note(spec, args.preregistration)
        refuse_duplicate_launches(args.ledger, specs, relaunch=args.relaunch)

    run = modal.Function.from_name(APP_NAME, args.function)
    launched_utc = utc_now()
    register_path = launch_register_path(args.register_dir, args.label, launched_utc)
    entries = []
    for path, spec in zip(paths, specs, strict=True):
        call = run.spawn(spec.to_dict(), commit)
        try:
            dashboard = call.get_dashboard_url()
        except Exception:  # noqa: BLE001
            dashboard = None
        if args.function == "probe":
            from pairedrl.train.probe import probe_ledger_row

            run_id = spec.probe_id
            model, condition, arm = trajectory_meta(spec.trajectory)
            row = probe_ledger_row(spec, launched_utc, call.object_id, model, condition, arm, preregistration=args.preregistration)
            extra = {"function": "probe", "trajectory": spec.trajectory}
        else:
            run_id = spec.run_id
            row = ledger_row(spec, launched_utc, call.object_id, "LAUNCHED", preregistration=args.preregistration)
            extra = {}
        entries.append({
            "run_id": run_id,
            "spec_path": str(path),
            "call_id": call.object_id,
            "spawned_utc": utc_now(),
            "dashboard_url": dashboard,
            **extra,
        })
        write_launch_register(register_path, commit, APP_NAME, launched_utc, entries, preregistration=args.preregistration)
        append_rows(args.ledger, [row])
        print(f"SPAWNED {run_id} {call.object_id}")

    print(f"commit {commit}")
    print(f"wrote {register_path}")
    print(f"appended {len(entries)} rows to {args.ledger}")


if __name__ == "__main__":
    main()

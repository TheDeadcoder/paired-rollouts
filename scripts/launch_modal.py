"""Spawn run specs onto the deployed pairedrl-train app and record the launch.

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", required=True, help="comma-separated run spec JSON files")
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
    specs = [RunSpec.from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    for spec in specs:
        preregistration_note(spec, args.preregistration)
    refuse_duplicate_launches(args.ledger, specs, relaunch=args.relaunch)

    run = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    launched_utc = utc_now()
    register_path = launch_register_path(args.register_dir, args.label, launched_utc)
    entries = []
    for path, spec in zip(paths, specs, strict=True):
        call = run.spawn(spec.to_dict(), commit)
        try:
            dashboard = call.get_dashboard_url()
        except Exception:  # noqa: BLE001
            dashboard = None
        entries.append({
            "run_id": spec.run_id,
            "spec_path": str(path),
            "call_id": call.object_id,
            "spawned_utc": utc_now(),
            "dashboard_url": dashboard,
        })
        write_launch_register(register_path, commit, APP_NAME, launched_utc, entries, preregistration=args.preregistration)
        append_rows(args.ledger, [ledger_row(spec, launched_utc, call.object_id, "LAUNCHED", preregistration=args.preregistration)])
        print(f"SPAWNED {spec.run_id} {call.object_id}")

    print(f"commit {commit}")
    print(f"wrote {register_path}")
    print(f"appended {len(entries)} rows to {args.ledger}")


if __name__ == "__main__":
    main()

"""Launch and collect the Modal job that archives run weights to Hugging Face (the app in scripts/archive_modal.py).

    MODAL_PROFILE=<profile> python scripts/archive_weights.py launch --runs <run_id>,<run_id> --label <label> \
        [--accept-missing <run_id>=20,40,60 --accept-reason "<why they are absent>"] [--allow-dirty]
    MODAL_PROFILE=<profile> python scripts/archive_weights.py collect --register registers/launches/<utc>_<label>.json

launch spawns the deployed function and returns within seconds; the job runs on Modal whatever happens to this machine.
The launch register records the call, the runs, the accepted gaps, the commit and the sha256 of the archiving code.
collect reports the call's state and each run's status from the pairedrl-archive volume, copies every verified record
byte for byte into registers/weights/<run_id>.json (refusing to replace a different existing record), and prints the
commit URL, file count, size and the check of the last checkpoint's adapter. Exit code 0 when every run is VERIFIED.
"""

import argparse
import json
import pathlib
import sys

import modal
import modal.exception

from pairedrl.ops.archive import code_digest
from pairedrl.ops.ledger import git_state, launch_register_path, read_launch_register, utc_now

APP_NAME = "pairedrl-archive"
FUNCTION_NAME = "archive"
ARCHIVE_VOLUME = "pairedrl-archive"
REPO = "Melikshah/paired-rollouts-weights"
CODE_FILES = ("scripts/upload_weights.py", "scripts/verify_run.py", "src/pairedrl/ops/archive.py",
              "src/pairedrl/analysis/curves.py")


def parse_accept(items: list[str], reason: str | None) -> dict:
    """["<run_id>=20,40,60", ...] -> {run_id: {"steps": [20, 40, 60], "reason": reason}}."""
    out = {}
    for item in items:
        run_id, _, steps = item.partition("=")
        out[run_id.strip()] = {"steps": sorted({int(s) for s in steps.split(",") if s.strip()}), "reason": reason}
    if out and not (reason or "").strip():
        raise SystemExit("--accept-missing needs --accept-reason")
    return out


def launch(args) -> int:
    run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
    accept = parse_accept(args.accept_missing, args.accept_reason)
    unknown = sorted(set(accept) - set(run_ids))
    if unknown:
        raise SystemExit(f"--accept-missing names runs that are not being archived: {unknown}")
    existing = [r for r in run_ids if (pathlib.Path(args.weights_dir) / f"{r}.json").exists()]
    if existing:
        raise SystemExit(f"refusing to launch: {existing} already have records under {args.weights_dir}")
    head, dirty = git_state()
    if dirty and not args.allow_dirty:
        raise SystemExit(f"refusing to launch: the working tree has uncommitted changes in {dirty}; commit and deploy first")
    commit = head + ("-dirty" if dirty else "")
    fn = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    launched_utc = utc_now()
    call = fn.spawn(run_ids, args.repo, accept, args.label, commit)
    try:
        dashboard = call.get_dashboard_url()
    except Exception:  # noqa: BLE001
        dashboard = None
    entries = [{"run_id": r, "call_id": call.object_id, "function": FUNCTION_NAME, "label": args.label,
                "repo": args.repo, "accept_missing": accept.get(r), "spawned_utc": launched_utc,
                "dashboard_url": dashboard} for r in run_ids]
    register = {"launched_utc": launched_utc, "commit": commit, "app": APP_NAME, "preregistration": None,
                "dirty_paths": dirty, "code_sha256": code_digest({f: f for f in CODE_FILES}), "entries": entries}
    path = launch_register_path(args.register_dir, args.label, launched_utc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(register, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"SPAWNED {FUNCTION_NAME} {call.object_id} for {len(run_ids)} runs: {', '.join(run_ids)}")
    print(f"commit {commit}")
    print(f"dashboard {dashboard}")
    print(f"wrote {path}")
    return 0


def call_state(call_id: str) -> tuple[str, dict | None]:
    call = modal.FunctionCall.from_id(call_id)
    try:
        result = call.get(timeout=0)
    except TimeoutError:
        return "RUNNING", None
    except modal.exception.OutputExpiredError:
        return "EXPIRED", None
    except Exception as e:  # noqa: BLE001
        return f"INFRA_FAILED ({type(e).__name__}: {e})", None
    return result.get("status", "?"), result


def read_volume_json(volume, path: str) -> tuple[dict | None, bytes | None]:
    try:
        data = b"".join(volume.read_file(path))
    except Exception:  # noqa: BLE001
        return None, None
    return json.loads(data.decode("utf-8")), data


def collect(args) -> int:
    register = read_launch_register(args.register)
    entries = [e for e in register["entries"] if e.get("function") == FUNCTION_NAME]
    if not entries:
        raise SystemExit(f"{args.register} is not an archive launch register")
    label = entries[0]["label"]
    state, result = call_state(entries[0]["call_id"])
    print(f"register {args.register} (commit {register['commit']}, launched {register['launched_utc']}, label {label})")
    print(f"call {entries[0]['call_id']}: {state}")
    if result is not None and result.get("code_sha256") != register.get("code_sha256"):
        print(f"  WARNING: deployed code {result.get('code_sha256')} differs from the launch-side code {register.get('code_sha256')}")
    if result is not None and result.get("error"):
        print(f"  error: {result['error']}")
    volume = modal.Volume.from_name(ARCHIVE_VOLUME)
    weights_dir = pathlib.Path(args.weights_dir)
    all_verified = True
    for entry in entries:
        run_id = entry["run_id"]
        status, _ = read_volume_json(volume, f"{label}/status/{run_id}.json")
        if status is None:
            print(f"{run_id}: no status on the volume yet")
            all_verified = False
            continue
        line = f"{run_id}: {status.get('status')}"
        if status.get("files") is not None:
            line += f" | {status['files']} files | {status['total_bytes'] / 1e9:.2f} GB | {status.get('commit_url')}"
        check = status.get("verification")
        if check:
            line += (f" | {check['path']} at {check['revision'][:12]}: sha256 {check['sha256_downloaded'][:16]}... "
                     f"{'matches' if check['match'] else 'DOES NOT MATCH'} the record")
        if status.get("error"):
            line += f" | error {status['error']}"
        print(line)
        if status.get("status") != "VERIFIED":
            all_verified = False
            continue
        record, raw = read_volume_json(volume, f"{label}/records/{run_id}.json")
        if record is None or record.get("revision") != status.get("revision"):
            print("  record missing or at another revision on the volume; not copied")
            all_verified = False
            continue
        out = weights_dir / f"{run_id}.json"
        if out.exists() and out.read_bytes() != raw:
            print(f"  {out} exists with different content; not replaced")
            all_verified = False
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(raw)
        print(f"  wrote {out}")
    return 0 if all_verified else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("launch", help="spawn the deployed archive function")
    p.add_argument("--runs", required=True, help="comma-separated run ids on the pairedrl-runs volume")
    p.add_argument("--label", required=True, help="short name for this launch; also the folder on the archive volume")
    p.add_argument("--repo", default=REPO)
    p.add_argument("--accept-missing", action="append", default=[], help="<run_id>=<steps>, for instance gate1-c2-paired-s0=20,40,60")
    p.add_argument("--accept-reason", default=None, help="why the accepted checkpoints are absent; required with --accept-missing")
    p.add_argument("--register-dir", default="registers/launches")
    p.add_argument("--weights-dir", default="registers/weights")
    p.add_argument("--allow-dirty", action="store_true", help="launch from a tree with uncommitted changes (recorded as <commit>-dirty)")
    c = sub.add_parser("collect", help="report the job and copy verified records into registers/weights/")
    c.add_argument("--register", required=True)
    c.add_argument("--weights-dir", default="registers/weights")
    args = parser.parse_args()
    return launch(args) if args.command == "launch" else collect(args)


if __name__ == "__main__":
    sys.exit(main())

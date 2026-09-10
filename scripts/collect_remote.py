"""Report and download the runs of a remote launch register (runs started by `scripts/launch_remote.py`).

    python scripts/collect_remote.py --register registers/launches/<utc>_<label>.json [--no-download] [--with-weights]

Reads each run's manifest on the host over SSH (or rsyncs the run directory without weights into outputs/runs/),
prints one line per run, and fills in the ledger status once a run is terminal. Exit code 0 when every run in the
register is terminal, 1 otherwise. A manifest that has not progressed for two hours is flagged STALE.
"""

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys

from pairedrl.ops.ledger import TERMINAL, latest_launch_register, read_launch_register, set_status
from pairedrl.ops.remote import rsync_command, ssh_command

STALE_AFTER_S = 2 * 3600


def read_remote_manifest(host: str, host_run_dir: str) -> dict | None:
    proc = subprocess.run(ssh_command(host, f"cat {host_run_dir}/run_manifest.json"), capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return json.loads(proc.stdout)


def download(host: str, host_run_dir: str, local_run_dir: pathlib.Path, with_weights: bool) -> bool:
    local_run_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(rsync_command(host, host_run_dir, str(local_run_dir), with_weights), check=False)
    return proc.returncode == 0


def describe(run_id: str, manifest: dict | None, now: dt.datetime) -> tuple[str, str]:
    if manifest is None:
        return "RUNNING", f"{run_id}: RUNNING | no manifest yet"
    status = manifest.get("status", "?")
    state = status if status in TERMINAL else "RUNNING"
    progress = manifest.get("progress_utc")
    stale = ""
    if state == "RUNNING" and progress:
        age = (now - dt.datetime.fromisoformat(progress)).total_seconds()
        if age > STALE_AFTER_S:
            stale = f" | STALE (no progress for {age / 3600:.1f} h)"
    line = (
        f"{run_id}: {state} | manifest {status} | attempt {manifest.get('attempt')} | episodes {manifest.get('episodes_logged', 0)}"
        f" | train steps {manifest.get('train_steps_logged', 0)} | wall {manifest.get('wall_time_s')} s"
        f" | est {manifest.get('estimated_cost_usd')} USD | progress {progress}{stale}"
    )
    if manifest.get("error"):
        line += f" | error {manifest['error']}"
    return state, line


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", help="launch register JSON; defaults to the newest under --register-dir")
    parser.add_argument("--register-dir", default="registers/launches")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--ledger", default="docs/RUN_LEDGER.md")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--with-weights", action="store_true")
    args = parser.parse_args()

    register_path = pathlib.Path(args.register) if args.register else latest_launch_register(args.register_dir)
    register = read_launch_register(register_path)
    if not register["entries"] or "host" not in register["entries"][0]:
        raise SystemExit(f"{register_path} is not a remote launch register (use scripts/collect_modal.py)")
    print(f"register {register_path} (commit {register['commit']}, host {register['app']}, launched {register['launched_utc']})")
    now = dt.datetime.now(dt.UTC)
    all_terminal = True
    for entry in register["entries"]:
        run_id, host, host_run_dir = entry["run_id"], entry["host"], entry["host_run_dir"]
        if entry.get("cancelled_utc"):
            print(f"{run_id}: CANCELLED before it started on {host} at {entry['cancelled_utc']} ({entry.get('cancelled_reason', '')})")
            continue
        local_run_dir = pathlib.Path(args.runs_dir) / run_id
        if args.no_download:
            manifest = read_remote_manifest(host, host_run_dir)
        else:
            ok = download(host, host_run_dir, local_run_dir, args.with_weights)
            manifest_path = local_run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if ok and manifest_path.exists() else None
        state, line = describe(run_id, manifest, now)
        print(line)
        if state in TERMINAL:
            set_status(args.ledger, entry["call_id"], state)
        else:
            all_terminal = False
    return 0 if all_terminal else 1


if __name__ == "__main__":
    sys.exit(main())

"""Start run specs on a remote GPU host over SSH, one after another in a detached shell, and record the launch.

    python scripts/launch_remote.py --host root@203.0.113.5 --specs configs/a.json,configs/b.json --label do-tier1-a

The remote checkout is moved to this tree's commit first (refusing a dirty local tree), the package is reinstalled
in the remote container, and the chain starts with `docker exec -d`, so it survives this machine going offline.
Every spec gets a LAUNCHED ledger row whose call id is `<host>:<run_id>`; `scripts/collect_remote.py` collects.
"""

import argparse
import json
import pathlib
import subprocess

from pairedrl.ops.ledger import (
    append_rows,
    launch_register_path,
    ledger_row,
    utc_now,
    write_launch_register,
)
from pairedrl.ops.remote import checkout_command, install_command, run_chain_command, ssh_command
from pairedrl.train.runner import RunSpec


def git_state() -> tuple[str, bool]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False).stdout.strip()
    return head or "unknown", bool(dirty)


def remote(host: str, command: str) -> str:
    proc = subprocess.run(ssh_command(host, command), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"remote command failed on {host}: {command}\n{proc.stdout}{proc.stderr}")
    return proc.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, help="ssh target, for instance root@203.0.113.5")
    parser.add_argument("--specs", required=True, help="comma-separated run spec JSON files, run in this order")
    parser.add_argument("--label", required=True)
    parser.add_argument("--provider", default="digitalocean")
    parser.add_argument("--usd-per-hour", type=float, default=1.99)
    parser.add_argument("--remote-repo", default="/root/work/paired-rollouts", help="checkout on the host")
    parser.add_argument("--container", default="pairedrl", help="docker container name; 'none' runs on the host")
    parser.add_argument("--container-repo", default="/work/paired-rollouts", help="the checkout's path inside the container")
    parser.add_argument("--runs-dir", default="/work/runs", help="run directory root as seen by the job")
    parser.add_argument("--host-runs-dir", default="/root/work/runs", help="the same directory as seen from the host (for rsync)")
    parser.add_argument("--hf-home", default="/work/hf")
    parser.add_argument("--ledger", default="docs/RUN_LEDGER.md")
    parser.add_argument("--register-dir", default="registers/launches")
    args = parser.parse_args()

    head, dirty = git_state()
    if dirty:
        raise SystemExit("refusing to launch: the working tree has uncommitted changes; commit and push first")
    container = None if args.container == "none" else args.container
    repo_in_job = args.remote_repo if container is None else args.container_repo
    spec_paths = [p.strip() for p in args.specs.split(",") if p.strip()]
    specs = [RunSpec.from_dict(json.loads(pathlib.Path(p).read_text(encoding="utf-8"))) for p in spec_paths]

    remote_head = remote(args.host, checkout_command(args.remote_repo, head))
    if remote_head.splitlines()[-1] != head:
        raise SystemExit(f"remote checkout is at {remote_head}, expected {head}")
    remote(args.host, install_command(repo_in_job, container))
    remote(args.host, run_chain_command(
        repo_in_job, spec_paths, args.runs_dir, args.provider, args.usd_per_hour, head, args.hf_home, container,
    ))

    launched_utc = utc_now()
    register_path = launch_register_path(args.register_dir, args.label, launched_utc)
    entries = []
    rows = []
    for path, spec in zip(spec_paths, specs, strict=True):
        call_id = f"{args.host}:{spec.run_id}"
        entries.append({
            "run_id": spec.run_id, "spec_path": path, "call_id": call_id, "host": args.host,
            "container": container, "host_run_dir": f"{args.host_runs_dir}/{spec.run_id}",
            "log": f"{args.host_runs_dir}/{pathlib.Path(path).stem}.log", "spawned_utc": launched_utc,
        })
        rows.append(ledger_row(spec, launched_utc, call_id, "LAUNCHED", provider=args.provider))
        print(f"STARTED {spec.run_id} on {args.host}")
    write_launch_register(register_path, head, args.host, launched_utc, entries)
    append_rows(args.ledger, rows)
    print(f"commit {head}")
    print(f"wrote {register_path}")
    print(f"appended {len(rows)} rows to {args.ledger}")


if __name__ == "__main__":
    main()

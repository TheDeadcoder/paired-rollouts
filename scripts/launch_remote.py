"""Start run specs on a remote GPU host over SSH, detached, and record the launch.

    python scripts/launch_remote.py --host root@203.0.113.5 --specs configs/a.json,configs/b.json --label do-tier1-a
    python scripts/launch_remote.py --host root@203.0.113.5 --entry configs/solo.json \
        --entry configs/paired.json,configs/independent.json --label do-a [--append]

With --specs (a chain) the remote checkout is moved to this tree's commit first (refusing a dirty local tree), the
package is reinstalled in the remote container, and the chain starts with `docker exec -d`, so it survives this
machine going offline. With --entry the specs go to the droplet's job queue instead (scripts/run_queue.py): each
--entry is one queue entry, two comma-separated specs in one --entry run as a concurrent pair on one GPU, the entry
runs from a checkout of this commit of its own (so later appends never change running code), and the queue's runner
is started detached if none is alive. --append adds entries while the queue runs; without it a running queue is
refused. Every spec gets a LAUNCHED ledger row whose call id is `<host>:<run_id>`; `scripts/collect_remote.py`
collects and reports the queue.
"""

import argparse
import json
import pathlib
import subprocess

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
from pairedrl.ops.remote import (
    chain_status_command,
    checkout_command,
    ensure_checkout_command,
    install_command,
    queue_append_command,
    queue_host_status_command,
    queue_run_command,
    run_chain_command,
    ssh_command,
)
from pairedrl.train.runner import RunSpec


def remote(host: str, command: str) -> str:
    proc = subprocess.run(ssh_command(host, command), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"remote command failed on {host}: {command}\n{proc.stdout}{proc.stderr}")
    return proc.stdout.strip()


def load_specs(paths: list[str], preregistration: str | None) -> list[RunSpec]:
    specs = [RunSpec.from_dict(json.loads(pathlib.Path(p).read_text(encoding="utf-8"))) for p in paths]
    for spec in specs:
        preregistration_note(spec, preregistration)
    return specs


def launch_queue(args, head: str, container: str | None) -> tuple[list[tuple[str, RunSpec]], list[dict], dict]:
    """Append the --entry entries to the droplet's queue and start its runner when none is alive. Returns the
    (spec path, spec) pairs, their register entries and the register's queue block."""
    groups = [[p.strip() for p in item.split(",") if p.strip()] for item in args.entry]
    if any(not 1 <= len(g) <= 2 for g in groups):
        raise SystemExit("an --entry holds one spec, or two comma-separated specs that run as a pair")
    pairs = [(p, s) for g in groups for p, s in zip(g, load_specs(g, args.preregistration), strict=True)]
    refuse_duplicate_launches(args.ledger, [s for _, s in pairs], relaunch=args.relaunch)
    state = remote(args.host, queue_host_status_command(container)).splitlines()[-1]
    if state == "CHAIN":
        raise SystemExit(f"refusing to launch: jobs outside a queue are running on {args.host}")
    if state == "QUEUE" and not args.append:
        raise SystemExit(f"refusing to launch: the queue on {args.host} is running; pass --append to add entries")
    checkouts = args.checkouts_dir or f"{args.remote_repo.rstrip('/').rsplit('/', 1)[0]}/checkouts"
    job_checkouts = checkouts if container is None else (
        args.container_checkouts_dir or f"{args.container_repo.rstrip('/').rsplit('/', 1)[0]}/checkouts")
    remote_head = remote(args.host, ensure_checkout_command(args.remote_repo, checkouts, head))
    if remote_head.splitlines()[-1] != head:
        raise SystemExit(f"remote checkout of {head} is at {remote_head}")
    checkout = f"{job_checkouts.rstrip('/')}/{head}"
    queue_dir = args.queue_dir or f"{args.runs_dir.rstrip('/')}/queue"
    host_queue_dir = args.host_queue_dir or f"{args.host_runs_dir.rstrip('/')}/queue"
    by_path = dict(pairs)
    register_entries, runner_alive = [], True
    for group in groups:
        entry = {"kind": "pair" if len(group) == 2 else "run", "label": args.label, "commit": head, "checkout": checkout,
                 "jobs": [{"name": by_path[p].run_id, "spec": p} for p in group]}
        reply = remote(args.host, queue_append_command(container, checkout, queue_dir, entry)).splitlines()[-1].split()
        if len(reply) != 3 or reply[0] != "APPENDED":
            raise SystemExit(f"unexpected reply from the queue on {args.host}: {' '.join(reply)}")
        entry_id, runner_alive = reply[1], reply[2] == "RUNNER_ALIVE"
        for slot, path in enumerate(group):
            spec = by_path[path]
            register_entries.append({
                "run_id": spec.run_id, "spec_path": path, "call_id": f"{args.host}:{spec.run_id}", "host": args.host,
                "container": container, "host_run_dir": f"{args.host_runs_dir}/{spec.run_id}", "queue_entry": entry_id,
                "slot": slot, "pair": entry["kind"] == "pair",
            })
            print(f"QUEUED {spec.run_id} on {args.host} (entry {entry_id}, slot {slot})")
    if not runner_alive:
        remote(args.host, queue_run_command(container, checkout, queue_dir, args.runs_dir, args.provider,
                                            args.usd_per_hour, args.hf_home))
        print(f"started the queue runner on {args.host}")
    queue = {"queue_dir": queue_dir, "host_queue_dir": host_queue_dir, "checkout": checkout, "append": args.append}
    return pairs, register_entries, queue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, help="ssh target, for instance root@203.0.113.5")
    parser.add_argument("--specs", default=None, help="a chain: comma-separated run spec JSON files, run in this order")
    parser.add_argument("--entry", action="append", default=[],
                        help="a queue entry: one spec, or two comma-separated specs that run as a concurrent pair; repeatable")
    parser.add_argument("--append", action="store_true", help="with --entry: add to the droplet's queue while it runs")
    parser.add_argument("--queue-dir", default=None, help="the queue directory as seen by the jobs (default <runs-dir>/queue)")
    parser.add_argument("--host-queue-dir", default=None, help="the same directory as seen from the host (default <host-runs-dir>/queue)")
    parser.add_argument("--checkouts-dir", default=None, help="per-commit checkouts on the host (default next to --remote-repo)")
    parser.add_argument("--container-checkouts-dir", default=None, help="the same directory inside the container (default next to --container-repo)")
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
    parser.add_argument("--preregistration", default=None, help="label of the frozen pre-registration these specs belong to (for instance v1); every spec's notes must carry 'pre-registered <label>'; omit for exploratory runs")
    parser.add_argument("--relaunch", action="store_true", help="launch a run id whose newest ledger row is still LAUNCHED or RUNNING (only when that job is known to be dead)")
    args = parser.parse_args()
    if bool(args.specs) == bool(args.entry):
        raise SystemExit("give either --specs (a chain) or --entry (the droplet queue)")
    if args.append and not args.entry:
        raise SystemExit("--append applies to --entry")

    head, dirty = git_state()
    if dirty:
        raise SystemExit(f"refusing to launch: the working tree has uncommitted changes in {dirty}; commit and push first")
    container = None if args.container == "none" else args.container
    extra = None
    if args.entry:
        pairs, entries, queue = launch_queue(args, head, container)
        extra = {"queue": queue}
        launched_utc = utc_now()
        for e in entries:
            e["spawned_utc"] = launched_utc
    else:
        repo_in_job = args.remote_repo if container is None else args.container_repo
        spec_paths = [p.strip() for p in args.specs.split(",") if p.strip()]
        specs = load_specs(spec_paths, args.preregistration)
        refuse_duplicate_launches(args.ledger, specs, relaunch=args.relaunch)

        status = remote(args.host, chain_status_command(container))
        if status.splitlines()[-1] != "IDLE":
            raise SystemExit(f"refusing to launch: a job chain is already running on {args.host} ({status}); one chain per GPU")
        remote_head = remote(args.host, checkout_command(args.remote_repo, head))
        if remote_head.splitlines()[-1] != head:
            raise SystemExit(f"remote checkout is at {remote_head}, expected {head}")
        remote(args.host, install_command(repo_in_job, container))
        remote(args.host, run_chain_command(
            repo_in_job, spec_paths, args.runs_dir, args.provider, args.usd_per_hour, head, args.hf_home, container,
        ))
        launched_utc = utc_now()
        pairs = list(zip(spec_paths, specs, strict=True))
        entries = []
        for path, spec in pairs:
            entries.append({
                "run_id": spec.run_id, "spec_path": path, "call_id": f"{args.host}:{spec.run_id}", "host": args.host,
                "container": container, "host_run_dir": f"{args.host_runs_dir}/{spec.run_id}",
                "log": f"{args.host_runs_dir}/{pathlib.Path(path).stem}.log", "spawned_utc": launched_utc,
            })
            print(f"STARTED {spec.run_id} on {args.host}")

    register_path = launch_register_path(args.register_dir, args.label, launched_utc)
    rows = [ledger_row(spec, launched_utc, f"{args.host}:{spec.run_id}", "LAUNCHED", provider=args.provider,
                       preregistration=args.preregistration) for _, spec in pairs]
    write_launch_register(register_path, head, args.host, launched_utc, entries, preregistration=args.preregistration,
                          extra=extra)
    append_rows(args.ledger, rows)
    print(f"commit {head}")
    print(f"wrote {register_path}")
    print(f"appended {len(rows)} rows to {args.ledger}")


if __name__ == "__main__":
    main()

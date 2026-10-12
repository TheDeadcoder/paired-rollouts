"""The droplet-side job queue (pairedrl.ops.queue): run it, append to it, or print its status.

    python scripts/run_queue.py run --queue-dir /work/runs/queue --runs-dir /work/runs --provider digitalocean --usd-per-hour 1.99
    python scripts/run_queue.py append --queue-dir /work/runs/queue --entry-json '{"kind": "pair", ...}'
    python scripts/run_queue.py status --queue-dir /work/runs/queue

`run` holds the queue's runner lock, runs the entries in order (an entry is one job or a concurrent pair on slots 0
and 1) under the stall watchdog, and exits once no entry is left, recording when the queue went empty; a second
runner on the same queue exits at once. `append` adds an entry under the queue lock and prints
`APPENDED <id> RUNNER_ALIVE` or `APPENDED <id> RUNNER_IDLE`; on RUNNER_IDLE the caller starts a runner
(scripts/launch_remote.py does both). Entries record their commit and checkout, and every job runs from that
checkout. The entry format is in pairedrl/ops/queue.py.
"""

import argparse
import json
import pathlib
import sys

from pairedrl.ops.queue import (
    CHECKPOINT_EVERY,
    POLL_S,
    QueueRunner,
    append_entry,
    format_status,
    read_entries,
    read_status,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the queue until it is empty")
    run.add_argument("--queue-dir", required=True)
    run.add_argument("--runs-dir", required=True)
    run.add_argument("--provider", required=True)
    run.add_argument("--usd-per-hour", type=float, required=True)
    run.add_argument("--poll-s", type=float, default=POLL_S)
    run.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                     help="training steps between manifest writes of queued training jobs")
    run.add_argument("--cache-root", default=None, help="parent of the per-slot temp and compile-cache directories")
    append = sub.add_parser("append", help="append one entry; prints its id and whether a runner is alive")
    append.add_argument("--queue-dir", required=True)
    append.add_argument("--entry-json", required=True)
    status = sub.add_parser("status", help="print the queue's status")
    status.add_argument("--queue-dir", required=True)
    args = parser.parse_args()

    if args.command == "run":
        runner = QueueRunner(args.queue_dir, args.runs_dir, args.provider, args.usd_per_hour, poll_s=args.poll_s,
                             checkpoint_every=args.checkpoint_every, cache_root=args.cache_root)
        return runner.run()
    if args.command == "append":
        entry_id, alive = append_entry(pathlib.Path(args.queue_dir), json.loads(args.entry_json))
        print(f"APPENDED {entry_id} {'RUNNER_ALIVE' if alive else 'RUNNER_IDLE'}")
        return 0
    print(format_status(read_status(args.queue_dir), read_entries(args.queue_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

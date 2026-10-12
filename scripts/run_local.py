"""Run one spec on this machine (a DigitalOcean droplet, any GPU box) with the job body and outputs of the Modal
function: run_manifest.json, episodes.jsonl, groups.jsonl, trainer_log_history.json, summary.json, checkpoints.

    python scripts/run_local.py --spec configs/x.json --runs-dir /work/runs --provider digitalocean --usd-per-hour 1.99
        [--slot 0|1 --start-offset-s 0 --shares-gpu-with <run_id> --queue-entry <id>]

A relaunch under the same run id preserves the previous attempt under attempt<n>/ and resumes from the newest
trainer checkpoint. With --slot (two jobs on one GPU, scripts/run_queue.py) the job takes that slot's ports and temp
and compile-cache directories unless the environment already sets them, and the manifest records its placement. The
package must be imported from this checkout (the queue sets PYTHONPATH to it). Exit code 0 when the manifest says
COMPLETE, 1 otherwise.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys

import pairedrl
from pairedrl.ops.job import run_job
from pairedrl.ops.queue import SLOT_DIRS, slot_environment
from pairedrl.train.runner import RunSpec


def git_commit(repo: pathlib.Path) -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=False).stdout.strip()
    return (head or "unknown") + ("-dirty" if dirty else "")


def check_import_path(repo: pathlib.Path) -> None:
    """The commit recorded for the run is this checkout's, so the code that runs must be this checkout's too."""
    imported = pathlib.Path(pairedrl.__file__).resolve().parent
    expected = (repo / "src" / "pairedrl").resolve()
    if imported != expected:
        raise SystemExit(f"pairedrl is imported from {imported}, not from this checkout; run with PYTHONPATH={repo / 'src'}")


def placement(args) -> dict | None:
    """Apply the slot's environment (keeping whatever the caller already set) and describe where the job runs."""
    if args.slot is None:
        return None
    for key, value in slot_environment(args.slot).items():
        os.environ.setdefault(key, value)
    for key in SLOT_DIRS:
        pathlib.Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    return {"slot": args.slot, "ports": {k: os.environ[k] for k in ("MASTER_ADDR", "MASTER_PORT", "VLLM_PORT")},
            "dirs": {k: os.environ[k] for k in SLOT_DIRS}, "start_offset_s": args.start_offset_s,
            "shares_gpu_with": args.shares_gpu_with, "queue_entry": args.queue_entry}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--data-dir", default="data/tasks")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--usd-per-hour", type=float, required=True)
    parser.add_argument("--commit", default=None, help="the launcher's commit; defaults to this checkout's HEAD")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--slot", type=int, choices=(0, 1), default=None, help="GPU slot when two jobs share one GPU")
    parser.add_argument("--start-offset-s", type=float, default=0.0, help="seconds after the other job of its pair it started")
    parser.add_argument("--shares-gpu-with", default=None, help="run id of the other job on the GPU")
    parser.add_argument("--queue-entry", default=None, help="the queue entry this job belongs to")
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    check_import_path(repo)
    where = placement(args)
    spec = RunSpec.from_dict(json.loads(pathlib.Path(args.spec).read_text(encoding="utf-8")))
    code_commit = git_commit(repo)
    result = run_job(
        spec, pathlib.Path(args.runs_dir) / spec.run_id, args.data_dir, args.commit or code_commit, code_commit,
        provider=args.provider, usd_per_hour=args.usd_per_hour, checkpoint_every=args.checkpoint_every,
        placement=where,
    )
    manifest = result["manifest"]
    print(f"{spec.run_id}: {manifest['status']} | episodes {manifest.get('episodes_logged')} | "
          f"wall {manifest.get('wall_time_s')} s | est {manifest.get('estimated_cost_usd')} USD"
          + (f" | error {manifest['error']}" if manifest.get("error") else ""))
    return 0 if manifest["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())

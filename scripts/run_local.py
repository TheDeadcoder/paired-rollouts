"""Run one spec on this machine (a DigitalOcean droplet, any GPU box) with the job body and outputs of the Modal
function: run_manifest.json, episodes.jsonl, groups.jsonl, trainer_log_history.json, summary.json, checkpoints.

    python scripts/run_local.py --spec configs/x.json --runs-dir /work/runs --provider digitalocean --usd-per-hour 1.99

A relaunch under the same run id preserves the previous attempt under attempt<n>/ and resumes from the newest
trainer checkpoint. Exit code 0 when the manifest says COMPLETE, 1 otherwise.
"""

import argparse
import json
import pathlib
import subprocess
import sys

from pairedrl.ops.job import run_job
from pairedrl.train.runner import RunSpec


def git_commit(repo: pathlib.Path) -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=False).stdout.strip()
    return (head or "unknown") + ("-dirty" if dirty else "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--data-dir", default="data/tasks")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--usd-per-hour", type=float, required=True)
    parser.add_argument("--commit", default=None, help="the launcher's commit; defaults to this checkout's HEAD")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    spec = RunSpec.from_dict(json.loads(pathlib.Path(args.spec).read_text(encoding="utf-8")))
    code_commit = git_commit(repo)
    result = run_job(
        spec, pathlib.Path(args.runs_dir) / spec.run_id, args.data_dir, args.commit or code_commit, code_commit,
        provider=args.provider, usd_per_hour=args.usd_per_hour, checkpoint_every=args.checkpoint_every,
    )
    manifest = result["manifest"]
    print(f"{spec.run_id}: {manifest['status']} | episodes {manifest.get('episodes_logged')} | "
          f"wall {manifest.get('wall_time_s')} s | est {manifest.get('estimated_cost_usd')} USD"
          + (f" | error {manifest['error']}" if manifest.get("error") else ""))
    return 0 if manifest["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())

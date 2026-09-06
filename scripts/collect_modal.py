"""Report, download and (optionally) cancel the runs recorded in a launch register.

Safe to re-run at any time from any machine with Modal credentials; it only reads the volume and the function-call
states unless --cancel is given. Exit code 0 when every run in the register has reached a terminal state, 1 otherwise.
"""

import argparse
import json
import pathlib
import sys

import modal
import modal.exception
from modal.volume import FileEntryType

from pairedrl.ops.ledger import TERMINAL, latest_launch_register, read_launch_register, set_status

RUNS_VOLUME = "pairedrl-runs"


def call_state(call_id: str) -> tuple[str, str | None]:
    """Terminal status from the function call if it has one, RUNNING while it runs, EXPIRED if Modal dropped the result."""
    call = modal.FunctionCall.from_id(call_id)
    try:
        result = call.get(timeout=0)
    except TimeoutError:
        return "RUNNING", None
    except modal.exception.OutputExpiredError:
        return "EXPIRED", None
    except Exception as e:  # noqa: BLE001
        return "INFRA_FAILED", f"{type(e).__name__}: {e}"
    return result["manifest"]["status"], None


def volume_view(volume, run_id: str) -> tuple[dict | None, int]:
    try:
        entries = volume.listdir(run_id)
    except Exception:  # noqa: BLE001
        return None, 0
    episodes_bytes = next((e.size for e in entries if e.path.endswith("episodes.jsonl")), 0)
    if not any(e.path.endswith("run_manifest.json") for e in entries):
        return None, episodes_bytes
    data = b"".join(volume.read_file(f"{run_id}/run_manifest.json"))
    return json.loads(data.decode("utf-8")), episodes_bytes


WEIGHT_DIRS = ("trainer/", "adapter_final/")


def download(volume, run_id: str, runs_dir: pathlib.Path, with_weights: bool = False) -> bool:
    """Copy the run's files from the volume; trainer checkpoints and the final adapter (hundreds of MB per run)
    stay on the volume unless `with_weights` is given."""
    try:
        entries = volume.listdir(run_id, recursive=True)
    except Exception as e:  # noqa: BLE001
        print(f"  listdir failed: {type(e).__name__}: {e}")
        return False
    copied = 0
    for entry in entries:
        if getattr(entry, "type", FileEntryType.FILE) != FileEntryType.FILE:
            continue
        rel = entry.path[len(run_id) + 1:] if entry.path.startswith(run_id + "/") else entry.path
        if not with_weights and rel.startswith(WEIGHT_DIRS):
            continue
        dest = runs_dir / run_id / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            f.writelines(volume.read_file(f"{run_id}/{rel}"))
        copied += 1
    return copied > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", help="launch register JSON; defaults to the newest under --register-dir")
    parser.add_argument("--register-dir", default="registers/launches")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--ledger", default="docs/RUN_LEDGER.md")
    parser.add_argument("--partial", action="store_true", help="also download runs that are still running")
    parser.add_argument("--cancel", action="store_true", help="cancel every run in the register that is still running")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--with-weights", action="store_true", help="also download trainer checkpoints and the final adapter")
    args = parser.parse_args()

    register_path = pathlib.Path(args.register) if args.register else latest_launch_register(args.register_dir)
    register = read_launch_register(register_path)
    volume = modal.Volume.from_name(RUNS_VOLUME)
    runs_dir = pathlib.Path(args.runs_dir)
    print(f"register {register_path} (commit {register['commit']}, launched {register['launched_utc']})")
    all_terminal = True
    for entry in register["entries"]:
        run_id, call_id = entry["run_id"], entry["call_id"]
        state, detail = call_state(call_id)
        manifest, episodes_bytes = volume_view(volume, run_id)
        status = manifest.get("status", "?") if manifest else "?"
        if state in ("RUNNING", "EXPIRED") and status in TERMINAL:
            state = status
        if state == "RUNNING" and args.cancel:
            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
            state = "CANCELLED"
        phases = [t["what"] for t in manifest.get("eval_timings", [])] if manifest else []
        line = f"{run_id}: {state}"
        if manifest:
            line += (
                f" | manifest {status} | episodes {manifest.get('episodes_logged', 0)} | train steps "
                f"{manifest.get('train_steps_logged', 0)} | phases {phases} | wall {manifest.get('wall_time_s')} s"
                f" | est {manifest.get('estimated_cost_usd')} USD | peak {manifest.get('peak_mem_gb')} GB"
                f" | progress {manifest.get('progress_utc')} | vllm {manifest.get('vllm')}"
            )
            if manifest.get("error"):
                line += f" | error {manifest['error']}"
        else:
            line += f" | no manifest on the volume yet (episodes.jsonl {episodes_bytes} bytes)"
        if detail:
            line += f" | {detail}"
        if entry.get("dashboard_url"):
            line += f" | {entry['dashboard_url']}"
        print(line)
        terminal = state in TERMINAL
        all_terminal = all_terminal and terminal
        if terminal:
            set_status(args.ledger, call_id, state)
        if not args.no_download and (terminal or args.partial) and manifest is not None:
            ok = download(volume, run_id, runs_dir, with_weights=args.with_weights)
            print(f"  downloaded to {runs_dir / run_id}" if ok else "  download FAILED")
    return 0 if all_terminal else 1


if __name__ == "__main__":
    sys.exit(main())

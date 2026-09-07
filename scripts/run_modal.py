"""Deployed Modal app that trains or evaluates one run spec per container.

Deploy once per commit with `modal deploy scripts/run_modal.py`, spawn runs with `scripts/launch_modal.py`, and
collect them with `scripts/collect_modal.py`. Runs never depend on the launching machine: they are spawned onto the
deployed app, write everything to the pairedrl-runs volume, and checkpoint after every evaluation phase and every
few training steps.
"""

import datetime as dt
import importlib.metadata as md
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import time

import modal

APP_NAME = "pairedrl-train"
FUNCTION_NAME = "run"
HF_CACHE = "/cache/hf"
RUNS = "/runs"
RUNS_VOLUME = "pairedrl-runs"
H100_USD_PER_HOUR = 3.95
CHECKPOINT_EVERY_STEPS = 10


def local_commit() -> str:
    """HEAD of the deploying checkout, suffixed with -dirty when the tree has uncommitted changes."""
    if not modal.is_local():
        return os.environ.get("PAIREDRL_GIT_COMMIT", "")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False).stdout.strip()
    if not head:
        return "unknown"
    return head + ("-dirty" if dirty else "")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.27.1",
        "trl[vllm,peft]==1.12.0",
        "torch==2.13.0",
        "transformers==5.16.1",
        "peft==0.20.0",
        "accelerate==1.14.0",
        "datasets==5.0.1",
        "flash-linear-attention==0.5.2",
        "numpy>=1.26",
        "pydantic>=2.7",
        "pyyaml>=6.0",
    )
    .env({
        "HF_HOME": HF_CACHE,
        "TRL_EXPERIMENTAL_SILENCE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PAIREDRL_GIT_COMMIT": local_commit(),
    })
    .add_local_python_source("pairedrl")
    .add_local_dir("data/tasks", remote_path="/root/data/tasks")
)
app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("pairedrl-hf-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)


def package_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def preserve_previous_attempt(out_dir: pathlib.Path) -> tuple[int, pathlib.Path | None]:
    """A retried container must not overwrite the previous attempt's evidence: its manifest and logs move to
    attempt<n>/, and the newest trainer checkpoint (if any) is returned so training can resume from it."""
    manifest_path = out_dir / "run_manifest.json"
    if not manifest_path.exists():
        return 1, None
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    n = int(previous.get("attempt", 1))
    keep = out_dir / f"attempt{n}"
    keep.mkdir(exist_ok=True)
    for name in ("run_manifest.json", "summary.json", "trainer_log_history.json", "episodes.jsonl", "groups.jsonl"):
        src = out_dir / name
        if src.exists():
            shutil.move(str(src), str(keep / name))
    checkpoints = sorted((out_dir / "trainer").glob("checkpoint-*"), key=lambda c: int(c.name.split("-")[1]))
    return n + 1, (checkpoints[-1] if checkpoints else None)


def vllm_facts(trainer) -> dict:
    try:
        cfg = trainer.vllm_generation.llm.llm_engine.vllm_config
        return {
            "enable_prefix_caching": cfg.cache_config.enable_prefix_caching,
            "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens,
            "max_num_seqs": cfg.scheduler_config.max_num_seqs,
            "max_model_len": cfg.model_config.max_model_len,
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


@app.function(
    image=image,
    gpu="H100",
    timeout=24 * 3600,
    volumes={HF_CACHE: hf_cache, RUNS: runs_volume},
    single_use_containers=True,
    retries=modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=60.0),
)
def run(spec_dict: dict, git_commit: str) -> dict:
    import torch

    from pairedrl.train.env_adapter import BackOfficeEnv
    from pairedrl.train.runner import (
        EvalSchedule,
        RunSpec,
        build_trainer,
        luck_share_tables_by_phase,
        read_episode_log,
        summarize_episodes,
        summarize_groups,
    )

    spec = RunSpec.from_dict(spec_dict)
    out_dir = pathlib.Path(RUNS) / spec.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    deployed_commit = os.environ.get("PAIREDRL_GIT_COMMIT", "")
    provenance_ok = git_commit == deployed_commit and not git_commit.endswith("-dirty")
    attempt, checkpoint = (preserve_previous_attempt(out_dir) if provenance_ok else (1, None))
    checkpoint = checkpoint if (checkpoint is not None and not spec.eval_only) else None
    BackOfficeEnv.attempt = attempt
    log_path = out_dir / "episodes.jsonl"
    group_log_path = out_dir / "groups.jsonl"
    manifest = {
        "run_id": spec.run_id,
        "spec": spec.to_dict(),
        "git_commit": git_commit,
        "deployed_commit": deployed_commit,
        "attempt": attempt,
        "resumed_from_checkpoint": str(checkpoint) if checkpoint else None,
        "started_utc": utc_now(),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {p: package_version(p) for p in ["torch", "transformers", "trl", "vllm", "peft", "flash-linear-attention"]},
        "fla_importable": importlib.util.find_spec("fla") is not None,
        "status": "RUNNING",
        "eval_timings": [],
        "step_timings": [],
        "vllm": {},
    }
    t0 = time.perf_counter()
    trainer = None
    schedule = None

    def count_lines(path: pathlib.Path) -> int:
        if not path.exists():
            return 0
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def write_state() -> None:
        manifest["progress_utc"] = utc_now()
        manifest["wall_time_s"] = round(time.perf_counter() - t0, 1)
        manifest["estimated_cost_usd"] = round(manifest["wall_time_s"] / 3600 * H100_USD_PER_HOUR, 2)
        manifest["eval_timings"] = schedule.timings if schedule is not None else []
        manifest["step_timings"] = schedule.step_timings if schedule is not None else []
        manifest["episodes_logged"] = count_lines(log_path)
        manifest["groups_logged"] = count_lines(group_log_path)
        manifest["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        history = list(trainer.state.log_history) if trainer is not None else []
        manifest["train_steps_logged"] = sum(1 for h in history if "loss" in h)
        (out_dir / "trainer_log_history.json").write_text(json.dumps(history, indent=1) + "\n")
        (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        runs_volume.commit()

    write_state()
    try:
        if not provenance_ok:
            raise RuntimeError(f"provenance: launched with commit {git_commit!r} but the deployed image is {deployed_commit!r}")
        trainer, splits, periodic = build_trainer(
            spec, "/root/data/tasks", out_dir / "trainer", log_path, group_log_path=group_log_path
        )
        manifest["vllm"] = vllm_facts(trainer)
        schedule = EvalSchedule(spec, trainer, splits, periodic, on_checkpoint=write_state, checkpoint_every=CHECKPOINT_EVERY_STEPS)
        write_state()
        if spec.eval_only:
            schedule.run_final(splits)
            if 0 in spec.diagnostic_steps:
                schedule.run_diagnostic(0)
        else:
            if not spec.train_only and checkpoint is None:
                schedule.run_periodic(0)
                if 0 in spec.diagnostic_steps:
                    schedule.run_diagnostic(0)
            trainer.add_callback(schedule.callback)
            trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
            if not spec.train_only:
                schedule.run_final(splits)
                if spec.steps in spec.diagnostic_steps:
                    schedule.run_diagnostic(spec.steps)
            trainer.model.save_pretrained(str(out_dir / "adapter_final"))
        manifest["status"] = "COMPLETE"
    except Exception as e:  # noqa: BLE001
        import traceback

        manifest["status"] = "FAILED"
        manifest["error"] = f"{type(e).__name__}: {e}"
        manifest["traceback"] = traceback.format_exc()[-8000:]
    manifest["finished_utc"] = utc_now()
    records = read_episode_log(log_path)
    summary = {
        "episodes_logged": len(records),
        "attempt": attempt,
        "by_phase_condition": summarize_episodes(records),
        "luck_share_tables": {phase: len(tables) for phase, tables in luck_share_tables_by_phase(records).items()},
        "training_groups": summarize_groups(read_episode_log(group_log_path)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_state()
    return {"manifest": manifest, "summary": summary}

"""Deployed Modal app that trains or evaluates one run spec per container.

Deploy once per commit with `modal deploy scripts/run_modal.py`, spawn runs with `scripts/launch_modal.py`, and
collect them with `scripts/collect_modal.py`. Runs never depend on the launching machine: they are spawned onto the
deployed app, write everything to the pairedrl-runs volume, and checkpoint after every evaluation phase and every
few training steps.
"""

import datetime as dt
import json
import os
import pathlib
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


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


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
    import importlib.metadata as md

    import torch

    from pairedrl.train.runner import (
        EvalSchedule,
        RunSpec,
        build_trainer,
        luck_share_tables,
        read_episode_log,
        summarize_episodes,
    )

    spec = RunSpec.from_dict(spec_dict)
    out_dir = pathlib.Path(RUNS) / spec.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "episodes.jsonl"
    if log_path.exists():
        log_path.unlink()
    deployed_commit = os.environ.get("PAIREDRL_GIT_COMMIT", "")
    manifest = {
        "run_id": spec.run_id,
        "spec": spec.to_dict(),
        "git_commit": git_commit,
        "deployed_commit": deployed_commit,
        "started_utc": utc_now(),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {p: md.version(p) for p in ["torch", "transformers", "trl", "vllm", "peft"]},
        "status": "RUNNING",
        "eval_timings": [],
        "vllm": {},
    }
    t0 = time.perf_counter()
    trainer = None
    schedule = None

    def count_episodes() -> int:
        if not log_path.exists():
            return 0
        with open(log_path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def write_state() -> None:
        manifest["progress_utc"] = utc_now()
        manifest["wall_time_s"] = round(time.perf_counter() - t0, 1)
        manifest["estimated_cost_usd"] = round(manifest["wall_time_s"] / 3600 * H100_USD_PER_HOUR, 2)
        manifest["eval_timings"] = schedule.timings if schedule is not None else []
        manifest["episodes_logged"] = count_episodes()
        manifest["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        history = list(trainer.state.log_history) if trainer is not None else []
        manifest["train_steps_logged"] = sum(1 for h in history if "loss" in h)
        (out_dir / "trainer_log_history.json").write_text(json.dumps(history, indent=1) + "\n")
        (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        runs_volume.commit()

    write_state()
    try:
        if git_commit != deployed_commit or git_commit.endswith("-dirty"):
            raise RuntimeError(f"provenance: launched with commit {git_commit!r} but the deployed image is {deployed_commit!r}")
        trainer, splits, periodic = build_trainer(spec, "/root/data/tasks", out_dir / "trainer", log_path)
        manifest["vllm"] = vllm_facts(trainer)
        schedule = EvalSchedule(spec, trainer, splits, periodic, on_checkpoint=write_state, checkpoint_every=CHECKPOINT_EVERY_STEPS)
        write_state()
        if spec.eval_only:
            schedule.run_final(splits)
            if 0 in spec.diagnostic_steps:
                schedule.run_diagnostic(0)
        else:
            if not spec.train_only:
                schedule.run_periodic(0)
                if 0 in spec.diagnostic_steps:
                    schedule.run_diagnostic(0)
            trainer.add_callback(schedule.callback)
            trainer.train()
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
        "by_phase_condition": summarize_episodes(records),
        "luck_share_tables": len(luck_share_tables(records)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_state()
    return {"manifest": manifest, "summary": summary}

"""One run spec, executed to completion in a run directory: the job body shared by the Modal function and the
local runner. Writes run_manifest.json, episodes.jsonl, groups.jsonl, trainer_log_history.json, summary.json,
trainer checkpoints and adapter_final/, checkpointing the state files after every evaluation phase and every
`checkpoint_every` training steps; a relaunch under the same run id preserves the previous attempt and resumes."""

import datetime as dt
import importlib.metadata as md
import importlib.util
import json
import pathlib
import shutil
import time

from pairedrl.train.runner import RunSpec

STATE_FILES = ("run_manifest.json", "summary.json", "trainer_log_history.json", "episodes.jsonl", "groups.jsonl")
VERSION_PACKAGES = ["torch", "transformers", "trl", "vllm", "peft", "flash-linear-attention"]


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def package_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def preserve_previous_attempt(out_dir: pathlib.Path) -> tuple[int, pathlib.Path | None]:
    """A relaunched run must not overwrite the previous attempt's evidence: its manifest and logs move to
    attempt<n>/, and the newest trainer checkpoint (if any) is returned so training can resume from it."""
    manifest_path = out_dir / "run_manifest.json"
    if not manifest_path.exists():
        return 1, None
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    n = int(previous.get("attempt", 1))
    keep = out_dir / f"attempt{n}"
    keep.mkdir(exist_ok=True)
    for name in STATE_FILES:
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


def count_lines(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run_job(
    spec: RunSpec,
    out_dir: pathlib.Path,
    data_dir: str,
    git_commit: str,
    deployed_commit: str,
    provider: str,
    usd_per_hour: float,
    commit_fn=None,
    checkpoint_every: int = 10,
) -> dict:
    """Execute one run. `git_commit` is the launcher's commit and `deployed_commit` the code's own; they must be
    equal and clean or the run refuses to start. `commit_fn` (optional) persists the run directory after every
    state write, for volumes that need an explicit commit."""
    import torch

    from pairedrl.train.env_adapter import BackOfficeEnv
    from pairedrl.train.runner import (
        EvalSchedule,
        build_trainer,
        luck_share_tables_by_phase,
        read_episode_log,
        summarize_episodes,
        summarize_groups,
    )

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
        "provider": provider,
        "usd_per_hour": usd_per_hour,
        "attempt": attempt,
        "resumed_from_checkpoint": str(checkpoint) if checkpoint else None,
        "started_utc": utc_now(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_backend": "hip" if getattr(torch.version, "hip", None) else "cuda",
        "torch_backend_version": getattr(torch.version, "hip", None) or torch.version.cuda,
        "versions": {p: package_version(p) for p in VERSION_PACKAGES},
        "fla_importable": importlib.util.find_spec("fla") is not None,
        "status": "RUNNING",
        "eval_timings": [],
        "step_timings": [],
        "vllm": {},
    }
    t0 = time.perf_counter()
    trainer = None
    schedule = None

    def write_state() -> None:
        manifest["progress_utc"] = utc_now()
        manifest["wall_time_s"] = round(time.perf_counter() - t0, 1)
        manifest["estimated_cost_usd"] = round(manifest["wall_time_s"] / 3600 * usd_per_hour, 2)
        manifest["eval_timings"] = schedule.timings if schedule is not None else []
        manifest["step_timings"] = schedule.step_timings if schedule is not None else []
        manifest["episodes_logged"] = count_lines(log_path)
        manifest["groups_logged"] = count_lines(group_log_path)
        manifest["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None
        history = list(trainer.state.log_history) if trainer is not None else []
        manifest["train_steps_logged"] = sum(1 for h in history if "loss" in h)
        (out_dir / "trainer_log_history.json").write_text(json.dumps(history, indent=1) + "\n", encoding="utf-8")
        (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if commit_fn is not None:
            commit_fn()

    write_state()
    try:
        if not provenance_ok:
            raise RuntimeError(f"provenance: launched with commit {git_commit!r} but the code is at {deployed_commit!r}")
        trainer, splits, periodic = build_trainer(
            spec, data_dir, out_dir / "trainer", log_path, group_log_path=group_log_path
        )
        manifest["vllm"] = vllm_facts(trainer)
        schedule = EvalSchedule(spec, trainer, splits, periodic, on_checkpoint=write_state, checkpoint_every=checkpoint_every)
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
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_state()
    return {"manifest": manifest, "summary": summary}

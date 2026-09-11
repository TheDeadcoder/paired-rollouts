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
import socket
import time

from pairedrl.train.runner import RunSpec

STATE_FILES = ("run_manifest.json", "summary.json", "trainer_log_history.json", "episodes.jsonl", "groups.jsonl")
VERSION_PACKAGES = ["torch", "transformers", "trl", "vllm", "peft", "flash-linear-attention"]
CHECKPOINT_MARKER = "trainer_state.json"
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_model.bin", "model.safetensors", "pytorch_model.bin")
ADAPTER_CONFIG = "adapter_config.json"
CHECKPOINT_STATE_FILES = ("optimizer.pt", "scheduler.pt")
RNG_FILE = "rng_state.pth"
RELAUNCH_FIELDS = ("relaunch_from_commit",)
MIN_COMMIT_PREFIX = 7


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def package_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def checkpoint_step(path: pathlib.Path) -> int:
    return int(path.name.split("-")[1])


def checkpoint_defects(path: pathlib.Path) -> list[str]:
    """Why a checkpoint directory cannot be resumed from or archived as complete. The pinned trainer saves the
    adapter (with its config), optimizer.pt, scheduler.pt and rng_state.pth, then trainer_state.json last, so an
    interrupted save lacks the marker; a transferred or pruned copy can lack any file, and resuming without the
    optimizer state would silently restart Adam at the recorded step. The marker must parse and its global_step
    must be the directory's step."""
    defects = []
    weights = [f for f in ADAPTER_FILES if (path / f).exists()]
    if not weights:
        defects.append("no weights")
    elif weights[0].startswith("adapter_") and not (path / ADAPTER_CONFIG).exists():
        defects.append(f"{ADAPTER_CONFIG} missing")
    defects.extend(f"{name} missing" for name in CHECKPOINT_STATE_FILES if not (path / name).exists())
    if not (path / RNG_FILE).exists() and not list(path.glob("rng_state_*.pth")):
        defects.append("rng_state missing")
    marker = path / CHECKPOINT_MARKER
    if not marker.exists():
        defects.append(f"{CHECKPOINT_MARKER} missing")
    else:
        try:
            state = json.loads(marker.read_text(encoding="utf-8"))
            step = int(state.get("global_step"))
        except (ValueError, TypeError, OSError):
            defects.append(f"{CHECKPOINT_MARKER} unreadable or without global_step")
        else:
            expected = checkpoint_step(path)
            if step != expected:
                defects.append(f"global_step {step} != {expected}")
    return defects


def is_complete_checkpoint(path: pathlib.Path) -> bool:
    return not checkpoint_defects(path)


def list_checkpoints(trainer_dir: pathlib.Path) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """(complete, incomplete) checkpoint directories under trainer/, each sorted by step."""
    dirs = [c for c in trainer_dir.glob("checkpoint-*") if c.is_dir() and c.name.split("-")[-1].isdigit()]
    complete = sorted((c for c in dirs if is_complete_checkpoint(c)), key=checkpoint_step)
    incomplete = sorted((c for c in dirs if not is_complete_checkpoint(c)), key=checkpoint_step)
    return complete, incomplete


def newest_complete_checkpoint(trainer_dir: pathlib.Path) -> pathlib.Path | None:
    complete, _ = list_checkpoints(trainer_dir)
    return complete[-1] if complete else None


def preserve_previous_attempt(out_dir: pathlib.Path) -> tuple[int, pathlib.Path | None]:
    """A relaunched run must not overwrite the previous attempt's evidence: its manifest and logs move to
    attempt<n>/, and the newest complete trainer checkpoint (if any) is returned so training can resume from it."""
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
    return n + 1, newest_complete_checkpoint(out_dir / "trainer")


def commit_named(named: str | None, actual: str | None) -> bool:
    """Whether `named` (a full hash or a prefix of at least MIN_COMMIT_PREFIX characters) designates `actual`."""
    if not named or not actual or len(named) < MIN_COMMIT_PREFIX:
        return False
    return actual.startswith(named) or named.startswith(actual)


def pool_hashes(task_pools: dict | None) -> dict[str, str | None]:
    return {name: (v or {}).get("sha256") for name, v in (task_pools or {}).items()}


def refusal_reason(
    spec: RunSpec, out_dir: pathlib.Path, git_commit: str, deployed_commit: str, task_pools: dict | None = None
) -> str | None:
    """Why this invocation must not touch the run directory: a launch commit that is not the code's own, a repeat
    of a COMPLETE run, a relaunch under a run id whose previous attempt had a different spec, a relaunch at a
    different commit than the previous attempt unless the spec names that commit in `relaunch_from_commit` (an
    infrastructure-only change recorded in docs/DEVIATIONS.md), or task pools that differ from the previous
    attempt's. A spec with `extend_previous` (smoke tests of the resume path) may continue a completed or
    differently specified run."""
    if git_commit != deployed_commit or git_commit.endswith("-dirty"):
        return f"provenance: launched with commit {git_commit!r} but the code is at {deployed_commit!r}"
    manifest_path = out_dir / "run_manifest.json"
    if not manifest_path.exists() or spec.extend_previous:
        return None
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    attempt = previous.get("attempt")
    if previous.get("status") == "COMPLETE":
        return f"{spec.run_id} is already COMPLETE (attempt {attempt}); a repeat needs a new run id"
    previous_spec = RunSpec.from_dict(previous["spec"]).to_dict()
    current = spec.to_dict()
    changed = sorted(
        k for k in set(previous_spec) | set(current) if previous_spec.get(k) != current.get(k) and k not in RELAUNCH_FIELDS
    )
    if changed:
        return f"{spec.run_id}: relaunched with a different spec (fields {changed}); a different protocol needs a new run id"
    previous_commit = previous.get("deployed_commit") or previous.get("git_commit")
    if previous_commit and previous_commit != deployed_commit and not commit_named(spec.relaunch_from_commit, previous_commit):
        return (
            f"{spec.run_id}: attempt {attempt} ran at commit {previous_commit!r} and this launch is at {deployed_commit!r}; "
            f"a relaunch across commits must name the previous commit in relaunch_from_commit"
        )
    previous_pools = pool_hashes(previous.get("task_pools"))
    if task_pools is not None and previous_pools and previous_pools != pool_hashes(task_pools):
        differing = sorted(k for k in set(previous_pools) | set(pool_hashes(task_pools)) if previous_pools.get(k) != pool_hashes(task_pools).get(k))
        return f"{spec.run_id}: task pools {differing} differ from attempt {attempt}'s; the frozen pools are part of the protocol"
    return None


def write_refusal(spec: RunSpec, out_dir: pathlib.Path, reason: str, git_commit: str, deployed_commit: str, provider: str) -> dict:
    """Record a refused invocation beside the run without touching its state files."""
    record = {
        "run_id": spec.run_id,
        "spec": spec.to_dict(),
        "git_commit": git_commit,
        "deployed_commit": deployed_commit,
        "provider": provider,
        "status": "REFUSED",
        "error": reason,
        "refused_utc": utc_now(),
    }
    refused = out_dir / "refused"
    refused.mkdir(parents=True, exist_ok=True)
    stamp = record["refused_utc"].replace(":", "").replace("+0000", "Z")
    (refused / f"{stamp}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


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
    reraise_attempts: int = 0,
) -> dict:
    """Execute one run. `git_commit` is the launcher's commit and `deployed_commit` the code's own; they must be
    equal and clean or the invocation is refused before the run directory is touched, as is a repeat of a COMPLETE
    run or a relaunch with a different spec. `commit_fn` (optional) persists the run directory after every state
    write, for volumes that need an explicit commit. A FAILED attempt numbered at most `reraise_attempts` re-raises
    its error after recording it, so a platform retry policy (Modal) resumes it from the newest checkpoint; later
    attempts return normally, which bounds the retries of a deterministic failure."""
    import torch

    from pairedrl.train.env_adapter import BackOfficeEnv
    from pairedrl.train.runner import (
        EvalSchedule,
        build_trainer,
        luck_share_tables_by_phase,
        read_episode_log,
        summarize_episodes,
        summarize_groups,
        task_pool_digests,
    )

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_pools = task_pool_digests(data_dir)
    reason = refusal_reason(spec, out_dir, git_commit, deployed_commit, task_pools)
    if reason is not None:
        record = write_refusal(spec, out_dir, reason, git_commit, deployed_commit, provider)
        if commit_fn is not None:
            commit_fn()
        return {"manifest": record, "summary": {}}
    attempt, checkpoint = preserve_previous_attempt(out_dir)
    checkpoint = checkpoint if (checkpoint is not None and not spec.eval_only) else None
    _, incomplete = list_checkpoints(out_dir / "trainer")
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
        "resumed_from_step": checkpoint_step(checkpoint) if checkpoint else None,
        "incomplete_checkpoints_skipped": [c.name for c in incomplete],
        "incomplete_checkpoint_defects": {c.name: checkpoint_defects(c) for c in incomplete},
        "task_pools": task_pools,
        "started_utc": utc_now(),
        "hostname": socket.gethostname(),
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
        trainer, splits, periodic = build_trainer(
            spec, data_dir, out_dir / "trainer", log_path, group_log_path=group_log_path
        )
        manifest["vllm"] = vllm_facts(trainer)
        manifest["model_commit_hash"] = getattr(getattr(trainer.model, "config", None), "_commit_hash", None)
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
    if manifest["status"] == "FAILED" and attempt <= reraise_attempts:
        raise RuntimeError(f"attempt {attempt} failed and is handed to the platform retry policy: {manifest['error']}")
    return {"manifest": manifest, "summary": summary}

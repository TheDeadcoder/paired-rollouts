"""Train or evaluate one run spec on Modal. One container per run; outputs land on the pairedrl-runs volume."""

import datetime as dt
import json
import pathlib
import subprocess
import time

import modal

APP_NAME = "pairedrl-train"
HF_CACHE = "/cache/hf"
RUNS = "/runs"
H100_USD_PER_HOUR = 3.95

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
    .env({"HF_HOME": HF_CACHE, "TRL_EXPERIMENTAL_SILENCE": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("pairedrl")
    .add_local_dir("data/tasks", remote_path="/root/data/tasks")
)
app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("pairedrl-hf-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("pairedrl-runs", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=8 * 3600,
    volumes={HF_CACHE: hf_cache, RUNS: runs_volume},
    single_use_containers=True,
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
    manifest = {
        "run_id": spec.run_id,
        "spec": spec.to_dict(),
        "git_commit": git_commit,
        "started_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {p: md.version(p) for p in ["torch", "transformers", "trl", "vllm", "peft"]},
        "status": "RUNNING",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    t0 = time.perf_counter()
    trainer = None
    schedule = None
    try:
        trainer, splits, periodic = build_trainer(spec, "/root/data/tasks", out_dir / "trainer", log_path)
        schedule = EvalSchedule(spec, trainer, splits, periodic)
        if spec.eval_only:
            schedule.run_final(splits)
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
    wall = time.perf_counter() - t0
    manifest["finished_utc"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    manifest["wall_time_s"] = round(wall, 1)
    manifest["estimated_cost_usd"] = round(wall / 3600 * H100_USD_PER_HOUR, 2)
    manifest["eval_timings"] = schedule.timings if schedule else []
    manifest["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    history = getattr(trainer.state, "log_history", []) if trainer is not None else []
    (out_dir / "trainer_log_history.json").write_text(json.dumps(history, indent=1) + "\n")
    records = read_episode_log(log_path)
    summary = {
        "episodes_logged": len(records),
        "by_phase_condition": summarize_episodes(records),
        "luck_share_tables": len(luck_share_tables(records)),
        "step_times_s": [h["step_time"] for h in history if "step_time" in h][:200],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    runs_volume.commit()
    return {"manifest": manifest, "summary": summary}


@app.local_entrypoint()
def main(specs: str, download: bool = True):
    paths = [pathlib.Path(p.strip()) for p in specs.split(",") if p.strip()]
    spec_dicts = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    results = list(run.map(spec_dicts, [commit] * len(spec_dicts)))
    for result in results:
        run_id = result["manifest"]["run_id"]
        local = pathlib.Path("outputs") / "runs" / run_id
        local.mkdir(parents=True, exist_ok=True)
        (local / "summary.json").write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n")
        (local / "run_manifest.json").write_text(json.dumps(result["manifest"], indent=2, sort_keys=True) + "\n")
        m = result["manifest"]
        print(f"{run_id}: {m['status']} wall {m['wall_time_s']} s est {m['estimated_cost_usd']} USD peak {m['peak_mem_gb']} GB")
        if m["status"] != "COMPLETE":
            print(m.get("error"))
        for key, agg in result["summary"]["by_phase_condition"].items():
            print(f"  {key}: n={agg['episodes']} true={agg['true_success']:.3f} obs={agg['observed_reward']:.3f} "
                  f"recov={agg['recovery_success']} exposed={agg['exposed_frac']:.2f} calls={agg['mean_calls']:.1f}")
        if download:
            subprocess.run(["modal", "volume", "get", "pairedrl-runs", f"{run_id}", str(local.parent), "--force"], check=False)

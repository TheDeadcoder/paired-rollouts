"""Deployed Modal app that trains or evaluates one run spec per container.

Deploy once per commit with `modal deploy scripts/run_modal.py`, spawn runs with `scripts/launch_modal.py`, and
collect them with `scripts/collect_modal.py`. Runs never depend on the launching machine: they are spawned onto the
deployed app, write everything to the pairedrl-runs volume, and checkpoint after every evaluation phase and every
few training steps. The job body lives in `pairedrl.ops.job` and is shared with `scripts/run_local.py`.
"""

import os
import pathlib
import subprocess

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


@app.function(
    image=image,
    gpu="H100",
    timeout=24 * 3600,
    volumes={HF_CACHE: hf_cache, RUNS: runs_volume},
    single_use_containers=True,
    retries=modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=60.0),
)
def run(spec_dict: dict, git_commit: str) -> dict:
    from pairedrl.ops.job import run_job
    from pairedrl.train.runner import RunSpec

    spec = RunSpec.from_dict(spec_dict)
    return run_job(
        spec, pathlib.Path(RUNS) / spec.run_id, "/root/data/tasks", git_commit,
        os.environ.get("PAIREDRL_GIT_COMMIT", ""), provider="modal", usd_per_hour=H100_USD_PER_HOUR,
        commit_fn=runs_volume.commit, checkpoint_every=CHECKPOINT_EVERY_STEPS, reraise_attempts=1,
    )


@app.function(
    image=image,
    gpu="H100",
    timeout=24 * 3600,
    memory=65536,
    cpu=8.0,
    volumes={HF_CACHE: hf_cache, RUNS: runs_volume},
    single_use_containers=True,
    retries=modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=60.0),
)
def probe(spec_dict: dict, git_commit: str) -> dict:
    from pairedrl.train.probe import ProbeSpec, run_probe

    spec = ProbeSpec.from_dict(spec_dict)
    return run_probe(
        spec, pathlib.Path(RUNS), "/root/data/tasks", pathlib.Path(RUNS) / spec.probe_id, git_commit,
        os.environ.get("PAIREDRL_GIT_COMMIT", ""), provider="modal", usd_per_hour=H100_USD_PER_HOUR,
        commit_fn=runs_volume.commit,
    )

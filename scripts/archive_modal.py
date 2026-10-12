"""Deployed Modal app that archives run weights from the pairedrl-runs volume to the project's Hugging Face model
repository, so nothing passes through the laptop.

    MODAL_PROFILE=<profile> modal deploy scripts/archive_modal.py
    MODAL_PROFILE=<profile> python scripts/archive_weights.py launch --runs <run_id>,<run_id> --label <label>
    MODAL_PROFILE=<profile> python scripts/archive_weights.py collect --register registers/launches/<utc>_<label>.json

The function mounts the runs volume read-only and reads HF_TOKEN from the Modal secret HF_SECRET. For each run it calls
upload_run from scripts/upload_weights.py (verify_run with weights, one Hugging Face commit per run, the weights record)
and then downloads the adapter of the run's last checkpoint from the Hub at the recorded revision, into a fresh cache,
and checks its sha256 against the record (pairedrl.ops.archive). Records and per-run status files go to the
pairedrl-archive volume under the launch label; scripts/archive_weights.py collect copies the records into
registers/weights/. A retried container skips verified runs and never uploads a run whose record exists. Nothing on the
runs volume is written or deleted.
"""

import importlib.util
import os
import pathlib

import modal

APP_NAME = "pairedrl-archive"
FUNCTION_NAME = "archive"
RUNS = "/runs"
ARCHIVE = "/archive"
RUNS_VOLUME = "pairedrl-runs"
ARCHIVE_VOLUME = "pairedrl-archive"
HF_SECRET = "hf-token"
SCRIPTS = "/root/scripts"
DATA = "/root/data/tasks"


def local_commit() -> str:
    """HEAD of the deploying checkout, suffixed with -dirty exactly when scripts/archive_weights.py's launch would see
    uncommitted changes (launch records excluded), so the two sides agree on the commit string."""
    if not modal.is_local():
        return os.environ.get("PAIREDRL_GIT_COMMIT", "")
    from pairedrl.ops.ledger import git_state

    head, dirty = git_state()
    return head + ("-dirty" if dirty else "")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub==1.30.0", "numpy>=1.26", "pydantic>=2.7", "pyyaml>=6.0")
    .env({"PAIREDRL_GIT_COMMIT": local_commit()})
    .add_local_python_source("pairedrl")
    .add_local_dir("data/tasks", remote_path=DATA)
    .add_local_file("scripts/upload_weights.py", remote_path=f"{SCRIPTS}/upload_weights.py")
    .add_local_file("scripts/verify_run.py", remote_path=f"{SCRIPTS}/verify_run.py")
)
app = modal.App(APP_NAME)
runs_volume = modal.Volume.from_name(RUNS_VOLUME)
archive_volume = modal.Volume.from_name(ARCHIVE_VOLUME, create_if_missing=True)


def deployed_code() -> dict:
    """The archiving code as deployed, keyed like scripts/archive_weights.py's launch-side digest."""
    import pairedrl.ops.archive as archive_ops
    from pairedrl.analysis import curves

    return archive_ops.code_digest({"scripts/upload_weights.py": f"{SCRIPTS}/upload_weights.py",
                                    "scripts/verify_run.py": f"{SCRIPTS}/verify_run.py",
                                    "src/pairedrl/ops/archive.py": archive_ops.__file__,
                                    "src/pairedrl/analysis/curves.py": curves.__file__})


def _download(repo: str, path: str, revision: str) -> str:
    import tempfile

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=path, revision=revision, repo_type="model",
                           cache_dir=tempfile.mkdtemp(prefix="hf-verify-"), force_download=True)


@app.function(
    image=image,
    cpu=4.0,
    memory=8192,
    timeout=6 * 3600,
    volumes={RUNS: runs_volume.read_only(), ARCHIVE: archive_volume},
    secrets=[modal.Secret.from_name(HF_SECRET)],
    single_use_containers=True,
    retries=modal.Retries(max_retries=1, backoff_coefficient=1.0, initial_delay=60.0),
)
def archive(run_ids: list[str], repo: str, accept_missing: dict, label: str, git_commit: str) -> dict:
    from pairedrl.ops.archive import archive_runs

    deployed = os.environ.get("PAIREDRL_GIT_COMMIT", "")
    code = deployed_code()
    if git_commit != deployed:
        return {"status": "REFUSED", "error": f"launched at {git_commit!r} but the deployed code is at {deployed!r}",
                "label": label, "code_sha256": code}
    spec = importlib.util.spec_from_file_location("upload_weights", f"{SCRIPTS}/upload_weights.py")
    upload = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upload)
    results = archive_runs(
        run_ids, RUNS, pathlib.Path(ARCHIVE) / label, repo, upload.upload_run, _download,
        accept_missing=accept_missing, data_dir=DATA, persist=archive_volume.commit,
        extra={"label": label, "git_commit": git_commit, "hf_secret": HF_SECRET},
    )
    done = all(r.get("status") == "VERIFIED" for r in results.values())
    return {"status": "COMPLETE" if done else "FAILED", "label": label, "git_commit": git_commit,
            "code_sha256": code, "results": results}

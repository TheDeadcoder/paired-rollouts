"""The archive loop (pairedrl.ops.archive) behind scripts/archive_modal.py: one upload per run, a sha256 check of the
last checkpoint's adapter downloaded at the recorded revision, resumable status files, final refusals and bounded
retries of transient upload errors."""

import hashlib
import json
import pathlib

import pytest

from pairedrl.ops import archive

ADAPTER = b"adapter weights at step 100"


def write_record(run_id, out_dir, step=100):
    files = [
        {"path": "trainer/checkpoint-80/adapter_model.safetensors", "bytes": 3, "sha256": hashlib.sha256(b"a80").hexdigest()},
        {"path": f"trainer/checkpoint-{step}/adapter_model.safetensors", "bytes": len(ADAPTER),
         "sha256": hashlib.sha256(ADAPTER).hexdigest()},
    ]
    record = {"run_id": run_id, "repo": "user/repo", "revision": f"rev-{run_id}", "commit_url": f"https://hf.co/c/{run_id}",
              "uploaded_utc": "2026-10-12T00:00:00+00:00", "checkpoints": {"complete": [80, step]}, "files": files,
              "total_bytes": 3 + len(ADAPTER)}
    out = pathlib.Path(out_dir) / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record))
    return record


class FakeHub:
    def __init__(self, tmp_path, content=ADAPTER, failures=None):
        self.tmp, self.content, self.failures = tmp_path, content, failures or {}
        self.uploads, self.downloads = [], []

    def upload(self, run_dir, repo, data_dir=None, out_dir=None, accept_missing=(), accept_reason=None):
        run_id = pathlib.Path(run_dir).name
        self.uploads.append((run_id, tuple(accept_missing), accept_reason, data_dir))
        pending = self.failures.get(run_id, [])
        if pending:
            raise pending.pop(0)
        return write_record(run_id, out_dir)

    def download(self, repo, path, revision):
        self.downloads.append((repo, path, revision))
        local = self.tmp / "downloads" / revision / path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(self.content)
        return local


def test_each_run_is_uploaded_once_verified_and_skipped_on_resume(tmp_path):
    hub, persisted = FakeHub(tmp_path), []
    out = tmp_path / "archive" / "label"
    results = archive.archive_runs(["run-a", "run-b"], tmp_path / "runs", out, "user/repo", hub.upload, hub.download,
                                   data_dir="data", persist=lambda: persisted.append(1), extra={"label": "label"})
    assert {r: s["status"] for r, s in results.items()} == {"run-a": "VERIFIED", "run-b": "VERIFIED"}
    assert [u[0] for u in hub.uploads] == ["run-a", "run-b"] and all(u[3] == "data" for u in hub.uploads)
    assert hub.downloads[0] == ("user/repo", "run-a/trainer/checkpoint-100/adapter_model.safetensors", "rev-run-a")
    status = json.loads((out / "status" / "run-a.json").read_text())
    assert status["verification"]["match"] and status["files"] == 2 and status["label"] == "label"
    assert status["commit_url"] == "https://hf.co/c/run-a" and len(persisted) == 4
    again = archive.archive_runs(["run-a", "run-b"], tmp_path / "runs", out, "user/repo", hub.upload, hub.download)
    assert len(hub.uploads) == 2 and len(hub.downloads) == 2 and again["run-b"]["status"] == "VERIFIED"


def test_a_refusal_is_final_and_transient_errors_are_retried(tmp_path):
    hub = FakeHub(tmp_path, failures={"refused": [ValueError("refused is not accepted with weights: status FAILED")],
                                      "flaky": [RuntimeError("503"), RuntimeError("timeout")]})
    sleeps = []
    results = archive.archive_runs(["refused", "flaky"], tmp_path / "runs", tmp_path / "out", "user/repo", hub.upload,
                                   hub.download, sleep=sleeps.append)
    assert results["refused"]["status"] == "FAILED" and "not accepted with weights" in results["refused"]["error"]
    assert [u[0] for u in hub.uploads] == ["refused", "flaky", "flaky", "flaky"] and sleeps == [60, 120]
    assert results["flaky"]["status"] == "VERIFIED" and "error" not in results["flaky"]


def test_an_existing_record_is_verified_not_uploaded_and_a_mismatch_is_reported(tmp_path):
    out = tmp_path / "out"
    write_record("run-a", out / "records")
    hub = FakeHub(tmp_path, content=b"something else")
    results = archive.archive_runs(["run-a"], tmp_path / "runs", out, "user/repo", hub.upload, hub.download)
    assert hub.uploads == [] and results["run-a"]["status"] == "MISMATCH"
    check = results["run-a"]["verification"]
    assert not check["match"] and check["sha256_downloaded"] == hashlib.sha256(b"something else").hexdigest()


def test_accepted_gaps_reach_the_upload_and_the_last_checkpoint_is_checked(tmp_path):
    hub = FakeHub(tmp_path)
    accept = {"gate": {"steps": [20, 40, 60], "reason": "kept only 80 and 100"}}
    archive.archive_runs(["gate", "other"], tmp_path / "runs", tmp_path / "out", "user/repo", hub.upload, hub.download,
                         accept_missing=accept)
    assert hub.uploads[0][:3] == ("gate", (20, 40, 60), "kept only 80 and 100")
    assert hub.uploads[1][:3] == ("other", (), None)
    record = write_record("x", tmp_path / "r", step=60)
    assert archive.last_adapter(record) == ("x/trainer/checkpoint-80/adapter_model.safetensors",
                                            hashlib.sha256(b"a80").hexdigest())


def test_archive_launcher_parses_accepted_gaps():
    pytest.importorskip("modal", reason="the launcher needs the optional modal package")
    import importlib.util

    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("archive_weights", root / "scripts" / "archive_weights.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.parse_accept(["gate1-c2-paired-s0=20, 40,60"], "why") == {
        "gate1-c2-paired-s0": {"steps": [20, 40, 60], "reason": "why"}}
    with pytest.raises(SystemExit, match="needs --accept-reason"):
        module.parse_accept(["gate1-c2-paired-s0=20"], None)

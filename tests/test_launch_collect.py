import importlib.util
import json
import pathlib
import sys

import pytest

from pairedrl.ops.ledger import LEDGER_HEADER

pytest.importorskip("modal", reason="cloud scripts need the optional modal package (pip install -e '.[cloud]')")

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCall:
    def __init__(self, call_id, result=None, error=None, running=False):
        self.object_id = call_id
        self.result, self.error, self.running = result, error, running
        self.cancelled = False

    def get(self, timeout=None):
        if self.running:
            raise TimeoutError()
        if self.error:
            raise self.error
        return self.result

    def cancel(self, terminate_containers=False):
        self.cancelled = True

    def get_dashboard_url(self):
        return f"https://modal.test/{self.object_id}"


class FakeEntry:
    def __init__(self, path, size):
        from modal.volume import FileEntryType

        self.path, self.size, self.type = path, size, FileEntryType.FILE


class FakeVolume:
    def __init__(self, files):
        self.files = files

    def listdir(self, path, recursive=False):
        if not any(k.startswith(path + "/") for k in self.files):
            raise FileNotFoundError(path)
        return [FakeEntry(k, len(v)) for k, v in self.files.items() if k.startswith(path + "/")]

    def read_file(self, path):
        yield self.files[path]


def setup_repo(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    ledger = tmp_path / "docs" / "RUN_LEDGER.md"
    ledger.write_text("# Run ledger\n\n" + LEDGER_HEADER + "\n|---|---|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    for run_id in ("calib-a", "calib-b"):
        (tmp_path / "configs" / f"{run_id}.json").write_text(json.dumps({
            "run_id": run_id, "model": "Qwen/Qwen3.5-2B", "condition": "CALIB", "arm": "paired", "p": 0.25,
            "eval_only": True, "final_eval_tasks": 200, "diagnostic_steps": [0], "notes": "calibration",
        }))
    monkeypatch.chdir(tmp_path)
    return ledger


def test_launch_writes_register_and_ledger(tmp_path, monkeypatch):
    ledger = setup_repo(tmp_path, monkeypatch)
    launch = load_script("launch_modal")
    spawned = []

    class FakeFunction:
        def spawn(self, spec_dict, commit):
            spawned.append((spec_dict["run_id"], commit))
            return FakeCall(f"fc-{len(spawned)}")

    monkeypatch.setattr(launch, "git_state", lambda: ("abc1234", False))
    monkeypatch.setattr(launch.modal.Function, "from_name", staticmethod(lambda app, name: FakeFunction()))
    monkeypatch.setattr(sys, "argv", ["launch_modal.py", "--specs", "configs/calib-a.json,configs/calib-b.json", "--label", "calib 3"])
    launch.main()
    assert spawned == [("calib-a", "abc1234"), ("calib-b", "abc1234")]
    registers = list((tmp_path / "registers" / "launches").glob("*_calib-3.json"))
    assert len(registers) == 1
    register = json.loads(registers[0].read_text())
    assert register["commit"] == "abc1234" and [e["call_id"] for e in register["entries"]] == ["fc-1", "fc-2"]
    rows = [line for line in ledger.read_text().splitlines() if line.startswith("| calib-")]
    assert len(rows) == 2 and all("| LAUNCHED |" in r for r in rows) and "call fc-2" in rows[1]


def test_launch_refuses_dirty_tree(tmp_path, monkeypatch):
    setup_repo(tmp_path, monkeypatch)
    launch = load_script("launch_modal")
    monkeypatch.setattr(launch, "git_state", lambda: ("abc1234", True))
    monkeypatch.setattr(sys, "argv", ["launch_modal.py", "--specs", "configs/calib-a.json", "--label", "x"])
    with pytest.raises(SystemExit):
        launch.main()


def test_collect_reports_downloads_and_updates_ledger(tmp_path, monkeypatch, capsys):
    ledger = setup_repo(tmp_path, monkeypatch)
    launch = load_script("launch_modal")
    calls = {}

    class FakeFunction:
        def spawn(self, spec_dict, commit):
            call = FakeCall(f"fc-{spec_dict['run_id']}")
            calls[call.object_id] = call
            return call

    monkeypatch.setattr(launch, "git_state", lambda: ("abc1234", False))
    monkeypatch.setattr(launch.modal.Function, "from_name", staticmethod(lambda app, name: FakeFunction()))
    monkeypatch.setattr(sys, "argv", ["launch_modal.py", "--specs", "configs/calib-a.json,configs/calib-b.json", "--label", "x"])
    launch.main()

    done_manifest = {"status": "COMPLETE", "episodes_logged": 3424, "eval_timings": [{"what": "final_clean"}], "wall_time_s": 10.0,
                     "estimated_cost_usd": 0.01, "peak_mem_gb": 40.0, "progress_utc": "t", "vllm": {"enable_prefix_caching": True}}
    running_manifest = dict(done_manifest, status="RUNNING", episodes_logged=800)
    calls["fc-calib-a"].result = {"manifest": done_manifest, "summary": {}}
    calls["fc-calib-b"].running = True
    volume = FakeVolume({
        "calib-a/run_manifest.json": json.dumps(done_manifest).encode(),
        "calib-a/episodes.jsonl": b"x" * 100,
        "calib-b/run_manifest.json": json.dumps(running_manifest).encode(),
    })
    collect = load_script("collect_modal")
    monkeypatch.setattr(collect.modal.FunctionCall, "from_id", staticmethod(lambda call_id: calls[call_id]))
    monkeypatch.setattr(collect.modal.Volume, "from_name", staticmethod(lambda name: volume))
    downloaded = []
    monkeypatch.setattr(collect, "download", lambda volume, run_id, runs_dir, with_weights=False: downloaded.append(run_id) or True)

    monkeypatch.setattr(sys, "argv", ["collect_modal.py"])
    assert collect.main() == 1
    out = capsys.readouterr().out
    assert "calib-a: COMPLETE" in out and "calib-b: RUNNING" in out and "episodes 800" in out
    assert downloaded == ["calib-a"]
    rows = {line.split("|")[1].strip(): line.split("|")[9].strip() for line in ledger.read_text().splitlines() if line.startswith("| calib-")}
    assert rows == {"calib-a": "COMPLETE", "calib-b": "LAUNCHED"}

    monkeypatch.setattr(sys, "argv", ["collect_modal.py", "--cancel", "--no-download"])
    assert collect.main() == 0
    assert calls["fc-calib-b"].cancelled
    rows = {line.split("|")[1].strip(): line.split("|")[9].strip() for line in ledger.read_text().splitlines() if line.startswith("| calib-")}
    assert rows["calib-b"] == "CANCELLED"

    calls["fc-calib-b"].running = False
    calls["fc-calib-b"].error = collect.modal.exception.FunctionTimeoutError("timeout")
    monkeypatch.setattr(sys, "argv", ["collect_modal.py", "--no-download"])
    collect.main()
    out = capsys.readouterr().out
    assert "calib-b: INFRA_FAILED" in out and "FunctionTimeoutError" in out


def test_collect_uses_manifest_when_result_expired(tmp_path, monkeypatch, capsys):
    setup_repo(tmp_path, monkeypatch)
    collect = load_script("collect_modal")
    (tmp_path / "registers" / "launches").mkdir(parents=True)
    (tmp_path / "registers" / "launches" / "20260905T000000_x.json").write_text(json.dumps({
        "launched_utc": "t", "commit": "abc", "app": "pairedrl-train",
        "entries": [{"run_id": "calib-a", "call_id": "fc-1", "spec_path": "configs/calib-a.json", "spawned_utc": "t", "dashboard_url": None}],
    }))
    manifest = {"status": "FAILED", "episodes_logged": 0, "eval_timings": [], "error": "RuntimeError: provenance", "vllm": {}}
    volume = FakeVolume({"calib-a/run_manifest.json": json.dumps(manifest).encode()})
    call = FakeCall("fc-1", error=collect.modal.exception.OutputExpiredError())
    monkeypatch.setattr(collect.modal.FunctionCall, "from_id", staticmethod(lambda call_id: call))
    monkeypatch.setattr(collect.modal.Volume, "from_name", staticmethod(lambda name: volume))
    monkeypatch.setattr(sys, "argv", ["collect_modal.py", "--no-download"])
    assert collect.main() == 0
    assert "calib-a: FAILED" in capsys.readouterr().out


def test_download_skips_weights_unless_asked(tmp_path):
    collect = load_script("collect_modal")
    volume = FakeVolume({
        "run-a/run_manifest.json": b"{}", "run-a/episodes.jsonl": b"{}\n", "run-a/groups.jsonl": b"{}\n",
        "run-a/attempt1/episodes.jsonl": b"{}\n", "run-a/trainer/checkpoint-20/adapter_model.safetensors": b"w",
        "run-a/adapter_final/adapter_model.safetensors": b"w",
    })
    assert collect.download(volume, "run-a", tmp_path)
    got = sorted(str(p.relative_to(tmp_path / "run-a")) for p in (tmp_path / "run-a").rglob("*") if p.is_file())
    assert got == ["attempt1/episodes.jsonl", "episodes.jsonl", "groups.jsonl", "run_manifest.json"]
    assert collect.download(volume, "run-a", tmp_path / "full", with_weights=True)
    assert (tmp_path / "full" / "run-a" / "trainer" / "checkpoint-20" / "adapter_model.safetensors").read_bytes() == b"w"
    assert not collect.download(FakeVolume({}), "missing", tmp_path)

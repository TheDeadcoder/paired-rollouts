import datetime as dt
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

from pairedrl.ops.job import count_lines, preserve_previous_attempt
from pairedrl.ops.ledger import LEDGER_HEADER
from pairedrl.ops.remote import (
    checkout_command,
    in_container,
    install_command,
    rsync_command,
    run_chain_command,
    ssh_command,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_command_builders():
    assert ssh_command("root@h", "ls")[-2:] == ["root@h", "ls"]
    assert checkout_command("/root/work/repo", "abc") == "cd /root/work/repo && git fetch -q origin && git checkout -q abc && git rev-parse HEAD"
    assert in_container(None, "echo hi") == "echo hi"
    assert in_container("pairedrl", "echo 'hi'") == "docker exec pairedrl bash -c 'echo '\"'\"'hi'\"'\"''"
    assert install_command("/work/repo", "pairedrl").startswith("docker exec pairedrl bash -c ")
    chain = run_chain_command("/work/repo", ["configs/a.json", "configs/b.json"], "/work/runs", "digitalocean", 1.99, "abc", "/work/hf", "pairedrl")
    assert chain.startswith("docker exec -d pairedrl bash -c ")
    for piece in ("mkdir -p /work/runs", "--spec configs/a.json", "--spec configs/b.json", "/work/runs/a.log", "/work/runs/b.log",
                  "--provider digitalocean", "--usd-per-hour 1.99", "--commit abc", "HF_HOME=/work/hf"):
        assert piece in chain
    assert chain.index("configs/a.json") < chain.index("configs/b.json")
    native = run_chain_command("/root/repo", ["configs/a.json"], "/root/runs", "local", 0.0, "abc", "/root/hf", None)
    assert native.startswith("nohup bash -c ") and native.endswith("&")
    cmd = rsync_command("root@h", "/root/work/runs/r1", "outputs/runs/r1")
    assert cmd[0] == "rsync" and "--exclude" in cmd and "trainer/" in cmd and cmd[-2:] == ["root@h:/root/work/runs/r1/", "outputs/runs/r1/"]
    assert "--exclude" not in rsync_command("root@h", "/a", "/b", with_weights=True)


def test_preserve_previous_attempt_moves_state_and_finds_checkpoint(tmp_path):
    assert preserve_previous_attempt(tmp_path) == (1, None)
    for name in ("run_manifest.json", "episodes.jsonl", "groups.jsonl"):
        (tmp_path / name).write_text('{"attempt": 2}\n' if name.endswith("json") else "{}\n")
    (tmp_path / "trainer" / "checkpoint-20").mkdir(parents=True)
    (tmp_path / "trainer" / "checkpoint-100").mkdir()
    attempt, checkpoint = preserve_previous_attempt(tmp_path)
    assert attempt == 3 and checkpoint == tmp_path / "trainer" / "checkpoint-100"
    assert sorted(p.name for p in (tmp_path / "attempt2").iterdir()) == ["episodes.jsonl", "groups.jsonl", "run_manifest.json"]
    assert not (tmp_path / "episodes.jsonl").exists()
    assert count_lines(tmp_path / "attempt2" / "episodes.jsonl") == 1 and count_lines(tmp_path / "missing") == 0


def test_collect_remote_describe_and_stale():
    collect = load_script("collect_remote")
    now = dt.datetime(2026, 9, 8, 12, 0, tzinfo=dt.UTC)
    assert collect.describe("r", None, now)[0] == "RUNNING"
    fresh = {"status": "RUNNING", "attempt": 1, "progress_utc": "2026-09-08T11:30:00+00:00", "episodes_logged": 5}
    state, line = collect.describe("r", fresh, now)
    assert state == "RUNNING" and "STALE" not in line
    old = dict(fresh, progress_utc="2026-09-08T08:00:00+00:00")
    assert "STALE" in collect.describe("r", old, now)[1]
    done = {"status": "COMPLETE", "attempt": 1, "progress_utc": "2026-09-08T08:00:00+00:00"}
    state, line = collect.describe("r", done, now)
    assert state == "COMPLETE" and "STALE" not in line
    failed = dict(done, status="FAILED", error="boom")
    assert collect.describe("r", failed, now)[0] == "FAILED" and "boom" in collect.describe("r", failed, now)[1]


def test_launch_remote_records_register_and_ledger(tmp_path, monkeypatch, capsys):
    launch = load_script("launch_remote")
    (tmp_path / "docs").mkdir()
    ledger = tmp_path / "docs" / "RUN_LEDGER.md"
    ledger.write_text("# Run ledger\n\n" + LEDGER_HEADER + "\n|---|---|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    for run_id in ("run-a", "run-b"):
        (tmp_path / "configs" / f"{run_id}.json").write_text(json.dumps({
            "run_id": run_id, "model": "Qwen/Qwen3.5-2B", "condition": "C2", "arm": "paired", "p": 0.25, "steps": 100, "notes": "x",
        }))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launch, "git_state", lambda: ("abc123", False))
    calls = []

    def fake_run(cmd, capture_output=False, text=False, check=False):
        calls.append(cmd[-1])
        return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["launch_remote.py", "--host", "root@h", "--specs", "configs/run-a.json,configs/run-b.json", "--label", "do-a"])
    launch.main()
    out = capsys.readouterr().out
    assert "STARTED run-a on root@h" in out and "STARTED run-b on root@h" in out and "commit abc123" in out
    assert any("git checkout -q abc123" in c for c in calls) and any("pip install -q -e ." in c for c in calls)
    assert any("docker exec -d pairedrl" in c and "configs/run-a.json" in c and "configs/run-b.json" in c for c in calls)
    registers = list((tmp_path / "registers" / "launches").glob("*_do-a.json"))
    assert len(registers) == 1
    register = json.loads(registers[0].read_text())
    assert register["app"] == "root@h" and [e["run_id"] for e in register["entries"]] == ["run-a", "run-b"]
    assert register["entries"][0]["host_run_dir"] == "/root/work/runs/run-a" and register["entries"][0]["call_id"] == "root@h:run-a"
    text = ledger.read_text()
    assert "| run-a |" in text and "| digitalocean |" in text and "call root@h:run-a" in text


def test_launch_remote_refuses_dirty_tree(tmp_path, monkeypatch):
    launch = load_script("launch_remote")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launch, "git_state", lambda: ("abc123", True))
    monkeypatch.setattr(sys, "argv", ["launch_remote.py", "--host", "root@h", "--specs", "configs/x.json", "--label", "l"])
    with pytest.raises(SystemExit, match="uncommitted"):
        launch.main()

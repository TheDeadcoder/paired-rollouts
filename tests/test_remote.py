import datetime as dt
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

from pairedrl.ops.job import (
    checkpoint_defects,
    commit_named,
    count_lines,
    is_complete_checkpoint,
    list_checkpoints,
    preserve_previous_attempt,
    refusal_reason,
    write_refusal,
)
from pairedrl.ops.ledger import LEDGER_HEADER
from pairedrl.ops.remote import (
    chain_status_command,
    checkout_command,
    in_container,
    install_command,
    rsync_command,
    run_chain_command,
    ssh_command,
)
from pairedrl.train.runner import RunSpec

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_command_builders():
    assert ssh_command("root@h", "ls")[-2:] == ["root@h", "ls"]
    assert checkout_command("/root/work/repo", "abc") == (
        "cd /root/work/repo && git fetch -q origin && git status --porcelain && git checkout -q -f abc && git rev-parse HEAD"
    )
    assert in_container(None, "echo hi") == "echo hi"
    assert in_container("pairedrl", "echo 'hi'") == "docker exec pairedrl bash -c 'echo '\"'\"'hi'\"'\"''"
    assert install_command("/work/repo", "pairedrl").startswith("docker exec pairedrl bash -c ")
    chain = run_chain_command("/work/repo", ["configs/a.json", "configs/b.json"], "/work/runs", "digitalocean", 1.99, "abc", "/work/hf", "pairedrl")
    assert chain.startswith("docker exec -d pairedrl bash -c ")
    for piece in ("mkdir -p /work/runs", "--spec configs/a.json", "--spec configs/b.json", "/work/runs/a.log", "/work/runs/b.log",
                  "--provider digitalocean", "--usd-per-hour 1.99", "--commit abc", "HF_HOME=/work/hf"):
        assert piece in chain
    assert chain.index("configs/a.json") < chain.index("configs/b.json")
    assert chain.count("--spec configs/a.json") == 2 and "|| " in chain and ">> " in chain
    assert chain.index("configs/a.json", chain.index("configs/a.json") + 1) < chain.index("configs/b.json")
    native = run_chain_command("/root/repo", ["configs/a.json"], "/root/runs", "local", 0.0, "abc", "/root/hf", None)
    assert native.startswith("nohup bash -c ") and native.endswith("&")
    status = chain_status_command("pairedrl")
    assert status.startswith("docker exec pairedrl bash -c ") and "[s]cripts/run_local.py" in status and "echo IDLE" in status
    assert subprocess.run(["bash", "-c", chain_status_command(None)], capture_output=True, text=True, check=False).stdout.strip() == "IDLE"
    cmd = rsync_command("root@h", "/root/work/runs/r1", "outputs/runs/r1")
    assert cmd[0] == "rsync" and "--exclude" in cmd and "trainer/" in cmd and cmd[-2:] == ["root@h:/root/work/runs/r1/", "outputs/runs/r1/"]
    assert "--exclude" not in rsync_command("root@h", "/a", "/b", with_weights=True)


def make_checkpoint(path, complete=True):
    """The files the pinned trainer writes for a LoRA checkpoint, in its order; an interrupted save stops before
    trainer_state.json."""
    path.mkdir(parents=True)
    for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (path / name).write_text("x")
    if complete:
        (path / "trainer_state.json").write_text(json.dumps({"global_step": int(path.name.split("-")[1])}))


def test_preserve_previous_attempt_moves_state_and_finds_the_newest_complete_checkpoint(tmp_path):
    assert preserve_previous_attempt(tmp_path) == (1, None)
    for name in ("run_manifest.json", "episodes.jsonl", "groups.jsonl"):
        (tmp_path / name).write_text('{"attempt": 2}\n' if name.endswith("json") else "{}\n")
    make_checkpoint(tmp_path / "trainer" / "checkpoint-20")
    make_checkpoint(tmp_path / "trainer" / "checkpoint-80")
    make_checkpoint(tmp_path / "trainer" / "checkpoint-100", complete=False)
    (tmp_path / "trainer" / "checkpoint-60").mkdir()
    assert is_complete_checkpoint(tmp_path / "trainer" / "checkpoint-80") and not is_complete_checkpoint(tmp_path / "trainer" / "checkpoint-100")
    assert checkpoint_defects(tmp_path / "trainer" / "checkpoint-100") == ["trainer_state.json missing"]
    assert checkpoint_defects(tmp_path / "trainer" / "checkpoint-60") == ["no weights", "optimizer.pt missing", "scheduler.pt missing", "rng_state missing", "trainer_state.json missing"]
    (tmp_path / "trainer" / "checkpoint-20" / "optimizer.pt").unlink()
    assert checkpoint_defects(tmp_path / "trainer" / "checkpoint-20") == ["optimizer.pt missing"]
    (tmp_path / "trainer" / "checkpoint-20" / "optimizer.pt").write_text("x")
    complete, incomplete = list_checkpoints(tmp_path / "trainer")
    assert [c.name for c in complete] == ["checkpoint-20", "checkpoint-80"] and [c.name for c in incomplete] == ["checkpoint-60", "checkpoint-100"]
    attempt, checkpoint = preserve_previous_attempt(tmp_path)
    assert attempt == 3 and checkpoint == tmp_path / "trainer" / "checkpoint-80"
    assert sorted(p.name for p in (tmp_path / "attempt2").iterdir()) == ["episodes.jsonl", "groups.jsonl", "run_manifest.json"]
    assert not (tmp_path / "episodes.jsonl").exists()
    assert count_lines(tmp_path / "attempt2" / "episodes.jsonl") == 1 and count_lines(tmp_path / "missing") == 0


def test_refusal_reasons_leave_the_run_directory_untouched(tmp_path):
    spec = RunSpec(run_id="r", model="m", condition="C2", arm="paired", p=0.25, steps=100)
    pools = {"train": {"sha256": "t1"}, "heldout": {"sha256": "h1"}}
    assert refusal_reason(spec, tmp_path, "abc1234", "abc1234", pools) is None
    assert "provenance" in refusal_reason(spec, tmp_path, "abc1234-dirty", "abc1234-dirty")
    assert "provenance" in refusal_reason(spec, tmp_path, "abc1234", "def5678")
    manifest = {"run_id": "r", "status": "FAILED", "attempt": 1, "spec": spec.to_dict(), "git_commit": "abc1234",
                "deployed_commit": "abc1234", "task_pools": pools}
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "episodes.jsonl").write_text("{}\n")
    assert refusal_reason(spec, tmp_path, "abc1234", "abc1234", pools) is None
    changed = RunSpec(run_id="r", model="m", condition="C2", arm="paired", p=0.25, steps=100, learning_rate=2e-5)
    assert "different spec" in refusal_reason(changed, tmp_path, "abc1234", "abc1234") and "learning_rate" in refusal_reason(changed, tmp_path, "abc1234", "abc1234")
    # a relaunch at another commit must name the previous attempt's commit; the pools may never differ
    across = refusal_reason(spec, tmp_path, "def5678", "def5678", pools)
    assert "attempt 1 ran at commit 'abc1234'" in across and "relaunch_from_commit" in across
    named = RunSpec(run_id="r", model="m", condition="C2", arm="paired", p=0.25, steps=100, relaunch_from_commit="abc1234")
    assert refusal_reason(named, tmp_path, "def5678", "def5678", pools) is None
    wrong = RunSpec(run_id="r", model="m", condition="C2", arm="paired", p=0.25, steps=100, relaunch_from_commit="0000000")
    assert "relaunch_from_commit" in refusal_reason(wrong, tmp_path, "def5678", "def5678", pools)
    other_pools = {"train": {"sha256": "t2"}, "heldout": {"sha256": "h1"}}
    assert "task pools ['train'] differ from attempt 1's" in refusal_reason(spec, tmp_path, "abc1234", "abc1234", other_pools)
    assert commit_named("abc1234", "abc1234def") and commit_named("abc1234def", "abc1234") and not commit_named("abc", "abc1234") and not commit_named(None, "abc1234")
    (tmp_path / "run_manifest.json").write_text(json.dumps(dict(manifest, status="COMPLETE")))
    assert "already COMPLETE" in refusal_reason(spec, tmp_path, "abc1234", "abc1234")
    extension = RunSpec(run_id="r", model="m", condition="C2", arm="paired", p=0.25, steps=120, extend_previous=True)
    assert refusal_reason(extension, tmp_path, "abc1234", "abc1234") is None
    record = write_refusal(spec, tmp_path, "already COMPLETE", "abc1234", "abc1234", "digitalocean")
    assert record["status"] == "REFUSED" and record["error"] == "already COMPLETE"
    refused = list((tmp_path / "refused").glob("*.json"))
    assert len(refused) == 1 and json.loads(refused[0].read_text())["run_id"] == "r"
    assert json.loads((tmp_path / "run_manifest.json").read_text())["status"] == "COMPLETE"
    assert (tmp_path / "episodes.jsonl").read_text() == "{}\n" and not (tmp_path / "attempt1").exists()


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
    monkeypatch.setattr(launch, "git_state", lambda: ("abc123", []))
    calls = []

    def fake_run(cmd, capture_output=False, text=False, check=False):
        calls.append(cmd[-1])
        out = "IDLE\n" if "pgrep" in cmd[-1] else "abc123\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["launch_remote.py", "--host", "root@h", "--specs", "configs/run-a.json,configs/run-b.json", "--label", "do-a"])
    launch.main()
    out = capsys.readouterr().out
    assert "STARTED run-a on root@h" in out and "STARTED run-b on root@h" in out and "commit abc123" in out
    assert any("git checkout -q -f abc123" in c for c in calls) and any("pip install -q -e ." in c for c in calls)
    assert any("docker exec -d pairedrl" in c and "configs/run-a.json" in c and "configs/run-b.json" in c for c in calls)
    registers = list((tmp_path / "registers" / "launches").glob("*_do-a.json"))
    assert len(registers) == 1
    register = json.loads(registers[0].read_text())
    assert register["app"] == "root@h" and [e["run_id"] for e in register["entries"]] == ["run-a", "run-b"]
    assert register["entries"][0]["host_run_dir"] == "/root/work/runs/run-a" and register["entries"][0]["call_id"] == "root@h:run-a"
    text = ledger.read_text()
    assert "| run-a |" in text and "| digitalocean |" in text and "call root@h:run-a" in text
    assert calls.index(next(c for c in calls if "pgrep" in c)) < calls.index(next(c for c in calls if "git checkout" in c))

    # the same specs again: their ledger rows are still open
    with pytest.raises(SystemExit, match="LAUNCHED"):
        launch.main()

    # a chain already running on the host
    def busy_run(cmd, capture_output=False, text=False, check=False):
        out = "RUNNING\n" if "pgrep" in cmd[-1] else "abc123\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(launch.subprocess, "run", busy_run)
    monkeypatch.setattr(sys, "argv", ["launch_remote.py", "--host", "root@h", "--specs", "configs/run-a.json", "--label", "do-b", "--relaunch"])
    with pytest.raises(SystemExit, match="already running"):
        launch.main()


def test_launch_remote_refuses_dirty_tree(tmp_path, monkeypatch):
    launch = load_script("launch_remote")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launch, "git_state", lambda: ("abc123", ["src/x.py"]))
    monkeypatch.setattr(sys, "argv", ["launch_remote.py", "--host", "root@h", "--specs", "configs/x.json", "--label", "l"])
    with pytest.raises(SystemExit, match="uncommitted"):
        launch.main()

import json

import pytest

from pairedrl.ops.ledger import (
    LEDGER_HEADER,
    TERMINAL,
    append_rows,
    dirty_paths,
    latest_launch_register,
    launch_register_path,
    ledger_row,
    ledger_statuses,
    preregistration_note,
    read_launch_register,
    refuse_duplicate_launches,
    set_status,
    write_launch_register,
)
from pairedrl.train.runner import RunSpec

CALL = "fc-01ABC"


def make_ledger(tmp_path):
    path = tmp_path / "RUN_LEDGER.md"
    path.write_text("# Run ledger\n\n" + LEDGER_HEADER + "\n|---|---|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    return path


def test_ledger_row_fields():
    spec = RunSpec(run_id="calib-x", model="Qwen/Qwen3.5-2B", condition="CALIB", arm="paired", p=0.25,
                   eval_only=True, notes="zero-shot | calibration")
    row = ledger_row(spec, "2026-09-05T10:00:00+00:00", CALL, "LAUNCHED")
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    assert cells[:9] == ["calib-x", "2026-09-05T10:00:00+00:00", "modal", "Qwen/Qwen3.5-2B", "CALIB", "paired", "0", "0", "LAUNCHED"]
    assert cells[9] == f"call {CALL}; zero-shot / calibration; not pre-registered"
    train = RunSpec(run_id="g", model="m", condition="C2", arm="paired", p=0.25, steps=100)
    assert ledger_row(train, "t", CALL, "LAUNCHED").split("|")[8].strip() == "100"


def test_preregistration_label_requires_matching_spec_notes():
    frozen = RunSpec(run_id="t1-c2-paired-s1", model="m", condition="C2", arm="paired", p=0.25, steps=100,
                     notes="tier 1, pre-registered v1; C2 paired seed 1; provider modal")
    assert preregistration_note(frozen, None) == "not pre-registered"
    assert preregistration_note(frozen, "v1") == "pre-registered v1"
    row = ledger_row(frozen, "t", CALL, "LAUNCHED", preregistration="v1")
    assert row.strip().endswith("; provider modal; pre-registered v1 |")
    assert "not pre-registered" not in row
    unlabelled = RunSpec(run_id="g", model="m", condition="C2", arm="paired", p=0.25, steps=100)
    with pytest.raises(ValueError):
        preregistration_note(unlabelled, "v1")
    with pytest.raises(ValueError):
        preregistration_note(frozen, "v2")


def test_append_and_set_status(tmp_path):
    path = make_ledger(tmp_path)
    spec = RunSpec(run_id="r", model="m", condition="C2", arm="paired", p=0.25)
    append_rows(path, [ledger_row(spec, "t1", "fc-1", "LAUNCHED"), ledger_row(spec, "t2", "fc-2", "LAUNCHED")])
    assert set_status(path, "fc-2", "COMPLETE") is True
    assert set_status(path, "fc-2", "COMPLETE") is False
    assert set_status(path, "fc-9", "COMPLETE") is False
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("| r ")]
    assert [line.split("|")[9].strip() for line in lines] == ["LAUNCHED", "COMPLETE"]
    assert path.read_text(encoding="utf-8").endswith("\n")
    other = tmp_path / "other.md"
    other.write_text("no table", encoding="utf-8")
    with pytest.raises(ValueError):
        append_rows(other, ["| x |"])


def test_launch_register_round_trip(tmp_path):
    reg_dir = tmp_path / "launches"
    first = launch_register_path(reg_dir, "calib attempt/3", "2026-09-05T10:00:00+00:00")
    assert first.name == "20260905T100000_calib-attempt-3.json"
    entries = [{"run_id": "a", "call_id": "fc-1", "spec_path": "configs/a.json", "spawned_utc": "t", "dashboard_url": None}]
    register = write_launch_register(first, "abc123", "pairedrl-train", "2026-09-05T10:00:00+00:00", entries)
    assert read_launch_register(first) == register and json.loads(first.read_text())["commit"] == "abc123"
    assert register["preregistration"] is None
    labelled = write_launch_register(first, "abc123", "pairedrl-train", "2026-09-05T10:00:00+00:00", entries, preregistration="v1")
    assert labelled["preregistration"] == "v1" and json.loads(first.read_text())["preregistration"] == "v1"
    second = launch_register_path(reg_dir, "later", "2026-09-06T00:00:00+00:00")
    write_launch_register(second, "def456", "pairedrl-train", "2026-09-06T00:00:00+00:00", [])
    assert latest_launch_register(reg_dir) == second
    with pytest.raises(FileNotFoundError):
        latest_launch_register(tmp_path / "empty")


def test_dirty_paths_ignore_launch_records_only():
    porcelain = " M docs/RUN_LEDGER.md\n?? registers/launches/20260908T120000_x.json\n"
    assert dirty_paths(porcelain) == []
    assert dirty_paths(porcelain + " M src/pairedrl/ops/job.py\n") == ["src/pairedrl/ops/job.py"]
    assert dirty_paths("?? registers/runs/new.json\nR  a.py -> b.py\n") == ["registers/runs/new.json", "b.py"]
    assert dirty_paths("") == []


def test_ledger_statuses_and_duplicate_guard(tmp_path):
    path = make_ledger(tmp_path)
    spec = RunSpec(run_id="r", model="m", condition="C2", arm="paired", p=0.25)
    other = RunSpec(run_id="s", model="m", condition="C2", arm="paired", p=0.25)
    assert ledger_statuses(path, "r") == [] and "REFUSED" in TERMINAL
    refuse_duplicate_launches(path, [spec, other])
    append_rows(path, [ledger_row(spec, "t1", "fc-1", "LAUNCHED")])
    assert ledger_statuses(path, "r") == ["LAUNCHED"] and ledger_statuses(path, "s") == []
    with pytest.raises(SystemExit, match="LAUNCHED"):
        refuse_duplicate_launches(path, [other, spec])
    refuse_duplicate_launches(path, [spec], relaunch=True)
    set_status(path, "fc-1", "FAILED")
    refuse_duplicate_launches(path, [spec])
    append_rows(path, [ledger_row(spec, "t2", "fc-2", "LAUNCHED")])
    set_status(path, "fc-2", "COMPLETE")
    assert ledger_statuses(path, "r") == ["FAILED", "COMPLETE"]
    with pytest.raises(SystemExit, match="COMPLETE"):
        refuse_duplicate_launches(path, [spec], relaunch=True)
    extension = RunSpec(run_id="r", model="m", condition="C2", arm="paired", p=0.25, steps=120, extend_previous=True)
    refuse_duplicate_launches(path, [extension])

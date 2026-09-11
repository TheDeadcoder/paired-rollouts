"""The acceptance check must reject evidence that is complete by count but not the evidence the protocol asked
for: evaluations at the wrong step, a missing diagnostic task, substituted task or schedule identities, groups off
their rows, phases outside the plan, checkpoints without their state, attempts at unacknowledged commits."""

import importlib.util
import json
import pathlib
import shutil

import pytest
from synthetic import complete_run, manifest, small_spec, write_attempt

from pairedrl.analysis.curves import run_register
from pairedrl.ops.job import checkpoint_defects
from pairedrl.train.runner import read_episode_log

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rewrite(run_dir, episodes=None, groups=None, manifest_dict=None):
    write_attempt(
        run_dir,
        manifest_dict or json.loads((run_dir / "run_manifest.json").read_text()),
        episodes if episodes is not None else read_episode_log(run_dir / "episodes.jsonl"),
        groups if groups is not None else read_episode_log(run_dir / "groups.jsonl"),
    )


def write_checkpoint(path, step, **overrides):
    path.mkdir(parents=True, exist_ok=True)
    files = {"adapter_model.safetensors": "w", "adapter_config.json": "{}", "optimizer.pt": "o", "scheduler.pt": "s",
             "rng_state.pth": "r", "trainer_state.json": json.dumps({"global_step": step})}
    files.update(overrides)
    for name, content in files.items():
        if content is not None:
            (path / name).write_text(content)


def test_complete_synthetic_run_is_accepted(tmp_path):
    verify = load_script("verify_run")
    run_dir = complete_run(tmp_path, small_spec())
    entry = run_register(run_dir)
    ok, reasons = verify.verdict(entry, weights=False)
    assert ok and reasons == [], reasons
    comp = entry["completeness"]
    assert comp["episodes_found"] == comp["episodes_expected"] == 3 * (4 + 4 + 2) + (8 + 8 + 8 + 8 + 2) + 2 * 64 + 16
    assert comp["diagnostic_tasks"] == 16 and comp["final_step_expected"] == 4


def test_final_sets_at_an_earlier_step_are_rejected(tmp_path):
    run_dir = complete_run(tmp_path, small_spec())
    moved = [dict(r, phase=r["phase"].replace("step4", "step2")) if r["phase"].startswith("final_") else r
             for r in read_episode_log(run_dir / "episodes.jsonl")]
    rewrite(run_dir, episodes=moved)
    comp = run_register(run_dir)["completeness"]
    assert comp["complete"] is False
    assert {m["set"] for m in comp["periodic_and_final_missing"]} == {"final_clean", "final_noisy_p010", "final_noisy_p025", "final_heldout_types", "final_challenge"}
    assert set(comp["unexpected_phases"]) == {"final_clean:step2", "final_noisy_p010:step2", "final_noisy_p025:step2", "final_heldout_types:step2", "final_challenge:step2"}
    assert comp["episodes_found"] == comp["episodes_expected"] and comp["totals_match"] is True


def test_a_missing_diagnostic_task_is_rejected_even_with_intact_tables(tmp_path):
    run_dir = complete_run(tmp_path, small_spec())
    records = read_episode_log(run_dir / "episodes.jsonl")
    gone = next(r["task_id"] for r in records if r["phase"] == "diag:step4")
    rewrite(run_dir, episodes=[r for r in records if not (r["phase"] == "diag:step4" and r["task_id"] == gone)])
    entry = run_register(run_dir)
    comp = entry["completeness"]
    assert comp["complete"] is False and comp["diagnostic_missing"] == []
    assert comp["identity_defects"] == [{"phase": "diag:step4", "missing": 4, "extra": 0}]
    assert entry["luck_share"]["diag:step4"]["tasks"] == 15 and comp["diagnostic_tasks"] == 16


def test_substituted_identities_with_the_same_counts_are_rejected(tmp_path):
    run_dir = complete_run(tmp_path, small_spec())
    records = read_episode_log(run_dir / "episodes.jsonl")
    first = next(r for r in records if r["phase"] == "eval_noisy:step2")
    swapped = [dict(r, task_id=first["task_id"], schedule_seed=first["schedule_seed"]) if r["phase"] == "eval_noisy:step2" else r for r in records]
    rewrite(run_dir, episodes=swapped)
    comp = run_register(run_dir)["completeness"]
    assert comp["complete"] is False and comp["periodic_and_final_missing"] == []
    assert comp["identity_defects"] == [{"phase": "eval_noisy:step2", "missing": 3, "extra": 3}]


def test_groups_off_their_rows_and_wrong_sizes_are_rejected(tmp_path):
    run_dir = complete_run(tmp_path, small_spec())
    groups = read_episode_log(run_dir / "groups.jsonl")
    g0, g1 = groups[0], groups[1]
    groups[0], groups[1] = dict(g0, task_id=g1["task_id"], schedule_seed=g1["schedule_seed"]), dict(g1, true_success=[True])
    groups.append(dict(groups[2], step=9))
    rewrite(run_dir, groups=groups)
    comp = run_register(run_dir)["completeness"]
    assert comp["complete"] is False
    defects = comp["group_defects"]
    assert [d["step"] for d in defects] == [0, 0, 9]
    assert defects[0]["defect"].startswith("group 0 is ") and "expected" in defects[0]["defect"]
    assert defects[1]["defect"] == "group 1 has member counts [1, 2], expected 2"
    assert defects[2]["defect"] == "1 groups at a step outside the plan"


def test_other_pools_are_rejected(tmp_path):
    run_dir = complete_run(tmp_path, small_spec())
    other = tmp_path / "pools"
    shutil.copytree(ROOT / "data" / "tasks", other)
    heldout = other / "heldout.jsonl"
    lines = heldout.read_text().splitlines()
    heldout.write_text("\n".join([lines[2], lines[1], lines[0], *lines[3:]]) + "\n")
    entry = run_register(run_dir, data_dir=other)
    comp = entry["completeness"]
    assert comp["task_pools"]["heldout"]["match"] is False and comp["task_pools"]["train"]["match"] is True
    assert comp["task_pools_match"] is False and comp["complete"] is False
    assert comp["identity_defects"] and all(d["phase"].startswith("eval_") for d in comp["identity_defects"])


def test_checkpoints_need_their_state_files_and_matching_step(tmp_path):
    verify = load_script("verify_run")
    spec = small_spec()
    run_dir = complete_run(tmp_path, spec)
    write_checkpoint(run_dir / "trainer" / "checkpoint-2", 2)
    write_checkpoint(run_dir / "trainer" / "checkpoint-4", 4)
    (run_dir / "adapter_final").mkdir()
    (run_dir / "adapter_final" / "adapter_model.safetensors").write_text("w")
    entry = run_register(run_dir)
    assert entry["checkpoints"]["complete"] == [2, 4] and entry["checkpoints"]["adapter_final"] is False
    ok, reasons = verify.verdict(entry, weights=True)
    assert not ok and reasons == ["adapter_final missing or without adapter_config.json"]
    (run_dir / "adapter_final" / "adapter_config.json").write_text("{}")
    assert verify.verdict(run_register(run_dir), weights=True) == (True, [])
    assert verify.verdict(run_register(run_dir), weights=False) == (True, [])
    last = run_dir / "trainer" / "checkpoint-4"
    (last / "optimizer.pt").unlink()
    assert checkpoint_defects(last) == ["optimizer.pt missing"]
    write_checkpoint(last, 3)
    assert checkpoint_defects(last) == ["global_step 3 != 4"]
    (last / "trainer_state.json").write_text("{")
    assert checkpoint_defects(last) == ["trainer_state.json unreadable or without global_step"]
    entry = run_register(run_dir)
    assert entry["checkpoints"]["incomplete"] == ["checkpoint-4"] and entry["checkpoints"]["missing"] == [4]
    assert entry["checkpoints"]["defects"] == {"checkpoint-4": ["trainer_state.json unreadable or without global_step"]}
    ok, reasons = verify.verdict(entry, weights=True)
    assert not ok and "checkpoints missing [4], incomplete ['checkpoint-4']" in reasons[0]
    write_checkpoint(last, 4)
    (last / "rng_state.pth").rename(last / "rng_state_0.pth")
    assert checkpoint_defects(last) == []
    (last / "rng_state_0.pth").unlink()
    assert checkpoint_defects(last) == ["rng_state missing"]


def test_attempts_at_other_commits_need_acknowledgement(tmp_path):
    verify = load_script("verify_run")
    spec = small_spec()
    run_dir = complete_run(tmp_path, spec, commit="abc1234")
    root = json.loads((run_dir / "run_manifest.json").read_text())
    first = dict(manifest(spec, status="FAILED", attempt=1, commit="abc1234"), model_commit_hash="m1", task_pools={"train": {"sha256": "t"}})
    write_attempt(run_dir / "attempt1", first, [], [])
    later = dict(root, attempt=2, git_commit="def5678", deployed_commit="def5678", model_commit_hash="m1", task_pools={"train": {"sha256": "t"}})
    rewrite(run_dir, manifest_dict=later)
    entry = run_register(run_dir)
    consistency = entry["attempt_consistency"]
    assert consistency["commits"] == ["abc1234", "def5678"] and consistency["commits_consistent"] is False
    assert consistency["commit_changes"] == [{"attempt": 2, "from": "abc1234", "to": "def5678", "acknowledged": False}]
    ok, reasons = verify.verdict(entry, weights=False)
    assert not ok and reasons[0].startswith("attempt commits changed without acknowledgement")
    acknowledged = dict(later, spec=dict(spec.to_dict(), relaunch_from_commit="abc1234"))
    rewrite(run_dir, manifest_dict=acknowledged)
    entry = run_register(run_dir)
    assert entry["attempt_consistency"]["commits_consistent"] is True and verify.verdict(entry, weights=False) == (True, [])
    rewrite(run_dir, manifest_dict=dict(acknowledged, model_commit_hash="m2", task_pools={"train": {"sha256": "u"}}))
    entry = run_register(run_dir)
    assert entry["attempt_consistency"]["model_consistent"] is False and entry["attempt_consistency"]["task_pools_consistent"] is False
    ok, reasons = verify.verdict(entry, weights=False)
    assert reasons == ["model snapshot changed across attempts ['m1', 'm2']", "task pools changed across attempts"]


def test_decide_refuses_inconsistent_runs(tmp_path):
    from pairedrl.analysis.hypotheses import decide

    spec = small_spec()
    run_dir = complete_run(tmp_path, spec, commit="abc1234")
    first = manifest(spec, status="FAILED", attempt=1, commit="abc1234")
    write_attempt(run_dir / "attempt1", first, [], [])
    root = json.loads((run_dir / "run_manifest.json").read_text())
    rewrite(run_dir, manifest_dict=dict(root, attempt=2, git_commit="def5678", deployed_commit="def5678"))
    with pytest.raises(ValueError, match="consistent False"):
        decide("C2", [run_dir], n_boot=10)

import json

import pytest

from pairedrl.analysis.registers import (
    CALIBRATION_BANDS,
    build_calibration_register,
    calibration_entry,
    completion_stats,
    format_calibration,
)


def record(**kw):
    base = {
        "task_id": "heldout-00000", "phase": "final_clean:step0", "condition": "eval:clean",
        "true_success": True, "observed_reward": 1.0, "flipped": False, "calls": 6, "budget": 13,
        "budget_exceeded": False, "finished": True, "exposed": False, "faults": {}, "schedule_seed": 1,
    }
    base.update(kw)
    return base


def make_run_dir(tmp_path, model="Qwen/Qwen3.5-2B", clean_successes=(True, True, False, False)):
    run_dir = tmp_path / "calib-test"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": "calib-test", "spec": {"model": model}, "git_commit": "abc1234", "status": "COMPLETE",
        "wall_time_s": 1234.5, "estimated_cost_usd": 1.35, "peak_mem_gb": 40.0,
        "versions": {"torch": "2.13.0", "trl": "1.12.0"},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    records = [record(task_id=f"heldout-{i:05d}", true_success=s) for i, s in enumerate(clean_successes)]
    records += [
        record(phase="final_noisy:step0", condition="eval:noisy", true_success=False, exposed=True),
        record(phase="final_heldout_types:step0", condition="eval:heldout_types", true_success=True),
    ]
    for task in ("diagnostic-00000", "diagnostic-00001"):
        for k, seed in enumerate((11, 22, 33)):
            for m in range(4):
                records.append(record(
                    task_id=task, phase="diag:step0", condition="diag", schedule_seed=seed,
                    true_success=(k == 0) or (m == 0),
                ))
    (run_dir / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    history = [
        {"final_clean_loss": 0.0, "final_clean_runtime": 100.0, "eval_completions/mean_length": 900.0,
         "eval_completions/max_length": 3000.0, "eval_completions/clipped_ratio": 0.05,
         "eval_tools/call_frequency": 6.5, "eval_tools/failure_frequency": 0.1, "eval_reward": 0.5, "step": 0},
        {"diag_loss": 0.0, "diag_runtime": 50.0, "eval_completions/mean_length": 1100.0,
         "eval_completions/max_length": 6144.0, "eval_completions/clipped_ratio": 0.2,
         "eval_tools/call_frequency": 8.0, "eval_tools/failure_frequency": 0.3, "eval_reward": 0.4, "step": 0},
        {"loss": 0.1, "step": 1},
    ]
    (run_dir / "trainer_log_history.json").write_text(json.dumps(history))
    return run_dir


def test_calibration_entry_fields(tmp_path):
    entry = calibration_entry(make_run_dir(tmp_path))
    assert entry["run_id"] == "calib-test" and entry["status"] == "COMPLETE" and entry["episodes"] == 30
    assert set(entry["sets"]) == {"clean", "noisy", "heldout_types"}
    assert entry["sets"]["clean"]["true_success"] == pytest.approx(0.5)
    assert entry["sets"]["noisy"]["recovery_success"] == pytest.approx(0.0)
    assert entry["clean_band"] == list(CALIBRATION_BANDS["Qwen/Qwen3.5-2B"]) and entry["clean_in_band"]
    assert entry["luck_share"]["tasks"] == 2 and entry["luck_share"]["lam"] is not None
    assert 0.0 <= entry["luck_share"]["lam_pooled"] <= 1.0
    assert entry["luck_share_ok"] == (entry["luck_share"]["lam"] >= 0.15)
    assert set(entry["completions"]) == {"final_clean", "diag"}
    assert entry["completions"]["diag"]["clipped_ratio"] == pytest.approx(0.2)
    assert entry["completions"]["final_clean"]["runtime_s"] == pytest.approx(100.0)


def test_out_of_band_and_unknown_model(tmp_path):
    entry = calibration_entry(make_run_dir(tmp_path, clean_successes=(True,) * 9 + (False,)))
    assert entry["sets"]["clean"]["true_success"] == pytest.approx(0.9) and not entry["clean_in_band"]
    other = calibration_entry(make_run_dir(tmp_path / "other", model="Qwen/Qwen3-4B"))
    assert other["clean_band"] is None and not other["clean_in_band"]


def test_build_and_format(tmp_path):
    run_dir = make_run_dir(tmp_path)
    out = tmp_path / "registers" / "calibration.json"
    register = build_calibration_register([run_dir], out)
    assert json.loads(out.read_text())["entries"][0]["run_id"] == "calib-test"
    text = format_calibration(register)
    assert "calib-test (Qwen/Qwen3.5-2B): status COMPLETE, 30 episodes" in text
    assert "clean          true=0.500" in text and "luck share diag:step0: lam=" in text and "pooled=" in text
    assert "diag           mean_len=1100 max_len=6144 clipped=0.200" in text


def test_completion_stats_ignores_training_entries():
    assert completion_stats([{"loss": 0.1, "step": 3}]) == {}
    stats = completion_stats([{"eval_completions/mean_length": 10.0, "step": 0}])
    assert list(stats) == ["step0"] and stats["step0"]["clipped_ratio"] is None

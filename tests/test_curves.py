import json

import pytest

from pairedrl.analysis.curves import (
    area_under_curve,
    final_sets,
    format_run,
    periodic_curve,
    run_register,
    steps_to_threshold,
    training_curve,
)


def rec(phase, condition, task, success, **kw):
    base = {"task_id": task, "phase": phase, "condition": condition, "true_success": success,
            "observed_reward": float(success), "flipped": False, "calls": 6, "budget": 13, "budget_exceeded": False,
            "finished": True, "exposed": condition != "eval:clean", "faults": {}, "schedule_seed": 1}
    base.update(kw)
    return base


def make_records():
    records = []
    for step, rate in ((0, 0.5), (20, 0.75), (40, 1.0)):
        for i in range(4):
            records.append(rec(f"eval_noisy:step{step}", "eval:noisy", f"h{i}", i < rate * 4))
            records.append(rec(f"eval_clean:step{step}", "eval:clean", f"h{i}", True))
    for i in range(4):
        records.append(rec("final_clean:step40", "eval:clean", f"t{i}", i < 3))
        records.append(rec("final_challenge:step40", "eval:challenge", f"t{i}", i < 2))
    for step in (0, 1):
        for i in range(8):
            records.append(rec(f"train:step{step}", "C2", "tr", i < 4 + step))
    for task in ("d0", "d1"):
        for k in range(8):
            for m in range(8):
                records.append(rec("diag:step0", "diag", task, (k + m) % 3 == 0, schedule_seed=k))
    return records


def test_curves_areas_and_threshold():
    records = make_records()
    noisy = periodic_curve(records, "noisy")
    assert [(p["step"], p["true_success"]) for p in noisy] == [(0, 0.5), (20, 0.75), (40, 1.0)]
    assert area_under_curve(noisy) == pytest.approx((0.625 * 20 + 0.875 * 20) / 40)
    assert steps_to_threshold(noisy, 0.5) == 0.0
    assert steps_to_threshold(noisy, 0.625) == pytest.approx(10.0)
    assert steps_to_threshold(noisy, 1.5) is None
    assert area_under_curve(noisy[:1]) is None
    assert periodic_curve(records, "challenge") == []
    assert final_sets(records)["challenge"]["true_success"] == 0.5
    assert [p["true_success"] for p in training_curve(records)] == [0.5, 0.625]


def test_run_register_and_format(tmp_path):
    run_dir = tmp_path / "run-x"
    run_dir.mkdir()
    manifest = {"run_id": "run-x", "status": "COMPLETE", "attempt": 1, "git_commit": "abc", "wall_time_s": 10.0,
                "estimated_cost_usd": 1.0, "peak_mem_gb": 1.0, "versions": {},
                "spec": {"condition": "C2", "arm": "paired", "seed": 0, "p": 0.25, "q": 0.0, "steps": 40,
                         "prompts_per_step": 4, "num_generations": 8, "scale_rewards": "group", "loss_type": "dapo"},
                "step_timings": [{"step": 1, "seconds": 100.0, "generation_s": 60.0}, {"step": 2, "seconds": 120.0, "generation_s": 70.0}]}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    (run_dir / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in make_records()) + "\n")
    group = {"step": 0, "attempt": 1, "group": 0, "task_id": "tr", "condition": "C2", "schedule_seed": 1,
             "true_success": [True] * 8, "observed_reward": [1.0] * 8, "advantages": [0.0] * 8,
             "zero_variance": True, "spurious": False, "max_abs_advantage": 0.0}
    (run_dir / "groups.jsonl").write_text(json.dumps(group) + "\n")
    entry = run_register(run_dir, threshold=0.625)
    assert entry["auc"]["noisy"] == pytest.approx(0.75) and entry["auc"]["clean"] == pytest.approx(1.0)
    assert entry["steps_to_threshold"] == pytest.approx(10.0)
    assert entry["final"]["clean"]["true_success"] == 0.75 and entry["final"]["clean"]["step"] == 40
    assert entry["luck_share"]["diag:step0"]["tables"] == 2 and entry["luck_share"]["diag:step0"]["lam"] is not None
    assert entry["training_groups"]["C2"]["all_correct_groups"] == 1
    assert entry["timing"]["mean_step_s"] == 110.0 and entry["timing"]["mean_passes_s"] == pytest.approx(45.0)
    text = format_run(entry)
    assert "noisy      0:0.500 20:0.750 40:1.000  auc=0.750" in text and "steps to threshold 10.0" in text
    assert "groups C2: 1 groups" in text and "timing: 2 steps, mean 110.0 s" in text

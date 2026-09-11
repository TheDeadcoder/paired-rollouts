import json

import pytest
from synthetic import (
    complete_run,
    diagnostic_records,
    final_records,
    manifest,
    periodic_records,
    rec,
    small_spec,
    training_records,
    write_attempt,
)

from pairedrl.analysis.curves import (
    area_under_curve,
    final_sets,
    format_run,
    lineage_records,
    load_attempts,
    periodic_curve,
    run_register,
    steps_to_threshold,
    training_curve,
)
from pairedrl.train.runner import RunSpec, read_episode_log


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
    spec = RunSpec(run_id="run-x", model="m", condition="C2", arm="paired", p=0.25, seed=0, steps=40, prompts_per_step=4,
                   eval_every=20, eval_tasks=4, final_eval_tasks=4, diagnostic_steps=[0])
    m = manifest(spec, git_commit="abc", deployed_commit="abc", wall_time_s=10.0, estimated_cost_usd=1.0, peak_mem_gb=1.0, versions={},
                 step_timings=[{"step": 1, "seconds": 100.0, "generation_s": 60.0}, {"step": 2, "seconds": 120.0, "generation_s": 70.0}])
    (run_dir / "run_manifest.json").write_text(json.dumps(m))
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
    comp = entry["completeness"]
    assert comp["complete"] is False and comp["training_groups_expected"] == 160 and comp["training_groups_found"] == 1
    assert {(m["set"], m["step"]) for m in comp["periodic_and_final_missing"]} >= {("eval_challenge", 0), ("final_noisy_p010", 40)}
    # synthetic identities are none of the frozen pools' tasks: every present phase is an identity defect
    assert {d["phase"] for d in comp["identity_defects"]} >= {"eval_noisy:step0", "final_clean:step40", "diag:step0", "train:step0"}
    assert comp["diagnostic_tasks"] == 16 and comp["task_pools_match"] is True and comp["totals_match"] is False
    assert entry["attempts"][0]["attempt"] == 1 and entry["total_estimated_cost_usd"] == 1.0 and entry["checkpoints"] is None
    assert entry["attempt_consistency"]["commits"] == ["abc"] and entry["attempt_consistency"]["commits_consistent"]
    assert "INCOMPLETE" in text and "identity defects" in text


def test_lineage_across_attempts_and_completeness(tmp_path):
    spec = small_spec()
    run_dir = tmp_path / "run-l"

    def up(name, step, ti, k):
        return True

    # attempt 1: evaluated at 0 and 2, saved checkpoint-2, evaluated at step 4 and generated rollouts for the step
    # after the checkpoint (train:step2, train:step3), then died before saving checkpoint-4
    first = periodic_records(spec, [0, 2, 4], up) + diagnostic_records(spec, [0], lambda s, ti, kk, m: (kk + m) % 2 == 0)
    train1, groups1 = training_records(spec, [0, 1, 2, 3], lambda step, g: [False, False])
    m1 = manifest(spec, status="FAILED", attempt=1, resumed_from_step=None, error="boom", wall_time_s=100.0, estimated_cost_usd=0.4)
    write_attempt(run_dir / "attempt1", m1, first + train1, groups1)
    # attempt 2 resumed from checkpoint-2: regenerated steps 2 and 3, evaluated at 4, ran the final sets and the
    # step-4 diagnostic
    second = periodic_records(spec, [4], up, attempt=2) + final_records(spec, lambda name, ti, k: True, attempt=2)
    second += diagnostic_records(spec, [4], lambda s, ti, kk, m: (kk + m) % 2 == 0, attempt=2)
    train2, groups2 = training_records(spec, [2, 3], lambda step, g: [True, True], attempt=2)
    m2 = dict(m1, status="COMPLETE", attempt=2, resumed_from_step=2, wall_time_s=60.0, estimated_cost_usd=0.2, error=None)
    write_attempt(run_dir, m2, second + train2, groups2)
    attempts = load_attempts(run_dir)
    assert [a["manifest"]["attempt"] for a in attempts] == [1, 2]
    records, book = lineage_records(attempts, "episodes.jsonl")
    assert book["attempts"][0]["cutoff_step"] == 2 and book["attempts"][1]["cutoff_step"] is None
    phases = sorted({r["phase"] for r in records})
    assert "eval_noisy:step4" in phases and "train:step3" in phases
    assert all(r["attempt"] == 2 for r in records if r["phase"] in ("eval_noisy:step4", "train:step2", "train:step3"))
    assert all(r["attempt"] == 1 for r in records if r["phase"] in ("eval_noisy:step0", "eval_noisy:step2", "train:step0", "train:step1"))
    assert book["attempts"][0]["superseded"] == 8 + 4 + 4 + 2  # train steps 2 and 3, then clean, noisy and challenge at step 4
    entry = run_register(run_dir)
    comp = entry["completeness"]
    assert comp["complete"] is True, comp
    assert comp["training_groups_found"] == 8 and comp["episodes_found"] == comp["episodes_expected"]
    assert comp["identity_defects"] == [] and comp["unexpected_phases"] == {} and comp["group_defects"] == []
    assert [p["true_success"] for p in entry["training_curve"]] == [0.0, 0.0, 1.0, 1.0]
    assert entry["total_wall_time_s"] == 160.0 and entry["total_estimated_cost_usd"] == 0.6
    assert [a["status"] for a in entry["attempts"]] == ["FAILED", "COMPLETE"]
    assert set(entry["luck_share"]) == {"diag:step0", "diag:step4"} and not any("defect" in v for v in entry["luck_share"].values())
    assert entry["attempt_consistency"]["commits_consistent"] and entry["attempt_consistency"]["task_pools_consistent"]
    assert "COMPLETE;" in format_run(entry)

    # the naive reading of the root files alone would report an area over steps 4 to 4 only
    root_only = periodic_curve(read_episode_log(run_dir / "episodes.jsonl"), "noisy")
    assert [p["step"] for p in root_only] == [4] and area_under_curve(root_only) is None
    assert [p["step"] for p in entry["curves"]["noisy"]] == [0, 2, 4]


def test_completeness_flags_missing_sets_and_diagnostic_defects(tmp_path):
    spec = small_spec(run_id="run-m")
    run_dir = complete_run(tmp_path, spec)
    records = read_episode_log(run_dir / "episodes.jsonl")
    records = [r for r in records if not (r["phase"] == "eval_clean:step2" and r["slot"] == 0)]
    records = [r for r in records if not (r["phase"] == "diag:step4" and r["slot"] == 7)]
    groups = [g for g in read_episode_log(run_dir / "groups.jsonl") if g["step"] < 3]
    write_attempt(run_dir, json.loads((run_dir / "run_manifest.json").read_text()), records, groups)
    (run_dir / "trainer" / "checkpoint-2").mkdir(parents=True)
    (run_dir / "trainer" / "checkpoint-2" / "trainer_state.json").write_text("{}")
    (run_dir / "trainer" / "checkpoint-2" / "adapter_model.safetensors").write_text("w")
    (run_dir / "trainer" / "checkpoint-4").mkdir()
    entry = run_register(run_dir)
    comp = entry["completeness"]
    assert comp["complete"] is False
    assert comp["periodic_and_final_missing"] == [{"set": "eval_clean", "step": 2, "found": 3, "expected": 4},
                                                  {"set": "diag", "step": 4, "found": 63, "expected": 64}]
    assert comp["identity_defects"] == [{"phase": "eval_clean:step2", "missing": 1, "extra": 0}, {"phase": "diag:step4", "missing": 1, "extra": 0}]
    assert comp["diagnostic_missing"][0]["phase"] == "diag:step4" and "samples per schedule" in comp["diagnostic_missing"][0]["defect"]
    assert comp["training_groups_found"] == 6 and comp["training_groups_expected"] == 8
    assert [d["step"] for d in comp["group_defects"]] == [3] and comp["totals_match"] is False
    ck = entry["checkpoints"]
    assert ck["expected"] == [2, 4] and ck["complete"] == [] and ck["incomplete"] == ["checkpoint-2", "checkpoint-4"] and ck["missing"] == [2, 4]
    assert ck["defects"]["checkpoint-2"] == ["adapter_config.json missing", "optimizer.pt missing", "scheduler.pt missing",
                                             "rng_state missing", "trainer_state.json unreadable or without global_step"]
    assert ck["defects"]["checkpoint-4"][0] == "no weights" and ck["adapter_final"] is False
    text = format_run(entry)
    assert "INCOMPLETE" in text and "missing eval_clean step 2: 3 of 4" in text and "incomplete ['checkpoint-2', 'checkpoint-4']" in text
    assert "identity diag:step4: 1 expected records absent, 0 unexpected" in text and "groups step 3: groups [] instead of 0..1" in text

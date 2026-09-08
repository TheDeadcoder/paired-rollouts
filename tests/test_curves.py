import json

import pytest

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
from pairedrl.train.runner import RunSpec, expected_eval_sizes, read_episode_log


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
    spec = RunSpec(run_id="run-x", model="m", condition="C2", arm="paired", p=0.25, seed=0, steps=40, prompts_per_step=4,
                   eval_every=20, eval_tasks=4, final_eval_tasks=4, diagnostic_steps=[0])
    manifest = {"run_id": "run-x", "status": "COMPLETE", "attempt": 1, "git_commit": "abc", "wall_time_s": 10.0,
                "estimated_cost_usd": 1.0, "peak_mem_gb": 1.0, "versions": {}, "spec": spec.to_dict(),
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
    comp = entry["completeness"]
    assert comp["complete"] is False and comp["training_groups_expected"] == 160 and comp["training_groups_found"] == 1
    assert {(m["set"], m["step"]) for m in comp["periodic_and_final_missing"]} >= {("eval_challenge", 0), ("final_noisy_p010", 40)}
    assert entry["attempts"][0]["attempt"] == 1 and entry["total_estimated_cost_usd"] == 1.0 and entry["checkpoints"] is None
    assert "INCOMPLETE" in text


def spec_small(**kw):
    base = {"run_id": "run-l", "model": "m", "condition": "C2", "arm": "paired", "p": 0.25, "seed": 0, "steps": 4,
            "prompts_per_step": 2, "num_generations": 2, "eval_every": 2, "checkpoint_steps": 2, "eval_tasks": 2,
            "final_eval_tasks": 2, "diagnostic_steps": [0, 4], "diagnostic_schedules": 2, "diagnostic_samples": 2}
    base.update(kw)
    return RunSpec(**base)


def complete_records(spec, attempt, steps, final=True, diag_steps=None):
    """Synthetic episode records with exactly the counts a complete run produces for the given steps."""
    records = []
    sizes = expected_eval_sizes(spec)
    for step in steps:
        for name, n in sizes["periodic"].items():
            for i in range(n):
                records.append(rec(f"eval_{name}:step{step}", f"eval:{name}", f"h{i // 2}", True, schedule_seed=i % 2 + 1, slot=i, attempt=attempt))
    if final:
        for name, n in sizes["final"].items():
            for i in range(n):
                records.append(rec(f"final_{name}:step{spec.steps}", f"eval:{name}", f"t{i // 4}", True, schedule_seed=i % 4 + 1, slot=i, attempt=attempt))
    for step in diag_steps or []:
        for task in ("d0", "d1"):
            for k in range(spec.diagnostic_schedules):
                for m in range(spec.diagnostic_samples):
                    records.append(rec(f"diag:step{step}", "diag", task, (k + m) % 2 == 0, schedule_seed=k, slot=k * 2 + m, attempt=attempt))
    return records


def train_records(attempt, steps, success):
    return [rec(f"train:step{step}", "C2", "tr", success, slot=i, episode_index=step, attempt=attempt)
            for step in steps for i in range(4)]


def group_records(attempt, steps):
    return [{"step": step, "attempt": attempt, "group": g, "task_id": "tr", "condition": "C2", "schedule_seed": 1,
             "true_success": [True, False], "observed_reward": [1.0, 0.0], "advantages": [1.0, -1.0],
             "zero_variance": False, "spurious": False, "max_abs_advantage": 1.0} for step in steps for g in range(2)]


def write_attempt(directory, manifest, episodes, groups):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_manifest.json").write_text(json.dumps(manifest))
    (directory / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in episodes) + "\n")
    (directory / "groups.jsonl").write_text("\n".join(json.dumps(g) for g in groups) + "\n")


def test_lineage_across_attempts_and_completeness(tmp_path):
    spec = spec_small()
    run_dir = tmp_path / "run-l"
    # attempt 1: evaluated at 0 and 2, saved checkpoint-2, evaluated at step 4 and generated rollouts for the step
    # after the checkpoint (train:step2, train:step3), then died before saving checkpoint-4
    first = complete_records(spec, 1, [0, 2, 4], final=False, diag_steps=[0]) + train_records(1, [0, 1, 2, 3], False)
    first_groups = group_records(1, [0, 1, 2, 3])
    m1 = {"run_id": "run-l", "status": "FAILED", "attempt": 1, "git_commit": "abc", "wall_time_s": 100.0,
          "estimated_cost_usd": 0.4, "spec": spec.to_dict(), "resumed_from_step": None, "error": "boom"}
    write_attempt(run_dir / "attempt1", m1, first, first_groups)
    # attempt 2 resumed from checkpoint-2: regenerated steps 2 and 3, evaluated at 4, ran the final sets and the
    # step-4 diagnostic
    second = complete_records(spec, 2, [4], final=True, diag_steps=[4]) + train_records(2, [2, 3], True)
    m2 = dict(m1, status="COMPLETE", attempt=2, resumed_from_step=2, wall_time_s=60.0, estimated_cost_usd=0.2, error=None)
    write_attempt(run_dir, m2, second, group_records(2, [2, 3]))
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
    assert entry["completeness"]["complete"] is True
    assert entry["completeness"]["training_groups_found"] == 8 and entry["completeness"]["episodes_found"] == entry["completeness"]["episodes_expected"]
    assert [p["true_success"] for p in entry["training_curve"]] == [0.0, 0.0, 1.0, 1.0]
    assert entry["total_wall_time_s"] == 160.0 and entry["total_estimated_cost_usd"] == 0.6
    assert [a["status"] for a in entry["attempts"]] == ["FAILED", "COMPLETE"]
    assert set(entry["luck_share"]) == {"diag:step0", "diag:step4"} and not any("defect" in v for v in entry["luck_share"].values())
    assert "COMPLETE;" in format_run(entry)

    # the naive reading of the root files alone would report an area over steps 4 to 4 only
    root_only = periodic_curve(read_episode_log(run_dir / "episodes.jsonl"), "noisy")
    assert [p["step"] for p in root_only] == [4] and area_under_curve(root_only) is None
    assert [p["step"] for p in entry["curves"]["noisy"]] == [0, 2, 4]


def test_completeness_flags_missing_sets_and_diagnostic_defects(tmp_path):
    spec = spec_small()
    run_dir = tmp_path / "run-m"
    records = complete_records(spec, 1, [0, 2, 4], final=True, diag_steps=[0, 4]) + train_records(1, [0, 1, 2, 3], True)
    records = [r for r in records if not (r["phase"] == "eval_clean:step2" and r["slot"] == 0)]
    records = [r for r in records if not (r["phase"] == "diag:step4" and r["task_id"] == "d1" and r["slot"] == 3)]
    manifest = {"run_id": "run-m", "status": "COMPLETE", "attempt": 1, "spec": spec.to_dict(), "wall_time_s": 1.0, "estimated_cost_usd": 0.1}
    write_attempt(run_dir, manifest, records, group_records(1, [0, 1, 2]))
    (run_dir / "trainer" / "checkpoint-2").mkdir(parents=True)
    (run_dir / "trainer" / "checkpoint-2" / "trainer_state.json").write_text("{}")
    (run_dir / "trainer" / "checkpoint-2" / "adapter_model.safetensors").write_text("w")
    (run_dir / "trainer" / "checkpoint-4").mkdir()
    entry = run_register(run_dir)
    comp = entry["completeness"]
    assert comp["complete"] is False
    assert comp["periodic_and_final_missing"] == [{"set": "eval_clean", "step": 2, "found": 3, "expected": 4}]
    assert comp["diagnostic_missing"][0]["phase"] == "diag:step4" and "samples per schedule" in comp["diagnostic_missing"][0]["defect"]
    assert comp["training_groups_found"] == 6 and comp["training_groups_expected"] == 8
    assert entry["checkpoints"] == {"expected": [2, 4], "complete": [2], "incomplete": ["checkpoint-4"], "missing": [4], "adapter_final": False}
    text = format_run(entry)
    assert "INCOMPLETE" in text and "missing eval_clean step 2: 3 of 4" in text and "incomplete ['checkpoint-4']" in text

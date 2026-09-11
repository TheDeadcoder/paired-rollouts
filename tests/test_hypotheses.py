import json

import numpy as np
import pytest

from pairedrl.analysis.hypotheses import (
    RunData,
    decide,
    format_decision,
    group_seed_sharing,
    h1a,
    h2,
    h3,
    h5,
    pair_runs,
    trapezoid,
    write_decision,
)
from pairedrl.train.runner import RunSpec, expected_eval_sizes, periodic_steps

STEPS_H3 = (40, 60, 80)


def make_spec(arm, seed, condition="C2"):
    p, q = (0.25, 0.0) if condition == "C2" else (0.0, 0.10)
    return RunSpec(run_id=f"{condition}-{arm}-s{seed}", model="m", condition=condition, arm=arm, p=p, q=q, seed=seed,
                   steps=80, prompts_per_step=2, num_generations=2, eval_every=20, checkpoint_steps=40, eval_tasks=4,
                   final_eval_tasks=4, diagnostic_steps=[0], diagnostic_schedules=2, diagnostic_samples=2)


def rec(phase, condition, task, success, **kw):
    base = {"task_id": task, "phase": phase, "condition": condition, "true_success": bool(success),
            "observed_reward": float(success), "flipped": False, "calls": 6, "budget": 13, "budget_exceeded": False,
            "finished": True, "exposed": True, "faults": {}, "schedule_seed": 1, "attempt": 1}
    base.update(kw)
    return base


def write_run(tmp_path, spec, success_fn, final_fn, resolved_seed_fn, spurious_groups=0, status="COMPLETE", drop_final=False):
    """A complete synthetic run whose periodic task means are success_fn(set, step, task) and final task means
    final_fn(set, task); training groups (2 per step, 2 rollouts) with resolved seeds from resolved_seed_fn."""
    run_dir = tmp_path / spec.run_id
    run_dir.mkdir()
    sizes = expected_eval_sizes(spec)
    records = []
    per_task = {"clean": 2, "noisy": 2, "challenge": 1}
    for step in periodic_steps(spec):
        for name, n in sizes["periodic"].items():
            k = per_task[name]
            for i in range(n):
                task = f"h{i // k}"
                records.append(rec(f"eval_{name}:step{step}", f"eval:{name}", task, success_fn(name, step, task, i % k), schedule_seed=i % k + 1, slot=i))
    if not drop_final:
        for name, n in sizes["final"].items():
            k = 1 if name == "challenge" else 4
            for i in range(n):
                task = f"t{i // k}"
                records.append(rec(f"final_{name}:step{spec.steps}", f"eval:{name}", task, final_fn(name, task, i % k), schedule_seed=i % k + 1, slot=i))
    for task in ("d0", "d1"):
        for kk in range(2):
            for m in range(2):
                records.append(rec("diag:step0", "diag", task, (kk + m) % 2 == 0 if task == "d0" else kk == 0, schedule_seed=kk, slot=kk * 2 + m))
    groups = []
    for step in range(spec.steps):
        for g in range(2):
            outcomes = [True, True] if (step + g) % 3 else [True, False]
            spurious = bool(all(outcomes)) and g < spurious_groups
            observed = [1.0, 0.0] if spurious else [float(o) for o in outcomes]
            groups.append({"step": step, "attempt": 1, "group": g, "task_id": f"tr{g}", "condition": spec.condition,
                           "schedule_seed": 100 + step * 2 + g, "true_success": outcomes, "observed_reward": observed,
                           "advantages": [0.0, 0.0], "zero_variance": len(set(observed)) == 1, "spurious": spurious, "max_abs_advantage": 0.0})
            for i, o in enumerate(outcomes):
                records.append(rec(f"train:step{step}", spec.condition, f"tr{g}", o, schedule_seed=100 + step * 2 + g,
                                   resolved_seed=resolved_seed_fn(100 + step * 2 + g, i), slot=i, episode_index=step))
    manifest = {"run_id": spec.run_id, "status": status, "attempt": 1, "git_commit": "abc", "wall_time_s": 1.0,
                "estimated_cost_usd": 0.1, "spec": spec.to_dict(), "step_timings": []}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    (run_dir / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (run_dir / "groups.jsonl").write_text("\n".join(json.dumps(g) for g in groups) + "\n")
    return run_dir


def paired_seed_fn(schedule_seed, i):
    return schedule_seed


def independent_seed_fn(schedule_seed, i):
    return schedule_seed * 10 + i


def build_condition(tmp_path, condition, gaps, final_gaps, heldout_gaps=None):
    """Three seed pairs. Both arms score 0.5 at step 0; from step 20 on the paired arm scores 1.0 wherever
    gaps[seed] > 0 while the independent arm stays at 0.5; final sets likewise at 1.0 against 0.5 wherever
    final_gaps[seed] (heldout_gaps[seed] for the held-out types) is positive."""
    heldout_gaps = heldout_gaps or final_gaps
    dirs = []
    for seed, gap in enumerate(gaps):
        for arm in ("paired", "independent"):
            spec = make_spec(arm, seed, condition)
            bonus = gap if arm == "paired" else 0.0

            def success(name, step, task, k, bonus=bonus):
                base = (int(task[1:]) + k) % 2 == 0
                return True if (step > 0 and bonus > 0) else base

            def final(name, task, k, seed=seed, arm=arm):
                gap_f = (heldout_gaps if name == "heldout_types" else final_gaps)[seed] if arm == "paired" else 0.0
                return True if gap_f > 0 else (int(task[1:]) + k) % 2 == 0

            dirs.append(write_run(tmp_path, spec, success, final, paired_seed_fn if arm == "paired" else independent_seed_fn))
    return dirs


def test_trapezoid_matches_area_under_curve():
    assert trapezoid([0, 20, 40], np.array([0.0, 0.5, 1.0])) == pytest.approx(0.5)
    assert trapezoid([0, 40, 80], np.array([[0.2, 0.2, 0.2], [0.0, 1.0, 0.0]])).tolist() == pytest.approx([0.2, 0.5])


def test_h2_h3_h5_rules_and_bootstrap(tmp_path):
    dirs = build_condition(tmp_path, "C2", gaps=[0.5, 0.5, 0.5], final_gaps=[0.1, 0.1, 0.1])
    runs = [RunData(d) for d in dirs]
    pairs = pair_runs(runs)
    assert [p.spec.seed for p, _ in pairs] == [0, 1, 2] and all(p.spec.arm == "paired" and i.spec.arm == "independent" for p, i in pairs)
    result = h2(pairs, "C2", n_boot=500, seed=1)
    # paired arm: task means 1.0 at steps 20 to 80, 0.5 at step 0 -> AUC 0.9375; independent: 0.5 at every step
    assert result["auc_paired"] == pytest.approx([0.9375] * 3) and result["auc_independent"] == pytest.approx([0.5] * 3)
    assert result["auc_difference_by_seed"] == pytest.approx([0.4375] * 3) and result["rule"]["every_seed_positive"]
    assert result["final_difference_mean"] == pytest.approx(0.5) and result["passes"] is True
    lo, hi = result["auc_difference_ci95"]
    assert lo <= 0.4375 <= hi and lo > 0
    assert result["tasks"] == 4 and result["steps"] == [0, 20, 40, 60, 80]
    h3r = h3(pairs, n_boot=200, seed=1)
    assert h3r["difference_by_seed"] == pytest.approx([0.5] * 3) and h3r["passes"] is True
    h5r = h5(pairs, n_boot=200, seed=1)
    assert h5r["difference_by_seed"] == pytest.approx([0.5] * 3) and h5r["passes"] is True and h5r["difference_ci90"][0] > -0.03


def test_h2_misses_when_one_seed_is_not_positive(tmp_path):
    dirs = build_condition(tmp_path, "C2", gaps=[0.5, 0.0, 0.5], final_gaps=[0.1, 0.0, 0.1])
    pairs = pair_runs([RunData(d) for d in dirs])
    result = h2(pairs, "C2", n_boot=200, seed=2)
    assert result["auc_difference_by_seed"][1] == pytest.approx(0.0)
    assert result["rule"]["mean_criterion"] is True and result["rule"]["every_seed_positive"] is False
    assert result["passes"] is False


def test_h1a_c2_luck_share_and_seed_sharing(tmp_path):
    dirs = build_condition(tmp_path, "C2", gaps=[0.5, 0.5, 0.5], final_gaps=[0.1] * 3)
    runs = [RunData(d) for d in dirs]
    paired = next(r for r in runs if r.spec.arm == "paired")
    independent = next(r for r in runs if r.spec.arm == "independent")
    assert group_seed_sharing(paired) == {"groups": 160, "groups_sharing_one_seed": 160, "groups_with_several_seeds": 0}
    assert group_seed_sharing(independent) == {"groups": 160, "groups_sharing_one_seed": 0, "groups_with_several_seeds": 160}
    result = h1a(runs, "C2", n_boot=200, seed=3)
    for run_id, v in result["runs"].items():
        assert v["sharing_criterion"] is True, run_id
        assert v["luck_share_step0"]["tasks"] == 2
    assert "passes" in result


def test_h1a_c4_spurious_rates(tmp_path):
    dirs = []
    for seed in range(2):
        for arm, spurious in (("paired", 0), ("independent", 1)):
            spec = make_spec(arm, seed, "C4")
            dirs.append(write_run(tmp_path, spec, lambda n, s, t, k: k == 0, lambda n, t, k: k < 2,
                                  paired_seed_fn if arm == "paired" else independent_seed_fn, spurious_groups=spurious))
    runs = [RunData(d) for d in dirs]
    result = h1a(runs, "C4", n_boot=10, seed=0)
    assert result["spurious_rates"]["paired"]["rate"] == 0.0 and result["rule"]["paired_criterion"] is True
    ind = result["spurious_rates"]["independent"]
    assert ind["spurious"] > 0 and ind["rate"] == pytest.approx(ind["spurious"] / ind["all_correct_groups"])
    assert result["rule"]["independent_criterion"] == (abs(ind["rate"] - result["expected_independent_rate"]) <= 0.07)


def test_pair_runs_and_decide_guards(tmp_path):
    dirs = build_condition(tmp_path, "C2", gaps=[0.5, 0.5, 0.5], final_gaps=[0.1] * 3)
    runs = [RunData(d) for d in dirs]
    with pytest.raises(ValueError, match="seeds without both arms"):
        pair_runs(runs[:-1])
    decision = decide("C2", dirs, n_boot=100, seed=4)
    assert decision["h2"]["passes"] and decision["h3"]["passes"] and decision["h5"]["passes"]
    assert len(decision["seed_pairs"]) == 3
    text = format_decision(decision)
    assert "H2a (noisy AUC)" in text and "H3 (challenge 40/60/80)" in text and "H5 (heldout types)" in text
    out = tmp_path / "decisions" / "c2.json"
    write_decision(decision, out)
    assert json.loads(out.read_text())["condition"] == "C2"
    incomplete = write_run(tmp_path, make_spec("paired", 7), lambda n, s, t, k: True, lambda n, t, k: True, paired_seed_fn, drop_final=True)
    with pytest.raises(ValueError, match="not accepted"):
        decide("C2", [*dirs, incomplete], n_boot=10)
    with pytest.raises(ValueError, match="condition"):
        decide("C4", dirs, n_boot=10)

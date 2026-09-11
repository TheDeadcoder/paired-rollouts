import json

import numpy as np
import pytest
from synthetic import complete_run

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
from pairedrl.train.runner import RunSpec

STEPS_H3 = (40, 60, 80)


def make_spec(arm, seed, condition="C2"):
    p, q = (0.25, 0.0) if condition == "C2" else (0.0, 0.10)
    return RunSpec(run_id=f"{condition}-{arm}-s{seed}", model="m", condition=condition, arm=arm, p=p, q=q, seed=seed,
                   steps=80, prompts_per_step=2, num_generations=2, eval_every=20, checkpoint_steps=40, eval_tasks=4,
                   final_eval_tasks=4, diagnostic_steps=[0], diagnostic_schedules=2, diagnostic_samples=2)


def independent_seed_fn(schedule_seed, i):
    return schedule_seed * 10 + i


def write_run(tmp_path, spec, periodic, final, diagnostic=None, outcomes=None, spurious_groups=0, status="COMPLETE"):
    """A complete synthetic run on the frozen pools: periodic(set, step, task_index, schedule_index), final(set,
    task_index, schedule_index); the independent arm resolves a different seed per member."""
    return complete_run(
        tmp_path, spec, status=status, periodic=periodic, final=final, diagnostic=diagnostic, outcomes=outcomes,
        resolved_seed=None if spec.arm == "paired" else independent_seed_fn, spurious_groups=spurious_groups,
    )


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

            def periodic(name, step, ti, k, bonus=bonus):
                return True if (step > 0 and bonus > 0) else (ti + k) % 2 == 0

            def final(name, ti, k, seed=seed, arm=arm):
                gap_f = (heldout_gaps if name == "heldout_types" else final_gaps)[seed] if arm == "paired" else 0.0
                return True if gap_f > 0 else (ti + k) % 2 == 0

            dirs.append(write_run(tmp_path, spec, periodic, final))
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
        assert v["luck_share_step0"]["tasks"] == 16
    assert "passes" in result


def test_h1a_c4_spurious_rates(tmp_path):
    dirs = []
    for seed in range(2):
        for arm, spurious in (("paired", 0), ("independent", 1)):
            spec = make_spec(arm, seed, "C4")
            dirs.append(write_run(tmp_path, spec, lambda n, s, ti, k: k == 0, lambda n, ti, k: k < 2,
                                  outcomes=lambda step, g: [True, True] if (step + g) % 3 else [True, False], spurious_groups=spurious))
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
    assert set(decision["acceptance"]) == set(decision["runs"]) and all(v["commits"] == ["abc1234"] for v in decision["acceptance"].values())
    assert set(decision["task_pools"]) == {"train", "heldout", "diagnostic", "test"} and all(len(v) == 64 for v in decision["task_pools"].values())
    text = format_decision(decision)
    assert "H2a (noisy AUC)" in text and "H3 (challenge 40/60/80)" in text and "H5 (heldout types)" in text
    out = tmp_path / "decisions" / "c2.json"
    write_decision(decision, out)
    assert json.loads(out.read_text())["condition"] == "C2"
    incomplete = write_run(tmp_path, make_spec("paired", 7), lambda n, s, ti, k: True, lambda n, ti, k: True)
    (incomplete / "episodes.jsonl").write_text("".join(
        line + "\n" for line in (incomplete / "episodes.jsonl").read_text().splitlines() if '"final_' not in line
    ))
    with pytest.raises(ValueError, match="not accepted"):
        decide("C2", [*dirs, incomplete], n_boot=10)
    with pytest.raises(ValueError, match="condition"):
        decide("C4", dirs, n_boot=10)

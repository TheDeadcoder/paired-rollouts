import math
import random

import pytest

from pairedrl.analysis.diagnostics import (
    effective_group_size,
    grpo_advantages,
    luck_share,
    luck_share_over_tasks,
    spurious_rate_all_correct,
    std_ddof1,
    summarize_group,
)


def test_advantages_match_trl_formula():
    rewards = [1.0] * 7 + [0.0]
    adv = grpo_advantages(rewards)
    assert math.isclose(std_ddof1(rewards), math.sqrt(0.125), rel_tol=1e-9)
    assert math.isclose(adv[-1], -0.875 / (math.sqrt(0.125) + 1e-4), rel_tol=1e-9)
    assert math.isclose(adv[0], 0.125 / (math.sqrt(0.125) + 1e-4), rel_tol=1e-9)
    assert round(adv[-1], 2) == -2.47 and round(adv[0], 2) == 0.35
    plain = grpo_advantages(rewards, scale="none")
    assert math.isclose(plain[-1], -0.875) and math.isclose(plain[0], 0.125)
    with pytest.raises(ValueError):
        grpo_advantages(rewards, scale="batch")


def test_summarize_group_flags_spurious_variance():
    flaked = summarize_group([1.0] * 7 + [0.0], true_outcomes=[True] * 8)
    assert flaked.spurious is True and not flaked.zero_variance
    assert round(flaked.max_abs_advantage, 2) == 2.47
    honest = summarize_group([1.0] * 5 + [0.0] * 3, true_outcomes=[True] * 5 + [False] * 3)
    assert honest.spurious is False
    paired_flake = summarize_group([0.0] * 8, true_outcomes=[True] * 8)
    assert paired_flake.zero_variance and paired_flake.spurious is False
    assert paired_flake.max_abs_advantage == 0.0
    assert summarize_group([1.0, 0.0]).spurious is None


def test_spurious_rate_closed_form():
    assert round(spurious_rate_all_correct(0.05, 8), 2) == 0.34
    assert round(spurious_rate_all_correct(0.10, 8), 2) == 0.57
    assert round(spurious_rate_all_correct(0.02, 8), 2) == 0.15
    assert spurious_rate_all_correct(0.0, 8) == 0.0
    rng = random.Random(0)
    hits = 0
    n = 20000
    for _ in range(n):
        obs = [0.0 if rng.random() < 0.1 else 1.0 for _ in range(8)]
        hits += summarize_group(obs, true_outcomes=[True] * 8).spurious
    assert abs(hits / n - spurious_rate_all_correct(0.1, 8)) < 0.01


def test_luck_share_recovers_known_share_gaussian():
    rng = random.Random(1)
    for sigma_env, sigma_pol in ((0.5, 0.5), (0.3, 0.9), (0.9, 0.3)):
        target = sigma_env**2 / (sigma_env**2 + sigma_pol**2)
        tables = []
        for _ in range(400):
            a = [rng.gauss(0, sigma_env) for _ in range(8)]
            tables.append([[ak + rng.gauss(0, sigma_pol) for _ in range(8)] for ak in a])
        est = luck_share_over_tasks(tables)
        assert abs(est["lam"] - target) < 0.05, (sigma_env, sigma_pol, est)


def test_luck_share_recovers_known_share_binary():
    rng = random.Random(2)
    tables = []
    p_by_schedule = []
    for _ in range(600):
        probs = [rng.choice((0.2, 0.9)) for _ in range(8)]
        p_by_schedule.extend(probs)
        tables.append([[1.0 if rng.random() < p else 0.0 for _ in range(8)] for p in probs])
    mean_p = sum(p_by_schedule) / len(p_by_schedule)
    var_p = sum((p - mean_p) ** 2 for p in p_by_schedule) / len(p_by_schedule)
    within = sum(p * (1 - p) for p in p_by_schedule) / len(p_by_schedule)
    target = var_p / (var_p + within)
    est = luck_share_over_tasks(tables)
    assert abs(est["lam"] - target) < 0.05


def test_luck_share_edge_cases():
    zero = luck_share([[1.0] * 4 for _ in range(4)])
    assert zero.lam is None and zero.sigma2_env == 0.0 and zero.sigma2_pol == 0.0
    pure_policy = luck_share([[1.0, 0.0, 1.0, 0.0] for _ in range(4)])
    assert pure_policy.lam == 0.0
    pure_luck = luck_share([[1.0] * 4, [0.0] * 4, [1.0] * 4, [0.0] * 4])
    assert pure_luck.lam == 1.0
    with pytest.raises(ValueError):
        luck_share([[1.0, 0.0]])
    with pytest.raises(ValueError):
        luck_share([[1.0, 0.0], [1.0]])
    summary = luck_share_over_tasks([[[1.0] * 4] * 4, [[1.0, 0.0, 1.0, 0.0]] * 4])
    assert summary["tasks"] == 2 and summary["tasks_defined"] == 1 and summary["lam"] == 0.0


def test_effective_group_size():
    assert effective_group_size(8, 0.0) == 8
    assert math.isclose(effective_group_size(8, 0.5), 16)
    with pytest.raises(ValueError):
        effective_group_size(8, 1.0)

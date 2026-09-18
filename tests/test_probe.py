import pathlib
import sys

import numpy as np
import pytest

from pairedrl.analysis.probe import (
    bandit_enumeration,
    decide_trajectory,
    group_terms,
    outcome_noise_group,
    trace_variance,
    transition_designs,
)
from pairedrl.train.probe import ProbeSpec, probe_ledger_row

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import build_probe_register


def test_bandit_reproduces_theory():
    b = bandit_enumeration()
    assert round(b["contrast_variance_independent"], 3) == 0.495
    assert round(b["contrast_variance_paired"], 3) == 0.450
    assert round(b["trace_var_mean_centered_independent"], 6) == 0.002615
    assert round(b["trace_var_mean_centered_paired"], 6) == 0.005845
    assert round(b["trace_var_implemented_independent"], 6) == 0.007053
    assert round(b["trace_var_implemented_paired"], 6) == 0.019724
    assert round(b["expected_update_implemented_independent"], 4) == 0.3918
    assert round(b["expected_update_implemented_paired"], 4) == 0.3905
    assert round(b["lhs_mean"], 4) == 0.0496
    assert round(b["rhs_mean"], 4) == 0.0137


def test_mean_centered_identity_and_implemented_monte_carlo():
    rng = np.random.default_rng(0)
    g, dim, q = 8, 50, 0.1
    scores = rng.standard_normal((g, dim))
    r = (rng.random(g) < 0.7).astype(float)
    r[0] = 1.0
    gram = scores @ scores.T
    tokens = np.ones(g)
    res = outcome_noise_group(gram, r, tokens, q)
    lhs, rhs = res["lhs"], res["rhs"]
    diff = res["mean_centered"]["paired"]["expected_sq_norm"] - res["mean_centered"]["independent"]["expected_sq_norm"]
    assert abs(diff - q * (1 - q) * (lhs - rhs)) < 1e-9

    succ = [i for i in range(g) if r[i] > 0.5]
    n_mc = 200_000
    kept = rng.random((n_mc, len(succ))) < (1 - q)
    obs = np.zeros((n_mc, g))
    obs[:, succ] = kept.astype(float)
    mean = obs.mean(axis=1, keepdims=True)
    std = obs.std(axis=1, ddof=1, keepdims=True)
    weights = (obs - mean) / (std + 1e-4) / tokens.sum()
    quad = np.einsum("ni,ij,nj->n", weights, gram, weights)
    mc_mean, mc_se = quad.mean(), quad.std() / np.sqrt(n_mc)
    assert abs(res["implemented"]["independent"]["expected_sq_norm"] - mc_mean) < 3 * mc_se


def test_orthogonal_scores_condition():
    g = 8
    scores = np.eye(g) * 2.0
    gram = scores @ scores.T
    tokens = np.ones(g)
    for k in range(g + 1):
        r = np.zeros(g)
        r[:k] = 1.0
        res = outcome_noise_group(gram, r, tokens, 0.1)
        lhs, rhs = res["lhs"], res["rhs"]
        if k == 0:
            assert lhs == 0.0 and rhs == 0.0
        elif k == 1:
            assert abs(lhs - rhs) < 1e-12
        else:
            assert lhs < rhs
        if k == g:
            assert abs(lhs) < 1e-12


def test_transition_designs_shapes():
    k_sched, m_samp, resamples = 8, 8, 16
    rng = np.random.default_rng(1)
    designs = transition_designs(k_sched, m_samp, resamples, rng)
    assert designs["paired"] == [[k * m_samp + m for m in range(m_samp)] for k in range(k_sched)]
    assert len(designs["independent_resampled"]) == resamples
    for group in designs["independent_resampled"]:
        assert len(group) == m_samp and len(set(group)) == m_samp
        for sched in {i // m_samp for i in group}:
            same = [i for i in group if i // m_samp == sched]
            assert len(set(same)) == len(same) and len(same) <= m_samp
    assert len(designs["stratified"]) == resamples
    for group in designs["stratified"]:
        assert sorted(i // m_samp for i in group) == list(range(k_sched))


def test_group_terms_and_trace_variance_match_numpy():
    rng = np.random.default_rng(3)
    k_sched, m_samp = 4, 4
    scores = rng.standard_normal((k_sched * m_samp, 6))
    gram = scores @ scores.T
    rewards = (rng.random(k_sched * m_samp) < 0.5).astype(float)
    tokens = rng.integers(1, 5, size=k_sched * m_samp).astype(float)
    designs = transition_designs(k_sched, m_samp, 8, np.random.default_rng(4))
    terms = group_terms(gram, rewards, tokens, designs["paired"])
    mc = terms["mean_centered"]
    explicit = []
    for group in designs["paired"]:
        obs = rewards[group]
        w = (obs - obs.mean()) / len(group)
        explicit.append(sum(w[a] * w[b] * scores[group[a]] @ scores[group[b]]
                            for a in range(len(group)) for b in range(len(group))))
    assert abs(mc["sum_sq_norm"] - sum(explicit)) < 1e-9
    running = mc["summed_weights"] @ scores
    tv = trace_variance(mc["sum_sq_norm"], running @ running, mc["n_groups"])
    group_vecs = np.array([sum((rewards[g_] - rewards[group].mean()) / len(group) * scores[g_] for g_ in group)
                           for group in designs["paired"]])
    expected = np.mean([v @ v for v in group_vecs]) - group_vecs.mean(axis=0) @ group_vecs.mean(axis=0)
    assert abs(tv - expected) < 1e-9


def _summary(out_p, out_i, tr_p, tr_i, lhs=0.01, rhs=0.05):
    return {
        "outcome": {"lhs_mean": lhs, "rhs_mean": rhs,
                    "mean_centered": {"trace_var": {"paired": out_p, "independent": out_i}},
                    "implemented": {"trace_var": {"paired": out_p, "independent": out_i}}},
        "transition": {"mean_centered": {"trace_var": {"paired": tr_p, "independent_resampled": tr_i, "stratified": tr_i}},
                       "implemented": {"trace_var": {"paired": tr_p, "independent_resampled": tr_i, "stratified": tr_i}}},
    }


def test_decide_trajectory():
    passing = {0: _summary(1, 2, 1, 2), 20: _summary(1, 2, 1, 2), 100: _summary(1, 2, 3, 2)}
    d = decide_trajectory(passing)
    assert d["outcome_noise_every_checkpoint"] and d["transition_noise_majority"] and d["passes"]

    outcome_fail = {0: _summary(1, 2, 1, 2), 100: _summary(3, 2, 1, 2)}
    d2 = decide_trajectory(outcome_fail)
    assert not d2["outcome_noise_every_checkpoint"] and not d2["passes"]

    transition_minority = {0: _summary(1, 2, 3, 2), 20: _summary(1, 2, 3, 2), 100: _summary(1, 2, 1, 2)}
    d3 = decide_trajectory(transition_minority)
    assert not d3["transition_noise_majority"] and not d3["passes"]


def test_probespec_roundtrip_and_register_smoke_refusal():
    spec = ProbeSpec(probe_id="p", model="m", trajectory="t1-c2-paired-s1", steps=[0], smoke=True,
                     notes="end-to-end smoke; not pre-registered")
    assert ProbeSpec.from_dict(spec.to_dict()) == spec
    manifest = {"spec": {"smoke": True}, "status": "COMPLETE", "steps": [0]}
    with pytest.raises(SystemExit):
        build_probe_register.check(manifest, {0: {}}, "outputs/runs/probe-smoke")


def test_probe_ledger_row_has_ten_columns():
    spec = ProbeSpec(probe_id="probe-x", model="Qwen/Qwen3.5-2B", trajectory="t1-c2-paired-s1", steps=[0, 20, 40],
                     seed=1, notes="gradient probe H1(b) trajectory B; pre-registered v1")
    row = probe_ledger_row(spec, "2026-09-18T00:00:00+00:00", "fc-1", "Qwen/Qwen3.5-2B", "C2", "paired",
                           preregistration="v1")
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    assert len(cells) == 10
    assert cells[0] == "probe-x" and cells[2] == "modal" and cells[4] == "C2" and cells[5] == "paired"
    assert cells[6] == "1" and cells[7] == "3" and cells[8] == "LAUNCHED"

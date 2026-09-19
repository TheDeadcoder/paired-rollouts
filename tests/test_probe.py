import gzip
import json
import pathlib
import sys

import numpy as np
import pytest

from pairedrl.analysis.probe import (
    bandit_enumeration,
    checkpoint_summary,
    decide_trajectory,
    group_terms,
    implemented_weights,
    outcome_noise_group,
    transition_designs,
)
from pairedrl.train import probe as tp
from pairedrl.train.probe import ProbeSpec, probe_ledger_row

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import build_probe_register
import recompute_probe_step

FULL_DESIGN = build_probe_register.FULL_DESIGN


# ---- arithmetic (analysis.probe) --------------------------------------------

def test_bandit_reproduces_theory():
    b = bandit_enumeration()
    assert round(b["contrast_variance_independent"], 3) == 0.495
    assert round(b["contrast_variance_paired"], 3) == 0.450
    assert round(b["trace_var_mean_centered_independent"], 6) == 0.002615
    assert round(b["trace_var_mean_centered_paired"], 6) == 0.005845
    assert round(b["trace_var_implemented_independent"], 6) == 0.007053
    assert round(b["trace_var_implemented_paired"], 6) == 0.019724
    assert round(b["lhs_mean"], 4) == 0.0496
    assert round(b["rhs_mean"], 4) == 0.0137


def test_mean_centered_identity_and_implemented_monte_carlo():
    rng = np.random.default_rng(0)
    g, dim, q = 8, 50, 0.1
    scores = rng.standard_normal((g, dim))
    r = (rng.random(g) < 0.7).astype(float)
    r[0] = 1.0
    gram = scores @ scores.T
    rho, t_ref = np.ones(g), float(g)
    res = outcome_noise_group(gram, r, rho, t_ref, q)
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
    weights = (obs - mean) / (std + 1e-4) / t_ref
    quad = np.einsum("ni,ij,nj->n", weights, gram, weights)
    assert abs(res["implemented"]["independent"]["expected_sq_norm"] - quad.mean()) < 3 * quad.std() / np.sqrt(n_mc)


def test_orthogonal_scores_condition():
    g = 8
    gram = (np.eye(g) * 2.0) @ (np.eye(g) * 2.0)
    rho, t_ref = np.ones(g), float(g)
    for k in range(g + 1):
        r = np.zeros(g)
        r[:k] = 1.0
        res = outcome_noise_group(gram, r, rho, t_ref, 0.1)
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
    designs = transition_designs(k_sched, m_samp, resamples, np.random.default_rng(1))
    assert designs["paired"] == [[k * m_samp + m for m in range(m_samp)] for k in range(k_sched)]
    assert len(designs["independent_resampled"]) == resamples
    for group in designs["independent_resampled"]:
        assert len(group) == m_samp and len(set(group)) == m_samp
        for sched in {i // m_samp for i in group}:
            same = [i for i in group if i // m_samp == sched]
            assert len(set(same)) == len(same) and len(same) <= m_samp
    for group in designs["stratified"]:
        assert sorted(i // m_samp for i in group) == list(range(k_sched))


def test_group_terms_and_trace_variance_match_numpy():
    rng = np.random.default_rng(3)
    k_sched, m_samp = 4, 4
    scores = rng.standard_normal((k_sched * m_samp, 6))
    gram = scores @ scores.T
    rewards = (rng.random(k_sched * m_samp) < 0.5).astype(float)
    rho, t_ref = np.ones(k_sched * m_samp), 100.0
    designs = transition_designs(k_sched, m_samp, 8, np.random.default_rng(4))
    mc = group_terms(gram, rewards, rho, t_ref, designs["paired"])["mean_centered"]
    explicit = 0.0
    for group in designs["paired"]:
        obs = rewards[group]
        w = (obs - obs.mean()) / len(group)
        explicit += sum(w[a] * w[b] * scores[group[a]] @ scores[group[b]] for a in range(len(group)) for b in range(len(group)))
    assert abs(mc["sum_sq_norm"] - explicit) < 1e-9


def test_implemented_weights_batch_vs_per_group_counterexample():
    r, rho = [1.0, 0.0], np.array([1.0, 1.0])
    t_a, t_b = 10.0, 100.0
    batch = np.concatenate([implemented_weights(r, rho, t_a + t_b), implemented_weights(r, rho, t_a + t_b)])
    per_group = np.concatenate([implemented_weights(r, rho, t_a), implemented_weights(r, rho, t_b)])
    cos = batch @ per_group / (np.linalg.norm(batch) * np.linalg.norm(per_group))
    assert cos < 0.999


def test_within_task_decomposition_matches_bruteforce():
    rng = np.random.default_rng(5)
    k_sched, m_samp = 4, 4
    scores = rng.standard_normal((k_sched * m_samp, 6))
    gram = scores @ scores.T
    rewards = (rng.random(k_sched * m_samp) < 0.5).astype(float)
    designs = transition_designs(k_sched, m_samp, 6, np.random.default_rng(6))["independent_resampled"]
    mc = group_terms(gram, rewards, np.ones(k_sched * m_samp), 100.0, designs)["mean_centered"]
    within = tp.within_task_trace_transition(mc["sum_sq_norm"], mc["summed_weights"], gram, mc["n_groups"])
    gvecs = np.array([sum((rewards[g_] - rewards[grp].mean()) / len(grp) * scores[g_] for g_ in grp) for grp in designs])
    brute = np.mean([v @ v for v in gvecs]) - gvecs.mean(axis=0) @ gvecs.mean(axis=0)
    assert abs(within - brute) < 1e-9


def _outcome_block(indep):
    return {"paired": _pd(1.0), indep: _pd(2.0), "cross_vec": 0.1}


def _pd(ss):
    return {"sum_sq_norm": ss, "vec_sq_norm": 0.5, "n_groups": 10, "n_within": 5, "within_sum": ss * 0.4, "reward_variance": 0.3}


def _accumulators(step):
    tr = {"paired": _pd(1.0), "independent_resampled": _pd(2.0), "stratified": _pd(2.0), "cross_vec": 0.1}
    return {"step": step, "q": 0.1,
            "outcome": {"n_groups": 10, "lhs_sum": 0.5, "rhs_sum": 2.0,
                        "by_k": {8: {"count": 10, "lhs_sum": 0.5, "rhs_sum": 2.0, "ratio_sum": 2.5}},
                        "mean_centered": _outcome_block("independent"), "implemented": _outcome_block("independent")},
            "transition": {"mean_centered": dict(tr), "implemented": dict(tr)}}


def test_checkpoint_summary_within_and_by_k():
    s = checkpoint_summary(_accumulators(0))
    oc = s["outcome"]["mean_centered"]
    assert abs(oc["trace_var"]["paired"] - 0.095) < 1e-9
    assert abs(oc["trace_var_within"]["paired"] - 0.08) < 1e-9
    assert abs(oc["trace_var_between"]["paired"] - 0.015) < 1e-9
    assert s["outcome"]["by_k"][8]["count"] == 10 and abs(s["outcome"]["by_k"][8]["lhs_over_rhs"] - 0.25) < 1e-9


def test_decide_trajectory():
    def summ(op, oi, tp_, ti):
        return {"outcome": {"lhs_mean": 0.01, "rhs_mean": 0.05,
                            "mean_centered": {"trace_var": {"paired": op, "independent": oi}}},
                "transition": {"mean_centered": {"trace_var": {"paired": tp_, "independent_resampled": ti, "stratified": ti}}}}
    passing = {0: summ(1, 2, 1, 2), 20: summ(1, 2, 1, 2), 100: summ(1, 2, 3, 2)}
    d = decide_trajectory(passing)
    assert d["outcome_noise_every_checkpoint"] and d["transition_noise_majority"] and d["passes"]
    d2 = decide_trajectory({0: summ(1, 2, 1, 2), 100: summ(3, 2, 1, 2)})
    assert not d2["outcome_noise_every_checkpoint"] and not d2["passes"]
    d3 = decide_trajectory({0: summ(1, 2, 3, 2), 20: summ(1, 2, 3, 2), 100: summ(1, 2, 1, 2)})
    assert not d3["transition_noise_majority"] and not d3["passes"]


# ---- train.probe helpers ----------------------------------------------------

class _FakeEnv:
    def __init__(self, episode):
        self.last_episode = episode


class _FakeTrainer:
    def __init__(self):
        self.environments = []

    def _generate_and_score_completions(self, inputs):
        self.environments = [_FakeEnv(ep) for ep in inputs]
        return {"batch": list(inputs)}


def test_capture_isolation():
    trainer = _FakeTrainer()
    with tp.capture_batches(trainer) as diag:
        trainer._generate_and_score_completions([{"task_id": "a"}, {"task_id": "b"}])
    with tp.capture_batches(trainer) as clean:
        trainer._generate_and_score_completions([{"task_id": "c"}])
    assert tp.flatten_episodes(diag) == [{"task_id": "a"}, {"task_id": "b"}]
    assert tp.flatten_episodes(clean) == [{"task_id": "c"}]
    assert trainer._generate_and_score_completions([{"task_id": "d"}]) == {"batch": [{"task_id": "d"}]}


def _episode(task, sched, phase, exposed=False):
    return {"task_id": task, "schedule_seed": sched, "resolved_seed": sched, "phase": phase, "exposed": exposed}


def test_identity_assertions():
    design = {"diagnostic_tasks": 2, "schedules": 2, "samples": 2, "clean_tasks": 2, "clean_rollouts": 2, "resamples": 4}
    diag = [(None, [_episode(t, s, "probe_diag:step0") for t in ("d0", "d1") for s in (10, 11) for _ in range(2)])]
    clean = [(None, [_episode(t, 0, "probe_clean:step0") for t in ("c0", "c1") for _ in range(2)])]
    tp.assert_capture_identities(diag, clean, design)
    bad = [(None, [_episode("d0", 0, "probe_clean:step0") for _ in range(2)] + [_episode("c1", 0, "probe_clean:step0") for _ in range(2)])]
    with pytest.raises(AssertionError):
        tp.assert_capture_identities(diag, bad, design)


def test_transition_member_ordering():
    def mem(sched, idx):
        return (None, idx, {"schedule_seed": sched, "task_id": "t"})
    shuffled = [mem(20, 0), mem(10, 1), mem(20, 2), mem(10, 3), mem(10, 4), mem(20, 5)]
    seeds = [m[2]["schedule_seed"] for m in tp.order_transition_members(shuffled)]
    assert seeds == [10, 10, 10, 20, 20, 20]
    tp.assert_schedule_layout(seeds, 2, 3, "t")
    with pytest.raises(AssertionError):
        tp.assert_schedule_layout([10, 20, 10, 20, 10, 20], 2, 3, "t")


def test_coordinate_stability():
    named = [("m.lora_A", True), ("m.lora_B", True), ("vision.lora_A", False)]
    coords = tp.select_coordinates(named)
    assert coords == ["m.lora_A", "m.lora_B"]
    tp.check_coordinates_stable(named, coords)
    with pytest.raises(AssertionError):
        tp.check_coordinates_stable([("m.lora_A", True), ("m.lora_B", False), ("vision.lora_A", True)], coords)


def test_chunked_gram_and_weighted_sum():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    scores = rng.standard_normal((8, 40))
    gram = tp.gram_chunked(torch.as_tensor(scores), 3)
    assert np.allclose(gram, scores @ scores.T)
    w = rng.standard_normal(8)
    assert np.allclose(tp.weighted_sum_chunked(w, torch.as_tensor(scores), 3), w @ scores)


def test_lora_fingerprint_on_bf16_parameters():
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.randn(6, 4, dtype=torch.bfloat16))
            self.lora_B = torch.nn.Parameter(torch.randn(4, 6, dtype=torch.bfloat16))
            self.base = torch.nn.Parameter(torch.randn(3, 3))

    torch.manual_seed(0)
    model = Tiny()
    coords = ["lora_A", "lora_B"]
    assert len(tp._bf16_bytes(model.lora_A.data)) == 2 * model.lora_A.numel()
    assert len(tp._bf16_bytes(model.base.data)) == 2 * model.base.numel()
    assert len(tp._bf16_bytes(model.lora_A.data.T)) == 2 * model.lora_A.numel()
    before = tp.lora_fingerprint(model, coords)
    assert len(before) == 64 and before == tp.lora_fingerprint(model, coords)
    assert before != tp.lora_fingerprint(model, coords[::-1])
    with torch.no_grad():
        model.lora_B[1, 2] += 1.0
    assert tp.lora_fingerprint(model, coords) != before


def test_atomic_write_and_resume(tmp_path):
    spec = ProbeSpec(probe_id="p", model="m", trajectory="t1-c2-paired-s1", steps=[0])
    path = tmp_path / "step0.json"
    tp.atomic_write_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    prov = tp._provenance(spec, "abc", None, "commitX", spec.design())
    ok, _ = tp.resume_matches({"provenance": prov}, prov)
    assert ok
    other = tp._provenance(spec, "abc", None, "commitY", spec.design())
    ok2, reason = tp.resume_matches({"provenance": prov}, other)
    assert not ok2 and "git_commit" in reason


def test_run_probe_loop(tmp_path):
    spec = ProbeSpec(probe_id="p", model="m", trajectory="t1-c2-paired-s1", steps=[0, 20])

    def good(step):
        payload = {"step": step, "trajectory": spec.trajectory,
                   "provenance": tp._provenance(spec, None, None, "c", spec.design()),
                   "counts": {"clean_rollouts": 10, "diagnostic_rollouts": 20},
                   "generation_seconds": 1, "scoring_seconds": 1}
        tp.atomic_write_json(tmp_path / f"step{step}.json", payload)
        return 0

    res = tp.run_probe(spec, "runs", "data", tmp_path, "c", "c", "modal", 3.5, step_runner=good)
    assert res["manifest"]["status"] == "COMPLETE"
    res2 = tp.run_probe(spec, "runs", "data", tmp_path, "c", "c", "modal", 3.5, step_runner=lambda s: pytest.fail("should skip"))
    assert res2["manifest"]["status"] == "COMPLETE"
    fail_dir = tmp_path / "fail"
    with pytest.raises(RuntimeError):
        tp.run_probe(spec, "runs", "data", fail_dir, "c", "c", "modal", 3.5, reraise_attempts=1, step_runner=lambda s: 1)
    assert tp.run_probe(spec, "runs", "data", tmp_path, "c", "cx", "modal", 3.5, step_runner=good)["manifest"]["status"] == "REFUSED"


# ---- register builder and recompute -----------------------------------------

def _summary(step):
    return checkpoint_summary(_accumulators(step))


def _write_probe_dir(base, probe_id, steps, trajectory, coord="cc", git="gg", smoke=False, design=None):
    d = base / probe_id
    d.mkdir()
    (d / "run_manifest.json").write_text(json.dumps({"spec": {"smoke": smoke}, "status": "COMPLETE", "steps": list(steps), "git_commit": git}))
    for s in steps:
        payload = {"step": s, "trajectory": trajectory,
                   "provenance": {"design_counts": design or FULL_DESIGN, "spec_sha256": f"spec-{probe_id}", "base_fingerprint": None},
                   "coordinate_names_sha256": coord, "git_commit": git,
                   "counts": {"rho_zero_fraction": 0.0, "clean_rollouts": 512, "diagnostic_rollouts": 1024}, "summary": _summary(s)}
        (d / f"step{s}.json").write_text(json.dumps(payload))
    return d


def _run_register(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["build_probe_register.py", *args])
    build_probe_register.main()


def test_register_accepts_declared_set(monkeypatch, tmp_path):
    a = _write_probe_dir(tmp_path, "probe-a", [0, 20, 40], "t1-c2-paired-s1")
    b = _write_probe_dir(tmp_path, "probe-b", [60, 80, 100], "t1-c2-paired-s1")
    out = tmp_path / "reg.json"
    _run_register(monkeypatch, ["--probes", str(a), str(b), "--trajectory", "t1-c2-paired-s1", "--base-from", str(a), "--out", str(out)])
    reg = json.loads(out.read_text())
    assert reg["steps"] == [0, 20, 40, 60, 80, 100] and reg["verdict"]["passes"]


def test_register_shares_the_base_across_trajectories(monkeypatch, tmp_path):
    a = _write_probe_dir(tmp_path, "probe-a", [0, 20, 40], "t1-c2-paired-s1")
    g = _write_probe_dir(tmp_path, "probe-g", [80, 100], "gate1-c2-paired-s0")
    out = tmp_path / "gate.json"
    _run_register(monkeypatch, ["--probes", str(g), "--trajectory", "gate1-c2-paired-s0", "--base-from", str(a), "--out", str(out)])
    reg = json.loads(out.read_text())
    assert reg["steps"] == [0, 80, 100] and reg["base_trajectory"] == "t1-c2-paired-s1" and reg["base_from"] == str(a)
    assert reg["checkpoints"]["0"]["trajectory"] == "t1-c2-paired-s1" and reg["checkpoints"]["80"]["trajectory"] == "gate1-c2-paired-s0"
    a0 = _write_probe_dir(tmp_path, "probe-a0", [0], "t1-c2-paired-s1")
    with pytest.raises(SystemExit, match="trajectory"):
        _run_register(monkeypatch, ["--probes", str(g), str(a0), "--trajectory", "gate1-c2-paired-s0", "--out", str(tmp_path / "x.json")])
    adapted = _write_probe_dir(tmp_path, "probe-adapted", [0], "t1-c2-paired-s1")
    payload = json.loads((adapted / "step0.json").read_text())
    payload["adapter_path"] = "runs/t1-c2-paired-s1/trainer/checkpoint-20"
    (adapted / "step0.json").write_text(json.dumps(payload))
    with pytest.raises(SystemExit):
        _run_register(monkeypatch, ["--probes", str(g), "--trajectory", "gate1-c2-paired-s0", "--base-from", str(adapted), "--out", str(tmp_path / "y.json")])


def test_register_refuses_missing_wrong_smoke_duplicate(monkeypatch, tmp_path):
    a = _write_probe_dir(tmp_path, "probe-a", [0, 20, 40], "t1-c2-paired-s1")
    with pytest.raises(SystemExit):
        _run_register(monkeypatch, ["--probes", str(a), "--trajectory", "t1-c2-paired-s1", "--out", str(tmp_path / "m.json")])
    wrong = _write_probe_dir(tmp_path, "probe-wrong", [0, 80, 100], "gate1-c2-paired-s0")
    with pytest.raises(SystemExit):
        _run_register(monkeypatch, ["--probes", str(wrong), "--trajectory", "t1-c2-paired-s1", "--out", str(tmp_path / "w.json")])
    smoke = _write_probe_dir(tmp_path, "probe-smk", [0, 80, 100], "gate1-c2-paired-s0", smoke=True)
    with pytest.raises(SystemExit):
        _run_register(monkeypatch, ["--probes", str(smoke), "--trajectory", "gate1-c2-paired-s0", "--out", str(tmp_path / "s.json")])
    dup1 = _write_probe_dir(tmp_path, "probe-d1", [0, 80], "gate1-c2-paired-s0")
    dup2 = _write_probe_dir(tmp_path, "probe-d2", [80, 100], "gate1-c2-paired-s0", coord="different")
    with pytest.raises(SystemExit):
        _run_register(monkeypatch, ["--probes", str(dup1), str(dup2), "--trajectory", "gate1-c2-paired-s0", "--out", str(tmp_path / "d.json")])


def test_recompute_reproduces_summary(tmp_path):
    rng = np.random.default_rng(11)
    t_ref, q = 200.0, 0.1
    outcome_groups, transition_tasks = [], []
    for _ in range(4):
        sc = rng.standard_normal((8, 12))
        gram = sc @ sc.T
        true = (rng.random(8) < 0.7).astype(float)
        true[0] = 1.0
        rho = np.ones(8)
        outcome_groups.append({"gram": gram.tolist(), "true": true.tolist(), "tokens": [10] * 8, "rho": rho.tolist(), "identities": []})
    for _ in range(3):
        sc = rng.standard_normal((16, 12))
        gram = sc @ sc.T
        rewards = (rng.random(16) < 0.5).astype(float)
        lists = transition_designs(4, 4, 6, np.random.default_rng(1))
        transition_tasks.append({"gram": gram.tolist(), "rewards": rewards.tolist(), "tokens": [10] * 16, "rho": [1.0] * 16, "index_lists": lists, "identities": []})
    run_oc = {e: {"paired": 0.3, "independent": 0.7, "cross_vec": 0.05} for e in ("mean_centered", "implemented")}
    run_tr = {e: {"paired": 0.3, "independent_resampled": 0.7, "stratified": 0.6, "cross_vec": 0.05} for e in ("mean_centered", "implemented")}
    evidence = {"step": 20, "q": q, "t_ref": t_ref, "P": 12, "coordinate_names": ["x"],
                "outcome": {"groups": outcome_groups, "running": run_oc},
                "transition": {"tasks": transition_tasks, "running": run_tr}}
    summary = recompute_probe_step.recompute_summary(evidence)
    path = tmp_path / "step20.json"
    tp.atomic_write_json(path, {"step": 20, "summary": json.loads(json.dumps(summary, sort_keys=True))})
    tp.atomic_write_gzip(tmp_path / "step20_evidence.json.gz", evidence)
    with gzip.open(tmp_path / "step20_evidence.json.gz", "rt") as f:
        again = recompute_probe_step.recompute_summary(json.load(f))
    assert not recompute_probe_step.compare(json.loads(json.dumps(summary, sort_keys=True)), json.loads(json.dumps(again, sort_keys=True)), 1e-9)


def test_probespec_roundtrip_and_ledger_row():
    spec = ProbeSpec(probe_id="probe-x", model="Qwen/Qwen3.5-2B", trajectory="t1-c2-paired-s1", steps=[0, 20, 40], seed=0,
                     notes="gradient probe H1(b); pre-registered v1")
    assert ProbeSpec.from_dict(spec.to_dict()) == spec
    row = probe_ledger_row(spec, "2026-09-18T00:00:00+00:00", "fc-1", "Qwen/Qwen3.5-2B", "C2", "paired", preregistration="v1")
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    assert len(cells) == 10 and cells[0] == "probe-x" and cells[4] == "C2" and cells[8] == "LAUNCHED"

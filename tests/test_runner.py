import json
from collections import Counter

import pytest

from pairedrl.analysis.diagnostics import luck_share_over_tasks
from pairedrl.noise.config import NoiseConfig
from pairedrl.train.dataset import eval_schedule_seed
from pairedrl.train.runner import (
    RunSpec,
    assemble_diagnostic_rows,
    assemble_eval_sets,
    assemble_training_rows,
    expected_eval_sizes,
    load_task_splits,
    luck_share_tables,
    luck_share_tables_by_phase,
    periodic_steps,
    read_episode_log,
    summarize_episodes,
    summarize_groups,
    training_group_records,
    vllm_engine_overrides,
)

SPLITS = load_task_splits("data/tasks")


def spec(**overrides) -> RunSpec:
    base = {"run_id": "t", "model": "Qwen/Qwen3.5-2B", "condition": "C2", "arm": "paired", "p": 0.25}
    base.update(overrides)
    return RunSpec(**base)


def test_spec_validation_and_round_trip():
    s = spec()
    assert RunSpec.from_dict(json.loads(json.dumps(s.to_dict()))) == s
    with pytest.raises(ValueError):
        spec(arm="sometimes")
    with pytest.raises(ValueError):
        spec(arm="clean", p=0.25)
    with pytest.raises(ValueError):
        spec(arm="paired", p=0.0)
    with pytest.raises(ValueError):
        spec(scale_rewards="batch")
    with pytest.raises(ValueError):
        spec(eval_only=True, train_only=True)
    with pytest.raises(ValueError):
        spec(prompts_per_step=5, num_generations=8, micro_batch=16)
    with pytest.raises(ValueError):
        spec(logprob_chunk=0)
    assert (s.micro_batch, s.logprob_chunk, s.per_device_eval_batch_size) == (2, 1, 128)
    assert (s.max_completion_length, s.vllm_max_model_length, s.max_tool_calling_iterations) == (6144, 12288, 24)
    assert vllm_engine_overrides(s) == {}
    assert vllm_engine_overrides(spec(vllm_enable_prefix_caching=True, vllm_max_num_batched_tokens=8192)) == {
        "enable_prefix_caching": True, "max_num_batched_tokens": 8192}
    with pytest.raises(ValueError):
        spec(vllm_max_num_batched_tokens=512)
    assert spec(arm="clean", p=0.0).training_noise().is_clean
    assert spec(arm="blocking").training_noise().mode == "independent"
    assert spec(arm="independent").eval_noise().mode == "paired"
    assert spec(arm="paired", p=0.0, q=0.1).training_noise() == NoiseConfig(p=0.0, q=0.1, mode="paired")


def test_training_rows_per_arm():
    s = spec(steps=10, prompts_per_step=4)
    rows = assemble_training_rows(s, SPLITS["train"])
    assert len(rows) == 40 and all(r["condition"] == "C2" for r in rows)
    assert all(json.loads(r["noise"])["mode"] == "paired" for r in rows)
    indep = assemble_training_rows(spec(arm="independent", steps=10, prompts_per_step=4), SPLITS["train"])
    assert [r["task_id"] for r in indep] == [r["task_id"] for r in rows]
    assert all(json.loads(r["noise"])["mode"] == "independent" for r in indep)
    block = assemble_training_rows(spec(arm="blocking", steps=10, prompts_per_step=4), SPLITS["train"])
    assert len(block) == 40 and Counter(r["condition"] for r in block) == {"C2:clean": 20, "C2:noisy": 20}
    clean = assemble_training_rows(spec(arm="clean", p=0.0, steps=10, prompts_per_step=4), SPLITS["train"])
    assert all(NoiseConfig.from_dict(json.loads(r["noise"])).is_clean for r in clean)


def test_eval_sets_periodic_and_final():
    s = spec()
    periodic = assemble_eval_sets(s, SPLITS["heldout"], final=False)
    assert set(periodic) == {"clean", "noisy", "challenge"}
    assert len(periodic["clean"]) == 200 and len(periodic["noisy"]) == 200 and len(periodic["challenge"]) == 100
    assert all(json.loads(r["noise"])["challenge"] == "writes_once" for r in periodic["challenge"])
    seeds_by_task = Counter(r["task_id"] for r in periodic["noisy"])
    assert set(seeds_by_task.values()) == {2}
    assert len({r["schedule_seed"] for r in periodic["noisy"]}) == 200
    assert periodic["noisy"][0]["schedule_seed"] == eval_schedule_seed(periodic["noisy"][0]["task_id"], 1)
    assert periodic["noisy"][1]["schedule_seed"] == eval_schedule_seed(periodic["noisy"][1]["task_id"], 2)
    assert all(json.loads(r["noise"])["mode"] == "paired" for r in periodic["noisy"])
    final = assemble_eval_sets(s, SPLITS["test"], final=True)
    assert set(final) == {"clean", "noisy_p010", "noisy_p025", "heldout_types", "challenge"}
    assert all(len(rows) == 800 for name, rows in final.items() if name != "challenge") and len(final["challenge"]) == 200
    assert all(r["task_id"].startswith("test-") for r in final["clean"])
    assert json.loads(final["noisy_p010"][0]["noise"])["p"] == 0.1 and json.loads(final["noisy_p025"][0]["noise"])["p"] == 0.25
    assert json.loads(final["heldout_types"][0]["noise"])["weights"] == {"timeout_after_commit": 0.5, "field_dropout": 0.5}
    clean_arm = assemble_eval_sets(spec(arm="clean", p=0.0), SPLITS["heldout"], final=False)
    assert json.loads(clean_arm["noisy"][0]["noise"])["p"] == 0.25
    for other in (spec(arm="independent"), spec(arm="clean", p=0.0), spec(arm="paired", p=0.0, q=0.1)):
        same = assemble_eval_sets(other, SPLITS["test"], final=True)
        for name in ("clean", "noisy_p010", "noisy_p025", "heldout_types", "challenge"):
            assert [r["schedule_seed"] for r in same[name]] == [r["schedule_seed"] for r in final[name]]
    outcome = assemble_eval_sets(spec(arm="paired", p=0.0, q=0.1), SPLITS["test"], final=True)
    assert "at_training" in outcome and json.loads(outcome["at_training"][0]["noise"])["q"] == 0.1
    assert "at_training" not in final and "at_training" not in assemble_eval_sets(spec(arm="clean", p=0.0), SPLITS["test"], final=True)


def test_recoverable_only_mixture_flows_into_training_diagnostic_and_final_sets():
    weights = {"transient": 0.5, "rate_limit": 0.5 / 3, "stale": 0.5 / 3, "truncate": 0.5 / 3}
    s = spec(condition="C2r", fault_weights=weights)
    assert s.training_noise().weights == weights and s.eval_noise().weights == weights
    assert "outage" not in json.loads(assemble_training_rows(s, SPLITS["train"])[0]["noise"])["weights"]
    assert json.loads(assemble_diagnostic_rows(s, SPLITS["diagnostic"])[0]["noise"])["weights"] == weights
    final = assemble_eval_sets(s, SPLITS["test"], final=True)
    assert set(final) == {"clean", "noisy_p010", "noisy_p025", "heldout_types", "challenge", "at_training"}
    assert json.loads(final["at_training"][0]["noise"])["weights"] == weights
    assert json.loads(final["noisy_p025"][0]["noise"])["weights"]["outage"] == 0.1
    assert RunSpec.from_dict(json.loads(json.dumps(s.to_dict()))) == s
    for bad in ({"outage": -1.0}, {"transient": 0.0}, {"field_dropout": 1.0}):
        with pytest.raises(ValueError, match="fault_weights"):
            spec(fault_weights=bad)


def test_eval_schedules_flake_at_rate_q_and_differ_across_tasks():
    from pairedrl.noise.schedule import NoiseSchedule

    rows = assemble_eval_sets(spec(arm="paired", p=0.0, q=0.1), SPLITS["test"], final=True)["at_training"]
    cfg = NoiseConfig.outcome(0.1, "paired")
    flips = sum(NoiseSchedule(r["schedule_seed"], cfg).outcome_flip() for r in rows)
    assert 0.07 <= flips / len(rows) <= 0.13
    assert eval_schedule_seed("test-00000", 1) != eval_schedule_seed("test-00001", 1) != eval_schedule_seed("test-00000", 2)


def test_diagnostic_rows_pair_samples_by_schedule():
    rows = assemble_diagnostic_rows(spec(), SPLITS["diagnostic"])
    assert len(rows) == 16 * 8 * 8
    seeds = Counter((r["task_id"], r["schedule_seed"]) for r in rows)
    assert len(seeds) == 128 and set(seeds.values()) == {8}
    assert all(r["condition"] == "diag" and json.loads(r["noise"])["mode"] == "paired" for r in rows)
    q_only = assemble_diagnostic_rows(spec(p=0.0, q=0.1), SPLITS["diagnostic"])
    assert json.loads(q_only[0]["noise"])["p"] == 0.25


def make_record(**kw):
    base = {
        "task_id": "heldout-00000", "phase": "train", "condition": "C2", "true_success": True,
        "observed_reward": 1.0, "flipped": False, "calls": 5, "budget": 13, "budget_exceeded": False,
        "finished": True, "exposed": False, "faults": {}, "schedule_seed": 1,
    }
    base.update(kw)
    return base


def test_summarize_episodes():
    records = [
        make_record(),
        make_record(true_success=False, observed_reward=0.0, exposed=True, calls=13, budget_exceeded=True, finished=False),
        make_record(exposed=True, observed_reward=0.0, flipped=True),
        make_record(phase="eval_noisy:step20", condition="eval:noisy"),
    ]
    out = summarize_episodes(records)
    assert set(out) == {"train|C2", "eval_noisy:step20|eval:noisy"}
    train = out["train|C2"]
    assert train["episodes"] == 3 and train["true_success"] == pytest.approx(2 / 3)
    assert train["observed_reward"] == pytest.approx(1 / 3) and train["flipped_frac"] == pytest.approx(1 / 3)
    assert train["exposed_frac"] == pytest.approx(2 / 3) and train["recovery_success"] == pytest.approx(0.5)
    assert train["budget_exceeded_frac"] == pytest.approx(1 / 3) and train["finished_frac"] == pytest.approx(2 / 3)
    assert out["eval_noisy:step20|eval:noisy"]["recovery_success"] is None


def test_luck_share_tables_keep_phases_apart_and_check_cardinality():
    records = []
    for phase, flip in (("diag:step0", 0), ("diag:step50", 1)):
        for task in ("diagnostic-00000", "diagnostic-00001"):
            for k, seed in enumerate((11, 22, 33)):
                for m in range(4):
                    records.append(make_record(task_id=task, phase=phase, condition="diag", schedule_seed=seed, true_success=(k + m + flip) % 2 == 0))
    records.append(make_record(task_id="diagnostic-00000", condition="eval:noisy", schedule_seed=99))
    by_phase = luck_share_tables_by_phase(records, schedules=3, samples=4)
    assert set(by_phase) == {"diag:step0", "diag:step50"}
    assert all(len(t) == 2 and all(len(tb) == 3 and all(len(row) == 4 for row in tb) for tb in t) for t in by_phase.values())
    assert luck_share_over_tasks(by_phase["diag:step0"])["tasks"] == 2
    assert luck_share_tables(records, phase="diag:step50") == by_phase["diag:step50"]
    with pytest.raises(ValueError):
        luck_share_tables(records)
    with pytest.raises(ValueError):
        luck_share_tables_by_phase(records, schedules=4, samples=4)
    with pytest.raises(ValueError):
        luck_share_tables_by_phase(records + [make_record(task_id="diagnostic-00000", phase="diag:step0", condition="diag", schedule_seed=11)], schedules=3, samples=4)
    only_first = [r for r in records if r["phase"] != "diag:step50"]
    assert len(luck_share_tables(only_first)) == 2


def test_read_episode_log(tmp_path):
    path = tmp_path / "episodes.jsonl"
    assert read_episode_log(path) == []
    path.write_text(json.dumps(make_record()) + "\n\n" + json.dumps(make_record(calls=2)) + "\n")
    assert [r["calls"] for r in read_episode_log(path)] == [5, 2]


def episode(task_id, true, observed, **kw):
    base = {"task_id": task_id, "true_success": true, "observed_reward": observed, "flipped": true and not observed,
            "resolved_seed": 7, "slot": 0, "episode_index": 0, "exposed": False, "calls": 5}
    base.update(kw)
    return base


def test_training_group_records_follow_the_trainer_batch():
    rows = [{"task_id": "a", "condition": "train:independent", "schedule_seed": 3}] * 4
    rows += [{"task_id": "b", "condition": "train:independent", "schedule_seed": 4}] * 4
    episodes = [episode("a", True, 1.0, slot=i) for i in range(4)]
    episodes[1] = episode("a", True, 0.0, slot=1)
    episodes += [episode("b", False, 0.0, slot=4 + i) for i in range(4)]
    rewards = [e["observed_reward"] for e in episodes]
    advantages = [0.35, -2.47, 0.35, 0.35, 0.0, 0.0, 0.0, 0.0]
    records = training_group_records(12, 2, rows, episodes, rewards, advantages, 4)
    assert [r["task_id"] for r in records] == ["a", "b"] and [r["group"] for r in records] == [0, 1]
    first, second = records
    assert first["step"] == 12 and first["attempt"] == 2 and first["schedule_seed"] == 3
    assert first["spurious"] and not first["zero_variance"] and first["advantages"] == advantages[:4]
    assert first["slots"] == [0, 1, 2, 3] and first["flipped"] == [False, True, False, False]
    assert first["max_abs_advantage"] == pytest.approx(2.47)
    assert second["zero_variance"] and not second["spurious"] and second["max_abs_advantage"] == 0.0
    summary = summarize_groups(records)["train:independent"]
    assert summary["groups"] == 2 and summary["steps"] == [12] and summary["zero_variance_frac"] == 0.5
    assert summary["all_correct_groups"] == 1 and summary["spurious_among_all_correct"] == 1.0


def test_training_group_records_refuse_broken_linkage():
    rows = [{"task_id": "a", "condition": "c", "schedule_seed": 1}] * 2
    good = [episode("a", True, 1.0), episode("a", False, 0.0)]
    training_group_records(0, 1, rows, good, [1.0, 0.0], [1.0, -1.0], 2)
    with pytest.raises(ValueError, match="does not match"):
        training_group_records(0, 1, rows, [episode("b", True, 1.0), good[1]], [1.0, 0.0], [1.0, -1.0], 2)
    with pytest.raises(ValueError, match="does not match"):
        training_group_records(0, 1, rows, good, [0.0, 0.0], [1.0, -1.0], 2)
    with pytest.raises(ValueError, match="does not match"):
        training_group_records(0, 1, rows, [None, good[1]], [1.0, 0.0], [1.0, -1.0], 2)
    with pytest.raises(ValueError, match="group register"):
        training_group_records(0, 1, rows, good, [1.0], [1.0, -1.0], 2)
    with pytest.raises(ValueError, match="group size"):
        training_group_records(0, 1, rows, good, [1.0, 0.0], [1.0, -1.0], 3)


def test_expected_eval_sizes_match_assembled_sets_and_periodic_steps():
    for s in (spec(eval_tasks=64, final_eval_tasks=200), spec(condition="C4", p=0.0, q=0.10, eval_tasks=64, final_eval_tasks=200),
              spec(condition="C0", arm="clean", p=0.0, eval_tasks=64, final_eval_tasks=200)):
        sizes = expected_eval_sizes(s)
        periodic = assemble_eval_sets(s, SPLITS["heldout"], final=False)
        final = assemble_eval_sets(s, SPLITS["test"], final=True)
        assert {k: len(v) for k, v in periodic.items()} == sizes["periodic"]
        assert {k: len(v) for k, v in final.items()} == sizes["final"]
    assert sum(expected_eval_sizes(spec(eval_tasks=64)).get("periodic").values()) == 320
    c2, c4 = expected_eval_sizes(spec(final_eval_tasks=200)), expected_eval_sizes(spec(condition="C4", p=0.0, q=0.10, final_eval_tasks=200))
    assert sum(c2["final"].values()) == 3400 and sum(c4["final"].values()) == 4200 and "at_training" in c4["final"]
    assert periodic_steps(spec(steps=100, eval_every=20)) == [0, 20, 40, 60, 80, 100]
    assert periodic_steps(spec(steps=5, eval_every=2)) == [0, 2, 4, 5]
    assert periodic_steps(spec(eval_only=True)) == [] and periodic_steps(spec(train_only=True)) == []

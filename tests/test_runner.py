import json
from collections import Counter

import pytest

from pairedrl.analysis.diagnostics import luck_share_over_tasks
from pairedrl.noise.config import NoiseConfig
from pairedrl.train.runner import (
    RunSpec,
    assemble_diagnostic_rows,
    assemble_eval_sets,
    assemble_training_rows,
    load_task_splits,
    luck_share_tables,
    read_episode_log,
    summarize_episodes,
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
    assert set(periodic) == {"clean", "noisy"}
    assert len(periodic["clean"]) == 200 and len(periodic["noisy"]) == 200
    assert {r["schedule_seed"] for r in periodic["noisy"]} == {1, 2}
    assert all(json.loads(r["noise"])["mode"] == "paired" for r in periodic["noisy"])
    final = assemble_eval_sets(s, SPLITS["heldout"], final=True)
    assert set(final) == {"clean", "noisy", "heldout_types"}
    assert all(len(rows) == 800 for rows in final.values())
    assert json.loads(final["heldout_types"][0]["noise"])["weights"] == {"timeout_after_commit": 0.5, "field_dropout": 0.5}
    clean_arm = assemble_eval_sets(spec(arm="clean", p=0.0), SPLITS["heldout"], final=False)
    assert json.loads(clean_arm["noisy"][0]["noise"])["p"] == 0.25
    same_seeds_other_arm = assemble_eval_sets(spec(arm="independent"), SPLITS["heldout"], final=True)
    assert [r["schedule_seed"] for r in same_seeds_other_arm["noisy"]] == [r["schedule_seed"] for r in final["noisy"]]


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


def test_luck_share_tables_from_records():
    records = []
    for task in ("diagnostic-00000", "diagnostic-00001"):
        for k, seed in enumerate((11, 22, 33)):
            for m in range(4):
                records.append(make_record(task_id=task, condition="diag", schedule_seed=seed, true_success=(k + m) % 2 == 0))
    records.append(make_record(task_id="diagnostic-00000", condition="eval:noisy", schedule_seed=99))
    tables = luck_share_tables(records)
    assert len(tables) == 2 and all(len(t) == 3 and all(len(row) == 4 for row in t) for t in tables)
    assert luck_share_over_tasks(tables)["tasks"] == 2


def test_read_episode_log(tmp_path):
    path = tmp_path / "episodes.jsonl"
    assert read_episode_log(path) == []
    path.write_text(json.dumps(make_record()) + "\n\n" + json.dumps(make_record(calls=2)) + "\n")
    assert [r["calls"] for r in read_episode_log(path)] == [5, 2]

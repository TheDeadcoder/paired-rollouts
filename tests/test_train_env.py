import inspect
import json
import typing

import pytest

from pairedrl.env.backoffice import TOOL_NAMES, ToolAPI, read_jsonl
from pairedrl.noise.config import NoiseConfig
from pairedrl.noise.oracle import run_oracle
from pairedrl.train.dataset import (
    SYSTEM_PROMPT,
    build_blocking_rows,
    build_diagnostic_rows,
    build_eval_rows,
    build_training_rows,
    make_row,
)
from pairedrl.train.env_adapter import BackOfficeEnv, budget_for, make_env_factory

TASK_PATHS = ["data/tasks/heldout.jsonl"]
TASKS = read_jsonl(TASK_PATHS[0])


def row_for(task, config, seed=1234, condition="test"):
    return make_row(task, seed, config, condition)


def test_env_exposes_exactly_the_tools_with_matching_schemas():
    env = BackOfficeEnv(TASK_PATHS)
    public = {n for n, _ in inspect.getmembers(env, predicate=inspect.ismethod) if not n.startswith("_")}
    assert public == set(TOOL_NAMES) | {"reset", "get_reward"}
    for name in TOOL_NAMES:
        assert inspect.signature(getattr(env, name)) == inspect.signature(getattr(ToolAPI(None), name))
        assert typing.get_type_hints(getattr(env, name)) == typing.get_type_hints(getattr(ToolAPI, name))
        assert inspect.getdoc(getattr(env, name)) == inspect.getdoc(getattr(ToolAPI, name))
        assert getattr(env, name).__name__ == name
    assert inspect.signature(env.reset).parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD


def test_reset_receives_row_and_builds_episode():
    env = BackOfficeEnv(TASK_PATHS)
    task = TASKS[0]
    row = row_for(task, NoiseConfig.clean())
    assert env.reset(**row) is None
    assert env.task.task_id == task.task_id and env.budget == budget_for(task)
    assert env.schedule is None and env.calls == 0
    out = json.loads(env.get_customer(customer_id=task.customer_id))
    assert out["customer_id"] == task.customer_id and env.calls == 1
    assert json.loads(env.search_customers("x"))["error"]["code"] == "INVALID_ARGUMENT"
    assert json.loads(BackOfficeEnv(TASK_PATHS).get_order(order_id="O0001"))["error"]["code"] == "NO_EPISODE"


def test_budget_is_enforced_and_finish_still_works_before_exhaustion():
    env = BackOfficeEnv(TASK_PATHS)
    task = TASKS[0]
    env.reset(**row_for(task, NoiseConfig.clean(), condition="c"))
    for _ in range(env.budget):
        assert "error" not in json.loads(env.get_customer(customer_id=task.customer_id))
    err = json.loads(env.get_customer(customer_id=task.customer_id))
    assert err["error"]["code"] == "BUDGET_EXCEEDED" and env.budget_exceeded
    assert json.loads(env.finish(summary="x"))["error"]["code"] == "EPISODE_FINISHED"
    env.reset(**row_for(task, NoiseConfig.clean()))
    assert json.loads(env.finish(summary="done"))["status"] == "finished"
    assert json.loads(env.get_order(order_id="O0001"))["error"]["code"] == "EPISODE_FINISHED"
    assert env.calls == 0


def test_reward_from_oracle_plan_and_from_nothing():
    env = BackOfficeEnv(TASK_PATHS)
    task = TASKS[1]
    env.reset(**row_for(task, NoiseConfig.clean()))
    for call in task.oracle_plan:
        assert "error" not in json.loads(getattr(env, call.tool)(**call.args))
    env.finish(summary="done")
    assert env.get_reward() == 1.0
    rec = env.last_episode
    assert rec["true_success"] and rec["observed_reward"] == 1.0 and not rec["flipped"]
    assert rec["finished"] and rec["calls"] == len(task.oracle_plan) and not rec["exposed"]
    assert [t["tool"] for t in rec["tool_sequence"]] == [c.tool for c in task.oracle_plan] + ["finish"]
    env.reset(**row_for(task, NoiseConfig.clean()))
    assert env.get_reward() == 0.0 and not env.last_episode["changed_from_initial"]


def test_paired_rows_share_fates_and_independent_rows_do_not():
    cfg_paired = NoiseConfig.transition(0.5, "paired")
    cfg_indep = NoiseConfig.transition(0.5, "independent")
    task = TASKS[2]
    logs = {}
    for label, cfg in (("paired", cfg_paired), ("independent", cfg_indep)):
        seqs = []
        for _ in range(2):
            env = BackOfficeEnv(TASK_PATHS)
            env.reset(**row_for(task, cfg, seed=99))
            for _ in range(12):
                env.get_order(order_id="O0001")
                env.list_orders(customer_id=task.customer_id)
            seqs.append([(f["tool"], f["index"], f["kind"]) for f in env.api.fault_log])
        logs[label] = seqs
    assert logs["paired"][0] == logs["paired"][1] and logs["paired"][0]
    assert logs["independent"][0] != logs["independent"][1]


def test_independent_seeds_change_across_episodes_of_one_instance():
    cfg = NoiseConfig.transition(0.5, "independent")
    env = BackOfficeEnv(TASK_PATHS)
    env.reset(**row_for(TASKS[3], cfg, seed=5))
    first = env.schedule.seed
    env.reset(**row_for(TASKS[3], cfg, seed=5))
    assert env.schedule.seed != first and env.episode_index == 1
    paired = BackOfficeEnv(TASK_PATHS)
    paired.reset(**row_for(TASKS[3], NoiseConfig.transition(0.5, "paired"), seed=5))
    assert paired.schedule.seed == 5


def test_outcome_noise_flips_observed_reward_but_logs_truth():
    cfg = NoiseConfig.outcome(1.0, "paired")
    env = BackOfficeEnv(TASK_PATHS)
    task = TASKS[4]
    env.reset(**row_for(task, cfg, seed=3))
    for call in task.oracle_plan:
        getattr(env, call.tool)(**call.args)
    assert env.get_reward() == 0.0
    assert env.last_episode["true_success"] and env.last_episode["flipped"]


def test_recovering_oracle_through_env_under_transition_noise(tmp_path):
    cfg = NoiseConfig.transition(0.25, "paired")
    log = tmp_path / "episodes.jsonl"
    factory = make_env_factory(TASK_PATHS, log_path=log)
    env = factory()
    successes = 0
    outage_free = 0
    for i, task in enumerate(TASKS[:40]):
        env.reset(**row_for(task, cfg, seed=i))
        result = run_oracle(task, env.api, recover=True)
        reward = env.get_reward()
        successes += reward
        if not any(k.startswith("outage") for k in result.faults):
            outage_free += 1
            assert reward == 1.0
    assert 20 <= successes <= 40 and outage_free >= 20
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 40 and any(r["exposed"] for r in records)
    assert all("failed_calls" in r for r in records)
    assert all(r["condition"] == "test" and r["mode"] == "paired" for r in records)


def test_training_rows_are_deterministic_and_well_formed():
    cfg = NoiseConfig.transition(0.25, "paired")
    rows = build_training_rows(TASKS, cfg, "C2", n_rows=50, run_seed=7)
    again = build_training_rows(TASKS, cfg, "C2", n_rows=50, run_seed=7)
    assert rows == again and len(rows) == 50
    assert rows != build_training_rows(TASKS, cfg, "C2", n_rows=50, run_seed=8)
    assert len({r["schedule_seed"] for r in rows}) == 50
    assert all(set(r) == {"prompt", "task_id", "schedule_seed", "noise", "condition", "budget"} for r in rows)
    assert all(r["prompt"][0] == {"role": "system", "content": SYSTEM_PROMPT} for r in rows)
    assert all(r["prompt"][1]["role"] == "user" and r["prompt"][1]["content"].startswith("Today is day") for r in rows)
    assert all(NoiseConfig.from_dict(json.loads(r["noise"])) == cfg for r in rows)
    json.dumps(rows)


def test_blocking_rows_pair_clean_and_noisy_groups():
    cfg = NoiseConfig.transition(0.25, "independent")
    rows = build_blocking_rows(TASKS, cfg, "B2", n_rows=20, run_seed=1)
    assert len(rows) == 20
    for clean, noisy in zip(rows[0::2], rows[1::2]):
        assert clean["task_id"] == noisy["task_id"] and clean["schedule_seed"] == noisy["schedule_seed"]
        assert NoiseConfig.from_dict(json.loads(clean["noise"])).is_clean
        assert NoiseConfig.from_dict(json.loads(noisy["noise"])) == cfg
        assert clean["condition"] == "B2:clean" and noisy["condition"] == "B2:noisy"
    with pytest.raises(ValueError):
        build_blocking_rows(TASKS, NoiseConfig.transition(0.25, "paired"), "B2", 4, 1)


def test_eval_and_diagnostic_rows():
    cfg = NoiseConfig.transition(0.25, "paired")
    rows = build_eval_rows(TASKS[:10], cfg, "eval", schedule_seeds=[1, 2, 3, 4])
    assert len(rows) == 40 and [r["schedule_seed"] for r in rows[:4]] == [1, 2, 3, 4]
    diag = build_diagnostic_rows(TASKS[:16], cfg, "diag", n_schedules=8, base_seed=0)
    assert len(diag) == 128 and len({r["schedule_seed"] for r in diag}) == 128
    assert diag == build_diagnostic_rows(TASKS[:16], cfg, "diag", n_schedules=8, base_seed=0)


def test_budget_formula():
    assert budget_for(next(t for t in TASKS if t.n_subgoals == 1)) == 9
    assert budget_for(next(t for t in TASKS if t.n_subgoals == 3)) == 13
    assert all(9 <= budget_for(t) <= 21 for t in TASKS)

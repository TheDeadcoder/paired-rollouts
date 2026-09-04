import json
import random
from collections import Counter

import pytest

from pairedrl.env.backoffice import ToolAPI, World
from pairedrl.env.backoffice.grader import expected_snapshot, grade, normalize
from pairedrl.env.backoffice.tasks import (
    SPLIT_SEED_RANGES,
    TEMPLATES,
    Task,
    describe_order,
    generate_tasks,
    read_jsonl,
    simulate_plan,
    write_jsonl,
)
from pairedrl.env.backoffice.tools import WRITE_TOOLS


@pytest.fixture(scope="module")
def train_tasks():
    return generate_tasks(120, "train", seed=1)


@pytest.fixture(scope="module")
def heldout_tasks():
    return generate_tasks(40, "heldout", seed=1)


def test_generation_is_deterministic():
    a = [t.to_json() for t in generate_tasks(10, "train", seed=3)]
    b = [t.to_json() for t in generate_tasks(10, "train", seed=3)]
    assert a == b
    assert a != [t.to_json() for t in generate_tasks(10, "train", seed=4)]


def test_ids_splits_and_seed_ranges(train_tasks, heldout_tasks):
    assert [t.task_id for t in train_tasks] == [f"train-{i:05d}" for i in range(120)]
    lo, hi = SPLIT_SEED_RANGES["train"]
    assert all(lo <= t.world_seed < hi and t.split == "train" for t in train_tasks)
    lo, hi = SPLIT_SEED_RANGES["heldout"]
    assert all(lo <= t.world_seed < hi and t.split == "heldout" for t in heldout_tasks)
    assert not ({t.world_seed for t in train_tasks} & {t.world_seed for t in heldout_tasks})


def test_every_oracle_plan_executes_cleanly(train_tasks):
    for task in train_tasks:
        _, outcomes = simulate_plan(task.world_seed, task.oracle_plan)
        assert all(o["ok"] for o in outcomes), (task.task_id, outcomes)
        assert task.n_subgoals == len(task.oracle_plan) >= 1
        assert all(c.tool in WRITE_TOOLS for c in task.oracle_plan)


def test_instruction_mentions_today_and_finish(train_tasks):
    for task in train_tasks:
        assert task.instruction.startswith("Today is day 100.")
        assert "then call finish" in task.instruction
        world = World.generate(task.world_seed)
        assert world.customers[task.customer_id].name in task.instruction


def test_template_coverage_and_subgoal_mix(train_tasks):
    counts = Counter(t for task in train_tasks for t in task.templates)
    assert set(counts) == set(TEMPLATES), counts
    n_sub = Counter(len(t.templates) for t in train_tasks)
    assert set(n_sub) == {1, 2, 3}
    assert max(t.n_subgoals for t in train_tasks) >= 4
    assert {t.reference for t in train_tasks} == {"description", "id"}


def test_describe_order_is_unique_per_customer():
    world = World.generate(5)
    for customer_id in list(world.customers)[:10]:
        orders = [o for o in world.orders.values() if o.customer_id == customer_id]
        descriptions = [describe_order(world, o, "description") for o in orders]
        assert len(set(descriptions)) == len(descriptions)


def test_oracle_end_state_passes_and_initial_fails(train_tasks):
    for task in train_tasks[:40]:
        world, _ = simulate_plan(task.world_seed, task.oracle_plan)
        initial = World.generate(task.world_seed).snapshot()
        assert grade(task, world.snapshot(), initial).success
        result = grade(task, initial, initial)
        assert not result.success and result.diffs and not result.changed_from_initial


def test_extra_side_effect_fails(train_tasks):
    task = next(t for t in train_tasks if "open_ticket" not in t.templates)
    world, _ = simulate_plan(task.world_seed, task.oracle_plan)
    api = ToolAPI(world)
    other = next(
        o for o in world.orders.values()
        if o.customer_id != task.customer_id and o.status == "pending"
    )
    assert "error" not in json.loads(api.cancel_order(other.order_id, "oops"))
    result = grade(task, world.snapshot())
    assert not result.success and any(other.order_id in d for d in result.diffs)


def test_free_text_fields_do_not_matter(train_tasks):
    task = next(t for t in train_tasks if t.templates == ["cancel_paid_refund"])
    world = World.generate(task.world_seed)
    api = ToolAPI(world)
    order_id = task.oracle_plan[0].args["order_id"]
    total = task.oracle_plan[1].args["amount_cents"]
    api.cancel_order(order_id, "different wording")
    api.issue_refund(order_id, total, "another reason")
    assert grade(task, world.snapshot()).success
    task = next(t for t in train_tasks if t.templates == ["open_ticket"])
    world = World.generate(task.world_seed)
    api = ToolAPI(world)
    args = dict(task.oracle_plan[0].args)
    args["summary"] = "paraphrased summary"
    api.create_ticket(**args)
    assert grade(task, world.snapshot()).success


def test_wrong_amount_or_partial_completion_fails(train_tasks):
    task = next(t for t in train_tasks if t.templates == ["cancel_paid_refund"])
    world = World.generate(task.world_seed)
    api = ToolAPI(world)
    order_id = task.oracle_plan[0].args["order_id"]
    total = task.oracle_plan[1].args["amount_cents"]
    api.cancel_order(order_id, "customer request")
    assert not grade(task, world.snapshot()).success
    api.issue_refund(order_id, total - 1, "cancelled order")
    result = grade(task, world.snapshot())
    assert not result.success and any("refunds" in d for d in result.diffs)


def test_shipment_grading_ignores_ids_but_not_content(train_tasks):
    task = next(t for t in train_tasks if t.templates == ["ship_reserved"])
    call = task.oracle_plan[-1]
    world = World.generate(task.world_seed)
    api = ToolAPI(world)
    other_carrier = "express" if call.args["carrier"] == "standard" else "standard"
    api.schedule_shipment(call.args["order_id"], other_carrier, call.args["day"])
    assert not grade(task, world.snapshot()).success
    world = World.generate(task.world_seed)
    ToolAPI(world).schedule_shipment(call.args["order_id"], call.args["carrier"], call.args["day"])
    assert grade(task, world.snapshot()).success


def test_normalize_drops_volatile_fields():
    world = World.generate(9)
    a = normalize(world.snapshot())
    world.clock_seconds += 100
    world.commit()
    assert normalize(world.snapshot()) == a
    assert "version" not in a and "clock_seconds" not in a


def test_expected_snapshot_rejects_broken_plan(train_tasks):
    task = train_tasks[0]
    broken = Task.from_dict(task.to_dict())
    broken.oracle_plan = [c for c in broken.oracle_plan] + [
        type(broken.oracle_plan[0])("cancel_order", {"order_id": "O9999", "reason": "x"})
    ]
    with pytest.raises(ValueError):
        expected_snapshot(broken)


def test_jsonl_round_trip(tmp_path, train_tasks):
    path = tmp_path / "tasks.jsonl"
    write_jsonl(train_tasks[:15], path)
    back = read_jsonl(path)
    assert [t.to_json() for t in back] == [t.to_json() for t in train_tasks[:15]]
    assert json.loads(path.read_text().splitlines()[0])["task_id"] == "train-00000"


def test_random_agent_rarely_succeeds(train_tasks):
    rng = random.Random(0)
    successes = 0
    for task in train_tasks[:60]:
        world = World.generate(task.world_seed)
        api = ToolAPI(world)
        order_ids = list(world.orders)
        for _ in range(6):
            tool = rng.choice(sorted(WRITE_TOOLS))
            order_id = rng.choice(order_ids)
            if tool == "update_shipping_address":
                api.update_shipping_address(order_id, "1 X", "Dhaka", "1000")
            elif tool == "cancel_order":
                api.cancel_order(order_id, "r")
            elif tool == "issue_refund":
                api.issue_refund(order_id, rng.randint(1, 5000), "r")
            elif tool == "reserve_stock":
                api.reserve_stock(rng.choice(list(world.products)), 1, order_id)
            elif tool == "schedule_shipment":
                api.schedule_shipment(order_id, "standard", 101)
            else:
                api.create_ticket(world.orders[order_id].customer_id, "", "billing", "low", "x")
        successes += grade(task, world.snapshot()).success
    assert successes <= 1

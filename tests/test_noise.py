import json
from collections import Counter

import pytest

from pairedrl.env.backoffice import ToolAPI, World, read_jsonl
from pairedrl.noise.config import DEFAULT_WEIGHTS, FAULT_TYPES, NoiseConfig
from pairedrl.noise.noisy_api import NoisyToolAPI
from pairedrl.noise.oracle import make_api, observe, run_oracle
from pairedrl.noise.schedule import (
    NoiseSchedule,
    applicable_types,
    fault_key,
    request_key,
    resolve_seed,
)

TASKS = read_jsonl("data/tasks/heldout.jsonl")[:80]


def api_with(seed: int, config: NoiseConfig, world_seed: int = 11) -> NoisyToolAPI:
    return NoisyToolAPI(World.generate(world_seed), NoiseSchedule(seed, config))


def test_config_validation():
    with pytest.raises(ValueError):
        NoiseConfig(p=0.1, mode="sometimes")
    with pytest.raises(ValueError):
        NoiseConfig(p=1.5, mode="paired")
    with pytest.raises(ValueError):
        NoiseConfig(p=0.1, weights={"nope": 1.0}, mode="paired")
    assert NoiseConfig.clean().is_clean
    assert NoiseConfig.transition(0.25, "paired").weights == DEFAULT_WEIGHTS
    cfg = NoiseConfig.outcome(0.1, "independent")
    assert NoiseConfig.from_dict(cfg.to_dict()) == cfg


def test_applicable_types_partition():
    assert applicable_types("wait") == () and applicable_types("finish") == ()
    assert "stale" in applicable_types("get_order") and "truncate" not in applicable_types("get_order")
    assert "truncate" in applicable_types("list_orders")
    assert "timeout_after_commit" in applicable_types("issue_refund")
    assert "timeout_after_commit" not in applicable_types("create_ticket")
    assert set(FAULT_TYPES) >= {t for tool in ("get_order", "list_orders", "issue_refund") for t in applicable_types(tool)}


def test_fate_depends_only_on_tool_event_and_repeat():
    cfg = NoiseConfig.transition(0.5, "paired")
    a = NoiseSchedule(123, cfg)
    b = NoiseSchedule(123, cfg)
    event = fault_key("get_order", {"order_id": "O0001"})
    for k in range(50):
        assert a.fate("get_order", event, k) == b.fate("get_order", event, k)
    other = fault_key("get_order", {"order_id": "O0002"})
    assert [a.fate("get_order", event, k) for k in range(50)] != [a.fate("get_order", other, k) for k in range(50)]
    c = NoiseSchedule(124, cfg)
    assert [a.fate("get_order", event, k) for k in range(50)] != [c.fate("get_order", event, k) for k in range(50)]
    assert request_key({"amount_cents": 1500, "order_id": "O0001"}) == request_key({"order_id": "O0001", "amount_cents": "1500"})


def test_fault_key_ignores_free_text_and_argument_form():
    assert fault_key("cancel_order", {"order_id": "O0073", "reason": "customer request"}) == fault_key(
        "cancel_order", {"order_id": "O0073", "reason": "please cancel"}
    )
    assert fault_key("issue_refund", {"order_id": "O0073", "amount_cents": 100, "reason": "a"}) == fault_key(
        "issue_refund", {"order_id": "O0073", "amount_cents": "250", "reason": "b"}
    )
    assert fault_key("create_ticket", {"customer_id": "C0001", "order_id": "O0001", "summary": "x", "category": "billing", "priority": "low"}) == fault_key(
        "create_ticket", {"customer_id": "C0001", "order_id": "", "summary": "y", "category": "other", "priority": "high"}
    )
    assert fault_key("search_customers", {"query": "Mateo"}) == fault_key("search_customers", {"query": "khan", "offset": 5})
    assert fault_key("reserve_stock", {"order_id": "O0001", "sku": "SKU-0001", "quantity": 2}) == fault_key(
        "reserve_stock", {"quantity": "3", "sku": "SKU-0001", "order_id": "O0001"}
    )
    assert fault_key("reserve_stock", {"order_id": "O0001", "sku": "SKU-0001", "quantity": 2}) != fault_key(
        "reserve_stock", {"order_id": "O0001", "sku": "SKU-0002", "quantity": 2}
    )
    assert fault_key("update_shipping_address", {"order_id": "O0001", "postal_code": 1200}) == fault_key(
        "update_shipping_address", {"order_id": "O0001", "postal_code": "1200"}
    )


def test_rewording_cannot_dodge_an_outage_and_ordering_does_not_change_fates():
    cfg = NoiseConfig(p=1.0, weights={"outage": 1.0}, mode="paired")
    api = api_with(5, cfg)
    order = pending_order(api)
    for reason in ("customer request", "please cancel", "duplicate order"):
        assert json.loads(api.cancel_order(order.order_id, reason))["error"]["code"] == "SERVICE_UNAVAILABLE"
    assert api.world.orders[order.order_id].status == "pending"
    cfg = NoiseConfig(p=0.5, weights={"transient": 0.6, "stale": 0.4}, mode="paired")
    a, b = api_with(7, cfg), api_with(7, cfg)
    for _ in range(8):
        a.get_order("O0001")
    for _ in range(8):
        b.get_customer("C0001")
        b.search_customers("anything")
        b.get_order("O0002")
        b.get_order("O0001")
    fa = [(c.get("fault"), c.get("repeat")) for c in a.call_log if c["tool"] == "get_order"]
    fb = [(c.get("fault"), c.get("repeat")) for c in b.call_log if c["tool"] == "get_order" and c["args"] == {"order_id": "O0001"}]
    assert fa == fb and any(f for f, _ in fa)


def test_resolve_seed_modes():
    paired = NoiseConfig.transition(0.25, "paired")
    indep = NoiseConfig.transition(0.25, "independent")
    assert resolve_seed(paired, 777, 0, 0) == resolve_seed(paired, 777, 5, 3) == 777
    seeds = {resolve_seed(indep, 777, slot, ep) for slot in range(8) for ep in range(3)}
    assert len(seeds) == 24 and 777 not in seeds
    assert resolve_seed(indep, 777, 2, 1) == resolve_seed(indep, 777, 2, 1)


def test_fault_rate_and_type_mix():
    cfg = NoiseConfig.transition(0.25, "paired")
    kinds = Counter()
    n = 20000
    req = request_key({"order_id": "O0001"})
    for seed in range(200):
        s = NoiseSchedule(seed, cfg)
        for k in range(100):
            kinds[s.fate("get_order", req, k).kind] += 1
    faulted = 1 - kinds[None] / n
    assert abs(faulted - 0.25) < 0.02
    applicable = applicable_types("get_order")
    total_w = sum(DEFAULT_WEIGHTS.get(t, 0.0) for t in applicable)
    for kind in ("transient", "rate_limit", "outage", "stale"):
        expected = 0.25 * DEFAULT_WEIGHTS[kind] / total_w
        assert abs(kinds[kind] / n - expected) < 0.02, kind
    assert kinds["field_dropout"] == 0 and kinds["truncate"] == 0
    clean = NoiseSchedule(1, NoiseConfig.clean())
    assert all(clean.fate("get_order", req, k).kind is None for k in range(100))
    assert NoiseSchedule(1, cfg).fate("wait", request_key({"seconds": 5}), 0).kind is None


def test_outcome_flip_rate():
    cfg = NoiseConfig.outcome(0.1, "paired")
    flips = sum(NoiseSchedule(seed, cfg).outcome_flip() for seed in range(5000))
    assert abs(flips / 5000 - 0.1) < 0.015
    assert not NoiseSchedule(3, NoiseConfig.transition(0.5, "paired")).outcome_flip()
    assert observe(True, NoiseSchedule(3, NoiseConfig.clean())) == (1.0, False)
    assert observe(False, NoiseSchedule(3, cfg)) == (0.0, False)


def pending_order(api):
    return next(o for o in sorted(api.world.orders.values(), key=lambda o: o.order_id) if o.status == "pending")


def test_transient_fault_does_not_apply_write():
    cfg = NoiseConfig(p=1.0, weights={"transient": 1.0}, mode="paired")
    api = api_with(1, cfg)
    order = pending_order(api)
    out = json.loads(api.cancel_order(order.order_id, "x"))
    assert out["error"]["code"] == "SERVICE_UNAVAILABLE" and order.status == "pending"
    assert api.exposed and api.exposure_counts() == {"transient": 1}
    assert api.call_log[-1]["fault"] == "transient" and api.world.version == 0


def test_rate_limit_blocks_until_wait():
    cfg = NoiseConfig(p=1.0, weights={"rate_limit": 1.0}, mode="paired")
    api = api_with(1, cfg)
    out = json.loads(api.get_customer("C0001"))
    assert out["error"]["code"] == "RATE_LIMITED"
    retry = out["error"]["retry_after_seconds"]
    again = json.loads(api.get_customer("C0001"))
    assert again["error"]["code"] == "RATE_LIMITED" and again["error"]["retry_after_seconds"] == retry
    api.wait(retry)
    cfg2 = NoiseConfig.clean()
    fresh = NoisyToolAPI(api.world, None)
    assert "error" not in json.loads(fresh.get_customer("C0001"))
    assert cfg2.is_clean
    assert api.exposure_counts() == {"rate_limit": 1, "rate_limit_ongoing": 1}


def test_stale_read_shows_pre_write_state_then_recovers():
    cfg = NoiseConfig(p=1.0, weights={"stale": 1.0}, mode="paired")
    api = api_with(1, cfg)
    order = pending_order(api)
    clean = ToolAPI(api.world)
    assert "error" not in json.loads(clean.cancel_order(order.order_id, "x"))
    stale = json.loads(api.get_order(order.order_id))
    assert stale["status"] == "pending"
    assert api.exposure_counts() == {"stale": 1}
    api.schedule = NoiseSchedule(1, NoiseConfig.clean())
    assert json.loads(api.get_order(order.order_id))["status"] == "cancelled"


def test_stale_read_before_any_write_is_fresh():
    cfg = NoiseConfig(p=1.0, weights={"stale": 1.0}, mode="paired")
    api = api_with(1, cfg)
    assert "error" not in json.loads(api.get_order("O0001")) and not api.exposed


def test_truncation_shortens_pages_and_sets_next_offset():
    cfg = NoiseConfig(p=1.0, weights={"truncate": 1.0}, mode="paired")
    api = api_with(1, cfg)
    page = json.loads(api.search_customers("a"))
    assert page["truncated"] is True and len(page["results"]) == 2 and page["next_offset"] == 2
    page2 = json.loads(api.search_customers("a", offset=page["next_offset"]))
    assert page2["offset"] == 2 and len(page2["results"]) == 2
    assert "error" not in json.loads(api.get_order("O0001"))
    assert api.exposure_counts() == {"truncate": 2}


def test_timeout_after_commit_applies_the_write():
    cfg = NoiseConfig(p=1.0, weights={"timeout_after_commit": 1.0}, mode="paired")
    api = api_with(1, cfg)
    order = pending_order(api)
    out = json.loads(api.cancel_order(order.order_id, "x"))
    assert out["error"]["code"] == "TIMEOUT" and order.status == "cancelled"
    assert api.exposure_counts() == {"timeout_after_commit": 1}
    ticket = json.loads(api.create_ticket(order.customer_id, "", "billing", "low", "s"))
    assert "error" not in ticket


def test_field_dropout_removes_one_key():
    cfg = NoiseConfig(p=1.0, weights={"field_dropout": 1.0}, mode="paired")
    api = api_with(1, cfg)
    full = json.loads(ToolAPI(api.world).get_order("O0001"))
    dropped = json.loads(api.get_order("O0001"))
    assert len(dropped) == len(full) - 1 and set(dropped) < set(full)
    assert api.exposure_counts() == {"field_dropout": 1}


def test_control_tools_and_clean_mode_are_never_faulted():
    cfg = NoiseConfig(p=1.0, weights={"transient": 1.0}, mode="paired")
    api = api_with(1, cfg)
    assert "error" not in json.loads(api.wait(5))
    api2 = NoisyToolAPI(World.generate(11), None)
    for _ in range(20):
        assert "error" not in json.loads(api2.get_order("O0001"))
    assert not api2.exposed


def test_paired_apis_share_fates_and_independent_do_not():
    cfg = NoiseConfig(p=0.5, weights={"transient": 0.6, "stale": 0.4}, mode="paired")
    a, b = api_with(42, cfg), api_with(42, cfg)
    c = api_with(resolve_seed(NoiseConfig.transition(0.5, "independent"), 42, 1, 0), cfg)
    logs = []
    for api in (a, b, c):
        for _ in range(15):
            api.get_order("O0001")
            api.search_customers("a")
        logs.append([(f["tool"], f["index"], f["kind"]) for f in api.fault_log])
    assert logs[0] == logs[1] and logs[0] and logs[0] != logs[2]


def test_request_key_makes_string_and_number_arguments_the_same_request():
    cfg = NoiseConfig(p=0.5, weights={"transient": 1.0}, mode="paired")
    a, b = api_with(9, cfg), api_with(9, cfg)
    ka = [json.loads(a.wait(5)) and json.loads(a.get_order("O0001")).get("error", {}).get("code") for _ in range(12)]
    kb = [json.loads(b.wait(5)) and json.loads(b.get_order("O0001")).get("error", {}).get("code") for _ in range(12)]
    assert ka == kb and "SERVICE_UNAVAILABLE" in ka


def test_outage_persists_for_the_request_and_only_that_request():
    cfg = NoiseConfig(p=1.0, weights={"outage": 1.0}, mode="paired")
    api = api_with(3, cfg)
    order = pending_order(api)
    for _ in range(3):
        assert json.loads(api.cancel_order(order.order_id, "customer request"))["error"]["code"] == "SERVICE_UNAVAILABLE"
    assert api.world.orders[order.order_id].status == "pending"
    counts = api.exposure_counts()
    assert counts["outage"] == 1 and counts["outage_ongoing"] == 2
    api2 = api_with(3, NoiseConfig(p=0.0, mode="paired"))
    assert "error" not in json.loads(api2.cancel_order(order.order_id, "customer request"))


def test_naive_oracle_suffers_and_recovering_oracle_reaches_the_ceiling():
    cfg = NoiseConfig.transition(0.25, "paired")
    naive = [run_oracle(t, make_api(t, cfg, seed=i), recover=False, lookups=True) for i, t in enumerate(TASKS)]
    recovering = [run_oracle(t, make_api(t, cfg, seed=i), recover=True, lookups=True) for i, t in enumerate(TASKS)]
    naive_rate = sum(r.success for r in naive) / len(TASKS)
    recovering_rate = sum(r.success for r in recovering) / len(TASKS)
    assert naive_rate < 0.75 and recovering_rate > naive_rate + 0.1 and recovering_rate >= 0.75
    unlucky = [r for r in recovering if not r.success and not any(k.startswith("outage") for k in r.faults)]
    assert len(unlucky) <= 0.05 * len(TASKS)
    assert any(r.exposed for r in recovering)
    assert all(r.observed_reward == float(r.success) and not r.flipped for r in recovering)
    no_outage = NoiseConfig(p=0.25, weights={"transient": 0.6, "rate_limit": 0.2, "stale": 0.2}, mode="paired")
    without = [run_oracle(t, make_api(t, no_outage, seed=i), recover=True, lookups=True).success for i, t in enumerate(TASKS)]
    assert sum(without) / len(TASKS) >= 0.97


def test_oracle_respects_the_call_budget():
    cfg = NoiseConfig.transition(0.25, "paired")
    tight = [run_oracle(t, make_api(t, cfg, seed=i), recover=True, budget=2, lookups=True) for i, t in enumerate(TASKS)]
    assert all(r.calls <= 3 for r in tight) and any(r.budget_exceeded for r in tight)
    assert sum(r.success for r in tight) / len(TASKS) < 0.3


def test_recovering_oracle_handles_heldout_types():
    cfg = NoiseConfig.heldout_types(0.25, "paired")
    results = [run_oracle(t, make_api(t, cfg, seed=i), recover=True, lookups=True) for i, t in enumerate(TASKS)]
    assert all(r.success for r in results)
    kinds = Counter(k for r in results for k in r.faults)
    assert "timeout_after_commit" in kinds


def test_outcome_noise_flips_about_q_of_successes():
    cfg = NoiseConfig.outcome(0.1, "paired")
    results = [run_oracle(TASKS[i % len(TASKS)], make_api(TASKS[i % len(TASKS)], cfg, seed=i), recover=True) for i in range(600)]
    assert all(r.success for r in results)
    observed = sum(r.observed_reward for r in results) / 600
    assert abs(observed - 0.9) < 0.03
    assert all(r.exposed is False for r in results)


def test_challenge_fails_every_write_exactly_once_and_nothing_else():
    from pairedrl.env.backoffice.tools import READ_TOOLS, WRITE_TOOLS

    config = NoiseConfig.challenge_writes()
    assert not config.is_clean and config.challenge == "writes_once"
    for seed in (1, 999):
        schedule = NoiseSchedule(seed, config)
        for tool in sorted(WRITE_TOOLS):
            assert schedule.fate(tool, "{}", 0).kind == "transient" and schedule.fate(tool, "{}", 1).kind is None
        for tool in sorted(READ_TOOLS) + ["wait", "finish"]:
            assert schedule.fate(tool, "{}", 0).kind is None
        assert not schedule.outcome_flip()
    for bad in ({"challenge": "nope"}, {"p": 0.1}, {"q": 0.1}, {"mode": "clean"}):
        with pytest.raises(ValueError, match="challenge"):
            NoiseConfig(**{"mode": "paired", "challenge": "writes_once", **bad})
    assert NoiseConfig.from_dict(config.to_dict()) == config


def test_challenge_is_recoverable_within_budget_and_exposes_everyone():
    from pairedrl.train.env_adapter import budget_for

    tasks = read_jsonl("data/tasks/heldout.jsonl")[:40]
    config = NoiseConfig.challenge_writes()
    recovering = [run_oracle(t, make_api(t, config, seed=1), recover=True, budget=budget_for(t), lookups=True) for t in tasks]
    naive = [run_oracle(t, make_api(t, config, seed=1), recover=False, budget=budget_for(t), lookups=True) for t in tasks]
    assert all(r.success and r.exposed and not r.budget_exceeded for r in recovering)
    assert all(r.exposed and not r.success for r in naive)

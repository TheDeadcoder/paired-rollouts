"""Scripted oracles that execute a task's plan through a (possibly noisy) tool API."""

import json
from dataclasses import dataclass, field

from pairedrl.env.backoffice.grader import grade
from pairedrl.env.backoffice.tasks import Task, ToolCall
from pairedrl.env.backoffice.tools import ToolAPI
from pairedrl.env.backoffice.world import World
from pairedrl.noise.config import NoiseConfig
from pairedrl.noise.noisy_api import NoisyToolAPI
from pairedrl.noise.schedule import NoiseSchedule

MAX_ATTEMPTS = 4
MAX_VERIFY_READS = 2


class BudgetExhausted(Exception):
    pass


@dataclass
class OracleResult:
    success: bool
    observed_reward: float
    flipped: bool
    calls: int
    faults: dict[str, int] = field(default_factory=dict)
    exposed: bool = False
    budget_exceeded: bool = False


def observe(true_success: bool, schedule: NoiseSchedule | None) -> tuple[float, bool]:
    flipped = bool(schedule is not None and true_success and schedule.outcome_flip())
    return (0.0 if flipped else float(true_success)), flipped


def make_api(task: Task, config: NoiseConfig, seed: int) -> NoisyToolAPI:
    world = World.generate(task.world_seed)
    schedule = None if config.is_clean else NoiseSchedule(seed, config)
    return NoisyToolAPI(world, schedule)


def _applied(read, call: ToolCall, before: dict) -> bool | None:
    """Check whether a write took effect, using fresh reads through `read(order_id)`; None means undecidable."""
    args = call.args
    if call.tool == "create_ticket":
        return None
    for _ in range(MAX_VERIFY_READS):
        rec = read(args["order_id"])
        if "error" in rec:
            continue
        if call.tool == "update_shipping_address":
            addr = rec.get("shipping_address")
            if addr is None:
                continue
            return (addr["street"], addr["city"], addr["postal_code"]) == (
                args["street"], args["city"], args["postal_code"]
            )
        if call.tool == "cancel_order":
            if "status" not in rec:
                continue
            return rec["status"] == "cancelled"
        if call.tool == "issue_refund":
            if "refunded_cents" not in rec:
                continue
            if rec["refunded_cents"] >= before["refunded"] + args["amount_cents"]:
                return True
            continue
        if call.tool == "reserve_stock":
            if "reservations" not in rec:
                continue
            reserved = sum(r["quantity"] for r in rec["reservations"] if r["sku"] == args["sku"])
            if reserved >= before["reserved"] + args["quantity"]:
                return True
            continue
        if call.tool == "schedule_shipment":
            if "status" not in rec:
                continue
            return rec["status"] == "shipped"
    return None


def _before(api: ToolAPI, call: ToolCall) -> dict:
    order = api.world.orders.get(call.args.get("order_id", ""))
    if order is None:
        return {"refunded": 0, "reserved": 0}
    return {
        "refunded": sum(r.amount_cents for r in order.refunds),
        "reserved": sum(r.quantity for r in order.reservations if r.sku == call.args.get("sku")),
    }


def _lookup_plan(task: Task, api: ToolAPI) -> list[ToolCall]:
    """The reads a competent agent needs before writing: find the customer, then read each order it will touch."""
    customer = api.world.customers[task.customer_id]
    reads = [ToolCall("search_customers", {"query": customer.name})]
    seen: set[str] = set()
    for call in task.oracle_plan:
        order_id = call.args.get("order_id")
        if order_id and order_id not in seen:
            seen.add(order_id)
            reads.append(ToolCall("get_order", {"order_id": order_id}))
    return reads


def run_oracle(
    task: Task, api: NoisyToolAPI, recover: bool, budget: int | None = None, lookups: bool = False
) -> OracleResult:
    """Execute the task's plan (optionally after realistic lookups) with or without recovery, under a call budget.

    Recovery retries SERVICE_UNAVAILABLE up to MAX_ATTEMPTS times, waits out rate limits, and verifies writes after a
    TIMEOUT. A write that stays unavailable is abandoned, so the recovering oracle's success rate under a schedule is
    the ceiling any policy can reach on that schedule, and its failures are irreducible luck.
    """
    calls = 0
    exceeded = False

    def invoke(tool: str, **args) -> dict:
        nonlocal calls
        if budget is not None and calls >= budget:
            raise BudgetExhausted()
        calls += 1
        return json.loads(getattr(api, tool)(**args))

    plan = (_lookup_plan(task, api) if lookups else []) + list(task.oracle_plan)
    try:
        for call in plan:
            before = _before(api, call)
            for _attempt in range(MAX_ATTEMPTS if recover else 1):
                response = invoke(call.tool, **call.args)
                if "error" not in response:
                    break
                if not recover:
                    break
                code = response["error"]["code"]
                if code == "RATE_LIMITED":
                    invoke("wait", seconds=int(response["error"].get("retry_after_seconds", 30)))
                elif code == "SERVICE_UNAVAILABLE":
                    continue
                elif code == "TIMEOUT":
                    if _applied(lambda oid: invoke("get_order", order_id=oid), call, before):
                        break
                else:
                    break
    except BudgetExhausted:
        exceeded = True
    api.finish("done")
    result = grade(task, api.world.snapshot())
    observed, flipped = observe(result.success, api.schedule)
    return OracleResult(
        success=result.success,
        observed_reward=observed,
        flipped=flipped,
        calls=len(api.call_log),
        faults=api.exposure_counts(),
        exposed=api.exposed,
        budget_exceeded=exceeded,
    )

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

MAX_ATTEMPTS = 8
MAX_VERIFY_READS = 4


@dataclass
class OracleResult:
    success: bool
    observed_reward: float
    flipped: bool
    calls: int
    faults: dict[str, int] = field(default_factory=dict)
    exposed: bool = False


def observe(true_success: bool, schedule: NoiseSchedule | None) -> tuple[float, bool]:
    flipped = bool(schedule is not None and true_success and schedule.outcome_flip())
    return (0.0 if flipped else float(true_success)), flipped


def make_api(task: Task, config: NoiseConfig, seed: int) -> NoisyToolAPI:
    world = World.generate(task.world_seed)
    schedule = None if config.is_clean else NoiseSchedule(seed, config)
    return NoisyToolAPI(world, schedule)


def _applied(api: ToolAPI, call: ToolCall, before: dict) -> bool | None:
    """Check whether a write took effect, using a fresh read; None means undecidable."""
    args = call.args
    if call.tool == "create_ticket":
        return None
    for _ in range(MAX_VERIFY_READS):
        rec = json.loads(api.get_order(args["order_id"]))
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


def run_oracle(task: Task, api: NoisyToolAPI, recover: bool) -> OracleResult:
    for call in task.oracle_plan:
        before = _before(api, call)
        for _attempt in range(MAX_ATTEMPTS if recover else 1):
            response = json.loads(getattr(api, call.tool)(**call.args))
            if "error" not in response:
                break
            if not recover:
                break
            code = response["error"]["code"]
            if code == "RATE_LIMITED":
                api.wait(int(response["error"].get("retry_after_seconds", 30)))
            elif code == "SERVICE_UNAVAILABLE":
                continue
            elif code == "TIMEOUT":
                if _applied(api, call, before):
                    break
            else:
                break
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
    )

"""State-based grading: the final world must equal the world the oracle plan produces."""

import copy
from dataclasses import dataclass, field

from pairedrl.env.backoffice.tasks import Task, simulate_plan

VOLATILE_TOP_LEVEL = ("clock_seconds", "version")


def normalize(snapshot: dict) -> dict:
    snap = copy.deepcopy(snapshot)
    for key in VOLATILE_TOP_LEVEL:
        snap.pop(key, None)
    for order in snap["orders"].values():
        order["refunds"] = sorted(r["amount_cents"] for r in order["refunds"])
        order["reservations"] = sorted((r["sku"], r["quantity"]) for r in order["reservations"])
        order["has_shipment"] = order.pop("shipment_id") is not None
    snap["shipments"] = sorted(
        (s["order_id"], s["carrier"], s["scheduled_day"], s["status"]) for s in snap["shipments"].values()
    )
    snap["tickets"] = sorted(
        (t["customer_id"], t["order_id"] or "", t["category"], t["priority"], t["status"])
        for t in snap["tickets"].values()
    )
    return snap


@dataclass
class GradeResult:
    success: bool
    diffs: list[str] = field(default_factory=list)
    changed_from_initial: bool = True

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "diffs": list(self.diffs),
            "changed_from_initial": self.changed_from_initial,
        }


def expected_snapshot(task: Task) -> dict:
    world, outcomes = simulate_plan(task.world_seed, task.oracle_plan)
    if not all(o["ok"] for o in outcomes):
        raise ValueError(f"oracle plan for {task.task_id} does not execute cleanly")
    return world.snapshot()


def _diff(expected: dict, actual: dict, prefix: str, out: list[str], limit: int = 20) -> None:
    if len(out) >= limit:
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                out.append(f"{prefix}/{key}: unexpected")
            elif key not in actual:
                out.append(f"{prefix}/{key}: missing")
            else:
                _diff(expected[key], actual[key], f"{prefix}/{key}", out, limit)
            if len(out) >= limit:
                return
    elif expected != actual:
        out.append(f"{prefix}: expected {expected!r}, got {actual!r}")


def grade(task: Task, final_snapshot: dict, initial_snapshot: dict | None = None) -> GradeResult:
    expected = normalize(expected_snapshot(task))
    actual = normalize(final_snapshot)
    diffs: list[str] = []
    _diff(expected, actual, "", diffs)
    changed = True
    if initial_snapshot is not None:
        changed = normalize(initial_snapshot) != actual
    return GradeResult(success=not diffs, diffs=diffs, changed_from_initial=changed)

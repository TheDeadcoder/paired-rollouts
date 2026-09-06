"""Rows for TRL: one row is one group. Each row carries the task, a schedule seed and the noise config."""

import json
import random

from pairedrl.env.backoffice.tasks import Task
from pairedrl.noise.config import NoiseConfig
from pairedrl.noise.schedule import derive_seed
from pairedrl.train.env_adapter import budget_for

SYSTEM_PROMPT = """You are the back-office operations agent of an online store. You act only through the tools; there is no other way to read or change records.

Records: customers (C0001 style ids), orders (O0001 style ids, status pending, paid, shipped, delivered or cancelled), products (SKU-0001 style ids), shipments and tickets. Money is in cents. Days are integers; the task states today's day number.

Business rules the tools enforce: a shipping address can change only before shipment; an order can be cancelled only while pending or paid, and a cancelled paid order must be refunded in full; refunds are possible on paid, delivered, or cancelled-after-payment orders, never while in transit, and never above the refundable balance; a shipment can be scheduled only for a paid order whose every item has reserved stock, on a day within seven days of today; a ticket may link only to an order of the same customer.

Tool responses are compact JSON. A failed call returns an object with an "error" field containing "code" and "message". Codes you may see: NOT_FOUND, INVALID_ARGUMENT, INVALID_STATE, RULE_VIOLATION, SERVICE_UNAVAILABLE (a temporary failure; the request was not applied), RATE_LIMITED (too many requests; the response includes retry_after_seconds and every call fails until you wait that long with the wait tool), TIMEOUT (the request may or may not have been applied), BUDGET_EXCEEDED (no more calls are accepted). Reads are eventually consistent: a read issued right after a write may still show the old state. List responses are pages; when next_offset is not null, more results follow at that offset, and a page may be marked truncated.

You have a limited budget of tool calls for each task. A request that fails with SERVICE_UNAVAILABLE more than once may stay unavailable for the whole task; do not spend the budget retrying it forever. Make exactly the requested changes and nothing else. When every requested change is done, call finish with a one-sentence summary."""


def make_prompt(task: Task) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.instruction},
    ]


def make_row(task: Task, schedule_seed: int, config: NoiseConfig, condition: str) -> dict:
    return {
        "prompt": make_prompt(task),
        "task_id": task.task_id,
        "schedule_seed": int(schedule_seed),
        "noise": json.dumps(config.to_dict(), sort_keys=True),
        "condition": condition,
        "budget": budget_for(task),
    }


def build_training_rows(
    tasks: list[Task], config: NoiseConfig, condition: str, n_rows: int, run_seed: int
) -> list[dict]:
    """n_rows rows cycling through the tasks in a seeded order, each with its own schedule seed."""
    order = list(range(len(tasks)))
    random.Random(f"rows-{run_seed}").shuffle(order)
    rows = []
    for i in range(n_rows):
        task = tasks[order[i % len(tasks)]]
        seed = derive_seed("schedule", run_seed, condition, i) % (2**31)
        rows.append(make_row(task, seed, config, condition))
    return rows


def build_blocking_rows(
    tasks: list[Task], noisy_config: NoiseConfig, condition: str, n_rows: int, run_seed: int
) -> list[dict]:
    """Fixed-mixture blocking baseline (inspired by NoisyAgent): every task appears as a clean group and as a
    noisy group, each normalized within itself. The noisy group is independent or paired per `noisy_config.mode`."""
    if noisy_config.mode not in ("independent", "paired"):
        raise ValueError("the blocking baseline needs an independent or paired noisy group")
    base = build_training_rows(tasks, noisy_config, condition, n_rows // 2, run_seed)
    clean = NoiseConfig.clean()
    rows = []
    for row in base:
        rows.append(dict(row, noise=json.dumps(clean.to_dict(), sort_keys=True), condition=f"{condition}:clean"))
        rows.append(dict(row, condition=f"{condition}:noisy"))
    return rows


def eval_schedule_seed(task_id: str, index: int) -> int:
    """Frozen evaluation schedule for (task, index): distinct across tasks, identical across arms and checkpoints."""
    return derive_seed("eval", task_id, index) % (2**31)


def build_eval_rows(
    tasks: list[Task], config: NoiseConfig, condition: str, schedule_indices: list[int]
) -> list[dict]:
    """Paired evaluation: every task under every fixed schedule index, identical across arms and checkpoints."""
    return [
        make_row(task, eval_schedule_seed(task.task_id, index), config, condition)
        for task in tasks
        for index in schedule_indices
    ]


def build_diagnostic_rows(
    tasks: list[Task], config: NoiseConfig, condition: str, n_schedules: int, base_seed: int
) -> list[dict]:
    """K schedules per diagnostic task; the trainer's num_generations supplies the M policy samples."""
    rows = []
    for task in tasks:
        for k in range(n_schedules):
            seed = derive_seed("diag", base_seed, task.task_id, k) % (2**31)
            rows.append(make_row(task, seed, config, condition))
    return rows

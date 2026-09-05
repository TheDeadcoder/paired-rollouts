"""Procedural task generation for the back-office environment."""

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any

from pairedrl.env.backoffice.tools import ToolAPI
from pairedrl.env.backoffice.world import (
    CARRIERS,
    SHIPPING_WINDOW_DAYS,
    TICKET_CATEGORIES,
    TICKET_PRIORITIES,
    TICKET_SUMMARIES,
    Order,
    World,
    order_fully_reserved,
    order_refunded_cents,
    order_reserved_quantity,
    order_total_cents,
)

TEMPLATES = (
    "address_change",
    "cancel_pending",
    "cancel_paid_refund",
    "partial_refund_delivered",
    "reserve_and_ship",
    "ship_reserved",
    "open_ticket",
    "open_ticket_no_order",
)
TEMPLATE_WEIGHTS = {
    "address_change": 3.0,
    "cancel_pending": 2.0,
    "cancel_paid_refund": 3.0,
    "partial_refund_delivered": 3.0,
    "reserve_and_ship": 5.0,
    "ship_reserved": 3.0,
    "open_ticket": 1.2,
    "open_ticket_no_order": 0.6,
}
SPLIT_SEED_RANGES = {"train": (0, 100_000), "heldout": (100_000, 200_000)}
SUBTASK_COUNT_WEIGHTS = ((1, 20), (2, 40), (3, 40))
STREETS = ("Lake Avenue", "Green Road", "Harbor Drive", "Elm Street", "Market Street", "River Road")
CITIES = ("Dhaka", "Sylhet", "Khulna", "Rajshahi", "Chattogram", "Rangpur")


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any]


@dataclass
class SubTask:
    template: str
    order_id: str | None
    calls: list[ToolCall]
    sentence: str


@dataclass
class Task:
    task_id: str
    split: str
    world_seed: int
    customer_id: str
    templates: list[str]
    instruction: str
    oracle_plan: list[ToolCall]
    n_subgoals: int
    reference: str
    order_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        data = dict(data)
        data["oracle_plan"] = [ToolCall(**c) for c in data["oracle_plan"]]
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def _item_name(world: World, sku: str) -> str:
    return world.products[sku].name


def describe_order(world: World, order: Order, reference: str) -> str:
    if reference == "id":
        return f"order {order.order_id}"
    siblings = [o for o in world.orders.values() if o.customer_id == order.customer_id]
    for item in order.items:
        clash = any(
            o.order_id != order.order_id
            and o.placed_on_day == order.placed_on_day
            and any(i.sku == item.sku for i in o.items)
            for o in siblings
        )
        if not clash:
            name = _item_name(world, item.sku)
            return f"the order placed on day {order.placed_on_day} that contains {name}"
    return f"order {order.order_id}"


def _new_address(rng: random.Random, current) -> tuple[str, str, str]:
    while True:
        street = f"{rng.randint(1, 250)} {rng.choice(STREETS)}"
        city = rng.choice(CITIES)
        postal = f"{rng.randint(1000, 9999)}"
        if (street, city, postal) != (current.street, current.city, current.postal_code):
            return street, city, postal


def _eligible(template: str, world: World, order: Order) -> bool:
    if template == "address_change":
        return order.status in ("pending", "paid")
    if template == "cancel_pending":
        return order.status == "pending"
    if template == "cancel_paid_refund":
        return order.status == "paid" and not order.refunds
    if template == "partial_refund_delivered":
        balance = order_total_cents(order) - order_refunded_cents(order)
        return order.status == "delivered" and any(
            i.quantity * i.unit_price_cents <= balance for i in order.items
        )
    if template == "reserve_and_ship":
        if order.status != "paid" or order_fully_reserved(order):
            return False
        return all(
            world.available(i.sku) >= i.quantity - order_reserved_quantity(order, i.sku)
            for i in order.items
        )
    if template == "ship_reserved":
        return order.status == "paid" and order_fully_reserved(order)
    if template in ("open_ticket", "open_ticket_no_order"):
        return True
    raise ValueError(template)


def _make_subtask(template: str, world: World, order: Order, rng: random.Random, reference: str) -> SubTask:
    customer = world.customers[order.customer_id]
    ref = describe_order(world, order, reference)
    if template == "address_change":
        street, city, postal = _new_address(rng, order.shipping_address)
        return SubTask(
            template,
            order.order_id,
            [
                ToolCall(
                    "update_shipping_address",
                    {"order_id": order.order_id, "street": street, "city": city, "postal_code": postal},
                )
            ],
            f"change the shipping address of {ref} to {street}, {city} {postal}",
        )
    if template == "cancel_pending":
        return SubTask(
            template,
            order.order_id,
            [ToolCall("cancel_order", {"order_id": order.order_id, "reason": "customer request"})],
            f"cancel {ref}, which is still unpaid",
        )
    if template == "cancel_paid_refund":
        total = order_total_cents(order)
        return SubTask(
            template,
            order.order_id,
            [
                ToolCall("cancel_order", {"order_id": order.order_id, "reason": "customer request"}),
                ToolCall(
                    "issue_refund",
                    {"order_id": order.order_id, "amount_cents": total, "reason": "cancelled order"},
                ),
            ],
            f"cancel {ref}, which was already paid, and refund the full amount",
        )
    if template == "partial_refund_delivered":
        balance = order_total_cents(order) - order_refunded_cents(order)
        items = [i for i in order.items if i.quantity * i.unit_price_cents <= balance]
        item = rng.choice(items)
        amount = item.quantity * item.unit_price_cents
        name = _item_name(world, item.sku)
        units = "unit" if item.quantity == 1 else "units"
        where = ref if reference == "id" else describe_order_by_item(world, order, item.sku)
        return SubTask(
            template,
            order.order_id,
            [
                ToolCall(
                    "issue_refund",
                    {"order_id": order.order_id, "amount_cents": amount, "reason": "damaged item"},
                )
            ],
            f"refund the full price of the {item.quantity} {units} of {name} that arrived damaged "
            f"in {where}",
        )
    if template in ("reserve_and_ship", "ship_reserved"):
        carrier = rng.choice(CARRIERS)
        day = world.today_day + rng.randint(0, SHIPPING_WINDOW_DAYS)
        calls = []
        for item in order.items:
            missing = item.quantity - order_reserved_quantity(order, item.sku)
            if missing > 0:
                calls.append(
                    ToolCall(
                        "reserve_stock",
                        {"sku": item.sku, "quantity": missing, "order_id": order.order_id},
                    )
                )
        calls.append(
            ToolCall("schedule_shipment", {"order_id": order.order_id, "carrier": carrier, "day": day})
        )
        if template == "reserve_and_ship":
            sentence = (
                f"reserve stock for every item on {ref}, which is paid, and schedule its shipment "
                f"with the {carrier} carrier on day {day}"
            )
        else:
            sentence = (
                f"schedule the shipment of {ref}, which is paid and fully reserved, "
                f"with the {carrier} carrier on day {day}"
            )
        return SubTask(template, order.order_id, calls, sentence)
    if template == "open_ticket":
        category = rng.choice(TICKET_CATEGORIES)
        priority = rng.choice(TICKET_PRIORITIES)
        summary = rng.choice(TICKET_SUMMARIES)
        return SubTask(
            template,
            order.order_id,
            [
                ToolCall(
                    "create_ticket",
                    {
                        "customer_id": customer.customer_id,
                        "order_id": order.order_id,
                        "category": category,
                        "priority": priority,
                        "summary": summary,
                    },
                )
            ],
            f"open a {priority}-priority ticket in the {category} category linked to {ref} "
            f"with the summary '{summary}'",
        )
    if template == "open_ticket_no_order":
        category = rng.choice(TICKET_CATEGORIES)
        priority = rng.choice(TICKET_PRIORITIES)
        summary = rng.choice(TICKET_SUMMARIES)
        return SubTask(
            template,
            None,
            [
                ToolCall(
                    "create_ticket",
                    {
                        "customer_id": customer.customer_id,
                        "order_id": "",
                        "category": category,
                        "priority": priority,
                        "summary": summary,
                    },
                )
            ],
            f"open a {priority}-priority ticket in the {category} category for the customer, "
            f"not linked to any order, with the summary '{summary}'",
        )
    raise ValueError(template)


def describe_order_by_item(world: World, order: Order, sku: str) -> str:
    siblings = [o for o in world.orders.values() if o.customer_id == order.customer_id]
    clash = any(
        o.order_id != order.order_id
        and o.placed_on_day == order.placed_on_day
        and any(i.sku == sku for i in o.items)
        for o in siblings
    )
    if clash:
        return f"order {order.order_id}"
    return f"the order placed on day {order.placed_on_day}"


def _customer_reference(world: World, customer_id: str, rng: random.Random) -> str:
    customer = world.customers[customer_id]
    style = rng.choice(("name", "email", "phone"))
    if style == "email":
        return f"customer {customer.name} ({customer.email})"
    if style == "phone":
        return f"customer {customer.name} (phone {customer.phone})"
    return f"customer {customer.name}"


def simulate_plan(world_seed: int, plan: list[ToolCall]) -> tuple[World, list[dict]]:
    world = World.generate(world_seed)
    api = ToolAPI(world)
    outcomes = []
    for call in plan:
        response = json.loads(getattr(api, call.tool)(**call.args))
        outcomes.append({"tool": call.tool, "ok": "error" not in response, "response": response})
    return world, outcomes


def generate_task(index: int, split: str, rng: random.Random) -> Task:
    lo, hi = SPLIT_SEED_RANGES[split]
    while True:
        world_seed = rng.randrange(lo, hi)
        world = World.generate(world_seed)
        n_sub = rng.choices([c for c, _ in SUBTASK_COUNT_WEIGHTS], [w for _, w in SUBTASK_COUNT_WEIGHTS])[0]
        reference = rng.choice(("description", "id"))
        customer_ids = sorted(world.customers)
        rng.shuffle(customer_ids)
        task = None
        for customer_id in customer_ids:
            task = _try_customer(index, split, world, world_seed, customer_id, n_sub, reference, rng)
            if task is not None:
                break
        if task is not None:
            return task


def _try_customer(index, split, world, world_seed, customer_id, n_sub, reference, rng) -> Task | None:
    orders = sorted(
        (o for o in world.orders.values() if o.customer_id == customer_id), key=lambda o: o.order_id
    )
    if not orders:
        return None
    subtasks: list[SubTask] = []
    used_orders: set[str] = set()
    ticket_used = False
    for _ in range(n_sub):
        options: dict[str, list[Order]] = {}
        for template in TEMPLATES:
            if template.startswith("open_ticket"):
                if not ticket_used:
                    options[template] = [o for o in orders if o.order_id not in used_orders] or orders
                continue
            candidates = [
                o for o in orders if o.order_id not in used_orders and _eligible(template, world, o)
            ]
            if candidates:
                options[template] = candidates
        if not options:
            break
        names = sorted(options)
        template = rng.choices(names, [TEMPLATE_WEIGHTS[n] for n in names])[0]
        order = rng.choice(options[template])
        subtasks.append(_make_subtask(template, world, order, rng, reference))
        if template.startswith("open_ticket"):
            ticket_used = True
        else:
            used_orders.add(order.order_id)
    if len(subtasks) != n_sub:
        return None
    rng.shuffle(subtasks)
    plan = [call for s in subtasks for call in s.calls]
    _, outcomes = simulate_plan(world_seed, plan)
    if not all(o["ok"] for o in outcomes):
        return None
    who = _customer_reference(world, customer_id, rng)
    if len(subtasks) == 1:
        body = f"{who[0].upper()}{who[1:]} asked us to {subtasks[0].sentence}."
    else:
        ordinals = ("First", "Second", "Third")
        parts = [f"{ordinals[i]}, {s.sentence}." for i, s in enumerate(subtasks)]
        body = f"{who[0].upper()}{who[1:]} has {len(subtasks)} requests. " + " ".join(parts)
    instruction = (
        f"Today is day {world.today_day}. {body} Use the tools to look up what you need and make "
        f"exactly these changes, then call finish."
    )
    return Task(
        task_id=f"{split}-{index:05d}",
        split=split,
        world_seed=world_seed,
        customer_id=customer_id,
        templates=[s.template for s in subtasks],
        instruction=instruction,
        oracle_plan=plan,
        n_subgoals=len(plan),
        reference=reference,
        order_ids=sorted({s.order_id for s in subtasks if s.order_id}),
    )


def generate_tasks(n: int, split: str, seed: int) -> list[Task]:
    rng = random.Random(f"{split}-{seed}")
    return [generate_task(i, split, rng) for i in range(n)]


def write_jsonl(tasks: list[Task], path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(task.to_json() + "\n" for task in tasks)


def read_jsonl(path) -> list[Task]:
    with open(path, encoding="utf-8") as f:
        return [Task.from_dict(json.loads(line)) for line in f if line.strip()]

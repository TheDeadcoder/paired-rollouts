"""Seeded fault schedules keyed by the fault event (tool, semantic resource key, repeat index), plus one
episode-level draw. Two rollouts that touch the same resource through the same tool for the k-th time meet the same
fate under the same schedule, whatever else they did before and however they word free-text arguments: this is the
common-random-numbers synchronization the paired design relies on."""

import hashlib
import json
import random
from dataclasses import dataclass

from pairedrl.env.backoffice.tools import CONTROL_TOOLS, READ_TOOLS, WRITE_TOOLS
from pairedrl.noise.config import RETRY_AFTER_CHOICES, NoiseConfig

LIST_TOOLS = frozenset({"search_customers", "list_orders"})


@dataclass(frozen=True)
class Fate:
    kind: str | None
    retry_after: int = 0
    dropped_field_index: int = 0

    @property
    def faulted(self) -> bool:
        return self.kind is not None


FAULT_KEY_FIELDS = {
    "search_customers": (),
    "get_customer": ("customer_id",),
    "list_orders": ("customer_id",),
    "get_order": ("order_id",),
    "check_inventory": ("sku",),
    "update_shipping_address": ("order_id",),
    "cancel_order": ("order_id",),
    "issue_refund": ("order_id",),
    "reserve_stock": ("order_id", "sku"),
    "schedule_shipment": ("order_id",),
    "create_ticket": ("customer_id",),
}


def _canonical(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def request_key(kwargs: dict) -> str:
    """Canonical form of all of a call's arguments; numbers and their string forms are the same request."""
    return json.dumps({k: _canonical(v) for k, v in kwargs.items()}, sort_keys=True, separators=(",", ":"))


def fault_key(tool: str, kwargs: dict) -> str:
    """The exogenous process a fault belongs to: the tool and the resource it touches. Free text (reasons,
    summaries, search queries) never changes the key, so rewording a request cannot dodge its fate."""
    fields = FAULT_KEY_FIELDS.get(tool, ())
    return json.dumps({f: _canonical(kwargs.get(f)) for f in fields}, sort_keys=True, separators=(",", ":"))


def applicable_types(tool: str) -> tuple[str, ...]:
    if tool in CONTROL_TOOLS:
        return ()
    types = ["transient", "rate_limit", "outage"]
    if tool in READ_TOOLS:
        types += ["stale", "field_dropout"]
    if tool in LIST_TOOLS:
        types.append("truncate")
    if tool in WRITE_TOOLS and tool != "create_ticket":
        types.append("timeout_after_commit")
    return tuple(types)


def derive_seed(*parts) -> int:
    text = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def resolve_seed(config: NoiseConfig, schedule_seed: int, instance_slot: int, episode_index: int, phase: str = "train") -> int:
    """Paired: every rollout of the row shares schedule_seed. Independent: each rollout gets its own, derived from
    the row's seed, the environment instance and that instance's episode counter within the phase family
    (training counters ignore evaluation episodes, so evaluation frequency never changes training draws)."""
    if config.mode == "independent":
        family = "train" if phase.startswith("train") else "eval"
        return derive_seed("independent", schedule_seed, instance_slot, episode_index, family)
    return int(schedule_seed)


class NoiseSchedule:
    def __init__(self, seed: int, config: NoiseConfig):
        self.seed = int(seed)
        self.config = config

    def _rng(self, *key) -> random.Random:
        return random.Random(derive_seed(self.seed, *key))

    def fate(self, tool: str, event: str, repeat: int) -> Fate:
        """Fate of the `repeat`-th occurrence of the fault event (`tool`, `event` = a `fault_key`) under this schedule."""
        if self.config.challenge == "writes_once":
            return Fate("transient") if tool in WRITE_TOOLS and repeat == 0 else Fate(None)
        types = applicable_types(tool)
        if self.config.is_clean or self.config.p <= 0 or not types:
            return Fate(None)
        rng = self._rng("event", tool, event, repeat)
        u = rng.random()
        retry_after = rng.choice(RETRY_AFTER_CHOICES)
        dropped = rng.randrange(0, 1_000_000)
        if u >= self.config.p:
            return Fate(None)
        weights = [self.config.weights.get(t, 0.0) for t in types]
        if sum(weights) <= 0:
            return Fate(None)
        kind = rng.choices(types, weights)[0]
        return Fate(kind, retry_after=retry_after, dropped_field_index=dropped)

    def outcome_flip(self) -> bool:
        if self.config.is_clean or self.config.q <= 0:
            return False
        return self._rng("outcome").random() < self.config.q

    def describe(self) -> dict:
        return {"seed": self.seed, "config": self.config.to_dict()}

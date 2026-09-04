"""Seeded fault schedules keyed by (tool, per-tool call index), plus one episode-level draw."""

import hashlib
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


def applicable_types(tool: str) -> tuple[str, ...]:
    if tool in CONTROL_TOOLS:
        return ()
    types = ["transient", "rate_limit"]
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


def resolve_seed(config: NoiseConfig, schedule_seed: int, instance_slot: int, episode_index: int) -> int:
    """Paired: every rollout of the row shares schedule_seed. Independent: each rollout gets its own."""
    if config.mode == "independent":
        return derive_seed("independent", schedule_seed, instance_slot, episode_index)
    return int(schedule_seed)


class NoiseSchedule:
    def __init__(self, seed: int, config: NoiseConfig):
        self.seed = int(seed)
        self.config = config

    def _rng(self, *key) -> random.Random:
        return random.Random(derive_seed(self.seed, *key))

    def fate(self, tool: str, call_index: int) -> Fate:
        types = applicable_types(tool)
        if self.config.is_clean or self.config.p <= 0 or not types:
            return Fate(None)
        rng = self._rng("call", tool, call_index)
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

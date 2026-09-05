"""Noise configuration: which faults can happen, how often, and how the schedule seed is chosen."""

from dataclasses import asdict, dataclass, field

MODES = ("paired", "independent", "clean")
TRANSITION_TYPES = ("transient", "rate_limit", "outage", "stale", "truncate")
HELDOUT_TYPES = ("timeout_after_commit", "field_dropout")
FAULT_TYPES = TRANSITION_TYPES + HELDOUT_TYPES
DEFAULT_WEIGHTS = {"transient": 0.45, "rate_limit": 0.15, "outage": 0.10, "stale": 0.15, "truncate": 0.15}
RETRY_AFTER_CHOICES = (15, 30, 45, 60)


@dataclass(frozen=True)
class NoiseConfig:
    """Per-call fault rate p, split across fault types by weight; episode-level outcome flip rate q."""

    p: float = 0.0
    q: float = 0.0
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    mode: str = "clean"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not 0.0 <= self.p <= 1.0 or not 0.0 <= self.q <= 1.0:
            raise ValueError("p and q must lie in [0, 1]")
        unknown = set(self.weights) - set(FAULT_TYPES)
        if unknown:
            raise ValueError(f"unknown fault types {sorted(unknown)}")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("weights must be non-negative")
        if self.mode != "clean" and self.p > 0 and sum(self.weights.values()) <= 0:
            raise ValueError("p > 0 requires at least one positive weight")

    @property
    def is_clean(self) -> bool:
        return self.mode == "clean" or (self.p == 0.0 and self.q == 0.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "NoiseConfig":
        return cls(**data)

    @classmethod
    def clean(cls) -> "NoiseConfig":
        return cls()

    @classmethod
    def transition(cls, p: float, mode: str, weights: dict[str, float] | None = None) -> "NoiseConfig":
        return cls(p=p, q=0.0, weights=dict(weights or DEFAULT_WEIGHTS), mode=mode)

    @classmethod
    def outcome(cls, q: float, mode: str) -> "NoiseConfig":
        return cls(p=0.0, q=q, weights=dict(DEFAULT_WEIGHTS), mode=mode)

    @classmethod
    def heldout_types(cls, p: float, mode: str) -> "NoiseConfig":
        return cls(p=p, q=0.0, weights={"timeout_after_commit": 0.5, "field_dropout": 0.5}, mode=mode)

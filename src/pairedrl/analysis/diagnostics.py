"""Group-level diagnostics: GRPO advantages, spurious-variance groups, and the luck-share estimator."""

import math
from dataclasses import asdict, dataclass

ADVANTAGE_EPS = 1e-4


def _mean(xs) -> float:
    return sum(xs) / len(xs)


def std_ddof1(xs) -> float:
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def grpo_advantages(rewards, scale: str = "group") -> list[float]:
    """TRL's group-relative advantage: (r - mean) / (std_ddof1 + 1e-4) for scale='group', r - mean for 'none'."""
    m = _mean(rewards)
    if scale == "none":
        return [r - m for r in rewards]
    if scale != "group":
        raise ValueError("scale must be 'group' or 'none'")
    s = std_ddof1(rewards)
    return [(r - m) / (s + ADVANTAGE_EPS) for r in rewards]


@dataclass
class GroupSummary:
    size: int
    mean: float
    std: float
    zero_variance: bool
    spurious: bool | None
    max_abs_advantage: float

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_group(observed, true_outcomes=None, scale: str = "group") -> GroupSummary:
    s = std_ddof1(observed)
    zero = math.isclose(s, 0.0, abs_tol=1e-12)
    spurious = None
    if true_outcomes is not None:
        spurious = len({bool(t) for t in true_outcomes}) == 1 and not zero
    adv = grpo_advantages(observed, scale) if not zero else [0.0] * len(observed)
    return GroupSummary(
        size=len(observed),
        mean=_mean(observed),
        std=s,
        zero_variance=zero,
        spurious=spurious,
        max_abs_advantage=max(abs(a) for a in adv),
    )


def spurious_rate_all_correct(q: float, group_size: int) -> float:
    """Probability that independent outcome flips at rate q give an all-correct group non-zero variance."""
    return 1.0 - (1.0 - q) ** group_size - q**group_size


@dataclass
class LuckShare:
    ms_between: float
    ms_within: float
    sigma2_env: float
    sigma2_pol: float
    lam: float | None
    schedules: int
    samples: int

    def to_dict(self) -> dict:
        return asdict(self)


def luck_share(rewards) -> LuckShare:
    """One-way random-effects decomposition of a K x M reward table (K schedules, M policy samples each)."""
    k = len(rewards)
    m = len(rewards[0])
    if k < 2 or m < 2 or any(len(row) != m for row in rewards):
        raise ValueError("need at least 2 schedules and 2 samples per schedule, rectangular")
    row_means = [_mean(row) for row in rewards]
    grand = _mean(row_means)
    ms_between = m / (k - 1) * sum((rm - grand) ** 2 for rm in row_means)
    ms_within = sum((x - rm) ** 2 for row, rm in zip(rewards, row_means) for x in row) / (k * (m - 1))
    sigma2_env = max(0.0, (ms_between - ms_within) / m)
    sigma2_pol = ms_within
    denom = sigma2_env + sigma2_pol
    lam = sigma2_env / denom if denom > 0 else None
    return LuckShare(ms_between, ms_within, sigma2_env, sigma2_pol, lam, k, m)


def luck_share_over_tasks(tables) -> dict:
    shares = [luck_share(t) for t in tables]
    defined = [s.lam for s in shares if s.lam is not None]
    return {
        "tasks": len(shares),
        "tasks_defined": len(defined),
        "lam": _mean(defined) if defined else None,
        "sigma2_env_mean": _mean([s.sigma2_env for s in shares]),
        "sigma2_pol_mean": _mean([s.sigma2_pol for s in shares]),
    }


def luck_share_pooled(tables) -> float | None:
    """Ratio of the averaged variance components across tasks (less clipping bias than the mean of task ratios)."""
    shares = [luck_share(t) for t in tables]
    env = _mean([s.sigma2_env for s in shares]) if shares else 0.0
    pol = _mean([s.sigma2_pol for s in shares]) if shares else 0.0
    return env / (env + pol) if env + pol > 0 else None


def contrast_precision_equivalent(group_size: int, lam: float) -> float:
    """Independent rollouts whose within-group reward contrasts are as precise as those of `group_size` paired
    ones: 2(sigma2_env + sigma2_pol) / (2 sigma2_pol) = 1 / (1 - lambda) per rollout. A statement about reward
    contrasts only; it is not an effective sample size for the policy gradient (see docs/THEORY.md, section 3)."""
    if not 0.0 <= lam < 1.0:
        raise ValueError("lam must lie in [0, 1)")
    return group_size / (1.0 - lam)

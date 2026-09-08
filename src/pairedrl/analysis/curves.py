"""Learning curves, areas and the per-run register built from a run directory (manifest, episodes, groups)."""

import json
import pathlib
from collections import defaultdict

from pairedrl.analysis.diagnostics import luck_share_over_tasks, luck_share_pooled
from pairedrl.train.runner import (
    luck_share_tables_by_phase,
    read_episode_log,
    summarize_episodes,
    summarize_groups,
)

PERIODIC_SETS = ("clean", "noisy", "challenge")


def _phase_step(phase: str) -> int | None:
    tail = phase.rsplit("step", 1)
    return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else None


def periodic_curve(records: list[dict], set_name: str) -> list[dict]:
    """Points (step, n, true_success, observed_reward, recovery_success, exposed_frac, budget_exceeded_frac)
    of one periodic evaluation set, in step order."""
    by_step: dict[int, list[dict]] = defaultdict(list)
    prefix = f"eval_{set_name}:step"
    for r in records:
        phase = r.get("phase", "")
        if phase.startswith(prefix):
            by_step[_phase_step(phase)].append(r)
    points = []
    for step in sorted(by_step):
        recs = by_step[step]
        agg = summarize_episodes(recs)
        (_key, value), = agg.items()
        points.append({"step": step, "n": value["episodes"], **{k: v for k, v in value.items() if k != "episodes"}})
    return points


def area_under_curve(points: list[dict], field: str = "true_success") -> float | None:
    """Trapezoidal mean of `field` over the evaluated steps: the average level along the run, in [0, 1]."""
    if len(points) < 2:
        return None
    steps = [p["step"] for p in points]
    values = [p[field] for p in points]
    total = sum((steps[i + 1] - steps[i]) * (values[i] + values[i + 1]) / 2 for i in range(len(points) - 1))
    return total / (steps[-1] - steps[0])


def steps_to_threshold(points: list[dict], threshold: float, field: str = "true_success") -> float | None:
    """First evaluated step at which `field` reaches `threshold`, linearly interpolated; None when censored."""
    for i, p in enumerate(points):
        if p[field] >= threshold:
            if i == 0:
                return float(p["step"])
            prev = points[i - 1]
            span = p[field] - prev[field]
            frac = (threshold - prev[field]) / span if span > 0 else 1.0
            return prev["step"] + frac * (p["step"] - prev["step"])
    return None


def final_sets(records: list[dict]) -> dict:
    out = {}
    for key, agg in summarize_episodes(records).items():
        phase, condition = key.split("|", 1)
        if phase.startswith("final_"):
            out[condition.replace("eval:", "")] = {"step": _phase_step(phase), **agg}
    return out


def training_curve(records: list[dict]) -> list[dict]:
    by_step: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        phase = r.get("phase", "")
        if phase.startswith("train:step"):
            by_step[_phase_step(phase)].append(r)
    points = []
    for step in sorted(by_step):
        recs = by_step[step]
        n = len(recs)
        points.append({
            "step": step, "n": n,
            "true_success": sum(r["true_success"] for r in recs) / n,
            "observed_reward": sum(r["observed_reward"] for r in recs) / n,
            "budget_exceeded_frac": sum(r["budget_exceeded"] for r in recs) / n,
            "mean_calls": sum(r["calls"] for r in recs) / n,
        })
    return points


def luck_by_phase(records: list[dict]) -> dict:
    out = {}
    for phase, tables in luck_share_tables_by_phase(records).items():
        over = luck_share_over_tasks(tables) if tables else {"tasks": 0, "tasks_defined": 0, "lam": None}
        over["lam_pooled"] = luck_share_pooled(tables) if tables else None
        over["tables"] = len(tables)
        out[phase] = over
    return out


def step_timing_summary(timings: list[dict]) -> dict:
    if not timings:
        return {}
    seconds = [t["seconds"] for t in timings]
    gen = [t["generation_s"] for t in timings if t.get("generation_s") is not None]
    return {
        "steps": len(timings),
        "mean_step_s": sum(seconds) / len(seconds),
        "mean_generation_s": (sum(gen) / len(gen)) if gen else None,
        "mean_passes_s": ((sum(seconds) - sum(gen)) / len(gen)) if gen and len(gen) == len(seconds) else None,
        "min_step_s": min(seconds),
        "max_step_s": max(seconds),
    }


def run_register(run_dir, threshold: float | None = None) -> dict:
    """Everything the paper reads from one run: spec facts, periodic curves with areas, final sets, luck share
    per checkpoint, training-group totals, training curve and timing summary."""
    run_dir = pathlib.Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    records = read_episode_log(run_dir / "episodes.jsonl")
    groups = read_episode_log(run_dir / "groups.jsonl")
    spec = manifest["spec"]
    curves = {name: periodic_curve(records, name) for name in PERIODIC_SETS}
    entry = {
        "run_id": manifest["run_id"],
        "provider": manifest.get("provider", "modal"),
        "status": manifest.get("status"),
        "attempt": manifest.get("attempt"),
        "git_commit": manifest.get("git_commit"),
        "condition": spec.get("condition"),
        "arm": spec.get("arm"),
        "seed": spec.get("seed"),
        "p": spec.get("p"),
        "q": spec.get("q"),
        "fault_weights": spec.get("fault_weights"),
        "steps": spec.get("steps"),
        "prompts_per_step": spec.get("prompts_per_step"),
        "num_generations": spec.get("num_generations"),
        "scale_rewards": spec.get("scale_rewards"),
        "loss_type": spec.get("loss_type"),
        "wall_time_s": manifest.get("wall_time_s"),
        "estimated_cost_usd": manifest.get("estimated_cost_usd"),
        "peak_mem_gb": manifest.get("peak_mem_gb"),
        "versions": manifest.get("versions"),
        "episodes": len(records),
        "curves": curves,
        "auc": {name: area_under_curve(points) for name, points in curves.items()},
        "final": final_sets(records),
        "luck_share": luck_by_phase(records),
        "training_groups": summarize_groups(groups) if groups else {},
        "training_curve": training_curve(records),
        "timing": step_timing_summary(manifest.get("step_timings", [])),
        "eval_timings": manifest.get("eval_timings", []),
    }
    if threshold is not None:
        entry["threshold"] = threshold
        entry["steps_to_threshold"] = steps_to_threshold(curves["noisy"], threshold) if curves["noisy"] else None
    return entry


def format_run(entry: dict) -> str:
    lines = [
        (
            f"{entry['run_id']} ({entry['condition']} {entry['arm']} seed {entry['seed']}, {entry['provider']}): "
            f"status {entry['status']}, {entry['episodes']} episodes, {entry['wall_time_s']} s, "
            f"est {entry['estimated_cost_usd']} USD"
        )
    ]
    for name in PERIODIC_SETS:
        points = entry["curves"].get(name, [])
        if points:
            values = " ".join(f"{p['step']}:{p['true_success']:.3f}" for p in points)
            auc = entry["auc"].get(name)
            lines.append(f"  {name:10s} {values}  auc={auc:.3f}" if auc is not None else f"  {name:10s} {values}")
    for name, agg in sorted(entry["final"].items()):
        rec = agg["recovery_success"]
        lines.append(
            f"  final {name:14s} true={agg['true_success']:.3f} recovery={'n/a' if rec is None else f'{rec:.3f}'} "
            f"exposed={agg['exposed_frac']:.2f} calls={agg['mean_calls']:.1f} budget_exceeded={agg['budget_exceeded_frac']:.2f}"
        )
    for phase, luck in sorted(entry["luck_share"].items(), key=lambda kv: _phase_step(kv[0]) or 0):
        lam, pooled = luck.get("lam"), luck.get("lam_pooled")
        lines.append(
            f"  luck share {phase}: lam={'n/a' if lam is None else f'{lam:.3f}'} "
            f"pooled={'n/a' if pooled is None else f'{pooled:.3f}'} over {luck.get('tasks_defined')} of {luck.get('tasks')} tasks"
        )
    for condition, g in sorted(entry["training_groups"].items()):
        spurious = g["spurious_among_all_correct"]
        lines.append(
            f"  groups {condition}: {g['groups']} groups, zero-variance {g['zero_variance_frac']:.3f}, "
            f"all-correct {g['all_correct_groups']}, spurious among all-correct "
            f"{'n/a' if spurious is None else f'{spurious:.3f}'}"
        )
    timing = entry.get("timing") or {}
    if timing:
        gen, passes = timing.get("mean_generation_s"), timing.get("mean_passes_s")
        lines.append(
            f"  timing: {timing['steps']} steps, mean {timing['mean_step_s']:.1f} s "
            f"(generation {'n/a' if gen is None else f'{gen:.1f}'} s, passes {'n/a' if passes is None else f'{passes:.1f}'} s)"
        )
    if "threshold" in entry:
        stt = entry["steps_to_threshold"]
        lines.append(f"  threshold {entry['threshold']:.3f}: steps to threshold {'censored' if stt is None else f'{stt:.1f}'}")
    return "\n".join(lines)

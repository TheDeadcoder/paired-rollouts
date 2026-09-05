"""Register builders: every number the paper cites is emitted here from run outputs, never typed."""

import json
import pathlib

from pairedrl.analysis.diagnostics import luck_share_over_tasks
from pairedrl.train.runner import luck_share_tables, read_episode_log, summarize_episodes

CALIBRATION_BANDS = {"Qwen/Qwen3.5-2B": (0.20, 0.50), "Qwen/Qwen3.5-4B": (0.40, 0.70)}
MIN_LUCK_SHARE = 0.15
COMPLETION_KEYS = {
    "mean_length": "eval_completions/mean_length",
    "max_length": "eval_completions/max_length",
    "clipped_ratio": "eval_completions/clipped_ratio",
    "call_frequency": "eval_tools/call_frequency",
    "failure_frequency": "eval_tools/failure_frequency",
    "reward": "eval_reward",
}


def _load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def completion_stats(history: list[dict]) -> dict:
    """One row per evaluate() call, keyed by its metric prefix (final_clean, diag, ...)."""
    out = {}
    for entry in history:
        if "eval_completions/mean_length" not in entry:
            continue
        prefixes = [k[: -len("_runtime")] for k in entry if k.endswith("_runtime")]
        name = prefixes[0] if prefixes else f"step{entry.get('step')}"
        out[name] = {field: entry.get(key) for field, key in COMPLETION_KEYS.items()}
        out[name]["runtime_s"] = entry.get(f"{name}_runtime")
    return out


def calibration_entry(run_dir) -> dict:
    """Summarize one eval-only calibration run: final sets, diagnostic luck share, manifest facts."""
    run_dir = pathlib.Path(run_dir)
    manifest = _load_json(run_dir / "run_manifest.json")
    records = read_episode_log(run_dir / "episodes.jsonl")
    history_path = run_dir / "trainer_log_history.json"
    history = _load_json(history_path) if history_path.exists() else []
    by_phase = summarize_episodes(records)
    sets = {}
    for key, agg in by_phase.items():
        phase, condition = key.split("|", 1)
        if phase.startswith("final_"):
            sets[condition.replace("eval:", "")] = agg
    tables = luck_share_tables(records)
    luck = luck_share_over_tasks(tables) if tables else {"tasks": 0, "tasks_defined": 0, "lam": None}
    observed_tables = luck_share_tables(records, field="observed_reward")
    luck_observed = luck_share_over_tasks(observed_tables) if observed_tables else {"tasks": 0, "tasks_defined": 0, "lam": None}
    model = manifest["spec"]["model"]
    band = CALIBRATION_BANDS.get(model)
    clean = sets.get("clean", {}).get("true_success")
    return {
        "run_id": manifest["run_id"],
        "model": model,
        "git_commit": manifest.get("git_commit"),
        "status": manifest.get("status"),
        "wall_time_s": manifest.get("wall_time_s"),
        "estimated_cost_usd": manifest.get("estimated_cost_usd"),
        "peak_mem_gb": manifest.get("peak_mem_gb"),
        "versions": manifest.get("versions"),
        "episodes": len(records),
        "sets": sets,
        "completions": completion_stats(history),
        "luck_share": luck,
        "luck_share_observed": luck_observed,
        "clean_band": list(band) if band else None,
        "clean_in_band": (band is not None and clean is not None and band[0] <= clean <= band[1]),
        "luck_share_ok": (luck.get("lam") is not None and luck["lam"] >= MIN_LUCK_SHARE),
    }


def build_calibration_register(run_dirs, out_path) -> dict:
    register = {"entries": [calibration_entry(d) for d in run_dirs]}
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(register, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return register


def _fmt(value, digits=3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_calibration(register: dict) -> str:
    lines = []
    for e in register["entries"]:
        lines.append(
            f"{e['run_id']} ({e['model']}): status {e['status']}, {e['episodes']} episodes, "
            f"{e['wall_time_s']} s, est {e['estimated_cost_usd']} USD, peak {e['peak_mem_gb']} GB"
        )
        for name, agg in sorted(e["sets"].items()):
            lines.append(
                f"  {name:14s} true={agg['true_success']:.3f} recovery={_fmt(agg['recovery_success'])} "
                f"exposed={agg['exposed_frac']:.2f} calls={agg['mean_calls']:.1f} "
                f"budget_exceeded={agg['budget_exceeded_frac']:.2f} finished={agg['finished_frac']:.2f}"
            )
        for name, stats in sorted(e["completions"].items()):
            lines.append(
                f"  {name:14s} mean_len={_fmt(stats['mean_length'], 0)} max_len={_fmt(stats['max_length'], 0)} "
                f"clipped={_fmt(stats['clipped_ratio'])} calls/ep={_fmt(stats['call_frequency'], 1)} "
                f"tool_fail={_fmt(stats['failure_frequency'])} runtime={_fmt(stats['runtime_s'], 0)} s"
            )
        luck = e["luck_share"]
        lines.append(
            f"  luck share lam={_fmt(luck.get('lam'))} over {luck.get('tasks_defined')} of {luck.get('tasks')} tasks"
            f" (observed reward: {_fmt(e['luck_share_observed'].get('lam'))})"
        )
        lines.append(
            f"  clean in band {e['clean_band']}: {e['clean_in_band']}; "
            f"luck share >= {MIN_LUCK_SHARE}: {e['luck_share_ok']}"
        )
    return "\n".join(lines)

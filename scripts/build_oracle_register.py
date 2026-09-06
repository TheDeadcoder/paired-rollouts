"""Scripted reference policies on the validation and test pools under the frozen evaluation schedules.

The recovering oracle is a strong scripted reference (hidden plan, fixed write order, bounded retries), not a
proven ceiling; the naive oracle never retries, waits, re-reads or paginates. Both use the same per-task
evaluation schedules as the model evaluations, so their numbers are directly comparable with a run's final sets.
"""

import argparse
import json
import pathlib
import statistics

from pairedrl.env.backoffice.tasks import read_jsonl
from pairedrl.noise.config import NoiseConfig
from pairedrl.noise.oracle import make_api, run_oracle
from pairedrl.train.dataset import eval_schedule_seed
from pairedrl.train.env_adapter import budget_for
from pairedrl.train.runner import EVAL_SCHEDULES_FINAL

NO_OUTAGE = {"transient": 0.5, "rate_limit": 0.5 / 3, "stale": 0.5 / 3, "truncate": 0.5 / 3}
CONDITIONS = {
    "clean": NoiseConfig.clean(),
    "C1_p0.10": NoiseConfig.transition(0.10, "paired"),
    "C2_p0.25": NoiseConfig.transition(0.25, "paired"),
    "C2r_p0.25_no_outage": NoiseConfig.transition(0.25, "paired", NO_OUTAGE),
    "heldout_types_p0.25": NoiseConfig.heldout_types(0.25, "paired"),
}
POOLS = {"heldout": "data/tasks/heldout.jsonl", "test": "data/tasks/test.jsonl"}


def evaluate(tasks, config, recover: bool) -> dict:
    results = [
        run_oracle(
            t, make_api(t, config, seed=eval_schedule_seed(t.task_id, k)), recover=recover,
            budget=budget_for(t), lookups=True,
        )
        for t in tasks
        for k in EVAL_SCHEDULES_FINAL
    ]
    n = len(results)
    outage = sum(1 for r in results if any(k.startswith("outage") for k in r.faults))
    return {
        "episodes": n,
        "success": sum(r.success for r in results) / n,
        "exposed_frac": sum(r.exposed for r in results) / n,
        "outage_hit_frac": outage / n,
        "budget_exceeded_frac": sum(r.budget_exceeded for r in results) / n,
        "mean_calls": statistics.fmean(r.calls for r in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="registers/oracle_reference.json")
    args = parser.parse_args()
    register = {"schedules": list(EVAL_SCHEDULES_FINAL), "seeding": "eval_schedule_seed(task_id, index)", "pools": {}}
    for pool, path in POOLS.items():
        tasks = read_jsonl(path)
        entry = {"tasks": path, "n_tasks": len(tasks), "conditions": {}}
        for name, config in CONDITIONS.items():
            entry["conditions"][name] = {
                "config": config.to_dict(),
                "naive": evaluate(tasks, config, recover=False),
                "recovering": evaluate(tasks, config, recover=True),
            }
            naive, rec = entry["conditions"][name]["naive"], entry["conditions"][name]["recovering"]
            print(
                f"{pool:8s} {name:22s} naive={naive['success']:.3f} recovering={rec['success']:.3f} "
                f"exposed={rec['exposed_frac']:.3f} outage_hit={rec['outage_hit_frac']:.3f} "
                f"budget_exceeded={rec['budget_exceeded_frac']:.3f} calls={rec['mean_calls']:.1f}"
            )
        register["pools"][pool] = entry
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(register, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()

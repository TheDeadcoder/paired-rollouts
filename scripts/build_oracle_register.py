"""Reference oracle statistics on the held-out tasks: the skill ceiling and the naive floor per noise condition."""

import argparse
import json
import pathlib
import statistics

from pairedrl.env.backoffice.tasks import read_jsonl
from pairedrl.noise.config import NoiseConfig
from pairedrl.noise.oracle import make_api, run_oracle
from pairedrl.train.env_adapter import budget_for

CONDITIONS = {
    "clean": NoiseConfig.clean(),
    "C1_p0.10": NoiseConfig.transition(0.10, "paired"),
    "C2_p0.25": NoiseConfig.transition(0.25, "paired"),
    "heldout_types_p0.25": NoiseConfig.heldout_types(0.25, "paired"),
}
SEEDS = (1, 2, 3, 4)


def evaluate(tasks, config, recover: bool) -> dict:
    results = [
        run_oracle(t, make_api(t, config, seed=s), recover=recover, budget=budget_for(t), lookups=True)
        for t in tasks
        for s in SEEDS
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
    parser.add_argument("--tasks", default="data/tasks/heldout.jsonl")
    parser.add_argument("--out", default="registers/oracle_reference.json")
    args = parser.parse_args()
    tasks = read_jsonl(args.tasks)
    register = {"tasks": args.tasks, "n_tasks": len(tasks), "seeds": list(SEEDS), "conditions": {}}
    for name, config in CONDITIONS.items():
        register["conditions"][name] = {
            "config": config.to_dict(),
            "naive": evaluate(tasks, config, recover=False),
            "recovering": evaluate(tasks, config, recover=True),
        }
        naive, rec = register["conditions"][name]["naive"], register["conditions"][name]["recovering"]
        print(
            f"{name:20s} naive={naive['success']:.3f} recovering={rec['success']:.3f} "
            f"exposed={rec['exposed_frac']:.3f} outage_hit={rec['outage_hit_frac']:.3f} "
            f"budget_exceeded={rec['budget_exceeded_frac']:.3f} calls={rec['mean_calls']:.1f}"
        )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(register, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()

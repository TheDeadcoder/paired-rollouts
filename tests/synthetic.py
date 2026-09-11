"""Synthetic run directories whose records carry the exact task and schedule identities of the frozen pools, so
the acceptance check and the decision rules can be tested against runs that differ from real ones only in their
outcomes."""

import json

from pairedrl.analysis.curves import DEFAULT_DATA_DIR
from pairedrl.train.runner import (
    RunSpec,
    assemble_diagnostic_rows,
    assemble_eval_sets,
    assemble_training_rows,
    final_step,
    load_task_splits,
    periodic_steps,
)

SPLITS = load_task_splits(DEFAULT_DATA_DIR)


def rec(phase, condition, task, success, **kw):
    base = {"task_id": task, "phase": phase, "condition": condition, "true_success": bool(success),
            "observed_reward": float(success), "flipped": False, "calls": 6, "budget": 13, "budget_exceeded": False,
            "finished": True, "exposed": condition != "eval:clean", "faults": {}, "schedule_seed": 1, "attempt": 1}
    base.update(kw)
    return base


def _indexed(rows):
    """(task index, index of the row within its task) for rows in task-major order."""
    tasks, seen = {}, {}
    for row in rows:
        ti = tasks.setdefault(row["task_id"], len(tasks))
        k = seen.get(row["task_id"], 0)
        seen[row["task_id"]] = k + 1
        yield row, ti, k


def periodic_records(spec, steps, success, attempt=1):
    """One record per row of every periodic set at each of `steps`; success(set, step, task_index, schedule_index)."""
    out = []
    for step in steps:
        for name, rows in assemble_eval_sets(spec, SPLITS["heldout"], final=False).items():
            for j, (row, ti, k) in enumerate(_indexed(rows)):
                out.append(rec(f"eval_{name}:step{step}", row["condition"], row["task_id"], success(name, step, ti, k),
                               schedule_seed=row["schedule_seed"], resolved_seed=row["schedule_seed"], slot=j, attempt=attempt))
    return out


def final_records(spec, success, attempt=1):
    """One record per row of every final set at the final step; success(set, task_index, schedule_index)."""
    out = []
    step = final_step(spec)
    for name, rows in assemble_eval_sets(spec, SPLITS["test"], final=True).items():
        for j, (row, ti, k) in enumerate(_indexed(rows)):
            out.append(rec(f"final_{name}:step{step}", row["condition"], row["task_id"], success(name, ti, k),
                           schedule_seed=row["schedule_seed"], resolved_seed=row["schedule_seed"], slot=j, attempt=attempt))
    return out


def diagnostic_records(spec, steps, success, attempt=1):
    """K x M records per diagnostic task at each of `steps`; success(step, task_index, schedule_index, sample_index)."""
    out = []
    rows = assemble_diagnostic_rows(spec, SPLITS["diagnostic"])
    m_samples = spec.diagnostic_samples
    for step in steps:
        for j, (row, ti, idx) in enumerate(_indexed(rows)):
            out.append(rec(f"diag:step{step}", "diag", row["task_id"], success(step, ti, idx // m_samples, idx % m_samples),
                           schedule_seed=row["schedule_seed"], resolved_seed=row["schedule_seed"], slot=j, attempt=attempt))
    return out


def training_records(spec, steps, outcomes, resolved_seed=None, spurious_groups=0, attempt=1):
    """Groups and rollouts of the given training steps in trainer order; outcomes(step, group) -> list of
    num_generations booleans; resolved_seed(schedule_seed, member) defaults to the paired arm (one seed per group);
    the first `spurious_groups` all-correct groups of every step are logged with a flipped observation."""
    rows = assemble_training_rows(spec, SPLITS["train"])
    per_step, size = spec.prompts_per_step, spec.num_generations
    resolved_seed = resolved_seed or (lambda seed, i: seed)
    episodes, groups = [], []
    for step in steps:
        for g in range(per_step):
            row = rows[step * per_step + g]
            true = [bool(o) for o in outcomes(step, g)]
            spurious = all(true) and g < spurious_groups
            observed = [1.0] * (size - 1) + [0.0] if spurious else [float(o) for o in true]
            mean = sum(observed) / size
            groups.append({"step": step, "attempt": attempt, "group": g, "task_id": row["task_id"], "condition": row["condition"],
                           "schedule_seed": row["schedule_seed"], "true_success": true, "observed_reward": observed,
                           "advantages": [o - mean for o in observed], "zero_variance": len(set(observed)) == 1,
                           "spurious": spurious, "max_abs_advantage": max(abs(o - mean) for o in observed)})
            for i, o in enumerate(true):
                episodes.append(rec(f"train:step{step}", row["condition"], row["task_id"], o, schedule_seed=row["schedule_seed"],
                                    resolved_seed=resolved_seed(row["schedule_seed"], i), slot=i, episode_index=step, attempt=attempt))
    return episodes, groups


def manifest(spec, status="COMPLETE", attempt=1, commit="abc1234", **kw):
    base = {"run_id": spec.run_id, "status": status, "attempt": attempt, "git_commit": commit, "deployed_commit": commit,
            "wall_time_s": 1.0, "estimated_cost_usd": 0.1, "spec": spec.to_dict(), "step_timings": []}
    base.update(kw)
    return base


def write_attempt(directory, manifest_dict, episodes, groups):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_manifest.json").write_text(json.dumps(manifest_dict))
    (directory / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in episodes) + "\n")
    (directory / "groups.jsonl").write_text("\n".join(json.dumps(g) for g in groups) + "\n")


def complete_run(tmp_path, spec, status="COMPLETE", periodic=None, final=None, diagnostic=None, outcomes=None,
                 resolved_seed=None, spurious_groups=0, commit="abc1234"):
    """A whole run with the exact expected identities and configurable outcomes (defaults: alternating)."""
    periodic = periodic or (lambda name, step, ti, k: (ti + k) % 2 == 0)
    final = final or (lambda name, ti, k: (ti + k) % 2 == 0)
    diagnostic = diagnostic or (lambda step, ti, kk, m: (ti + kk + m) % 2 == 0)
    outcomes = outcomes or (lambda step, g: [(step + g + i) % 3 != 0 for i in range(spec.num_generations)])
    diag_steps = [s for s in spec.diagnostic_steps if s == 0 or not spec.eval_only]
    episodes = periodic_records(spec, periodic_steps(spec), periodic) + final_records(spec, final)
    episodes += diagnostic_records(spec, diag_steps, diagnostic)
    train, groups = training_records(spec, range(spec.steps) if not spec.eval_only else [], outcomes, resolved_seed, spurious_groups)
    run_dir = tmp_path / spec.run_id
    write_attempt(run_dir, manifest(spec, status=status, commit=commit), episodes + train, groups)
    return run_dir


def small_spec(**kw) -> RunSpec:
    base = {"run_id": "run-l", "model": "m", "condition": "C2", "arm": "paired", "p": 0.25, "seed": 0, "steps": 4,
            "prompts_per_step": 2, "num_generations": 2, "eval_every": 2, "checkpoint_steps": 2, "eval_tasks": 2,
            "final_eval_tasks": 2, "diagnostic_steps": [0, 4], "diagnostic_schedules": 2, "diagnostic_samples": 2}
    base.update(kw)
    return RunSpec(**base)

"""The pre-registered decision rules of docs/PREREGISTRATION.md (H1a, H2a, H2b, H3, H5) computed from run
directories along the attempt lineage, with the section-8 bootstrap: seeds are resampled with replacement as
(paired s, independent s) pairs, tasks are resampled with replacement within an evaluation set with all of a
task's schedules kept together, and one task resample is applied to every checkpoint, both arms and every seed
of a replicate. Episodes are never pooled as independent samples. Every rule is applied exactly as written and
its outcome recorded; a miss is reported, never relaxed."""

import json
import pathlib
from collections import defaultdict

import numpy as np

from pairedrl.analysis.curves import _phase_step, lineage_records, load_attempts
from pairedrl.analysis.diagnostics import luck_share
from pairedrl.train.runner import RunSpec, luck_share_tables_by_phase, periodic_steps

H2_MEAN_MARGIN = 0.03
H3_MARGIN = 0.05
H3_STEPS = (40, 60, 80)
H5_MARGIN = -0.03
H1A_LAMBDA_MIN = 0.15
H1A_LAMBDA_LOWER = 0.05
H1A_SPURIOUS_RATE = 1 - 0.9**8 - 0.1**8
H1A_SPURIOUS_TOLERANCE = 0.07
H1A_PAIRED_SPURIOUS_MAX = 0.02
H2_SETS = {"C2": ("noisy", "noisy_p025"), "C4": ("clean", "clean")}


class RunData:
    """One run's evidence, keyed for task-level resampling: periodic[set][step][task] and final[set][task] are
    lists of true-success outcomes (every schedule of the task together)."""

    def __init__(self, run_dir):
        self.run_dir = pathlib.Path(run_dir)
        attempts = load_attempts(self.run_dir)
        self.manifest = attempts[-1]["manifest"]
        self.spec = RunSpec.from_dict(self.manifest["spec"])
        self.records, _ = lineage_records(attempts, "episodes.jsonl")
        self.groups, _ = lineage_records(attempts, "groups.jsonl")
        self.periodic: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.final: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in self.records:
            phase = r.get("phase", "")
            if phase.startswith("eval_"):
                name = phase[len("eval_"):].split(":", 1)[0]
                self.periodic[name][_phase_step(phase)][r["task_id"]].append(float(r["true_success"]))
            elif phase.startswith("final_"):
                name = phase[len("final_"):].split(":", 1)[0]
                self.final[name][r["task_id"]].append(float(r["true_success"]))
        self.diag_tables = luck_share_tables_by_phase(self.records)

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    def periodic_tasks(self, set_name: str) -> list[str]:
        steps = periodic_steps(self.spec)
        tasks = None
        for step in steps:
            present = set(self.periodic[set_name][step])
            tasks = present if tasks is None else tasks & present
        return sorted(tasks or [])

    def task_means(self, set_name: str, step: int) -> dict[str, float]:
        return {t: float(np.mean(v)) for t, v in self.periodic[set_name][step].items()}

    def final_task_means(self, set_name: str) -> dict[str, float]:
        return {t: float(np.mean(v)) for t, v in self.final[set_name].items()}


def trapezoid(steps: list[int], values: np.ndarray) -> np.ndarray:
    """Normalized trapezoidal area over `steps` for values of shape (..., len(steps))."""
    steps_arr = np.asarray(steps, dtype=float)
    widths = np.diff(steps_arr)
    return ((values[..., :-1] + values[..., 1:]) / 2 * widths).sum(axis=-1) / (steps_arr[-1] - steps_arr[0])


def curve_matrix(run: RunData, set_name: str, tasks: list[str]) -> np.ndarray:
    """Task-mean true success, shape (steps, tasks), in the registered step order."""
    steps = periodic_steps(run.spec)
    return np.array([[run.task_means(set_name, step)[t] for t in tasks] for step in steps])


def final_vector(run: RunData, set_name: str, tasks: list[str]) -> np.ndarray:
    means = run.final_task_means(set_name)
    return np.array([means[t] for t in tasks])


def paired_bootstrap(per_seed_values: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Given per-seed, per-task statistics of shape (seeds, tasks) whose mean over tasks then seeds is the point
    estimate, return `n_boot` replicate means with seeds and tasks resampled with replacement."""
    seeds, tasks = per_seed_values.shape
    seed_idx = rng.integers(0, seeds, size=(n_boot, seeds))
    task_idx = rng.integers(0, tasks, size=(n_boot, tasks))
    out = np.empty(n_boot)
    for b in range(n_boot):
        sub = per_seed_values[np.ix_(seed_idx[b], task_idx[b])]
        out[b] = sub.mean()
    return out


def h2(pairs: list[tuple[RunData, RunData]], condition: str, n_boot: int = 10000, seed: int = 0) -> dict:
    """H2a (C2, at-training-noise validation AUC) or H2b (C4, clean validation AUC): mean over seeds of the
    paired-minus-independent AUC difference at least 0.03, every seed-level difference positive, and the final test
    direction not contradicting (paired mean not below the independent mean)."""
    set_name, final_name = H2_SETS[condition]
    rng = np.random.default_rng(seed)
    steps = periodic_steps(pairs[0][0].spec)
    tasks = sorted(set.intersection(*[set(r.periodic_tasks(set_name)) for pr in pairs for r in pr]))
    curves_p = np.array([curve_matrix(p, set_name, tasks) for p, _ in pairs])
    curves_i = np.array([curve_matrix(i, set_name, tasks) for _, i in pairs])
    auc_p = trapezoid(steps, curves_p.mean(axis=2))
    auc_i = trapezoid(steps, curves_i.mean(axis=2))
    diffs = auc_p - auc_i
    final_tasks = sorted(set.intersection(*[set(r.final[final_name]) for pr in pairs for r in pr]))
    fin_p = np.array([final_vector(p, final_name, final_tasks).mean() for p, _ in pairs])
    fin_i = np.array([final_vector(i, final_name, final_tasks).mean() for _, i in pairs])
    n_seeds, n_tasks = len(pairs), len(tasks)
    seed_idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    task_idx = rng.integers(0, n_tasks, size=(n_boot, n_tasks))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        cp = curves_p[seed_idx[b]][:, :, task_idx[b]].mean(axis=2)
        ci = curves_i[seed_idx[b]][:, :, task_idx[b]].mean(axis=2)
        boot[b] = (trapezoid(steps, cp) - trapezoid(steps, ci)).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    passes = bool(diffs.mean() >= H2_MEAN_MARGIN and np.all(diffs > 0) and fin_p.mean() >= fin_i.mean())
    return {
        "hypothesis": "H2a" if condition == "C2" else "H2b",
        "condition": condition,
        "validation_set": set_name,
        "final_set": final_name,
        "steps": steps,
        "tasks": n_tasks,
        "seeds": [p.spec.seed for p, _ in pairs],
        "runs": [[p.run_id, i.run_id] for p, i in pairs],
        "auc_paired": [round(float(x), 4) for x in auc_p],
        "auc_independent": [round(float(x), 4) for x in auc_i],
        "auc_difference_by_seed": [round(float(x), 4) for x in diffs],
        "auc_difference_mean": round(float(diffs.mean()), 4),
        "auc_difference_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "final_paired_by_seed": [round(float(x), 4) for x in fin_p],
        "final_independent_by_seed": [round(float(x), 4) for x in fin_i],
        "final_difference_mean": round(float(fin_p.mean() - fin_i.mean()), 4),
        "rule": {
            "mean_at_least": H2_MEAN_MARGIN,
            "mean_criterion": bool(diffs.mean() >= H2_MEAN_MARGIN),
            "every_seed_positive": bool(np.all(diffs > 0)),
            "final_direction": bool(fin_p.mean() >= fin_i.mean()),
        },
        "passes": passes,
        "bootstrap": {"resamples": n_boot, "seed": seed},
    }


def h3(pairs: list[tuple[RunData, RunData]], n_boot: int = 10000, seed: int = 0) -> dict:
    """H3 (C2): on the matched challenge set, paired success exceeds independent success by at least 5 points
    averaged over steps 40, 60 and 80 and over seeds, with every seed-level difference positive."""
    rng = np.random.default_rng(seed)
    for pr in pairs:
        for r in pr:
            missing = [s for s in H3_STEPS if s not in r.periodic["challenge"]]
            if missing:
                raise ValueError(f"{r.run_id}: no challenge evaluation at steps {missing}")
    tasks = sorted(set.intersection(*[set(r.periodic_tasks("challenge")) for pr in pairs for r in pr]))
    mats_p = np.array([[[p.task_means("challenge", s)[t] for t in tasks] for s in H3_STEPS] for p, _ in pairs])
    mats_i = np.array([[[i.task_means("challenge", s)[t] for t in tasks] for s in H3_STEPS] for _, i in pairs])
    diffs = (mats_p - mats_i).mean(axis=(1, 2))
    per_seed_task = (mats_p - mats_i).mean(axis=1)
    boot = paired_bootstrap(per_seed_task, n_boot, rng)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "hypothesis": "H3",
        "steps": list(H3_STEPS),
        "tasks": len(tasks),
        "seeds": [p.spec.seed for p, _ in pairs],
        "difference_by_seed": [round(float(x), 4) for x in diffs],
        "difference_mean": round(float(diffs.mean()), 4),
        "difference_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "rule": {"mean_at_least": H3_MARGIN, "mean_criterion": bool(diffs.mean() >= H3_MARGIN), "every_seed_positive": bool(np.all(diffs > 0))},
        "passes": bool(diffs.mean() >= H3_MARGIN and np.all(diffs > 0)),
        "bootstrap": {"resamples": n_boot, "seed": seed},
    }


def h5(pairs: list[tuple[RunData, RunData]], n_boot: int = 10000, seed: int = 0) -> dict:
    """H5 (C2): on the held-out-fault-type test set the paired arm is non-inferior with margin 3 points: the lower
    bound of the 90 percent bootstrap interval of the paired-minus-independent difference is above -0.03."""
    rng = np.random.default_rng(seed)
    tasks = sorted(set.intersection(*[set(r.final["heldout_types"]) for pr in pairs for r in pr]))
    per_seed_task = np.array([final_vector(p, "heldout_types", tasks) - final_vector(i, "heldout_types", tasks) for p, i in pairs])
    diffs = per_seed_task.mean(axis=1)
    boot = paired_bootstrap(per_seed_task, n_boot, rng)
    lo, hi = np.percentile(boot, [5, 95])
    return {
        "hypothesis": "H5",
        "set": "heldout_types",
        "tasks": len(tasks),
        "seeds": [p.spec.seed for p, _ in pairs],
        "difference_by_seed": [round(float(x), 4) for x in diffs],
        "difference_mean": round(float(diffs.mean()), 4),
        "difference_ci90": [round(float(lo), 4), round(float(hi), 4)],
        "rule": {"lower_bound_above": H5_MARGIN},
        "passes": bool(lo > H5_MARGIN),
        "bootstrap": {"resamples": n_boot, "seed": seed},
    }


def luck_share_bootstrap(tables: list, n_boot: int, rng: np.random.Generator) -> dict:
    lams = np.array([s.lam if s.lam is not None else np.nan for s in (luck_share(t) for t in tables)])
    defined = lams[~np.isnan(lams)]
    if len(defined) == 0:
        return {"lam": None, "ci95": None, "tasks": len(tables), "tasks_defined": 0}
    idx = rng.integers(0, len(lams), size=(n_boot, len(lams)))
    boot = np.array([np.nanmean(lams[row]) if not np.all(np.isnan(lams[row])) else np.nan for row in idx])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"lam": round(float(defined.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "tasks": len(tables), "tasks_defined": len(defined)}


def group_seed_sharing(run: RunData) -> dict:
    """Implementation check of H1a: from the training episode records, the distinct resolved schedule seeds within
    each training group (step, task, schedule seed); paired groups share exactly one, independent groups never."""
    by_group: dict[tuple, set] = defaultdict(set)
    for r in run.records:
        if r.get("phase", "").startswith("train:") and r.get("resolved_seed") is not None:
            by_group[(r["phase"], r["task_id"], r["schedule_seed"])].add(r["resolved_seed"])
    counts = [len(v) for v in by_group.values()]
    return {
        "groups": len(counts),
        "groups_sharing_one_seed": sum(1 for c in counts if c == 1),
        "groups_with_several_seeds": sum(1 for c in counts if c > 1),
    }


def h1a(runs: list[RunData], condition: str, n_boot: int = 10000, seed: int = 0) -> dict:
    """H1a counts. C4: the independent arm's spurious-variance rate among all-correct training groups within 0.07
    of 0.57 and the paired arm's at most 0.02, over all steps and seeds. C2: zero-shot luck share at step 0 at least
    0.15 with a 95 percent bootstrap lower bound above 0.05, and every paired training group shares one resolved
    schedule seed while no independent group does."""
    rng = np.random.default_rng(seed)
    out = {"hypothesis": "H1a", "condition": condition, "runs": {}}
    if condition == "C4":
        rates = {}
        for arm in ("paired", "independent"):
            all_correct = [g for r in runs if r.spec.arm == arm for g in r.groups if all(g["true_success"])]
            spurious = sum(1 for g in all_correct if g.get("spurious"))
            rates[arm] = {"all_correct_groups": len(all_correct), "spurious": spurious, "rate": (spurious / len(all_correct)) if all_correct else None}
        ind, par = rates["independent"]["rate"], rates["paired"]["rate"]
        out["spurious_rates"] = rates
        out["expected_independent_rate"] = round(H1A_SPURIOUS_RATE, 4)
        out["rule"] = {
            "independent_within": H1A_SPURIOUS_TOLERANCE,
            "independent_criterion": None if ind is None else bool(abs(ind - H1A_SPURIOUS_RATE) <= H1A_SPURIOUS_TOLERANCE),
            "paired_at_most": H1A_PAIRED_SPURIOUS_MAX,
            "paired_criterion": None if par is None else bool(par <= H1A_PAIRED_SPURIOUS_MAX),
        }
        out["passes"] = bool(out["rule"]["independent_criterion"] and out["rule"]["paired_criterion"])
        return out
    checks = []
    for r in runs:
        tables = r.diag_tables.get("diag:step0", [])
        share = luck_share_bootstrap(tables, n_boot, rng)
        sharing = group_seed_sharing(r)
        share_ok = share["lam"] is not None and share["lam"] >= H1A_LAMBDA_MIN and share["ci95"][0] > H1A_LAMBDA_LOWER
        if r.spec.arm == "paired":
            sharing_ok = sharing["groups"] > 0 and sharing["groups_with_several_seeds"] == 0
        else:
            sharing_ok = sharing["groups"] > 0 and sharing["groups_sharing_one_seed"] == 0
        out["runs"][r.run_id] = {"arm": r.spec.arm, "luck_share_step0": share, "seed_sharing": sharing, "luck_criterion": bool(share_ok), "sharing_criterion": bool(sharing_ok)}
        checks.append(bool(share_ok and sharing_ok))
    out["rule"] = {"lambda_at_least": H1A_LAMBDA_MIN, "lower_bound_above": H1A_LAMBDA_LOWER}
    out["passes"] = bool(checks and all(checks))
    return out


def pair_runs(runs: list[RunData]) -> list[tuple[RunData, RunData]]:
    paired = {r.spec.seed: r for r in runs if r.spec.arm == "paired"}
    independent = {r.spec.seed: r for r in runs if r.spec.arm == "independent"}
    seeds = sorted(set(paired) & set(independent))
    missing = sorted((set(paired) | set(independent)) - set(seeds))
    if missing:
        raise ValueError(f"seeds without both arms: {missing}")
    return [(paired[s], independent[s]) for s in seeds]


def decide(condition: str, run_dirs: list, n_boot: int = 10000, seed: int = 0, data_dir=None) -> dict:
    """Every registered rule that applies to `condition`, from run directories that pass the acceptance check
    (COMPLETE, complete against the frozen pools, consistent across attempts), in one register."""
    from pairedrl.analysis.curves import run_register

    accepted = []
    acceptance = {}
    pools = None
    for d in run_dirs:
        entry = run_register(d, data_dir=data_dir)
        consistency = entry["attempt_consistency"]
        consistent = consistency["commits_consistent"] and consistency["model_consistent"] and consistency["task_pools_consistent"]
        if entry["status"] != "COMPLETE" or not entry["completeness"]["complete"] or not consistent:
            raise ValueError(
                f"{d}: not accepted (status {entry['status']}, complete {entry['completeness']['complete']}, consistent {consistent})"
            )
        acceptance[entry["run_id"]] = {"attempts": len(entry["attempts"]), "commits": consistency["commits"],
                                       "episodes": entry["completeness"]["episodes_found"]}
        pools = entry["completeness"]["task_pools"]
        accepted.append(RunData(d))
    if any(r.spec.condition != condition for r in accepted):
        raise ValueError("every run must belong to the condition")
    pairs = pair_runs(accepted)
    out = {
        "condition": condition,
        "runs": [r.run_id for r in accepted],
        "acceptance": acceptance,
        "task_pools": {name: v["sha256"] for name, v in (pools or {}).items()},
        "seed_pairs": [[p.run_id, i.run_id] for p, i in pairs],
        "h1a": h1a(accepted, condition, n_boot, seed),
        "h2": h2(pairs, condition, n_boot, seed),
    }
    if condition == "C2":
        out["h3"] = h3(pairs, n_boot, seed)
        out["h5"] = h5(pairs, n_boot, seed)
    return out


def format_decision(d: dict) -> str:
    lines = [f"condition {d['condition']}: pairs {d['seed_pairs']}"]
    h = d["h2"]
    lines.append(
        f"  {h['hypothesis']} ({h['validation_set']} AUC): paired {h['auc_paired']} independent {h['auc_independent']} "
        f"diff by seed {h['auc_difference_by_seed']} mean {h['auc_difference_mean']} ci95 {h['auc_difference_ci95']}; "
        f"final {h['final_set']} diff {h['final_difference_mean']}; rule {h['rule']}; passes {h['passes']}"
    )
    a = d["h1a"]
    if d["condition"] == "C4":
        lines.append(f"  H1a: spurious rates {a['spurious_rates']} expected {a['expected_independent_rate']}; rule {a['rule']}; passes {a['passes']}")
    else:
        for run_id, v in a["runs"].items():
            lines.append(f"  H1a {run_id}: lambda0 {v['luck_share_step0']} sharing {v['seed_sharing']} luck {v['luck_criterion']} sharing {v['sharing_criterion']}")
        lines.append(f"  H1a passes {a['passes']}")
    if "h3" in d:
        h = d["h3"]
        lines.append(f"  H3 (challenge 40/60/80): diff by seed {h['difference_by_seed']} mean {h['difference_mean']} ci95 {h['difference_ci95']}; rule {h['rule']}; passes {h['passes']}")
    if "h5" in d:
        h = d["h5"]
        lines.append(f"  H5 (heldout types): diff by seed {h['difference_by_seed']} mean {h['difference_mean']} ci90 {h['difference_ci90']}; passes {h['passes']}")
    return "\n".join(lines)


def write_decision(d: dict, path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n", encoding="utf-8")

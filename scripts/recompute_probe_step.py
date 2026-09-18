"""Recompute a probe step's summary from its gzipped evidence file, without a GPU, and check it against step<N>.json.

    python scripts/recompute_probe_step.py outputs/runs/<probe_id>/step<N>.json

The Gram-derived terms are recomputed from scratch (the fp64 Gram matrices, rewards, rho, index lists); the
running-vector norms and cross terms are read from the evidence, since per-task Grams cannot rebuild cross-task
inner products. Every number in the summary must match within a small tolerance or the exit code is 1.
"""

import argparse
import gzip
import json
import math
import pathlib
import sys
from collections import defaultdict

import numpy as np

from pairedrl.analysis import probe as pm


def _outcome(evidence) -> dict:
    q, t_ref, od = evidence["q"], evidence["t_ref"], pm.OUTCOME_DESIGNS
    sum_sq = {e: {d: 0.0 for d in od} for e in pm.ESTIMATORS}
    within = {e: {d: 0.0 for d in od} for e in pm.ESTIMATORS}
    reward = {e: {d: 0.0 for d in od} for e in pm.ESTIMATORS}
    by_k = defaultdict(lambda: {"count": 0, "lhs_sum": 0.0, "rhs_sum": 0.0, "ratio_sum": 0.0})
    lhs_sum = rhs_sum = 0.0
    n_groups = 0
    for grp in evidence["outcome"]["groups"]:
        gram, true, rho = np.array(grp["gram"]), np.array(grp["true"]), np.array(grp["rho"])
        res = pm.outcome_noise_group(gram, true, rho, t_ref, q)
        lhs_sum += res["lhs"]
        rhs_sum += res["rhs"]
        n_groups += 1
        k = res["k_success"]
        b = by_k[k]
        b["count"] += 1
        b["lhs_sum"] += res["lhs"]
        b["rhs_sum"] += res["rhs"]
        b["ratio_sum"] += res["lhs"] / res["rhs"] if res["rhs"] > 0 else 0.0
        for e in pm.ESTIMATORS:
            for d in od:
                blk = res[e][d]
                sum_sq[e][d] += blk["expected_sq_norm"]
                within[e][d] += blk["expected_sq_norm"] - pm.quadratic(np.array(blk["expected_weights"]), gram)
                reward[e][d] += blk["expected_contrast_variance"]
    running = evidence["outcome"]["running"]
    block = {}
    for e in pm.ESTIMATORS:
        blk = {"cross_vec": running[e]["cross_vec"]}
        for d in od:
            blk[d] = {"sum_sq_norm": sum_sq[e][d], "vec_sq_norm": running[e][d], "n_groups": n_groups,
                      "n_within": n_groups, "within_sum": within[e][d], "reward_variance": reward[e][d] / max(n_groups, 1)}
        block[e] = blk
    block.update({"n_groups": n_groups, "lhs_sum": lhs_sum, "rhs_sum": rhs_sum, "by_k": dict(by_k)})
    return block


def _transition(evidence) -> dict:
    t_ref, td = evidence["t_ref"], pm.TRANSITION_DESIGNS
    sum_sq = {e: {d: 0.0 for d in td} for e in pm.ESTIMATORS}
    within = {e: {d: 0.0 for d in td} for e in pm.ESTIMATORS}
    reward = {e: {d: 0.0 for d in td} for e in pm.ESTIMATORS}
    n_g = {e: {d: 0 for d in td} for e in pm.ESTIMATORS}
    n_tasks = 0
    for task in evidence["transition"]["tasks"]:
        gram, rewards, rho = np.array(task["gram"]), np.array(task["rewards"]), np.array(task["rho"])
        n_tasks += 1
        for d in td:
            terms = pm.group_terms(gram, rewards, rho, t_ref, task["index_lists"][d])
            for e in pm.ESTIMATORS:
                blk = terms[e]
                sum_sq[e][d] += blk["sum_sq_norm"]
                within[e][d] += pm.trace_variance(blk["sum_sq_norm"], pm.quadratic(blk["summed_weights"], gram), blk["n_groups"])
                reward[e][d] += 2.0 * blk["reward_var_sum"] / max(blk["n_groups"], 1)
                n_g[e][d] += blk["n_groups"]
    running = evidence["transition"]["running"]
    block = {}
    for e in pm.ESTIMATORS:
        blk = {"cross_vec": running[e]["cross_vec"]}
        for d in td:
            blk[d] = {"sum_sq_norm": sum_sq[e][d], "vec_sq_norm": running[e][d], "n_groups": n_g[e][d],
                      "n_within": n_tasks, "within_sum": within[e][d], "reward_variance": reward[e][d] / max(n_tasks, 1)}
        block[e] = blk
    return block


def recompute_summary(evidence: dict) -> dict:
    accumulators = {"step": evidence["step"], "q": evidence["q"], "outcome": _outcome(evidence),
                    "transition": _transition(evidence)}
    return pm.checkpoint_summary(accumulators)


def compare(reference, recomputed, tol: float, path: str = "") -> list:
    diffs = []
    if isinstance(reference, dict):
        for key in set(reference) | set(recomputed or {}):
            diffs += compare(reference.get(key), (recomputed or {}).get(key), tol, f"{path}.{key}")
    elif isinstance(reference, list):
        if not isinstance(recomputed, list) or len(reference) != len(recomputed):
            diffs.append(f"{path}: list mismatch")
        else:
            for i, (x, y) in enumerate(zip(reference, recomputed)):
                diffs += compare(x, y, tol, f"{path}[{i}]")
    elif isinstance(reference, (int, float)) and isinstance(recomputed, (int, float)):
        if math.isinf(reference) and math.isinf(recomputed):
            return diffs
        if not math.isclose(reference, recomputed, rel_tol=tol, abs_tol=tol):
            diffs.append(f"{path}: {reference} != {recomputed}")
    elif reference != recomputed:
        diffs.append(f"{path}: {reference!r} != {recomputed!r}")
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step_file", help="a probe step<N>.json; its step<N>_evidence.json.gz sits beside it")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    step_file = pathlib.Path(args.step_file)
    payload = json.loads(step_file.read_text(encoding="utf-8"))
    evidence_file = step_file.parent / (step_file.stem + "_evidence.json.gz")
    with gzip.open(evidence_file, "rt", encoding="utf-8") as f:
        evidence = json.load(f)
    recomputed = json.loads(json.dumps(recompute_summary(evidence), sort_keys=True))
    diffs = compare(payload["summary"], recomputed, args.tolerance)
    if diffs:
        print(f"MISMATCH recomputing {step_file.name} from evidence:")
        for d in diffs[:20]:
            print("  " + d)
        return 1
    print(f"recompute ok: {step_file.name} summary reproduced from evidence within {args.tolerance}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

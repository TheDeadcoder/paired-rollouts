"""Gradient-variance probe arithmetic for H1(b) (docs/PREREGISTRATION.md section 3, docs/THEORY.md section 3b,
docs/DEVIATIONS.md amendment A1). Pure numpy. Every quantity is a within-group inner product a^T K a on the Gram
matrix K_ij = S_i . S_j of per-rollout LoRA scores, plus running sums of the group gradients across groups:
Tr Var(g) over n groups = sum ||g||^2 / n - ||sum g||^2 / n^2. No torch, no GPU."""

import itertools

import numpy as np

from pairedrl.analysis.diagnostics import grpo_advantages

OUTCOME_DESIGNS = ("paired", "independent")
TRANSITION_DESIGNS = ("paired", "independent_resampled", "stratified")
ESTIMATORS = ("mean_centered", "implemented")


def mean_centered_weights(rewards) -> np.ndarray:
    """a_i = (R_i - mean(R)) / G, the weights of g = sum_i a_i S_i for the mean-centered estimator (THEORY 3b)."""
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / len(r)


def implemented_weights(rewards, rho, t_ref, scale: str = "group") -> np.ndarray:
    """a_i = grpo_advantages(R, scale) * rho_i / T_ref: TRL's std-normalized advantage times the per-sequence vLLM
    importance-sampling ratio rho_i, divided by one batch-level token normalizer T_ref shared across every group of
    a step (TRL's DAPO num_items_in_batch). The earlier per-group divisor sum(T_i) was wrong (docs/PROBE.md)."""
    a = np.asarray(grpo_advantages(list(rewards), scale), dtype=float)
    return a * np.asarray(rho, dtype=float) / float(t_ref)


def quadratic(weights, gram) -> float:
    """a^T K a = ||sum_i a_i S_i||^2 for the Gram matrix K_ij = S_i . S_j."""
    w = np.asarray(weights, dtype=float)
    return float(w @ np.asarray(gram, dtype=float) @ w)


def _centered_sq_norms(gram: np.ndarray) -> np.ndarray:
    """||S_i - S_bar||^2 for every i, from the Gram matrix (S_bar the mean of all G scores)."""
    g = gram.shape[0]
    row = gram.sum(axis=1)
    total = gram.sum()
    return np.diag(gram) - (2.0 / g) * row + total / (g * g)


def outcome_noise_group(gram, true_success, rho, t_ref, q) -> dict:
    """One clean group under one-sided outcome noise at rate q (THEORY 3b). Observed R_i = Z_i r_i with
    Z_i ~ Bernoulli(1 - q) independent (independent design) or shared as a single Z (paired). The implemented
    weights are grpo_advantages(R) * rho_i / T_ref (rho the per-sequence vLLM importance ratio, T_ref a batch-level
    constant). Returns lhs = ||g_c||^2, rhs = (1/G^2) sum_i r_i ||S_i - S_bar||^2, k_success, and per estimator and
    design the exact expected_sq_norm E||g||^2, expected_weights E a (length G) and expected_contrast_variance
    Var(R_i - R_j) = 2 E[within-group var(R), ddof 1]. Mean-centered by closed forms; implemented by enumeration."""
    k_mat = np.asarray(gram, dtype=float)
    r = np.asarray(true_success, dtype=float)
    g = len(r)
    rho = np.asarray(rho, dtype=float)
    a_c = (r - r.mean()) / g
    lhs = float(a_c @ k_mat @ a_c)
    rhs = float((r * _centered_sq_norms(k_mat)).sum() / (g * g))
    k = round(float(r.sum()))
    out = {"lhs": lhs, "rhs": rhs, "k_success": k}
    zero = np.zeros(g)
    if k == 0:
        for est in ESTIMATORS:
            out[est] = {d: {"expected_sq_norm": 0.0, "expected_weights": zero.tolist(),
                            "expected_contrast_variance": 0.0} for d in OUTCOME_DESIGNS}
        return out
    ea_mc = (1.0 - q) * a_c
    mc = {
        "paired": (1.0 - q) * lhs,
        "independent": (1.0 - q) ** 2 * lhs + q * (1.0 - q) * rhs,
    }
    succ = [i for i in range(g) if r[i] > 0.5]
    impl_sq = {"paired": 0.0, "independent": 0.0}
    impl_ea = {"paired": np.zeros(g), "independent": np.zeros(g)}
    contrast = {"paired": 0.0, "independent": 0.0}
    for kept in itertools.product((1, 0), repeat=k):
        n_kept = sum(kept)
        prob = (1.0 - q) ** n_kept * q ** (k - n_kept)
        obs = np.zeros(g)
        for idx, keep in zip(succ, kept):
            obs[idx] = float(keep)
        a = np.asarray(grpo_advantages(obs.tolist(), "group")) * rho / float(t_ref)
        impl_sq["independent"] += prob * float(a @ k_mat @ a)
        impl_ea["independent"] += prob * a
        contrast["independent"] += prob * 2.0 * float(np.var(obs, ddof=1))
    for keep_all, prob in ((True, 1.0 - q), (False, q)):
        obs = r.copy() if keep_all else np.zeros(g)
        a = np.asarray(grpo_advantages(obs.tolist(), "group")) * rho / float(t_ref)
        impl_sq["paired"] += prob * float(a @ k_mat @ a)
        impl_ea["paired"] += prob * a
        contrast["paired"] += prob * 2.0 * float(np.var(obs, ddof=1))
    out["mean_centered"] = {
        d: {"expected_sq_norm": mc[d], "expected_weights": ea_mc.tolist(),
            "expected_contrast_variance": contrast[d]} for d in OUTCOME_DESIGNS}
    out["implemented"] = {
        d: {"expected_sq_norm": impl_sq[d], "expected_weights": impl_ea[d].tolist(),
            "expected_contrast_variance": contrast[d]} for d in OUTCOME_DESIGNS}
    return out


def transition_designs(k_schedules, m_samples, resamples, rng) -> dict:
    """Index lists (rollout index = k*M + m) into a task's K*M rollouts for three group designs (amendment A1):
    paired = the M rollouts of each schedule (K exact groups); independent_resampled = R groups, each formed by
    drawing M schedule labels with replacement and then, per drawn schedule, that many distinct rollouts without
    replacement; stratified = R groups, each one rollout drawn from every schedule."""
    paired = [[k * m_samples + m for m in range(m_samples)] for k in range(k_schedules)]
    independent = []
    for _ in range(resamples):
        labels = rng.integers(0, k_schedules, size=m_samples)
        group = []
        for k in range(k_schedules):
            j = int(np.count_nonzero(labels == k))
            if j:
                chosen = rng.choice(m_samples, size=j, replace=False)
                group.extend(int(k * m_samples + c) for c in chosen)
        independent.append(group)
    stratified = [[int(k * m_samples + rng.integers(0, m_samples)) for k in range(k_schedules)]
                  for _ in range(resamples)]
    return {"paired": paired, "independent_resampled": independent, "stratified": stratified}


def group_terms(gram, rewards, rho, t_ref, index_lists) -> dict:
    """For each estimator, over the given groups of one task: the sum of ||g||^2, the summed weights placed back on
    the task's K*M rollout axis (so the caller forms sum_i w_i S_i once per task and design), the group count and
    the summed within-group reward variance (ddof 1). Implemented weights are grpo_advantages(R) * rho_i / T_ref."""
    k_mat = np.asarray(gram, dtype=float)
    r = np.asarray(rewards, dtype=float)
    rho = np.asarray(rho, dtype=float)
    n = len(r)
    reward_var_sum = float(sum(np.var(r[list(grp)], ddof=1) for grp in index_lists))
    out = {}
    for est in ESTIMATORS:
        ssn = 0.0
        summed = np.zeros(n)
        for grp in index_lists:
            idx = np.asarray(grp)
            obs = r[idx]
            if est == "mean_centered":
                w = (obs - obs.mean()) / len(idx)
            else:
                w = np.asarray(grpo_advantages(obs.tolist(), "group")) * rho[idx] / float(t_ref)
            sub = k_mat[np.ix_(idx, idx)]
            ssn += float(w @ sub @ w)
            summed[idx] += w
        out[est] = {"sum_sq_norm": ssn, "summed_weights": summed,
                    "n_groups": len(index_lists), "reward_var_sum": reward_var_sum}
    return out


def trace_variance(sum_sq_norm, sum_vector_sq_norm, n) -> float:
    """Tr Var(g) over n groups = E||g||^2 - ||E g||^2 = sum||g||^2 / n - ||sum g||^2 / n^2."""
    return sum_sq_norm / n - sum_vector_sq_norm / (n * n)


def _summarize_block(block: dict, designs, primary_independent: str, q=None, with_identity=False) -> dict:
    """One (noise, estimator) block. Per design: the pooled trace_var over all groups, trace_var_within (the mean
    over tasks of the within-task variance), trace_var_between = trace_var - trace_var_within, and the expected
    update norm. The registered rule uses the pooled paired/independent ratio; the within ratio is reported too,
    with the cosine between the paired and primary-independent expected gradients, the reward variances and, for the
    mean-centered outcome estimator, the exact identity residual (must be ~0)."""
    trace, trace_within, trace_between, update_norm = {}, {}, {}, {}
    for d in designs:
        acc = block[d]
        trace[d] = trace_variance(acc["sum_sq_norm"], acc["vec_sq_norm"], acc["n_groups"])
        trace_within[d] = acc["within_sum"] / acc["n_within"] if acc["n_within"] else 0.0
        trace_between[d] = trace[d] - trace_within[d]
        update_norm[d] = float(np.sqrt(acc["vec_sq_norm"])) / acc["n_groups"]
    ind, ind_within = trace[primary_independent], trace_within[primary_independent]
    denom = np.sqrt(block["paired"]["vec_sq_norm"] * block[primary_independent]["vec_sq_norm"])
    summary = {
        "trace_var": trace,
        "trace_var_within": trace_within,
        "trace_var_between": trace_between,
        "ratio_paired_over_independent": trace["paired"] / ind if ind != 0 else float("inf"),
        "ratio_within_paired_over_independent": trace_within["paired"] / ind_within if ind_within != 0 else float("inf"),
        "expected_update_norm": update_norm,
        "cosine_paired_independent": float(block["cross_vec"] / denom) if denom > 0 else 0.0,
        "reward_variance": {d: block[d]["reward_variance"] for d in designs},
    }
    if with_identity and q is not None:
        summary["identity_residual"] = trace["paired"] - trace["independent"] - q * (1.0 - q) * (
            block["lhs_mean"] - block["rhs_mean"])
    return summary


def checkpoint_summary(accumulators: dict) -> dict:
    """Reported numbers for one checkpoint from accumulated scalars and vector norms. Each (noise, estimator)
    per-design block carries sum_sq_norm, vec_sq_norm (||sum g||^2), n_groups (pooled), within_sum and n_within (the
    within-task decomposition), reward_variance, plus a block-level cross_vec (paired . primary-independent running
    sums). Outcome also carries n_groups, lhs_sum, rhs_sum and by_k (per success count k: count, lhs_sum, rhs_sum,
    ratio_sum). Adds lhs_mean, rhs_mean, lhs_over_rhs, the per-k table and the mean-centered identity residual."""
    q = accumulators["q"]
    out = {"step": accumulators.get("step")}
    oc = accumulators["outcome"]
    n_oc = oc["n_groups"]
    lhs_mean, rhs_mean = oc["lhs_sum"] / n_oc, oc["rhs_sum"] / n_oc
    by_k = {}
    for k, s in sorted(oc.get("by_k", {}).items()):
        c = s["count"]
        by_k[int(k)] = {"count": c, "lhs_mean": s["lhs_sum"] / c, "rhs_mean": s["rhs_sum"] / c,
                        "lhs_over_rhs": s["ratio_sum"] / c}
    out["outcome"] = {"lhs_mean": lhs_mean, "rhs_mean": rhs_mean,
                      "lhs_over_rhs": lhs_mean / rhs_mean if rhs_mean != 0 else float("inf"), "by_k": by_k}
    for est in ESTIMATORS:
        block = dict(oc[est])
        block["lhs_mean"], block["rhs_mean"] = lhs_mean, rhs_mean
        out["outcome"][est] = _summarize_block(
            block, OUTCOME_DESIGNS, "independent", q=q, with_identity=(est == "mean_centered"))
    tr = accumulators["transition"]
    out["transition"] = {est: _summarize_block(tr[est], TRANSITION_DESIGNS, "independent_resampled")
                         for est in ESTIMATORS}
    return out


def decide_trajectory(steps: dict) -> dict:
    """H1(b) and P13 (amendment A1) from step -> checkpoint_summary. Outcome noise passes if the mean-centered
    Tr Var(g_paired) < Tr Var(g_indep) at every checkpoint; transition noise if the resampled-independent design
    has Tr Var(g_paired) < Tr Var(g_indep) at a majority of checkpoints; H1(b) passes if both. P13: base-model
    lhs_mean <= rhs_mean / 2; the outcome ratio lower at the last checkpoint than at the base; the transition
    ratio below 1 at a majority of checkpoints."""
    ordered = sorted(steps)
    outcome_ratio = {}
    transition_ratio = {}
    outcome_flags = []
    transition_flags = []
    for s in ordered:
        cp = steps[s]
        o = cp["outcome"]["mean_centered"]["trace_var"]
        outcome_ratio[s] = o["paired"] / o["independent"] if o["independent"] != 0 else float("inf")
        outcome_flags.append(o["paired"] < o["independent"])
        t = cp["transition"]["mean_centered"]["trace_var"]
        transition_ratio[s] = (t["paired"] / t["independent_resampled"]
                               if t["independent_resampled"] != 0 else float("inf"))
        transition_flags.append(t["paired"] < t["independent_resampled"])
    outcome_every = all(outcome_flags)
    transition_majority = sum(transition_flags) > len(transition_flags) / 2
    p13 = {}
    if 0 in steps:
        base = steps[0]["outcome"]
        p13["base_lhs_at_most_half_rhs"] = bool(base["lhs_mean"] <= base["rhs_mean"] / 2)
        p13["outcome_ratio_falls"] = bool(outcome_ratio[ordered[-1]] < outcome_ratio[0])
    else:
        p13["base_lhs_at_most_half_rhs"] = None
        p13["outcome_ratio_falls"] = None
    p13["transition_ratio_below_one_majority"] = bool(
        sum(v < 1 for v in transition_ratio.values()) > len(transition_ratio) / 2)
    return {
        "outcome_noise_every_checkpoint": bool(outcome_every),
        "transition_noise_majority": bool(transition_majority),
        "passes": bool(outcome_every and transition_majority),
        "p13": p13,
        "ratios_by_step": {"outcome": outcome_ratio, "transition_resampled": transition_ratio},
    }


def bandit_enumeration(theta: float = 0.0, group_size: int = 8, q: float = 0.1) -> dict:
    """THEORY 3b's aligned-scalar bandit: a_i ~ Bernoulli(sigmoid(theta)), r_i = a_i, scalar score S_i = a_i - 1/2,
    one group of `group_size` rollouts, outcome noise q. Enumerate all 2^G action tuples; each tuple's numbers come
    from `outcome_noise_group` on the scalar Gram K_ij = S_i S_j with rho_i = 1 and T_ref = G; accumulate over the
    tuple ensemble Tr Var(g) = E||g||^2 - ||E g||^2, the expected-update norm ||E g|| and the contrast variance, per
    estimator and design. Reproduces THEORY's exact targets (contrast 0.495 / 0.450; Tr Var 0.002615 / 0.005845
    mean-centered, 0.007053 / 0.019724 implemented; updates 0.3918 / 0.3905; lhs 0.0496, rhs 0.0137)."""
    g = group_size
    p = 1.0 / (1.0 + np.exp(-theta))
    sum_sq = {e: {d: 0.0 for d in OUTCOME_DESIGNS} for e in ESTIMATORS}
    sum_eg = {e: {d: 0.0 for d in OUTCOME_DESIGNS} for e in ESTIMATORS}
    contrast = {d: 0.0 for d in OUTCOME_DESIGNS}
    lhs_mean = 0.0
    rhs_mean = 0.0
    rho = np.ones(g)
    for actions in itertools.product((0, 1), repeat=g):
        a_arr = np.asarray(actions, dtype=float)
        n_ones = int(a_arr.sum())
        prob = p ** n_ones * (1.0 - p) ** (g - n_ones)
        scores = a_arr - 0.5
        gram = np.outer(scores, scores)
        grp = outcome_noise_group(gram, a_arr, rho, g, q)
        lhs_mean += prob * grp["lhs"]
        rhs_mean += prob * grp["rhs"]
        for e in ESTIMATORS:
            for d in OUTCOME_DESIGNS:
                blk = grp[e][d]
                sum_sq[e][d] += prob * blk["expected_sq_norm"]
                sum_eg[e][d] += prob * float(np.dot(blk["expected_weights"], scores))
        for d in OUTCOME_DESIGNS:
            contrast[d] += prob * grp["mean_centered"][d]["expected_contrast_variance"]
    result = {"lhs_mean": float(lhs_mean), "rhs_mean": float(rhs_mean)}
    for d in OUTCOME_DESIGNS:
        result[f"contrast_variance_{d}"] = float(contrast[d])
    for e in ESTIMATORS:
        for d in OUTCOME_DESIGNS:
            result[f"trace_var_{e}_{d}"] = float(sum_sq[e][d] - sum_eg[e][d] ** 2)
            result[f"expected_update_{e}_{d}"] = float(abs(sum_eg[e][d]))
    return result

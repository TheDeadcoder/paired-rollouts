# Gradient probe (H1(b), amendment A1)

The probe measures, at a frozen checkpoint (the base model or a LoRA checkpoint `trainer/checkpoint-N` of a
trajectory), whether pairing lowers the trace of the policy-gradient covariance under outcome noise and under
transition noise, on the actual LLM policy. It is the gradient half of H1 (`docs/PREREGISTRATION.md` section 3
H1(b), P13; `docs/THEORY.md` section 3b) as amended by `docs/DEVIATIONS.md` 2026-09-08 amendment A1. The
arithmetic is in `src/pairedrl/analysis/probe.py` (pure numpy), the GPU job in `src/pairedrl/train/probe.py`.

## What is measured

For every rollout i the probe forms the score vector

    S_i = sum over completion tokens t of mask_it * grad_theta log pi_theta(y_it | prefix),

over the LoRA parameters only, with the training tool mask. A group gradient is g = sum_i a_i S_i with weights
a_i that depend on the estimator: mean-centered (THEORY 3b) a_i = (R_i - R_bar) / G; implemented (TRL's update)
a_i = A_i / T_group with A_i = `grpo_advantages(R, "group")` and T_group = sum_i T_i the group's completion-token
count. Every reported quantity is a within-group inner product a^T K a on the Gram matrix K_ij = S_i . S_j plus a
running sum of the group gradients across groups:

    Tr Var(g) over n groups = E||g||^2 - ||E g||^2 = (sum ||g||^2) / n - ||sum g||^2 / n^2.

Outcome noise (q = 0.10, exact) uses 64 validation tasks x 8 clean rollouts, one group per task, r_i the true
success. Per group lhs = ||g_c||^2 with g_c = (1/G) sum_i (r_i - r_bar) S_i, and rhs = (1/G^2) sum_i r_i
||S_i - S_bar||^2. The observed reward is R_i = Z_i r_i, Z_i ~ Bernoulli(1 - q) independent, or Z_i = Z shared for
the paired design. The mean-centered expectations are exact closed forms: E||g_paired||^2 = (1 - q) lhs,
E||g_indep||^2 = (1 - q)^2 lhs + q (1 - q) rhs, E g = (1 - q) g_c under both, so
Tr Var(g_paired) - Tr Var(g_indep) = q (1 - q) (lhs - rhs) (checked as an identity residual). The implemented
values are enumerated exactly over the grader masks: all 2^k masks over the k successful rollouts for the
independent design (probability (1 - q)^kept q^flipped), the two masks all-kept and all-flipped for the paired
design (1 - q and q); for each mask A(Z) and a(Z) are recomputed and E||g||^2 and E a accumulated. The
reward-contrast variance Var(R_i - R_j) = 2 E[within-group reward variance, ddof 1] is reported alongside (0.495
independent, 0.450 paired in the aligned-scalar bandit).

Transition noise (p = 0.25) uses 16 diagnostic tasks x K = 8 schedules x M = 8 rollouts (paired schedules), the
observed rewards being the true successes. Per task, from its 64 x 64 Gram matrix, groups of G = 8 are formed under
three designs: paired (the 8 rollouts of one schedule, 8 exact groups); independent resampled (A1, primary: 8
schedule labels drawn with replacement, a label drawn j times contributing j distinct rollouts without replacement
from its schedule; R = 64 resamples, seeded); stratified (the v1 text, secondary: one rollout from each schedule; R
resamples, seeded). Per design and estimator the sum of ||g||^2, the running gradient vector, the group count and
the within-group reward variance accumulate; Tr Var(g), the expected-update norm ||E g|| and the cosine between the
paired and resampled expected gradients are reported.

Decision, per trajectory over its available checkpoints (A1): outcome noise passes if Tr Var(g_paired) <
Tr Var(g_indep) at every checkpoint (mean-centered; implemented reported alongside); transition noise passes at a
majority of checkpoints (resampled independent; stratified alongside). H1(b) passes if both. P13 reports three
booleans: base-model lhs_mean <= rhs_mean / 2; the outcome ratio lower at the last checkpoint than at the base;
the transition ratio below 1 at a majority of checkpoints. Trajectory A is `gate1-c2-paired-s0` at the base model,
80 and 100; trajectory B is `t1-c2-paired-s1` at the base model, 20, 40, 60, 80, 100. The base-model probe is one
computation shared by both trajectories. Nothing is tuned toward a result: both sides of every inequality are
reported whatever their sign.

## How to run

    modal deploy scripts/run_modal.py
    python scripts/launch_modal.py --function probe --specs configs/probe-t1-c2-paired-s1-a.json --label probe-b1 --preregistration v1
    python scripts/collect_modal.py --register registers/launches/<utc>_probe-b1.json --with-weights
    python scripts/build_probe_register.py --probes outputs/runs/probe-t1-c2-paired-s1-a outputs/runs/probe-t1-c2-paired-s1-b \
        --trajectory t1-c2-paired-s1 --base-from outputs/runs/probe-t1-c2-paired-s1-a --out registers/probe/t1-c2-paired-s1.json

`configs/probe-smoke.json` runs the shrunk design end to end in about ten minutes; the register builder refuses it.

## Sizing

Per checkpoint the probe generates about 1,536 rollouts (1,024 diagnostic + 512 clean) and runs the same number of
single-sequence backward passes for the scores, about one hour on an H100. The base model is probed once and shared
by both trajectories.

## Adaptations

Departures from the letter of PREREGISTRATION section 3 H1(b) and DEVIATIONS amendment A1:

- Per-group DAPO normalization. The implemented weight is a_i = A_i / T_group with T_group the group's own
  completion-token count; TRL's DAPO loss divides by the token count of the whole optimizer step, a scalar common
  to both designs at equal token counts, so the per-group divisor changes only the shared scale and not the
  paired-versus-independent comparison. Recorded per A1.
- Both estimators. The mean-centered estimator of THEORY 3b decides; the implemented std-normalized update (DAPO
  aggregation, the training tool mask) is reported alongside, with its expected-update norm and direction and the
  reward-contrast variance; a smaller update norm is not read as a gain (A1).
- No random projection. With G = 8 every quantity is an exact inner product of per-rollout LoRA gradients within a
  group plus running sums across groups, computed in the LoRA parameter space; the v1 seeded 4096-dimensional
  projection is dropped (A1) and no projection-accuracy check is needed.
- Resampling seeds. `transition_designs` draws the resampled and stratified groups from
  `numpy.random.default_rng(spec.seed * 1000 + step)`, one stream per checkpoint, so the group draws are fixed and
  reproducible.
- Base model shared. Step 0 is one probe computation; trajectory A's and B's registers both take it, via
  `build_probe_register --base-from`.
- The probe RunSpec. The trainer is built eval-only at the frozen checkpoint with the trajectory's condition (C2)
  and arm (paired), which `RunSpec.__post_init__` accepts (a non-clean arm with p = 0.25); q is 0 in generation
  because the clean rollouts are generated clean and the grader flips are enumerated exactly afterward, not sampled.
- Score forward. The score reproduces `GRPOTrainer._compute_loss`: input_ids = cat(prompt_ids, completion_ids),
  attention_mask = cat(prompt_mask, completion_mask), logits_to_keep = completion_ids.size(1), the logits divided
  by the sampling temperature inside `_get_per_token_logps_and_entropies`, and the differentiated scalar
  (per_token_logps * mask).sum() with mask = completion_mask * tool_mask, the mask the loss itself uses (tool-result
  tokens excluded); T_i = mask.sum().
- Capture path. The generation output and the environments' `last_episode` records are captured by wrapping the
  trainer instance's `_generate_and_score_completions` (the `training_group_records` pattern), rather than a
  `ProbeTrainer.prediction_step` subclass, so `build_trainer` is reused unchanged; its `PhasedTrainer` only logs
  group records while training, and the probe runs eval-only.
- Gradient checkpointing stays on (inherited from `build_trainer`); the scores are single-sequence backward passes
  (batch size one), so peak memory is bounded regardless.

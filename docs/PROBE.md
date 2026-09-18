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
a_i that depend on the estimator: mean-centered (THEORY 3b) a_i = (R_i - R_bar) / G; implemented (TRL's update at
this commit) a_i = A_i rho_i / T_ref with A_i = `grpo_advantages(R, "group")`, rho_i the per-sequence vLLM
importance-sampling ratio the loss applies, and T_ref one batch-level token normalizer shared across every group of a
step (TRL's DAPO `num_items_in_batch`), not the group's own token total. Every reported quantity is a within-group
inner product a^T K a on the Gram matrix K_ij = S_i . S_j plus a running sum of the group gradients across groups:

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

Departures from the letter of PREREGISTRATION section 3 H1(b) and DEVIATIONS amendment A1, and the corrections that
supersede the earlier text of this file:

- Per-group normalization withdrawn. The earlier version of this file claimed the implemented weight was
  a_i = A_i / T_group with T_group the group's own token total. That was wrong: TRL's DAPO loss divides the whole
  generation batch by one `num_items_in_batch` (the batch's masked-token total), so groups are not reweighted by
  their own token totals, and the loss multiplies each sequence by the vLLM importance-sampling ratio rho_i. The
  implemented estimator is now a_i = A_i rho_i / T_ref with T_ref the mean masked-token total of a
  24-group-by-8-generation training batch (24 x 8 x the mean masked tokens per rollout over the checkpoint's
  rollouts), one constant that cancels from every paired-versus-independent ratio. The review's counterexample
  (numerators (10, 0) and (0, 100), token totals 10 and 100) is a test: the batch normalizer and the per-group one
  give different update directions, which is why the per-group one was dropped.
- Importance ratio. rho_i is read from the captured `output["importance_sampling_ratio"]` (`vllm_importance_sampling_
  correction=True`, mode `sequence_mask`, cap 3; 0 outside the bounds), present in eval mode too. Each step reports
  the fraction of rollouts with rho = 0 and the min, mean, max of the nonzero ratios.
- Both estimators. The mean-centered estimator of THEORY 3b decides; the implemented update (rho, the batch
  normalizer, the training tool mask) is reported alongside with its expected-update norm, direction and the
  reward-contrast variance; a smaller update norm is not read as a gain (A1).
- No random projection. With G = 8 every quantity is an exact inner product of per-rollout LoRA gradients within a
  group plus running sums across groups, computed in the LoRA parameter space; the v1 4096-dimensional projection is
  dropped (A1).
- Score coordinates. `PeftModel.from_pretrained` loads a checkpoint adapter with inference mode, so grad is turned
  on for the `lora_` parameters and off for the rest. The coordinate set is fixed once per checkpoint as the LoRA
  parameters whose gradient is not None after the first rollout's backward (the vision-tower LoRA layers get no
  gradient on text-only rollouts); an always-zero coordinate contributes nothing to any inner product, so this is
  the full-parameter measurement without the structural zeros. Every later rollout must have gradients on exactly
  that set. A sha256 fingerprint of the coordinate parameters is asserted unchanged after scoring, and P and the
  coordinate-name sha256 are written to the step file.
- Base model. Step 0 seeds `transformers.set_seed(spec.seed)` immediately before `build_trainer` and records the
  fingerprint; the step-0 probe is a fixed reference LoRA parameterization of the base policy, not the trajectories'
  own initial adapters, so cross-checkpoint trends of Euclidean traces are coordinate-dependent. Step 0 is one probe
  computation shared by both trajectories via `build_probe_register --base-from`.
- Memory. One rollout is scored at a time into a pinned CPU matrix (n x P, n at most 64); the fp64 Gram and the
  weighted sums are formed by moving row chunks (at most 2 GB fp64) to the GPU; `empty_cache` runs after each task.
  Before any scoring the model is put in train mode and `gradient_checkpointing_enable(use_reentrant=False)` is called
  (on the PeftModel, or its `base_model.model` if the wrapper does not forward it), `use_cache` is set False, and some
  module is asserted to have `gradient_checkpointing` True; the model returns to eval after the fingerprint check.
  Train mode changes only the checkpointing path here (LoRA dropout is 0 and the base has no dropout), and without it
  a single ~8000-token backward would materialize full activations next to vLLM's reservation.
  `torch.cuda.max_memory_allocated` after generation and after scoring is recorded.
- One process per checkpoint. `run_probe` runs each step as a `python -m pairedrl.train.probe` subprocess so a step's
  memory is released before the next; it resumes only a step whose recorded provenance (spec sha, trajectory, adapter
  sha or base fingerprint, git commit, design counts) matches, refuses a mismatch (a repaired rerun needs a new probe
  id), and mirrors `run_job`'s single re-raise so Modal's retry resumes the finished steps.
- Evidence and recompute. Every step writes step<N>.json and a gzipped step<N>_evidence.json.gz atomically (temp then
  rename) with the sufficient statistics (per group and task the fp64 Gram, identities, rewards, masked tokens, rho,
  index lists, and the running-vector norms and cross terms). `scripts/recompute_probe_step.py` reproduces the summary
  from the evidence without a GPU.
- Decomposition and register. `checkpoint_summary` reports, next to the pooled trace, the within-task trace (mean over
  tasks) and the between-task trace, and the outcome groups stratified by success count k; the registered rule stays
  on the pooled quantity. The register merge requires the same coordinate-name sha256 and full design across the
  merged step files (the two halves of trajectory B legitimately have different spec shas; those are recorded per
  step), the trajectory's declared checkpoint set, finite and non-negative traces, and no conflicting duplicate step.
- The probe RunSpec is eval-only at the frozen checkpoint with the trajectory's condition (C2) and arm (paired), which
  `RunSpec.__post_init__` accepts (a non-clean arm with p = 0.25); q is 0 in generation because the clean rollouts are
  generated clean and the grader flips are enumerated exactly afterward, not sampled.
- Score forward and capture. The score reproduces `GRPOTrainer._compute_loss` (input_ids = cat(prompt, completion),
  attention_mask = cat(prompt_mask, completion_mask), logits_to_keep = completion_ids.size(1), the temperature applied
  inside `_get_per_token_logps_and_entropies`, differentiated scalar (per_token_logps * mask).sum() with
  mask = completion_mask * tool_mask, T_i = mask.sum()). The diagnostic and clean generations are captured in separate
  context managers, each into its own buffer; the phase each records is `{prefix}:step{n}` (PhasedTrainer.evaluate
  stamps it from the metric prefix), so the identity check compares `phase.split(":")[0]` with probe_diag / probe_clean.
  The identities (counts, phases, no faults on clean, disjoint task ids) are asserted before any backward.
- Transition layout. Each task's captured members are ordered schedule-major by a stable sort on (schedule_seed,
  capture position) so that rollout index k*M + m shares schedule k whatever order the eval dataloader delivered them
  in; the `schedules` blocks of `samples` are asserted single-seeded and distinct, and the schedule-seed order is
  written to the evidence.
- Loss validation. On one mixed clean group (smoke, and once per full-size step) -grad of `_compute_loss` is checked
  against the reconstruction (1/T_group) sum_i A_i rho_i S_i. The advantages tensor is built on the mask's device, and
  `current_gradient_accumulation_steps` is set to `steps_per_generation` for the call (Transformers 5.16.1 sets it only
  inside the training loop, and this makes DAPO's `current / steps_per_generation` factor exactly 1) then deleted; the
  backward runs in the same train-plus-checkpointing mode as scoring, and `steps_per_generation` is recorded.

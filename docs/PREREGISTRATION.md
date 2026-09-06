# Pre-registration: Luck Is Not Skill

Paired rollouts for group-relative RL of LLM agents in stochastic environments.

Version: v0.2 (draft, not frozen; revised 2026-09-06 after an external review of v0 and the repository). This document becomes v1 and is frozen, with RFC 3161 timestamps and a pushed commit, after the gate-1 runs and the gradient probe and before any pre-registered run starts. Items marked SET AT v1 are filled in then and nowhere else. The changes from v0 are listed in section 11; the environment and evaluation defects that motivated them are recorded in `docs/DEVIATIONS.md`.

## 1. Thesis

Group-relative advantage estimation compares each rollout with its siblings from the same prompt. When each sibling draws its own environment randomness, the comparison mixes environment luck with policy skill. Sharing one noise schedule across all rollouts of a group (common random numbers, paired rollouts) removes the environment share of every within-group reward contrast and leaves the unnormalized estimator unbiased. Whether it lowers the variance of the policy gradient is not automatic: `docs/THEORY.md` section 3b gives the exact condition under one-sided outcome noise (pairing helps when the clean group gradient is small relative to the score spread of the successful rollouts, which holds for all-correct groups and for near-orthogonal trajectory scores, and fails for aligned scalar scores). The paper's claims are therefore conditional and measured: (1) on the actual LLM policy the condition holds and pairing lowers the measured gradient variance; (2) at matched rollout budget this translates into faster learning under transition noise and under outcome noise; (3) the cost in schedule diversity is immaterial for generalization to held-out fault types.

Prior art. Common random numbers across policy rollouts are not new (PEGASUS, TRPO's vine estimator, shared example seeds in GRPO course material, "identity consistency" in GRPO for driving simulation, event-keyed environment hashing); the paper cites them and does not claim first use. Its contribution is the estimator-level analysis, the condition and its boundary, the measurement of the mechanism on tool-using LLM agents, and the controlled evidence on learning.

## 2. Definitions

Arms.
- paired: all G rollouts of a task in a training step share one schedule seed; different tasks and different steps get different seeds.
- independent: every rollout draws its own schedule seed. This is the behavior of existing trainers.
- clean: no noise.
- blocking (fixed-mixture blocking baseline inspired by NoisyAgent, not a reproduction of it): each task appears twice per step, once as a clean group and once as a noisy group, each normalized within its own group, at an equal total rollout budget; blocking_paired is the same mixture with a paired noisy group, so the pair (blocking, blocking_paired) isolates coupling inside the hybrid design.

Noise.
- Transition noise, rate p per eligible tool call, a mixture of: transient failure (error response; a retry draws a fresh fate), rate limit (error with retry-after; a wait tool exists and every call fails until the wait is over), outage (the same logical request fails for the rest of the episode), stale read (a read issued after a write returns the pre-write state), truncation (a list response is cut and requires pagination). Default mixture, fixed at calibration on 2026-09-05: transient 0.45, rate limit 0.15, outage 0.10, stale 0.15, truncation 0.15; weights are renormalized over the types applicable to each tool. Outage-free mixture (condition C2r): the default without outage, renormalized (transient 0.50, the other three 1/6 each). Two further types are used only in evaluation: timeout after commit (a write is applied but reported as TIMEOUT) and field dropout (a read loses one field). Noise intensity is reported three ways: p, the realized per-type exposure counts, and the fraction of episodes exposed to at least one fault (about 0.72 to 0.79 at p = 0.25 for the current policies; p = 0.25 is not a 25 percent episode-failure setting).
- Outcome noise, rate q per episode: the grader reports failure for a correct end state. The environment records both the true (noise-free) outcome and the observed reward for every episode; under outcome noise the true outcome is the learning metric and the observed reward is reported separately.

Schedule. A seeded random stream keyed by the fault event: tool name, the resource the tool touches (customer id, order id, sku; free text such as reasons, summaries and search queries never enters the key) and the repeat index of that event within the episode, plus one episode-level draw for outcome noise. Two rollouts issuing the same request for the k-th time meet the same fate under the same schedule, whatever else they did before and however they word the request. Every logged tool call carries its event key, so alignment across siblings can be measured in the trajectories.

Evaluation schedules. Schedule index k of task t uses the seed `eval_schedule_seed(t, k)`, distinct across tasks and identical across arms and checkpoints; the outcome flip of an evaluation episode is a function of that seed, so at q = 0.10 about 10 percent of evaluation episodes are flipped (the v0 protocol reused four global seeds that never flipped, see DEVIATIONS).

Task pools (`data/tasks/MANIFEST.json`, generator seed 20260904): train 2000; validation ("heldout", 300; disjoint from training by customer pool and world seed, same templates, so generalization within the generator) used for calibration, periodic curves and threshold setting; test (300, generated after calibration and never used for any decision) for the final evaluation only; diagnostic (16 validation-pool tasks with 2 to 4 write calls, chosen by that structural rule).

Budget. Every episode may make at most 7 + 2 n tool calls (n = number of write calls in the task's plan), capped at 21; finish is free. Calls beyond the budget are refused and the episode ends. The recovering scripted reference (one customer lookup, one read per order, the writes, retries and waits as needed) uses 6.6 calls on the clean tasks and never exceeds the budget there; it is a strong scripted reference, not a proven ceiling (`registers/oracle_reference.json`, both pools, frozen evaluation schedules).

Luck share. For a fixed policy and task, with K schedules and M policy samples per schedule and rewards r[k, m]:
- MS_between = M / (K - 1) * sum_k (mean_k - mean)^2
- MS_within = 1 / (K * (M - 1)) * sum_k sum_m (r[k, m] - mean_k)^2
- sigma2_env = max(0, (MS_between - MS_within) / M)
- sigma2_pol = MS_within
- lambda_task = sigma2_env / (sigma2_env + sigma2_pol), undefined when the denominator is 0
- lambda = mean of lambda_task over tasks where it is defined; lambda_pooled = ratio of the averaged components; both are reported with 95 percent bootstrap intervals over tasks, the component estimates and the number of undefined tasks. Under the null (lambda = 0) the task-mean estimator has a clipping bias of about 0.03 for 8 x 8 tables.
Diagnostic design: K = 8, M = 8 on the 16 diagnostic tasks, under the run's training mixture at p = 0.25 (p = 0.25 default mixture for outcome-noise and clean runs, labelled as a transition probe), at step 0 and at the steps listed in the run spec; tables are kept separate per checkpoint and never merged or truncated. Reported for the true outcome and for the observed reward.

Spurious-variance group. Under outcome noise, a training group whose true outcomes are all equal but whose observed rewards are not. Measured from the training-group register (`groups.jsonl`: the trainer's own groups with task, schedule seeds, true outcomes, observed rewards and the advantages actually used), not from a concurrent log. The closed-form rate for an all-correct group of size G under independent outcome noise q is 1 - (1 - q)^G - q^G, which is 0.34 at q = 0.05 and 0.57 at q = 0.10 for G = 8.

Recovery. Two measures, always reported together: (a) success on the matched challenge set, in which the first attempt of every write request fails transiently and nothing else does (deterministic, identical for every policy, recoverable within budget by the scripted reference); (b) descriptive: success among episodes exposed to at least one fault under the at-training-noise set, together with the exposure rate, since exposure depends on the policy's own choices.

Learning curve and area. The at-training-noise validation set is evaluated at steps 0, 20, 40, 60, 80 and 100. AUC is the trapezoidal mean of true success over those steps (a number in [0, 1], the average success along the run).

Steps-to-threshold (secondary). The first evaluation step at which the at-training-noise validation true success reaches T, linearly interpolated; a run that never reaches T is censored (reported as "> 100", never imputed, and never entered into a ratio). T is SET AT v1 as the midpoint between the zero-shot at-training-noise success and the gate-1 C2 paired run's final at-training-noise success on the validation set, so that it is reachable under noise.

## 3. Hypotheses and decision rules

### H1: mechanism

(a) Counts. Under outcome noise (C4), the independent arm's spurious-variance rate among all-correct training groups lies within 0.07 of 0.57 and the paired arm's is at most 0.02, over all training steps and 3 seeds. Under transition noise (C2), the zero-shot luck share at step 0 is at least 0.15 with a 95 percent bootstrap lower bound above 0.05, and, as a check of the implementation rather than a result, every paired training group in the register shares one resolved schedule seed while no independent group does.

(b) Gradients. At the base model and at the gate-1 C2 paired checkpoints (steps 20, 40, 60, 80, 100), the gradient probe computes per-rollout scores S_i of the LoRA parameters through a fixed seeded random projection to 4096 dimensions, with the training tool mask, on K = 8 schedules x M = 8 rollouts of the 16 diagnostic tasks (the diagnostic design) and on 8 clean rollouts of 64 validation tasks. From these it reports, for outcome noise at q = 0.10 (exact, from clean rollouts and the section 3b formula): both sides of the inequality E||g_c||^2 versus (1/G^2) E sum_i r_i ||S_i - S_bar||^2 and the resulting Tr Var(g) under both designs; for transition noise at p = 0.25: Tr Var(g) under both designs from groups resampled from the same rollouts (paired: 8 rollouts of one schedule; independent: one rollout from each of 8 schedules), the reward-contrast variance alongside, and the expected update under both. H1(b) passes if Tr Var(g_paired) < Tr Var(g_indep) at every checkpoint for outcome noise and at a majority of checkpoints for transition noise. Whatever the outcome, both sides of the inequality are reported; a contrast reduction without a gradient benefit is reported as a boundary result.

### H2: learning at matched rollout budget

Primary endpoint: AUC of true success on the at-training-noise validation set (definition in section 2), 100 steps x 24 prompts x 8 rollouts for every arm.

H2a, transition noise, condition C2 (p = 0.25), Qwen3.5-2B, 3 seeds per arm. Seed s of the paired arm is compared with seed s of the independent arm (same training task order). H2a passes if the mean over seeds of the paired-minus-independent AUC difference is at least 0.03 and every seed-level difference is positive. The final test-set true success at p = 0.25 is reported alongside and must not contradict the direction (paired mean not below independent mean).

H2b, outcome noise, condition C4 (q = 0.10), Qwen3.5-2B, 3 seeds per arm, same rule with the clean validation set as the at-training-noise set (true success is the learning metric under outcome noise; the observed-reward curve is reported separately).

H2a and H2b are two different scientific claims (luck in transitions, luck in grading) and are reported as two results; neither substitutes for the other. Gate 2 rule: the method claim proceeds if H2a or H2b passes, and the paper says which; if neither passes the paper reports the boundary (contrast reduction, gradient measurement, no learning gain at this scale) and the change of framing is recorded in DEVIATIONS.md the same day.

### H3: recovery skill

In C2, on the matched challenge set (validation during training, test at the end), the paired arm's success exceeds the independent arm's by at least 5 points averaged over the step-40, step-60 and step-80 evaluations and over 3 seeds, with every seed-level difference positive. Unconditional success and exposure on the at-training-noise set are reported alongside.

### H4: normalization interaction

In C4, with loss aggregation fixed at the trainer default (loss_type dapo for every run in this document), crossing design (paired, independent) with reward scaling (group, none): the paired-minus-independent AUC gap is larger with scaling "group" than with "none", averaged over 2 seeds, and the gap with "none" remains positive. This is a 2 x 2 at the same loss; no other trainer setting changes.

### H5: no diversity cost (non-inferiority)

In C2, on the held-out-fault-type test set (timeout after commit, field dropout; never used in training), the paired arm is non-inferior to the independent arm with margin 3 points: the lower bound of the 90 percent bootstrap interval of the paired-minus-independent difference (resampling seeds and tasks, section 8) is above -0.03. A paired advantage larger than 3 points is not a failure of H5.

### H6: external validity (run if budget allows)

On the BFCL v3 multi-turn benchmark decomposed into turn episodes (explicitly labelled as an adaptation; scenarios grouped by source to prevent split leakage; clean checker parity verified before training), with transition noise p = 0.25 on the tool layer, Qwen3.5-2B, 2 seeds per arm: the paired arm's AUC exceeds the independent arm's in both seeds. The flaky-test census of v0 is dropped from this document (a pilot, if run, is descriptive and reported as such).

## 4. Numeric predictions (recorded before any pre-registered run)

| id | prediction | confidence |
|---|---|---|
| P1 | Independent arm spurious-variance rate among all-correct groups: 0.57 plus or minus 0.05 at q = 0.10; paired arm below 0.02 | 0.90 |
| P2 | C4 AUC gap paired minus independent at least 0.03; paired final clean test success within 3 points of the clean run | 0.65 |
| P3 | C3 (q = 0.05, run if budget allows) AUC gap between 0.01 and 0.03 | 0.55 |
| P4 | C2 AUC gap between 0.02 and 0.06; final test gap at p = 0.25 between 2 and 8 points | 0.60 |
| P5 | C2r (outage-free) AUC gap smaller than C2's and within 0.03 (dose response in lambda) | 0.55 |
| P6 | C2 challenge-set success at steps 40 to 80: paired ahead by at least 5 points | 0.55 |
| P7 | With scale_rewards none the C4 gap shrinks by 30 to 70 percent but stays positive | 0.65 |
| P8 | blocking_paired beats blocking at C2 and C4; blocking lands between independent and paired | 0.60 |
| P9 | Held-out fault types: paired non-inferior with margin 3 points; point difference within 2 points | 0.80 |
| P10 | BFCL (if run): paired minus independent AUC between 0.01 and 0.05 | 0.50 |
| P12 | Zero-shot luck share at p = 0.25 for Qwen3.5-2B is at least 0.15 (recorded 2026-09-04 before the environment v2 change) | 0.60 |
| P13 | Gradient probe at the base model, outcome noise q = 0.10: E||g_c||^2 is below (1/G^2) E sum r_i ||S_i - S_bar||^2 by a factor of at least 2, so Tr Var(g_paired) < Tr Var(g_indep); the ratio Tr Var(g_paired) / Tr Var(g_indep) falls further at later checkpoints as the all-correct fraction rises; under transition noise at p = 0.25 the ratio is below 1 at a majority of checkpoints | 0.70 |

P12 outcome (2026-09-05, run calib-qwen3.5-2b at commit 8fa9475, environment v1): lambda = 0.030 over 12 of 16 diagnostic tasks; the prediction failed. Causes (from the episode logs): saturated diagnostic tasks (11 of 16 at 0.94 or higher, 3 at 0.00 through a tool-interface defect), a fault mix recoverable by the zero-shot policy (0.499 clean versus 0.480 noisy at p = 0.25), and `update_shipping_address` rejecting numeric postal codes. The environment v2 changes (request-keyed schedule, outage type, budget 7 + 2n, numeric arguments, diagnostic tasks restricted to 2 to 4 writes, task mix shifted toward multi-request tasks) were made in response; the v2 re-measurement at commit a6fd4f4 gave lambda = 0.222 (16 of 16 tasks; 0.581 for Qwen3.5-4B, 15 of 16) at p = 0.25 with the default mixture (`registers/calibration_env_v2.json`). Environment v3 (2026-09-06: resource-keyed fault events, per-task evaluation schedules, untouched test pool) is re-measured before v1 and the v3 value is the one reported; the v1 and v2 registers are kept. The mixture is frozen at v1 and is not modified afterwards whatever the results.

## 5. Run matrix

Primary model Qwen3.5-2B. Training: TRL 1.12.0 GRPOTrainer with environment_factory, vLLM colocate, LoRA rank 32, 24 prompts x 8 rollouts per step, 100 steps, loss_type dapo (the trainer default; it is the DAPO token-level aggregation, not the DAPO algorithm's dynamic sampling), scale_rewards group unless stated, learning rate 1e-5, temperature 1.0, tool-result tokens masked from the loss. Every paired-versus-independent comparison runs on one provider and one stack; a condition is never split across providers. Runs are ordered by tier; the runs funded at the freeze are SET AT v1 from the gate-1 measured step time and the remaining credit; lower tiers are "run if budget allows" and are reported if run.

| tier | condition | noise | arms | seeds | hypotheses |
|---|---|---|---|---|---|
| 1 | C2 | transition p = 0.25, default mixture | paired, independent | 3 | H1a, H2a, H3, H5 |
| 1 | C4 | outcome q = 0.10 | paired, independent | 3 | H1a, H2b |
| 1 | C0 | clean | clean | 2 | reference curves, threshold T |
| 2 | C2r | transition p = 0.25, outage-free mixture | paired, independent | 2 | P5 (useful noise, dose response) |
| 2 | C2, C4 | as above | blocking, blocking_paired | 2 | P8 |
| 2 | C4 | as above | paired, independent with scale_rewards none | 2 | H4 |
| 3 | C1 | transition p = 0.10 | paired, independent | 2 | lower-noise boundary |
| 3 | C3 | outcome q = 0.05 | paired, independent | 2 | dose response in q |
| 4 | BFCL | transition p = 0.25 | paired, independent | 2 | H6 |

Dropped from v0: Qwen3.5-9B (single seed, no inferential value), Qwen3.5-4B replication (run only if tiers 1 to 3 are complete), the flaky-test census.

## 6. Evaluation protocol

- Periodic (validation pool, 64 tasks): clean and at-training-noise sets, 2 schedules per task (one batch of 128 each), and the matched challenge set (64 episodes), at steps 0, 20, 40, 60, 80 and 100. The clean arm's "at-training-noise" set is the default mixture at p = 0.25.
- Final (test pool, 200 tasks, 4 schedules per task, identical for every arm and condition): clean; p = 0.10 and p = 0.25 default mixtures; held-out fault types at p = 0.25; the matched challenge set (200 episodes); and, when the run's training noise is not one of those (outcome noise, outage-free mixture), the at-training-noise set. Every policy is therefore evaluated on every comparison regime.
- Luck-share diagnostic: 16 tasks x 8 schedules x 8 samples at steps 0, 50 and 100 for seed 0 of every cell and at steps 0 and 100 for the other seeds.
- Training-group register: every training group's task, schedule seeds, true outcomes, observed rewards, advantages, zero-variance and spurious flags, from the trainer's own batches.
- Metrics: true success (primary), observed reward, recovery (section 2), calls used, budget-exceeded rate, exposure by fault type, spurious-variance group rate, luck share with intervals, and deterministic trace codes from the logged event keys (retry after transient failure, wait after rate limit, re-read after stale read, paginate after truncation); the codes are validated against 50 hand-labelled trajectories (Cohen's kappa at least 0.75 required before use).

## 7. Exclusion and relaunch rules

- Runs checkpoint the trainer state (adapter, optimizer, scheduler, RNG) every 20 steps and their logs after every evaluation phase. A container that dies is retried by the platform up to three times; a retry preserves the previous attempt's logs in an `attempt<n>/` directory and resumes from the last trainer checkpoint if one exists, otherwise restarts from step 0. All attempts are kept and reported.
- No run is excluded, dropped or replaced because of its result. All seeds are reported, with every seed-level difference shown.
- Runs launched before the v1 freeze (stack checks, calibrations, speed checks, gate 1, the gradient probe on gate-1 checkpoints) are exploratory and are labelled as such in the paper.

## 8. Analysis plan

- Point estimates are means over seeds. Intervals are 95 percent (90 percent for the non-inferiority bound) percentile bootstrap intervals with 10,000 resamples that preserve the pairing and the clustering: seeds are resampled with replacement as (paired s, independent s) pairs, and tasks are resampled with replacement within each evaluation set with all of a task's schedules kept together. Episodes are never pooled as independent samples.
- Arm comparisons use paired differences on shared tasks and shared schedules.
- AUC is the trapezoidal mean over the six periodic evaluations; steps-to-threshold is secondary and censored runs are reported as such.
- The predictions of section 4 are compared with the outcomes in a table in the paper's appendix, including the misses.

## 9. Not pre-registered

Behavior examples chosen for the paper, figure design, the exact wording of the applicability boundary, and any analysis added after the v1 freeze are exploratory and are labelled as such. Any partial-coupling ablation (two or four schedules per group with within-schedule baselines) is exploratory and requires its own literature check before being described as new.

## 10. Freeze block (v1)

- Frozen on (UTC): SET AT v1
- Git commit of the frozen file: SET AT v1
- SHA-256 of the frozen file: SET AT v1
- RFC 3161 token 1 (TSA, time): SET AT v1
- RFC 3161 token 2 (TSA, time): SET AT v1

## 11. Changes from v0 (2026-09-06, before the freeze)

- Thesis: the general gradient-variance and effective-group-size claims are withdrawn; the conditional claim and the exact condition of THEORY.md section 3b replace them; direct prior art is acknowledged.
- H1 gains the gradient probe (b) and P13; the unbiasedness statement is restricted to the unnormalized estimator.
- H2: AUC replaces the final-gap-versus-pooled-SD rule; the minimum effect is 0.03 with all seed-level differences positive; steps-to-threshold is secondary with censoring instead of imputation at 120.
- H3 uses the matched challenge set instead of success conditional on exposure.
- H4 fixes the loss aggregation (dapo, the implemented default; v0 wrongly said grpo and switched to dr_grpo together with scaling).
- H5 is a non-inferiority test with margin 3 points.
- Evaluation: per-task evaluation schedules; a new untouched test pool for the final evaluation; the validation pool is the calibration pool; every policy is evaluated on every comparison regime; diagnostics per checkpoint; the training-group register.
- Environment: fault events keyed by tool and resource (free text excluded) so that rewording cannot dodge an outage; the outage-free mixture C2r added as a condition.
- Run matrix: tiers with a funded list set at v1; 9B and the census dropped; 4B only after tiers 1 to 3.
- Relaunch rules describe the implemented checkpoint and attempt behavior.

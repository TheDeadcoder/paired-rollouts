# Pre-registration: Luck Is Not Skill

Paired rollouts for group-relative RL of LLM agents in stochastic environments.

Version: v0 (draft, not frozen). This document becomes v1 and is frozen, with RFC 3161 timestamps and a pushed commit, after the calibration step and before any pre-registered run starts. Items marked SET AT v1 are filled in then and nowhere else.

## 1. Thesis

Group-relative advantage estimation compares each rollout with its siblings from the same prompt. When each sibling draws its own environment randomness, the comparison mixes environment luck with policy skill. Sharing one noise schedule across all rollouts of a group (common random numbers, paired rollouts) removes the environment share of every within-group contrast, leaves the policy-gradient estimator unbiased, eliminates spurious-variance groups under outcome noise, and should therefore improve sample efficiency and the learning of recovery behaviors under transition noise.

## 2. Definitions

Arms.
- paired: all G rollouts of a task in a training step share one schedule seed; different tasks and different steps get different seeds.
- independent: every rollout draws its own schedule seed. This is the behavior of existing trainers.
- clean: no noise.
- blocking (NoisyAgent-style baseline): each task appears twice per step, once as a clean group and once as an independent-noise group, each normalized within its own group, at an equal total rollout budget.

Noise.
- Transition noise, rate p per tool call, a mixture of: transient failure (error response; the next identical call succeeds), rate limit (error with retry-after; a wait tool exists), stale read (a read issued within d calls after a write returns the pre-write state), truncation (a list response is cut and requires pagination). The mixture weights are SET AT v1.
- Outcome noise, rate q per episode: the grader returns failure for a correct end state.

Schedule. A seeded random stream keyed by (tool name, per-tool call index), plus one episode-level draw used for outcome noise. Two rollouts making the same k-th call to the same tool meet the same fate under the same schedule.

Luck share. For a fixed policy and task, with K schedules and M policy samples per schedule and rewards r[k, m]:
- MS_between = M / (K - 1) * sum_k (mean_k - mean)^2
- MS_within = 1 / (K * (M - 1)) * sum_k sum_m (r[k, m] - mean_k)^2
- sigma2_env = max(0, (MS_between - MS_within) / M)
- sigma2_pol = MS_within
- lambda_task = sigma2_env / (sigma2_env + sigma2_pol), undefined when the denominator is 0
- lambda = mean of lambda_task over tasks where it is defined
Diagnostic design: 16 fixed tasks, K = 8, M = 8, measured at step 0 and every 20 steps.

Spurious-variance group. Under outcome noise, a group whose noise-free graded outcomes are all equal but whose observed rewards are not all equal. The environment records both the noise-free grade and the observed reward for every episode. The closed-form rate for an all-correct group of size G under independent outcome noise q is 1 - (1 - q)^G - q^G, which is 0.34 at q = 0.05 and 0.57 at q = 0.10 for G = 8.

Recovery success. Success rate over episodes in which at least one transition fault was injected.

Steps-to-threshold. The first evaluation step at which success on the at-training-noise evaluation set reaches the threshold T, with linear interpolation between evaluation steps; a run that never reaches T is assigned 120 and flagged as censored. T is SET AT v1 as the midpoint between the model's zero-shot clean success and the clean run's final clean success from the gate-1 run.

## 3. Hypotheses and decision rules

### H1: mechanism

Under outcome noise (C3, C4), the independent arm's measured spurious-variance rate among all-truly-correct groups lies within 0.07 of the closed form (0.34 at q = 0.05, 0.57 at q = 0.10), and the paired arm's rate is at most 0.02. Under transition noise (C2), the independent arm's within-group reward variance contains a positive environment component (lambda at step 0 at least 0.15) and the paired arm's within-group luck share is at most 0.05.

### H2: sample efficiency (gate hypothesis)

H2a, transition noise, condition C2 (p = 0.25), Qwen3.5-2B, 3 seeds per arm. Let D be the mean over seeds of the paired arm's final at-training-noise success minus the independent arm's, and let s be the pooled standard deviation across seeds, s = sqrt((s_paired^2 + s_independent^2) / 2). H2a passes if D is at least s, or if the ratio of median steps-to-threshold (independent over paired) is at least 1.3 with at least two of three paired seeds reaching T.

H2b, outcome noise, condition C4 (q = 0.10), Qwen3.5-2B, 3 seeds per arm. With D and s defined as above on the at-training-noise evaluation set, H2b passes if D is at least 8 percentage points and at least s.

Gate 2 rule: the method claim proceeds if H2a or H2b passes. If neither passes, the paper is re-scoped to a diagnostic and measurement paper and the change is recorded in DEVIATIONS.md the same day.

### H3: recovery skill

In C2, the paired arm's recovery success exceeds the independent arm's by at least 5 percentage points when averaged over the step-40 and step-60 evaluations and over 3 seeds.

### H4: normalization interaction

In C4, the paired-minus-independent gap in final at-training-noise success is larger with std-normalized advantages (scale_rewards = True) than without (scale_rewards = False, loss_type = dr_grpo), averaged over 2 seeds, and the gap without normalization remains positive.

### H5: no diversity cost

In C2, the absolute difference between the paired and independent arms on the held-out-noise-type evaluation set is at most 3 percentage points, averaged over 3 seeds. The two held-out noise types are SET AT v1 and are never used in training.

### H6: external validity

On the BFCL turn-episode environment with transition noise (Qwen3.5-4B, 2 seeds per arm), the paired arm's final at-training-noise success exceeds the independent arm's in both seeds. The flaky-test census is reported descriptively and carries no pass or fail rule.

## 4. Numeric predictions (recorded before any run)

| id | prediction | confidence |
|---|---|---|
| P1 | Independent arm spurious-variance rate among all-correct groups: 0.34 plus or minus 0.05 at q = 0.05 and 0.57 plus or minus 0.05 at q = 0.10; paired arm below 0.02 | 0.90 |
| P2 | C4 final gap paired minus independent at least 8 points; paired within 3 points of the clean run | 0.70 |
| P3 | C3 final gap between 3 and 8 points | 0.60 |
| P4 | C2 steps-to-threshold ratio at least 1.3; final gap between 3 and 10 points | 0.65 |
| P5 | C1 (p = 0.10) final gap within 3 points and not significant | 0.60 |
| P6 | C2 recovery success at steps 40 and 60: paired ahead by at least 8 points | 0.60 |
| P7 | With scale_rewards = False the C4 gap shrinks by 30 to 70 percent but stays positive | 0.70 |
| P8 | Blocking baseline lands between independent and paired at C2 and C4 | 0.65 |
| P9 | Held-out-schedule and held-out-noise-type generalization differ by at most 2 points between arms | 0.80 |
| P10 | BFCL: paired minus independent between 3 and 8 points at the final step | 0.55 |
| P11 | Census: 10 to 30 percent of instances show at least one flip in 10 runs; among those the median per-run flake rate is 10 to 20 percent | 0.50 |
| P12 | Zero-shot luck share at p = 0.25 for Qwen3.5-2B is at least 0.15 before any calibration change | 0.60 |

## 5. Run matrix

Primary model Qwen3.5-2B; secondary Qwen3.5-4B; scale check Qwen3.5-9B. Fallbacks Qwen3-1.7B and Qwen3-4B only if the day-1 stack check fails, recorded in DEVIATIONS.md. Training: TRL GRPOTrainer with environment_factory, vLLM colocate, LoRA rank 32, num_generations 8, 24 prompts per step, 100 steps, loss_type grpo, scale_rewards True unless stated.

| condition | noise | arms | model | seeds | notes |
|---|---|---|---|---|---|
| C0 | clean | clean | 2B | 3 | reference and threshold source |
| C1 | transition p = 0.10 | paired, independent | 2B | 3 | boundary of usefulness |
| C2 | transition p = 0.25 | paired, independent | 2B | 3 | H2a, H3, H5 |
| C3 | outcome q = 0.05 | paired, independent | 2B | 3 | H1 |
| C4 | outcome q = 0.10 | paired, independent | 2B | 3 | H1, H2b, H4 |
| C2, C4 | as above | blocking | 2B | 3 | P8 |
| C2, C4 | as above | paired, independent with scale_rewards False | 2B | 2 | H4 |
| C2, C4 | as above | paired, independent | 4B | 2 | scale |
| C2 | as above | paired, independent | 9B | 1 | scale |
| BFCL | transition p = 0.25 | paired, independent | 4B | 2 | H6 |

## 6. Evaluation protocol

- Held-out tasks: 200, disjoint from training by customer pool. Held-out schedules: 4 fixed seeds shared by every arm and every checkpoint (paired evaluation).
- Evaluation sets: clean; at-training-noise; held-out noise types (two types, SET AT v1).
- During training: every 20 steps on 100 held-out tasks x 2 schedules. Final: 200 tasks x 4 schedules x 3 sets.
- Metrics: success, recovery success, turns used, spurious-variance group fraction (training-time, outcome-noise conditions), luck share (diagnostic subset), behavior codes (retry after transient failure, wait after rate limit, re-read after stale read, paginate after truncation), judged by a fixed rubric with 50 hand-labelled trajectories for agreement (Cohen's kappa at least 0.75 required before use).

## 7. Exclusion and relaunch rules

- A run that crashes before step 50 is relaunched with the same seed; both rows stay in the ledger.
- A run that crashes at or after step 50 resumes from its last checkpoint if one exists, otherwise it is relaunched with the same seed.
- No run is excluded, dropped or replaced because of its result. All seeds are reported.
- Runs launched before the v1 freeze (stack check, calibration, gate-1 runs) are exploratory and are labelled as such in the paper.

## 8. Analysis plan

- Point estimates are means over seeds; intervals are 95 percent bootstrap intervals (10,000 resamples) over seeds and tasks.
- Arm comparisons use paired differences on shared tasks and shared schedules.
- Steps-to-threshold uses linear interpolation between evaluation steps.
- The pre-specified predictions in Section 4 are compared with outcomes in a table in the paper's appendix, including the misses.

## 9. Not pre-registered

Behavior examples chosen for the paper, figure design, the exact wording of the applicability boundary, and any analysis added after the v1 freeze are exploratory and are labelled as such.

## 10. Freeze block (v1)

- Frozen on (UTC): SET AT v1
- Git commit of the frozen file: SET AT v1
- SHA-256 of the frozen file: SET AT v1
- RFC 3161 token 1 (TSA, time): SET AT v1
- RFC 3161 token 2 (TSA, time): SET AT v1

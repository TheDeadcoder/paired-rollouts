# Luck Is Not Skill: When Do Paired Rollouts Help Group-Relative RL of LLM Agents?

Common random numbers for GRPO in stochastic tool environments: the estimator-level theory, a pre-registered three-seed study, and a gradient probe on the trained policies.

Paper: under review (link to follow). The pre-registration (`docs/PREREGISTRATION.md`) was frozen and timestamped (RFC 3161) before any registered run; every number on this page is emitted by a file under `registers/`.

**Findings.** Sharing one noise schedule across a GRPO group removes environment luck from reward contrasts exactly as the theory predicts, and a gradient probe on frozen checkpoints shows it removes luck from the gradients too: 21 to 30 percent less gradient variance under grader noise, 40 to 63 percent less under transient tool faults, at every checkpoint. Whether that reaches the learning curve depends on the luck share of the gradient variance: a robustness gain under faults (+5.1 points on every seed), no measurable learning gain under grader noise.

## The question

Group-relative policy optimization scores each rollout against its siblings from the same prompt. In an agentic environment each sibling normally draws its own environment randomness: a tool call fails, a read goes stale, a grader flips. The comparison then mixes environment luck with policy skill, and a rollout is penalized for its schedule rather than its behaviour.

Paired rollouts share one event-keyed noise schedule across all rollouts of a group, so that every within-group reward contrast reflects the policy alone. Different groups still draw different schedules. The device is common random numbers; the question this repository answers is what it does to the group-relative gradient estimator, and whether that reaches the learning curve.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://ik.imagekit.io/sakib61/Paired-Rollouts/fig1_design_dark.png">
  <img alt="Independent rollouts draw one schedule per rollout; paired rollouts draw one schedule per group." src="https://ik.imagekit.io/sakib61/Paired-Rollouts/fig1_design_light.png" width="100%">
</picture>

## Theory

For a prompt, a group of $G$ rollouts has rewards $R_i$ and scores $S_i = \nabla_\theta \log \pi_\theta(\tau_i)$ summed over the policy's own tokens. The group gradient is $g = \frac{1}{G}\sum_i A_i S_i$ with mean-centered advantages $A_i = R_i - \bar R$ (or the std-normalized ones a trainer applies). Split the reward variance of a fixed prompt into the part explained by the schedule and the rest, $\sigma^2_{\text{env}}$ and $\sigma^2_{\text{pol}}$, with luck share $\lambda = \sigma^2_{\text{env}} / (\sigma^2_{\text{env}} + \sigma^2_{\text{pol}})$.

**Reward contrasts.** For siblings $i \neq j$, $\mathrm{Var}(R_i - R_j) = 2(\sigma^2_{\text{env}} + \sigma^2_{\text{pol}})$ under independent schedules and $2\sigma^2_{\text{pol}}$ under a shared one: pairing removes exactly the environment's share. The unnormalized estimator stays unbiased under pairing, because the score function has zero mean conditional on any schedule.

**Gradients.** Whether the gradient variance drops is a separate question. Under one-sided grader noise (a correct answer is graded wrong with probability $q$, independently per rollout or once per group) the answer is exact:

$$
\mathrm{Tr}\,\mathrm{Var}(g_{\text{paired}}) - \mathrm{Tr}\,\mathrm{Var}(g_{\text{indep}}) = q(1-q)\left[\mathbb{E}\|g_c\|^2 - \frac{1}{G^2}\,\mathbb{E}\sum_i r_i \|S_i - \bar S\|^2\right],
$$

where $g_c$ is the clean group gradient and $r_i$ the true success. Pairing lowers the gradient variance if and only if the clean group gradient is small relative to the score spread of the correct rollouts. For an all-correct group the left side is zero, so pairing never loses there; for aligned scalar scores (a one-step bandit) the condition fails and pairing raises the gradient variance 2.2-fold while still lowering the reward-contrast variance. The sign is a property of the policy's score geometry, so it is measured rather than assumed.

**Spurious-variance groups.** Under independent grader flips, a group whose $G$ rollouts are all correct has non-zero observed variance with probability $1 - (1-q)^G - q^G$, 0.57 at $q = 0.10$ and $G = 8$; it then emits a full-magnitude update against a correct trajectory. Under pairing that probability is 0.

Full derivations, the bandit enumeration and the luck-share estimator: `docs/THEORY.md`.

## Environment and protocol

**Environment.** A back-office tool environment over customers, orders, inventory and tickets: five read tools, six write tools, `wait` and `finish`. A task is a customer request composed of one to three sub-requests (address changes, cancellations, refunds, reservations and shipments, tickets); the grader compares the final world state with the expected one, so success is exact and noise-free. Every episode has a call budget of $7 + 2n$ for $n$ planned writes.

**Noise.** Transition noise at rate $p$ per eligible call is a fixed mixture of transient failure, rate limit, outage, stale read and truncation; two further types (timeout after commit, field dropout) are held out for evaluation. Outcome noise flips a correct grade with probability $q$. A schedule is a seeded stream keyed by the fault event (tool, resource, repeat index), so two rollouts issuing the same request for the k-th time meet the same fate under the same schedule, whatever else they did.

**Training.** Qwen3.5-2B with LoRA (rank 32, all linear layers), TRL `GRPOTrainer` with vLLM generation, 24 prompts x 8 rollouts x 100 steps, std-normalized advantages, DAPO token-level aggregation, tool-result tokens masked from the loss. Conditions: C0 clean (2 seeds), C2 transient faults at $p = 0.25$ and C4 grader flips at $q = 0.10$, each with a paired and an independent arm at 3 seeds. Seed $s$ of one arm is compared with seed $s$ of the other on the same task order, and every such pair trained on one hardware and software stack.

**Evaluation.** Validation curves at steps 0, 20, ..., 100 on 64 held-out tasks (clean, at-training-noise, and a matched challenge set in which the first attempt of every write fails). Final evaluation on a test pool of 200 tasks x 4 schedules, never used for any decision: clean, faults at $p = 0.10$ and $0.25$, held-out fault types, matched challenge. The learning metric is always the true, noise-free success.

**Pre-registration.** Hypotheses, decision rules, numeric predictions and the run matrix were frozen before the first registered run (`docs/PREREGISTRATION.md`, tokens in `registers/freeze/v1/`). No run was excluded or replaced; every departure is logged in `docs/DEVIATIONS.md`.

## Results

### The mechanism is there

At step 0 the luck share of the reward under transient faults is 0.33 to 0.38 across the six C2 runs (16 diagnostic tasks x 8 schedules x 8 samples). Under grader flips, 55.8 percent of the independent arm's all-correct training groups carried spurious advantage variance (2,080 of 3,726; the closed form gives 57 percent), against 0 of 3,989 in the paired arm.

### Learning under transient faults (C2)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://ik.imagekit.io/sakib61/Paired-Rollouts/fig2_learning_curves_dark.png">
  <img alt="Validation learning curves for paired, independent and clean training under transient faults and under grader flips." src="https://ik.imagekit.io/sakib61/Paired-Rollouts/fig2_learning_curves_light.png" width="100%">
</picture>

Pairing raises final test success under faults at $p = 0.25$ from 61.8 to 66.8 percent, ahead on every seed (+4.0, +1.0, +10.3 points), and success on the matched challenge set by 5.4 points averaged over steps 40 to 80 (every seed positive; 72.2 against 68.7 percent at the end). On held-out fault types the paired arm is non-inferior with a 3-point margin (+2.3 points, 90 percent interval lower bound at -0.4). The pre-registered learning-curve criterion (mean AUC gain of at least 0.03 with every seed positive) was missed by one seed: +0.054, -0.002, +0.059, mean +0.037. Clean training transfers little to faults: 56.4 percent on the test pool at $p = 0.25$.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://ik.imagekit.io/sakib61/Paired-Rollouts/fig3_final_success_dark.png">
  <img alt="Final test-set success per evaluation set for every arm and seed." src="https://ik.imagekit.io/sakib61/Paired-Rollouts/fig3_final_success_light.png" width="100%">
</picture>

### Learning under grader flips (C4)

No measurable gain: the clean-validation AUC difference is +0.003 (95 percent interval -0.029 to +0.033), final clean test success 89.4 against 88.0 percent, with clean training at 89.9. The noise costs the independent arm about two points against clean training and pairing recovers most of that, which is less than the registered margin.

### The gradient probe

At eight frozen checkpoints (the base model and steps 20 to 100 of one C2 paired run, trajectory B; steps 80 and 100 of a second, trajectory A) the probe computes every rollout's score over the 33.6 million LoRA coordinates and forms all group quantities as exact within-group inner products. Grader flips are enumerated exactly over the corruption masks of 64 clean groups; transient faults are measured by regrouping 16 tasks x 8 schedules x 8 rollouts under the paired design and under independent resampling at the same rollout cost. Both the registered mean-centered estimator and the update the trainer implements are reported.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://ik.imagekit.io/sakib61/Paired-Rollouts/fig4_probe_ratios_dark.png">
  <img alt="Ratio of the trace of the gradient covariance, paired over independent, at every checkpoint under grader flips and under transient faults." src="https://ik.imagekit.io/sakib61/Paired-Rollouts/fig4_probe_ratios_light.png" width="100%">
</picture>

The condition holds at every checkpoint: the clean group gradient is 0.19 to 0.27 of the score spread. Pairing lowers the trace of the gradient covariance by 21 to 30 percent under grader flips and by 40 to 63 percent under transient faults, at every checkpoint of both trajectories, and the reduction grows with training. The implemented update shows the same sign at 15 of the 16 checkpoint-noise pairs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://ik.imagekit.io/sakib61/Paired-Rollouts/fig5_probe_decomposition_dark.png">
  <img alt="Decomposition of the gradient variance into a between-task part and a within-group part, per checkpoint, under both designs." src="https://ik.imagekit.io/sakib61/Paired-Rollouts/fig5_probe_decomposition_light.png" width="100%">
</picture>

The decomposition explains the two learning results. Pairing can only act on the within-group part of the gradient variance. Under grader flips that part is 29 to 37 percent of the total, because the spread of the tasks' expected gradients dominates and no design touches it; under transient faults it is 87 to 88 percent. In batch terms, pairing is worth 1.3 to 1.4 times the batch under grader flips and 1.7 to 2.7 times under transient faults. Removing luck from rewards does remove it from gradients; how much that buys depends on how much of the gradient was luck.

## Pre-registered outcomes

| Hypothesis | Rule | Outcome |
|---|---|---|
| H1a, mechanism | luck share at step 0 at least 0.15; spurious-variance rate within 0.07 of 0.57 (independent) and at most 0.02 (paired) | pass: 0.33 to 0.38; 0.558 and 0.000 |
| H1b, gradients | Tr Var(g) lower under pairing at every checkpoint (grader flips) and at a majority (transient faults), per trajectory | pass on both trajectories: 0.70 to 0.79 and 0.37 to 0.60 |
| H2a, learning under faults | AUC gain at least 0.03, every seed positive; final direction | miss: mean +0.037, seeds +0.054, -0.002, +0.059; final direction holds |
| H2b, learning under grader flips | same rule on the clean validation set | miss: mean +0.003 |
| H3, recovery | challenge-set gain at least 5 points over steps 40 to 80, every seed positive | pass: +5.4 |
| H5, no diversity cost | non-inferior on held-out fault types, margin 3 points | pass: +2.3, lower bound -0.4 |
| P13, probe prediction | clean group gradient at most half the score spread at the base model; outcome ratio lower at the last checkpoint than at the base; transition ratio below 1 at a majority | all three true |

## Repository

```
src/pairedrl/env/         back-office environment: world, tools, tasks, grader
src/pairedrl/noise/       event-keyed schedules and fault types
src/pairedrl/train/       GRPO integration, evaluation phases, the gradient probe job
src/pairedrl/analysis/    estimators, luck share, probe statistics, decision rules
scripts/                  launch, collect, verify, decide, build registers
configs/                  every run and probe specification
registers/                every number in the paper: runs, decisions, probe, weights, freeze
docs/                     THEORY, PREREGISTRATION, PROBE, DEVIATIONS, run ledger
```

Trained adapters and checkpoints are archived at [Melikshah/paired-rollouts-weights](https://huggingface.co/Melikshah/paired-rollouts-weights); `registers/weights/` holds the revision and the hash of every file.

```
pip install -e ".[dev]"
pytest -q
python scripts/verify_run.py outputs/runs/<run_id>
python scripts/decide.py --condition C2 --runs outputs/runs/<run_id> ... --out registers/decisions/tier1_c2.json
python scripts/build_probe_register.py --probes outputs/runs/probe-t1-c2-paired-s1-a outputs/runs/probe-t1-c2-paired-s1-b \
    --trajectory t1-c2-paired-s1 --base-from outputs/runs/probe-t1-c2-paired-s1-a --out registers/probe/t1-c2-paired-s1.json
python scripts/recompute_probe_step.py outputs/runs/probe-t1-c2-paired-s1-b/step100.json
```

The decision scripts accept only runs that pass `scripts/verify_run.py` (complete evidence at every registered step, frozen task pools, one code commit per run). The probe's step summaries are reproducible from the evidence files alone with `recompute_probe_step.py`.

## Citation

```bibtex
@misc{sakib2026luck,
  title  = {Luck Is Not Skill: When Do Paired Rollouts Help Group-Relative RL of LLM Agents?},
  author = {Nazmus Sakib Touhid},
  year   = {2026},
  url    = {https://github.com/TheDeadcoder/paired-rollouts}
}
```

Apache 2.0.
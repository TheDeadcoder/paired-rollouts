# Luck and skill in group-relative advantage estimation

Working notes for the paper's theory section, version 2 (2026-09-06). Plain notation; every claim states the estimator it is about, and every closed form here has a matching implementation in `src/pairedrl/analysis/diagnostics.py` or a matching test. Version 1 of these notes claimed a general gradient-variance reduction and an "effective group size" of G / (1 - lambda); both claims were wrong as stated and are replaced by Section 3b, which gives the exact condition under which pairing lowers gradient variance and shows a case where it raises it.

## 1. Setup and prior art

Fix a prompt x and a policy pi with parameters theta. An episode draws two independent sources of randomness: the environment's, omega (fault schedule, grader flakiness), and the policy's own sampling, xi. A rollout is tau = tau(omega, xi), its reward is R = R(omega, xi), and its score is S = grad_theta log pi(tau), the sum of the action log-probability gradients over the policy's own tokens.

A group is G rollouts of the same prompt. Group-relative methods score rollout i by a contrast against its siblings,

    A_i = R_i - mean_j R_j                        (mean-centered; TRL scale_rewards = "none"),
    A_i = (R_i - mean_j R_j) / (s + eps)          (std-normalized; s the Bessel-corrected group std, eps = 1e-4),

and the group's policy-gradient estimate is g = (1/G) sum_i A_i S_i.

Two designs differ only in how omega is drawn.

- Independent (the status quo of every trainer we know of): omega_1, ..., omega_G are i.i.d. draws. Every rollout lives in its own world.
- Paired (common random numbers): one omega is drawn per group and shared by its G rollouts; different groups still draw different omegas.

In both designs xi_1, ..., xi_G are independent and each rollout's marginal law is the same. Sharing omega across a group is the common-random-numbers device of simulation (Glasserman and Yao, 1992) and of policy search: PEGASUS (Ng and Jordan, 2000) fixes the random numbers of a simulator to turn a stochastic MDP into a deterministic one, TRPO's vine estimator (Schulman et al., 2015, Section 5.2) uses common random numbers across the actions branched from one state, and in RL for LLMs the same idea appears as a shared example seed in course material on GRPO for games (Fiorucci, 2025, chapter 03), as "identity consistency" for GRPO in driving simulation (Bangunharcana, Lee and Han, ASCC 2026) and as event-keyed hashing of environment randomness (arXiv 2603.11084). The paper does not claim the device is new. What it studies is the estimator-level question below, which none of those works states or measures for group-relative advantages: when does coupling the environment randomness of a group lower the variance of the group's gradient, and when does it not.

## 2. Unbiasedness (for the unnormalized estimator only)

Claim. For the leave-one-out, unscaled estimator g_LOO = (1/G) sum_i (R_i - mean_{j != i} R_j) S_i, the expectation is grad_theta J(theta), J = E_{omega, xi}[R], under both designs. For the mean-including-self version the expectation is (G - 1) / G times that gradient under both designs.

Why. The term R_i S_i has the score-function expectation grad_theta E[R], because omega enters the trajectory only through observations and S_i differentiates the policy's action probabilities alone. The baseline term b_i S_i, with b_i the mean of the other rollouts' rewards, has expectation zero: conditioned on omega and on xi_j for j != i, b_i is a constant, and E_{xi_i}[S_i | omega, xi_{-i}] = sum_t E[grad log pi(a_t | s_t) | s_t] = 0 at every step because sum_a pi(a | s) grad log pi(a | s) = grad sum_a pi(a | s) = 0 at any state, including states produced by a shared omega. Sharing omega correlates R_i with b_i, which is the point, and changes nothing about the expectation.

What this does not cover. With std-normalization the denominator s + eps is a function of all G rewards, including R_i, and cannot be pulled through the baseline expectation; pairing changes the joint law of the rewards and therefore the law of s. The two designs' expected updates can differ under normalization. In the bandit of Section 3b they differ in the third decimal (0.3918 versus 0.3905), and in general the normalized estimator estimates a reweighted objective in both designs, which is the difficulty bias that TRL's documentation describes. The paper states the unbiasedness theorem for the unnormalized estimator and measures, rather than asserts, what happens under the implemented loss (Section 6).

## 3. Variance of a within-group reward contrast

Decompose the reward variance for a fixed prompt:

    sigma2_env = Var_omega( E_xi[ R | omega ] )      between-schedule variance, the part of the reward explained by luck,
    sigma2_pol = E_omega( Var_xi[ R | omega ] )      within-schedule variance, the part explained by the policy's own sampling,
    Var(R) = sigma2_env + sigma2_pol.

Define the luck share lambda = sigma2_env / (sigma2_env + sigma2_pol), an intraclass correlation in [0, 1]. Note that sigma2_pol is "everything not explained by the schedule", which includes policy-environment interaction (the same schedule can be survived or not depending on what the policy does); it is not pure skill.

Claim. For any two siblings i != j,

    Var(R_i - R_j) = 2 (sigma2_env + sigma2_pol)     under the independent design,
    Var(R_i - R_j) = 2 sigma2_pol                     under the paired design.

Why. Var(R_i - R_j) = Var(R_i) + Var(R_j) - 2 Cov(R_i, R_j). Independent siblings are uncorrelated. Paired siblings share omega, so Cov(R_i, R_j) = Var_omega(E[R | omega]) = sigma2_env, and the covariance term cancels exactly the between-schedule part.

Contrast-precision equivalent. Matching the precision of a paired reward contrast with independent rollouts takes 1 / (1 - lambda) of them per rollout (`contrast_precision_equivalent`). This is a statement about reward contrasts and nothing else. It is not an effective sample size for estimating the unconditional reward mean (there the exchangeable-correlation formula G / (1 + (G - 1) lambda) applies and pairing makes things worse), and it is not an effective sample size for the policy gradient, whose variance is the subject of the next section.

## 3b. From reward contrasts to gradient variance

The gradient estimate is g = (1/G) sum_i A_i S_i. Its variance depends on the joint law of the advantages and the scores, not on the reward contrasts alone. By the law of total variance,

    Var(g) = E_omega[ Var(g | omega) ] + Var_omega( E[g | omega] ),

where under the paired design omega is the group's single schedule and under the independent design it is the G-tuple. Pairing removes the luck offsets from every advantage (the first term shrinks) but makes the whole group's expected gradient depend on one schedule (the second term grows: a schedule that makes the task hopeless or trivial makes the whole group uninformative at once, where independent siblings would have averaged over several schedules). Whether the sum falls is a quantitative question. For outcome noise it has an exact answer.

Exact condition under one-sided outcome noise. Let the environment be clean apart from the grader: a true success (r_i = 1) is observed as R_i = Z_i r_i with Z_i ~ Bernoulli(1 - q); a true failure is always observed as 0. Under the independent design the Z_i are independent; under the paired design Z_i = Z is shared by the group. Use mean-centered advantages and write g_c = (1/G) sum_i (r_i - r_bar) S_i = (1/G) sum_i r_i (S_i - S_bar) for the clean gradient of the group (S_bar the mean score). Then g_paired = Z g_c, g_indep = (1/G) sum_i Z_i r_i (S_i - S_bar), both have expectation (1 - q) E[g_c], and

    Tr Var(g_paired) - Tr Var(g_indep) = q (1 - q) [ E ||g_c||^2 - (1/G^2) E sum_i r_i ||S_i - S_bar||^2 ].

Why. Tr Var(Z g_c) = (1 - q) E||g_c||^2 - (1 - q)^2 ||E g_c||^2. For the independent design, conditioning on the clean episode, E||g_indep||^2 = (1 - q)^2 E||g_c||^2 + q (1 - q) (1/G^2) E sum_i r_i ||S_i - S_bar||^2 (the diagonal terms have E Z_i^2 = 1 - q, the off-diagonal ones (1 - q)^2), and the squared means agree. Subtracting gives the display.

So pairing lowers the gradient variance if and only if

    E ||g_c||^2  <  (1/G^2) E sum_i r_i ||S_i - S_bar||^2.

The left side is the size of the clean group gradient (squared mean plus variance); the right side is the score spread of the truly successful rollouts. Two regimes make the condition concrete.

- Aligned scalar scores (the reviewer's bandit). One-step bandit, action a ~ Bernoulli(sigmoid(theta)) at theta = 0, true reward r = a, score S = a - 1/2, G = 8, q = 0.1. Exact enumeration: contrast variance 0.495 (independent) versus 0.450 (paired), but Tr Var(g) 0.002615 (independent) versus 0.005845 (paired): pairing lowers the contrast variance by 9 percent and raises the gradient variance 2.2-fold. The condition explains it: E||g_c||^2 = 0.0496 (of which the squared mean is 0.0388, since every group pushes in the same direction) against (1/G^2) E sum r_i ||S_i - S_bar||^2 = 0.0137, and q (1 - q) times the difference is 0.00323, exactly the measured gap. With std-normalized advantages the gap is larger still (0.00705 versus 0.01972) and the two expected updates differ. Losing a whole informative group at once is worse than perturbing one of its members when the group's clean gradient is large and consistent.
- Orthogonal scores of equal norm. If the S_i are mutually orthogonal with ||S_i||^2 = s^2, which is the regime of high-dimensional trajectory scores whose per-token contributions rarely repeat across rollouts, then for a group with k true successes ||g_c||^2 = s^2 k (G - k) / G^3 and (1/G^2) sum r_i ||S_i - S_bar||^2 = s^2 k (G - 1) / G^3. The condition reads G - k < G - 1, that is k >= 2: pairing lowers the gradient variance for every group with at least two true successes and ties at k <= 1. In particular for an all-correct group (k = G) the left side is exactly zero and the right side is not, whatever the score geometry: this is the spurious-variance group of Section 4, where pairing wins unconditionally, and the fraction of such groups grows as the policy improves.

What the paper claims. Not a theorem that pairing lowers gradient variance, but the exact condition above, the two regimes, and a measurement (Section 6, H1) of both sides of the inequality on the actual policy: the score geometry of an LLM policy decides the sign, and it is measured rather than assumed. For transition noise there is no closed form; the same decomposition is measured directly by resampling groups at a frozen checkpoint under both designs.

What pairing does not do. It never reduces sigma2_pol, and it cannot pair randomness the trainer does not control. In this environment omega is the fault schedule, keyed by the logical request (tool, resource, repeat index), so two rollouts that issue the same request for the k-th time meet the same fate; whether a rollout issues that request at all is the policy's choice, so alignment is by event, not by call index.

## 4. Spurious-variance groups under outcome noise

Let a grader flip turn a true success into an observed failure with probability q, independently per rollout under the independent design and once per group under the paired design.

Claim. A group whose G rollouts are all truly successful has non-zero observed variance with probability

    P_spurious(q, G) = 1 - (1 - q)^G - q^G,

which is 0.15 at q = 0.02, 0.34 at q = 0.05 and 0.57 at q = 0.10 for G = 8 (`spurious_rate_all_correct`). Under the paired design the probability is 0: either nobody is flipped or everybody is, and both are zero-variance groups with zero advantages.

Magnitude. With TRL's normalization, one flipped rollout among 8 truly correct ones receives A = -0.875 / (sqrt(0.125) + 1e-4) = -2.47 and each sibling +0.35; without normalization -0.875 and +0.125. The group carries no information about skill and emits a full-magnitude update against a correct trajectory.

What this does and does not imply. The expected noisy reward is (1 - q) times the clean one, so for the unnormalized estimator the spurious groups do not change the optimum or bias the gradient; they add variance, of exactly the amount computed in Section 3b (the all-correct case), and under normalization they also change the expected update. The rate of spurious groups therefore predicts nothing about learning-curve magnitudes by itself; it is a mechanism check (H1) whose consequence for learning is measured (H2b). The fraction of all-correct groups rises with competence, so under the independent design the fraction of pure-noise updates rises as the policy improves; the paired design converts those groups into zero-variance groups, which is what an all-correct clean group already is.

## 5. Estimating the luck share

For a fixed policy and task, run K schedules and M policy samples per schedule, rewards r[k, m]. One-way random-effects estimates:

    MS_between = M / (K - 1) * sum_k (mean_k - mean)^2
    MS_within  = 1 / (K (M - 1)) * sum_k sum_m (r[k, m] - mean_k)^2
    sigma2_env = max(0, (MS_between - MS_within) / M)
    sigma2_pol = MS_within
    lambda     = sigma2_env / (sigma2_env + sigma2_pol), undefined when the denominator is 0.

Two summaries are reported: the mean of lambda over the diagnostic tasks where it is defined (`luck_share_over_tasks`), and the ratio of the averaged components (`luck_share_pooled`). The first has a clipping bias at the null: with independent Bernoulli(0.5) rewards and 8 x 8 tables it averages about 0.028 although lambda = 0, because sigma2_env is clipped at zero. Both are reported with bootstrap intervals over tasks, with the component estimates and with the number of undefined tasks. Tables are kept separate per checkpoint (`luck_share_tables_by_phase`); merging checkpoints would combine a changing policy into a supposed fixed-policy design. Two tests check recovery of a known lambda within 0.05 for Gaussian rewards and for Bernoulli rewards whose success probability varies with the schedule.

A paired arm's within-group luck share is close to zero by construction (its siblings share the schedule); that number is a check of the implementation, not a learning result.

## 6. What the experiments must show, in these terms

- H1 (mechanism) has two parts. (a) Counts: the independent arm's spurious-variance rate among all-correct training groups (from the training-group register, actual trainer groups with their advantages) matches P_spurious within 0.07, the paired arm's is at most 0.02, and the luck share at step 0 under transition noise is at least 0.15 with a bootstrap interval that excludes 0.05. (b) Gradients: at frozen checkpoints, the gradient probe measures both sides of the Section 3b inequality on the LoRA parameters (E||g_c||^2 against (1/G^2) E sum r_i ||S_i - S_bar||^2), and the trace of the gradient covariance under both designs at matched rollout cost, for outcome noise and for transition noise. Prediction P13: the inequality holds on the LLM policy (scores are far from the aligned-scalar regime), so pairing lowers the measured gradient variance, with the margin growing with the all-correct fraction.
- H2 (learning) is the claim that needs data: whether the variance difference of (b) translates into faster learning at matched rollout and token cost. Primary endpoint: area under the true-success curve on the validation set over the fixed step budget; secondary: steps to a threshold with censoring, never imputation.
- H3 (recovery) needs a controlled comparison: matched challenge episodes in which a reachable request receives a forced recoverable fault, reported together with unconditional success and exposure, since success conditional on exposure alone compares different subsets for different policies.
- H4 (normalization) crosses design (paired, independent) with reward scaling (group, none) at a fixed loss aggregation; it follows from the magnitude computation of Section 4 and from the normalized gap in the bandit.
- H5 (diversity) addresses the one cost of pairing that the theory makes visible: the number of distinct schedules per batch drops from B x G to B. It is a non-inferiority claim on held-out fault types with a pre-registered margin, not an absolute-gap test.

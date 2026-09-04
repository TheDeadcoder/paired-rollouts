# Luck and skill in group-relative advantage estimation

Working notes for the paper's theory section. Plain notation; every statement here has a matching implementation in `src/pairedrl/analysis/diagnostics.py` or a matching test.

## 1. Setup

Fix a prompt x and a policy pi with parameters theta. An episode draws two independent sources of randomness: the environment's, omega (fault schedule, simulated-user latent, grader flakiness), and the policy's own sampling, xi. A rollout is tau = tau(omega, xi) and its reward is R = R(omega, xi).

A group is G rollouts of the same prompt. Group-relative methods (GRPO, DAPO, Dr. GRPO, GSPO, RLOO) score rollout i by a contrast against its siblings:

    A_i = (R_i - mean_j R_j) / (s + eps)        with s the Bessel-corrected std of the group and eps = 1e-4 (TRL),

or A_i = R_i - mean_j R_j when scaling is off. The policy-gradient estimate for the group is

    g = sum_i A_i * grad_theta log pi(tau_i).

Two designs differ only in how omega is drawn.

- Independent (the status quo): omega_1, ..., omega_G are i.i.d. draws from P(omega). Every rollout lives in its own world.
- Paired (common random numbers): one omega is drawn per group and shared by all G rollouts. Different groups still get different omegas.

In both designs, xi_1, ..., xi_G are independent, and each rollout's marginal law is the same.

## 2. Unbiasedness under both designs

Claim. For both designs, E[g] is the gradient of the same objective J(theta) = E_{omega, xi}[R] up to the usual self-inclusion factor of the group mean.

Why. Write the contrast as A_i = c_i * (R_i - b_i) where b_i is the mean of the other rollouts' rewards and c_i collects the (1 - 1/G) factor and the scaling. The term R_i * grad log pi(tau_i) has the score-function expectation grad_theta E[R], because the environment's randomness enters the trajectory only through observations and grad_theta log pi differentiates the policy's action probabilities alone. The baseline term b_i * grad log pi(tau_i) has expectation zero whenever b_i is independent of xi_i conditional on everything else: conditioned on omega and on xi_j for j != i, b_i is a constant and

    E_{xi_i}[ grad_theta log pi(tau_i) | omega, xi_{-i} ] = sum_t E[ grad_theta log pi(a_t | s_t) | s_t ] = 0

at every step, because sum_a pi(a | s) grad log pi(a | s) = grad sum_a pi(a | s) = 0 for any state, including states produced by a shared omega. So sharing omega across siblings changes nothing about the expectation; it only correlates R_i with b_i, which is the whole point. The scaling factor c_i is a function of the group's rewards and introduces the same mild dependence in both designs; with the leave-one-out mean and no scaling (RLOO) the estimator is exactly unbiased in both.

Consequence for the paper: pairing is not a different objective and not a bias-variance trade; it is variance reduction for the same objective.

## 3. Variance of a within-group contrast

Decompose the reward variance for a fixed prompt:

    sigma2_env = Var_omega( E_xi[ R | omega ] )      between-schedule variance, the part of the reward explained by luck,
    sigma2_pol = E_omega( Var_xi[ R | omega ] )      within-schedule variance, the part explained by the policy's own sampling,
    Var(R) = sigma2_env + sigma2_pol.

Define the luck share lambda = sigma2_env / (sigma2_env + sigma2_pol), an intraclass correlation in [0, 1].

Claim. For any two siblings i != j,

    Var(R_i - R_j) = 2 (sigma2_env + sigma2_pol)     under the independent design,
    Var(R_i - R_j) = 2 sigma2_pol                     under the paired design.

Why. Var(R_i - R_j) = Var(R_i) + Var(R_j) - 2 Cov(R_i, R_j). Independent siblings are uncorrelated. Paired siblings share omega, so Cov(R_i, R_j) = Cov(E[R_i | omega], E[R_j | omega]) = Var_omega(E[R | omega]) = sigma2_env, and the covariance term cancels exactly the between-schedule part.

Effective sample size. The same cancellation applies to the baseline. The group mean has variance (sigma2_env + sigma2_pol) / G under the independent design and, conditional on the shared omega, sigma2_pol / G under the paired design; matching the paired precision with independent rollouts takes G / (1 - lambda) of them (`effective_group_size`). At lambda = 0.5 pairing doubles the effective group; at lambda = 0 it does nothing. Lambda is the quantity the diagnostic subset measures during training (`luck_share`, Section 5 below).

What pairing does not do. It never reduces sigma2_pol, and it cannot pair randomness the trainer does not control. In this environment omega is the fault schedule, so pairing is exact for the four training fault types; the rate limit is stateful (its realized responses depend on the agent's own behavior after the drawn fault), and pairing applies to the drawn fates, not to the realized responses, which is the intended meaning: same weather, different sailors.

## 4. Spurious-variance groups under outcome noise

Now let omega include a grader flip with probability q that turns a true success into an observed failure, independently per rollout in the independent design and once per group in the paired design.

Claim. Consider a group whose G rollouts are all truly successful. Under the independent design its observed rewards have non-zero variance with probability

    P_spurious(q, G) = 1 - (1 - q)^G - q^G,

which is 0.15 at q = 0.02, 0.34 at q = 0.05 and 0.57 at q = 0.10 for G = 8 (`spurious_rate_all_correct`). Under the paired design the probability is 0: either nobody is flipped (rewards all 1) or everybody is (rewards all 0), and both are zero-variance groups that produce zero advantages.

Magnitude. With TRL's normalization, one flipped rollout among 8 truly correct ones receives A = -0.875 / (sqrt(0.125) + 1e-4) = -2.47 and each sibling receives +0.35. The group carries no information about skill, yet it emits a full-magnitude update that pushes the policy away from a correct trajectory. Without scaling the same group emits -0.875 and +0.125, smaller but not zero. This is hypothesis H4: std-normalization amplifies spurious variance, so the independent design's penalty should shrink when scaling is off, without vanishing.

Why it grows during training. The fraction of all-correct groups rises as the policy improves, so under the independent design the fraction of updates that are pure noise rises with competence. The paired design converts those groups into zero-variance groups, exactly what a clean all-correct group already is.

Mixed groups. When some siblings are truly correct and others are not, independent flips corrupt labels at rate q while the paired design loses the whole group with probability q (all correct siblings flipped together) and keeps it intact otherwise. Both remain unbiased; the difference in this regime is the label noise versus a small loss of rollouts.

## 5. Estimating the luck share from a two-way design

For a fixed policy and task, run K schedules and M policy samples per schedule, giving rewards r[k, m]. Standard one-way random-effects estimates:

    MS_between = M / (K - 1) * sum_k (mean_k - mean)^2
    MS_within  = 1 / (K (M - 1)) * sum_k sum_m (r[k, m] - mean_k)^2
    sigma2_env = max(0, (MS_between - MS_within) / M)
    sigma2_pol = MS_within
    lambda     = sigma2_env / (sigma2_env + sigma2_pol), undefined when the denominator is 0.

The task-level estimate is averaged over the 16 diagnostic tasks where it is defined (`luck_share_over_tasks`). Two tests check recovery of a known lambda within 0.05 for Gaussian rewards and for Bernoulli rewards whose success probability varies with the schedule, which is the case that matters here.

## 6. What the experiments must show, in these terms

- H1 measures P_spurious directly on training groups (outcome noise) and lambda at step 0 (transition noise); both have closed forms or known targets, so this is a check that the implementation matches the theory.
- H2 and H3 are the claims that need data: that the variance reduction of Section 3 and the removal of the noise groups of Section 4 translate into faster learning and better recovery behavior at realistic noise levels, which theory alone cannot settle because it depends on lambda along the training trajectory and on how the optimizer responds to the spurious updates.
- H4 follows from the magnitude computation in Section 4.
- H5 addresses the one cost of pairing that the theory makes visible: the number of distinct schedules per batch drops from B x G to B.

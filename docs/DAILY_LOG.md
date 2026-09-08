# Daily log

One entry per working day. Fields are mandatory. Spend is cumulative in USD by provider.

## Template

### YYYY-MM-DD, day N: title

- Commits: <short sha and message, one per line>
- Spend to date: Modal 0.00 | DigitalOcean 0.00 | GCP 0.00 | Daytona 0.00 | API 0.00
- Checks passed: <list from the runbook's "Day passed when" block, each marked PASS or FAIL>
- Deviations: <none, or a reference to docs/DEVIATIONS.md>
- Notes: <anything the next day needs to know>

## Entries

Spend figures are the sum of `estimated_cost_usd` over the run manifests (wall time x 3.95 USD/h for an H100 on Modal); they exclude image builds and container start-up, so the dashboard balance is the authority and is entered by hand at the freeze.

### 2026-09-02, day 0: freeze and scaffold

- Commits: a14f5c6 init
- Spend to date: Modal 0.00 | DigitalOcean 0.00 | GCP 0.00 | Daytona 0.00 | API 0.00
- Checks passed: repository scaffold PASS; pre-registration v0 drafted PASS; freeze deferred to after calibration (see DEVIATIONS)
- Deviations: none
- Notes: Solo submission under the ICLR 2027 reciprocal-reviewing exemption (no qualifying author; one submission cap). No co-author will be added.

### 2026-09-04, day 2: environment, noise layer, trainer integration, stack check

- Commits: 708672f modal integration check; a6d3908 world state and tool API; 2ca3da8 task generator and grader; ca2616c task datasets; 8c6e16d noise layer; 32e839b group diagnostics; a45e11c TRL environment adapter; fa63c84 run specification, trainer, evaluation schedule, Modal entry point
- Spend to date: Modal 0.30 | DigitalOcean 0.00 | GCP 0.00 | Daytona 0.00 | API 0.00
- Checks passed: Modal stack check Qwen3.5-2B (3 toy steps, vLLM colocate, tool mask) PASS; Qwen3-1.7B stack check FAIL (fallback list closed); unit tests PASS
- Deviations: model fallback (DEVIATIONS 2026-09-04)
- Notes: TRL 1.12.0, vLLM 0.27.1, transformers 5.16.1, torch 2.13.0, peft 0.20.0 pinned in the train extra.

### 2026-09-05, day 3: calibration v1 and v2, detached launches, speed checks

- Commits: 2483d32 evaluation memory guard and calibration register; 8fa9475 bf16 policy and one-sequence log-prob chunks after the OOM; d6ec810 detached Modal protocol (deploy, spawn, collect, checkpoint to the volume) and the v1 calibration register; a6fd4f4 environment v2; 40d262d v2 calibration register; f9cf41b vLLM engine overrides, speed-check specs, retries, 24 tool iterations; 14c3ba7 speed checks launched
- Spend to date: Modal about 42 (two OOM attempts, calib v1 2B 7.63, calib v1 4B cancelled by the client disconnect, calib v2 2B 12.47, calib v2 4B 11.09, speed checks 3.43) | DigitalOcean 0.00 | GCP 0.00 | Daytona 0.00 | API 0.00
- Checks passed: calibration completes end to end PASS (second attempt); P12 (lambda at least 0.15) FAIL on v1 (0.030), PASS on v2 (0.222 for 2B, 0.581 for 4B); detached launch survives a client disconnect PASS; speed checks A/B/C indistinguishable (230 to 336 s per 128 episodes) PASS
- Deviations: OOM relaunch and launch protocol; environment v1 to v2 (DEVIATIONS 2026-09-05)
- Notes: generation time is set by the longest episode's decode; prefix caching and batched-token limits do not help. Modal preempted the v2 2B container once and retried it by itself; retries are now explicit and attempt-preserving.

### 2026-09-06, day 4: speed register, gate-1 specs, external review, environment v3

- Commits: bd2bf4e speed-check register and gate-1 specs at 100 steps; (this commit) review fixes: per-task evaluation schedules, resource-keyed fault events, per-checkpoint luck tables, training-group register, trainer checkpoints with resume and attempt directories, test pool, C2r mixture, matched challenge set, theory v2, pre-registration v0.2
- Spend to date: Modal about 42 | DigitalOcean 0.00 | GCP 0.00 | Daytona 0.00 | API 0.00
- Checks passed: review defects reproduced and fixed with regression tests PASS; scripted-reference register rebuilt on both pools under the frozen evaluation schedules PASS; unit tests and ruff PASS
- Deviations: environment v3 and protocol fixes; run matrix and budget (DEVIATIONS 2026-09-06)
- Notes: numbers measured before this commit were produced under the old fault keying and evaluation seeds and are not compared with v3 numbers. Next: v3 zero-shot calibration (eval only), gate 1 at this commit if not already running, gradient probe, then the v1 freeze.

### 2026-09-07, day 5: smoke test of the v3 training path, kernel fix, step timings

- Commits: 5c8b2ee review fixes (environment v3, protocol, theory v2, pre-registration v0.2); 67b296f smoke-test specs, collect skips weights; (this commit) per-step timings in the manifest, loss log and train mode restored across in-step evaluations, flash-linear-attention kernels in the training image
- Spend to date: Modal about 44 (smoke-v3-train 1.78) | DigitalOcean 0.00 | GCP 0.00 | Daytona 0.00 | API 0.00
- Checks passed: training-group register lines up with the trainer's batch (12 groups, TRL advantages to the last digit) PASS; trainer checkpoints saved every step PASS; run completes end to end (27 min) PASS; training loss logged at every step FAIL (1 of 3: in-step evaluations cleared the trainer's log flag; fixed in this commit); resume from a checkpoint NOT YET TESTED (smoke-v3-resume)
- Deviations: none new
- Notes: the three smoke steps of 32 rollouts took about 250 s each including their checkpoint save, far above the 90 s per step of training passes assumed for 192 rollouts. Likely cause: the image had no `fla` package, so transformers ran the Qwen3.5 GatedDeltaNet layers through its pure-torch chunked fallback during the training passes (vLLM has its own kernels, so generation was unaffected). flash-linear-attention 0.5.2 is added to the image and the train extra; the resume test measures the step time with it before gate 1 is launched. The manifest now records `fla_importable` and per-step `step_timings` (step seconds, generation seconds, save seconds). Gate 1 launches only after the step time is known; the clean gate-1 run follows the C2 run.

### 2026-09-08, day 6: gate 1 complete, v3 registers, DigitalOcean stack check

- Commits: 93f13ba kernels, timings, loss log; 62caa40 resume test launched; (Instruction 27) calibration and gate 1 launched; (Instruction 28) registers/calibration_env_v3.json; f1360e3 local stack check and DigitalOcean runbook; e94bbef DigitalOcean stack-check register; (this commit) run register for gate 1, pre-registration v0.3
- Spend to date: Modal about 119 (resume 0.97, calib-v3 8.13, gate 1 64.21 on top of 44) | DigitalOcean about 2 | GCP 0.00 | Daytona 0.00 | API 0.00
- Checks passed: resume from checkpoint PASS (attempt 2, checkpoint-3, attempt1/ preserved); gate 1 end to end PASS (100 steps, 16.26 h, 27,592 episodes, 2,400 groups, all 100 losses logged); learning under noise PASS (test pool, zero-shot to step 100: clean 0.789 to 0.895, p = 0.10 0.695 to 0.819, p = 0.25 0.509 to 0.654, held-out types 0.703 to 0.798, challenge 0.540 to 0.780); P12 on v3 PASS (0.347 default mixture, 0.236 outage-free); DigitalOcean MI300X stack check PASS (0.396 episodes per second, all toy checks except the first-step time)
- Deviations: none new
- Notes: validation at-training-noise curve 0.570, 0.648, 0.633, 0.664, 0.641, 0.664 (AUC 0.641); T = 0.617. 52 percent of the paired arm's training groups had zero variance (30 percent all-correct, 22 percent all-fail). Mean step 460 s (generation 270 s). Only checkpoints 80 and 100 survive save_total_limit 2; the sweep keeps all five. Next: local job runner for the MI300X, freeze v1, tier-1 launch on both providers, gradient probe on the gate-1 checkpoints.

### 2026-09-08, day 7: pre-registration v1 frozen, tier 1 launched

- Commits: 3cd1168 shared job body, local runner, remote launch and collect, all checkpoints kept; d368ad6 Freeze pre-registration v1 with the 13 tier-1 specs; (this commit) freeze commit recorded, pre-registration label on ledger rows and launch registers; (next commits) tier-1 launches on Modal (C2) and DigitalOcean (C4, C0)
- Spend to date: Modal about 119 | DigitalOcean about 2 | GCP 0.00 | Daytona 0.00 | API 0.00
- Checks passed: pre-registration v1 frozen at 2026-09-08T12:33:49Z PASS (SHA-256 6dc709de6f36f008da3a9f04e7de447f862e039afe39db550fda8a341430c2cf; RFC 3161 tokens: https://freetsa.org/tsr Sep  8 12:34:39 2026 GMT; http://timestamp.digicert.com Sep  8 12:34:40 2026 GMT; freeze commit d368ad6b964355a5df212d0912383341cba6056b); tier-1 specs validated PASS; ledger label check (a pre-registration label is refused for a spec whose notes do not carry it) PASS
- Deviations: none
- Notes: `ledger_row` appended "not pre-registered" to every row; the launch scripts now take `--preregistration v1` for the frozen specs. Tier 1: C2 paired seeds 1 and 2 and independent seeds 0 to 2 on Modal H100 (gate1-c2-paired-s0 is C2 paired seed 0), C4 paired and independent seeds 0 to 2 and C0 seeds 0 and 1 on DigitalOcean MI300X across three droplets. The gradient probe on the gate-1 checkpoints follows the launches.

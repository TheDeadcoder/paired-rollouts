# Deviations from the pre-registration and the runbook

Every departure from `docs/PREREGISTRATION.md` (after it is frozen) or from the runbook is recorded here before or at the time it happens, never afterwards. A deviation entry has: date, what was planned, what was done instead, why, and which results it affects. Entries before the v1 freeze record departures from the runbook and from the v0 draft; they are exploratory decisions and are listed so that the v1 document does not hide them.

## Entries

### 2026-09-04, model fallback list

- Planned: Qwen3-1.7B as the fallback model if the Qwen3.5 stack check failed.
- Done: the Qwen3-1.7B stack check failed and the Qwen3.5-2B check passed; the fallback list is closed and Qwen3.5-2B is the primary model.
- Affects: nothing pre-registered.

### 2026-09-05, calibration runs relaunched after CUDA OOM, and the detached launch protocol

- Planned: one calibration run per model, launched with `modal run`.
- Done: both first attempts failed with CUDA OOM (policy loaded in fp32 by the trainer default, 4-sequence log-prob chunk); relaunched after loading in bf16 with a 1-sequence chunk and a micro-batch of 2. The relaunch was cut off when the launching laptop lost its network and Modal cancelled the client's inputs; the 4B run was lost, the 2B run had committed its volume seconds earlier. Runs are now deployed once per commit and spawned detached (`scripts/launch_modal.py`), collected at any time (`scripts/collect_modal.py`), and retried by the platform on preemption with the previous attempt's evidence preserved.
- Affects: cost only (the failed attempts are in the ledger).

### 2026-09-05, environment v1 to v2 after the P12 miss

- Planned (runbook): freeze the environment after a single calibration.
- Done: the v1 calibration gave lambda = 0.030 (P12 failed) and exposed three defects (numeric postal codes rejected, saturated diagnostic tasks, benign fault mix keyed by call index). Environment v2 (request-keyed faults, outage type, budget 7 + 2n, task mix, diagnostic task rule) was built and re-calibrated (lambda = 0.222 for 2B). This is exploratory calibration; the v1 negative result stays in `registers/calibration_env_v1.json` and is reported in the paper.
- Affects: the noise mixture is a design choice informed by the model's zero-shot behavior; the paper says so and the outage-free condition C2r tests the mixture that was not tuned.

### 2026-09-06, environment v3 and protocol fixes after the external review (before the freeze)

- Found: (1) the four evaluation schedule seeds were shared by every task and never flipped at q = 0.05 or 0.10, so the v0 outcome-noise evaluation applied no outcome corruption; (2) fault events were keyed by all tool arguments including free text, so rewording a cancellation reason dodged a persistent outage (reproduced on heldout-00077, seed 0); (3) luck-share tables merged diagnostic checkpoints; (4) training episodes were stamped without step or attempt, and no training-group register existed; (5) trainer checkpoints were disabled, so a retry restarted from scratch and overwrote the previous attempt's logs; (6) `train_only` still ran the periodic evaluations; (7) the loss type in the v0 document (grpo) did not match the implemented default (dapo); (8) the validation pool had informed calibration and no separate test pool existed.
- Done: per-task evaluation schedules (`eval_schedule_seed`); fault events keyed by tool and resource (`fault_key`), with metamorphic tests; per-checkpoint luck-share tables with exact cardinality checks; phase and attempt stamps, event keys on every logged call, and the training-group register `groups.jsonl`; trainer checkpoints every 20 steps with resume and attempt directories; `train_only` honored; the document now states dapo; a test pool of 300 tasks generated after calibration and reserved for the final evaluation; the scripted-reference register rebuilt on both pools under the frozen evaluation schedules; the outage-free mixture C2r and the matched challenge set added; the theory notes corrected (THEORY.md section 3b).
- Affects: every number measured before 2026-09-06 (v1 and v2 calibrations, speed checks, and any gate-1 run launched at commit bd2bf4e) was produced under the old keying and the old evaluation seeds; they remain in their registers as exploratory measurements and are not compared with v3 numbers. The zero-shot levels are re-measured under v3 (`calib-v3-*`) before the freeze.

### 2026-09-06, run matrix and budget

- Planned (runbook): 27 pre-registered runs including a 9B scale check and a flaky-test census, at an assumed 1.5 hours per run.
- Done: measured generation speed (230 to 336 s per batch of 128 episodes, independent of prefix caching and batched-token limits) gives about 17 hours per 100-step run; the matrix is tiered and the funded list is set at the freeze from the gate-1 measured step time; 9B and the census are dropped, 4B is last.
- Affects: which conditions are confirmatory (tier 1) and which are "run if budget allows".

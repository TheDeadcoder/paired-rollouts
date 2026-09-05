# Run ledger

Every training or evaluation job that starts is appended here at launch time, whether or not it finishes. A run that crashes is relaunched with the same seed and both rows stay.

| run_id | launched_utc | provider | model | condition | arm | seed | steps | status | notes |
|---|---|---|---|---|---|---|---|---|---|
| stackcheck-modal-qwen3.5-2b | 2026-09-04T05:19:56+00:00 | modal | Qwen/Qwen3.5-2B | stack_check | toy | 0 | 3 | COMPLETE | exploratory, not pre-registered |
| stackcheck-modal-qwen3-1.7b | 2026-09-04T05:23:13+00:00 | modal | Qwen/Qwen3-1.7B | stack_check | toy | 0 | 3 | FAILED | exploratory, not pre-registered |
| calib-qwen3.5-2b | 2026-09-05T03:14:21+00:00 | modal | Qwen/Qwen3.5-2B | CALIB | paired | 0 | 0 | FAILED | zero-shot calibration, eval only, commit 2483d32; CUDA OOM in the first log-prob pass (policy loaded in fp32 by TRL's default, 4-sequence chunk over a 248k vocabulary); exploratory, not pre-registered |
| calib-qwen3.5-4b | 2026-09-05T03:18:54+00:00 | modal | Qwen/Qwen3.5-4B | CALIB | paired | 0 | 0 | FAILED | zero-shot calibration, eval only, commit 2483d32; same CUDA OOM as the 2B run; exploratory, not pre-registered |
| calib-qwen3.5-2b | 2026-09-05T04:32:26+00:00 | modal | Qwen/Qwen3.5-2B | CALIB | paired | 0 | 0 | COMPLETE | zero-shot calibration, eval only, second attempt after the bf16 fix, commit 8fa9475; finished 06:28:20 UTC, 6954 s, 7.63 USD, peak 53.6 GB; the container reached its final volume commit seconds before Modal cancelled the `run.map` inputs of the disconnected client; 3424 episodes verified complete; register: registers/calibration_env_v1.json (environment v1 numbers, superseded by the v2 calibration) |
| calib-qwen3.5-4b | 2026-09-05T04:32:00+00:00 | modal | Qwen/Qwen3.5-4B | CALIB | paired | 0 | 0 | CANCELLED | zero-shot calibration, eval only, second attempt, commit 8fa9475; launch time approximate; the launching laptop lost its network at 06:28 UTC and Modal cancelled the `run.map` input after two final sets (1600 episodes); nothing was committed to the volume |

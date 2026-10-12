# DigitalOcean MI300X

Second provider for the sweep (AMD MI300X, 192 GB, 1.99 USD/h). A condition is never split across providers; a run's manifest records the provider, the GPU and every package version.

## Droplet

GPU Droplet, 1 x AMD MI300X, quick-start package "vLLM 0.27.1" (ROCm 7.14 host, Ubuntu 24.04, Docker preinstalled; the same vLLM version as the Modal image), SSH key login as root. Billing runs while the droplet exists; destroy it when it is not needed.

## Container

The training stack runs on the package's vLLM 0.27.1 (its Docker image when the package ships one, otherwise the host Python), so that vLLM, Triton and torch come prebuilt for ROCm; the stack check records which. The remaining packages are installed on top at the Modal pins (`trl==1.12.0`, `transformers==5.16.1`, `peft==0.20.0`, `accelerate==1.14.0`, `datasets==5.0.1`, `flash-linear-attention[rocm]==0.5.2`) after a `pip install --dry-run` shows that neither torch nor vllm would be replaced; if the dry run wants to replace either, the check stops there and the image choice is revisited.

```
docker run -d --name pairedrl --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 32g \
  -v /root/work:/work --entrypoint bash <image> -c "sleep infinity"
docker exec pairedrl bash -c "cd /work/paired-rollouts && pip install -e . && HF_HOME=/work/hf python scripts/stack_check_local.py --provider digitalocean --model Qwen/Qwen3.5-2B --real-eval 32"
```

## Stack check

`scripts/stack_check_local.py` runs the day-1 toy checks (row fields reach reset, tool calls parsed, tool mask applied with no sentinel among model tokens, three training steps, step time) and, with `--real-eval N`, one final evaluation of the real back-office runner on N test tasks whose per-set seconds and episodes per second are comparable with the Modal speed checks (H100, 384 episodes in 777 to 831 s, about 0.47 episodes per second; the v3 calibration, 4200 episodes in 5857 s, about 0.72). The register `registers/stack_check_digitalocean_qwen3.5-2b.json` is copied back and committed; its ratio to the H100 throughput is the number the budget uses for MI300X hours.

## Result (2026-09-08)

`registers/stack_check_digitalocean_qwen3.5-2b.json`: ROCm (HIP 7.2), torch 2.11 ROCm build, vLLM 0.27.1 (rocm723), trl 1.12.0, transformers 5.16.1, peft 0.20.0, fla importable. Toy checks: row fields, tool calls, tool mask (no sentinel among model tokens) and three training steps all pass; step times 68.5 s (one-time kernel compilation and autotuning), 2.2 s, 2.2 s, so the 60 s first-step check reads False and is recorded as such. Real evaluation batch: 544 episodes at 0.396 episodes per second (clean 181 s, p = 0.10 275 s, p = 0.25 407 s, held-out types 339 s, challenge 174 s per set), against 0.47 episodes per second for the H100 speed checks at the same set size, about 0.85 of the H100 on generation-bound work at half the hourly price. The MI300X is the provider for C4 and C0 in tier 1.

## Rate

The list price of a 1 x MI300X droplet is 2.59 USD/h since 2026-08-01 (DigitalOcean pricing page); the first droplet of this project is billed at 1.99 USD/h on its account. The rate an account actually bills (its billing page, hourly rate of the running droplet) is what `launch_remote.py --usd-per-hour` records in every manifest, so estimates in the registers are per account, not a repository constant. Billing runs while a droplet exists, powered off included.

## Bootstrapping a droplet

```
bash scripts/bootstrap_droplet.sh root@<ip> <vllm image repository:tag>
```

From a clean committed tree: clones the repository at HEAD into `/root/work/paired-rollouts`, stops the container the quick-start package started on boot, starts the `pairedrl` container on the package's vLLM image (`/root/work` mounted at `/work`), installs the training stack at the Modal pins on top of the image's torch and vLLM (stopping if pip would replace either), installs the package, downloads the model into `/work/hf` and prints the versions. About 15 minutes; idempotent.

## Running sweep jobs

Jobs run through the same body as the Modal function (`pairedrl.ops.job.run_job`) via `scripts/run_local.py`, so the run directory (manifest with `provider`, episodes, groups, trainer checkpoints, `attempt<n>/` on relaunch, `refused/` for refused invocations) is identical. From the laptop, with a committed tree (the ledger and the launch registers may be uncommitted from an earlier launch of the same batch; nothing else):

```
python scripts/launch_remote.py --host root@<ip> --specs configs/a.json,configs/b.json --label do-<name> --usd-per-hour <account rate> [--preregistration v1]
python scripts/collect_remote.py --register registers/launches/<utc>_do-<name>.json [--no-download] [--with-weights]
python scripts/verify_run.py outputs/runs/<run_id> [--weights]
```

`launch_remote` refuses a run id whose newest ledger row is still LAUNCHED or RUNNING (`--relaunch` overrides, for a job known to be dead) or that the ledger records as COMPLETE, refuses a host where a chain is already running (one job per GPU), moves the droplet's checkout to the laptop's commit (`git checkout -f`: the checkout is a deployment target, so an untracked file left there by an earlier script, such as a stack-check register committed later from the laptop, is replaced by the committed version rather than aborting the launch; modified tracked files are listed before they are discarded), reinstalls the package in the container, and starts one detached shell that runs the specs one after another, logging to `/root/work/runs/<spec>.log`; it appends `LAUNCHED` ledger rows with call id `<host>:<run_id>`. A spec whose job does not end COMPLETE is run a second time at once (the relaunch resumes from the newest complete checkpoint and keeps the first attempt under `attempt<n>/`), then the chain moves on; there is no other retry on this provider, so a run that fails twice waits for a human. `collect_remote` rsyncs each run directory without weights into `outputs/runs/`, prints the state, flags a manifest that has not progressed for two hours as STALE, and fills in the ledger. `verify_run` rebuilds the register along the attempt lineage and prints ACCEPT or REJECT with the reasons (status, evaluation and training identities against the frozen pools, phases outside the plan, diagnostic cardinality, group rows and sizes, totals, pool hashes, attempt consistency, and with `--weights` the checkpoint files and `adapter_final/`).

## The droplet queue (two jobs per GPU)

From the revision on, a droplet runs one job queue (`scripts/run_queue.py`, `pairedrl/ops/queue.py`) instead of a chain. Entries run in order; an entry is one job or a concurrent pair on one GPU, and the next entry starts when every job of the current one has ended. Entries can be appended while the queue runs, so probes join after a training pair has started.

```
python scripts/launch_remote.py --host root@<ip> --entry configs/<solo>.json --entry configs/<paired>.json,configs/<independent>.json --label do-<name> --usd-per-hour <account rate> [--preregistration v2]
python scripts/launch_remote.py --host root@<ip> --entry configs/<later>.json --label do-<name>-2 --append
python scripts/collect_remote.py --register registers/launches/<utc>_do-<name>.json
```

- Each `--entry` is one entry; two comma-separated specs in one `--entry` are a pair. Without `--append` a running queue is refused, as is a host running jobs outside a queue.
- Every entry runs from a checkout of its own commit (`/root/work/checkouts/<commit>`, `/work/checkouts/<commit>` in the container) with `PYTHONPATH` set to it, so an entry appended at a later commit never changes the code of jobs already queued or running; `run_local.py` refuses to run if the package is imported from anywhere else.
- The two jobs of a pair take slots 0 and 1: their own `MASTER_PORT` (29500, 29600; TRL 1.12.0 keeps a set value), `VLLM_PORT` (31000, 31100) and temp, Triton, Inductor and vLLM cache directories. The second job starts once the first has finished vLLM initialization (its manifest carries the engine facts), or after 30 minutes. `vllm_gpu_memory_utilization` stays the spec field (0.20 for concurrent jobs); the optional `vllm_kv_cache_memory_bytes` fixes vLLM's KV cache (vLLM 0.27.1 then ignores the utilization), so a job's start-up does not depend on the other job's memory at that moment. Each manifest records its `placement` (slot, ports, directories, start offset, the run sharing the GPU, the queue entry), the KV cache vLLM allocated (`vllm.kv_cache_tokens`), the initial adapter hash and the evaluation-set hash.
- Watchdog: progress is the growth of a job's `episodes.jsonl` or its stdout log (never the manifest, rewritten only after evaluation phases and every few steps). A job is stalled after 45 minutes without progress; the clock starts at the end of a 45-minute start-up grace, so a job that never progresses is stalled 90 minutes after it starts. A stalled job's process group is killed, the event goes to its run directory (`queue_events.jsonl`) and the queue status, and the job is relaunched under the same run id, resuming from its newest complete checkpoint. A job gets one relaunch in total, after a stall or an exit without COMPLETE; then it is FAILED and the queue moves on. A stalled or failed job never touches the other job of its pair. Queued training jobs write their manifest every 2 steps, so `collect_remote`'s two-hour STALE flag stays meaningful.
- Step-0 identity check (training pairs): as soon as both arms' manifests exist, their inputs are compared (commit, the spec without arm, run id and notes, the model snapshot, the initial adapter hash, the evaluation-set hash, the package versions). If any input differs, both arms are stopped and the entry fails: fix the difference and relaunch the pair once. If the inputs agree, the step-0 scores of every periodic set are compared once both arms have them; a difference is recorded (GPU nondeterminism, which a relaunch cannot remove) and both arms continue. The two arms start from the same adapter because `run_job` redraws every LoRA A matrix from the run seed once the trainer is built (`initial_adapter_seeded_matrices` in the manifest): TRL 1.12 creates the adapter before it applies the seed, so runs launched before 2026-10-12, tier 1 included, started each arm from its own adapter (the same starting policy, the base model, because B is zero).
- The queue directory (`/root/work/runs/queue`) holds `entries.jsonl`, `status.json` (entry and job states, attempts, events, step-0 checks, when the queue went empty), one log per job attempt under `logs/`, and `runner.log`; `collect_remote` prints it. The entry format is documented in `pairedrl/ops/queue.py`.
- The core evaluation profile (`"eval_profile": "core"`, with `"diagnostic_steps": []`) keeps the periodic noisy validation and matched challenge on exactly tier 1's tasks and schedules (the H2a set) and the final `noisy_p025`, `heldout_types`, `clean` and `challenge` sets (plus `at_training` when the run's own noise is none of them): about 3,750 evaluation episodes per run against about 7,400 for a full C2 run with two diagnostics. `verify_run` accepts a complete core run and rejects one missing a core set; the absent clean curve is absent in the run register, and the decision code refuses a rule that needs it.

## Sizing the chains

The production-shape smoke (`configs/smoke-do-train.json`, three steps of 24 x 8 rollouts with the tier-1 sequence limits, checkpoints every step, small evaluation sets; then `configs/smoke-do-resume.json`, which extends the same run by two steps from checkpoint-3) gives the MI300X training step time and exercises evaluation, checkpointing and resume before any pre-registered run is launched there. Hours per tier-1 run = 100 x (measured step seconds) / 3600 plus the evaluation phases (about 1.5 h at 0.4 episodes per second for a seed-0 spec, 1 h otherwise). A droplet's chain is sized so that its runs finish inside the account's credit with margin for collection: an account with 100 USD of credit at 1.99 USD/h has 50 hours; the chains are planned at 40 hours of jobs at most.

### Measured (2026-09-09, smoke-do-train, `registers/runs/smoke-do-train.json`)

Five production-shape C4 steps of 24 x 8 rollouts on the MI300X: 543.5, 378.7, 419.5 s in attempt 1 and 668.7, 518.7 s in attempt 2 (steps 1 and 4 carry each attempt's model load and kernel warm-up), generation 237 to 495 s of each, training passes about 140 to 175 s (faster than the H100's 186 s; the flash-linear-attention kernels run on ROCm). Warmed steps average about 440 s against the H100's 461 s, so a tier-1 run on the MI300X is planned at 480 s per step (13.3 h of training) plus the evaluation phases at about 0.57 episodes per second (0.85 of the H100's 0.67 measured on gate 1): a C4 seed-0 run about 17.8 h, other C4 seeds 17.3 h, C0 runs about 15.8 and 15.3 h. Resume from checkpoint-3 into attempt 2 worked (`attempt1/` preserved, `resumed_from_step` 3); the two-attempt total was 4674 s, 2.59 USD at 1.99 USD/h.

### Tier-1 chains

Re-planned on 2026-09-10 after t1-c4-paired-s0 measured 24.1 h and 47.98 USD (534 s per step, 14.8 h of training; evaluation phases and diagnostics 9.2 h, about 0.28 episodes per second over the 9,192 non-training episodes). Outcome noise leaves the environment dynamics clean, so a C0 run has the same training step time and the same evaluation load minus the at-training set: about 23.5 h (seed 0, three diagnostics) or 22.4 h (seed 1). A C4 run without the step-50 diagnostic (seeds 1 and 2) is about 23.0 h. Two runs of any kind are about 46 to 47 h and 92 to 94 USD, so a 100 USD account carries two runs only with the account's payment method covering a few dollars of overage, and one run otherwise.

| droplet | account | chain (in order) | planned hours | planned USD at 1.99 |
|---|---|---|---|---|
| A (destroyed 2026-09-10) | friend 1 | t1-c4-paired-s0 (24.1 h, 47.98 USD, collected with weights); t1-c0-clean-s0 cancelled | 24 | 48 |
| B | friend 2 | t1-c4-independent-s0 (COMPLETE, 22.9 h, 45.63 USD, collected and accepted with weights 2026-09-16, weights at Melikshah/paired-rollouts-weights revision 641a2a72, droplet retained and reused for t1-c4-independent-s2), t1-c4-independent-s2 | 46 | 92 |
| C | own | t1-c4-paired-s1 (COMPLETE, 21.7 h, 43.21 USD, collected and accepted with weights 2026-09-16, weights at Melikshah/paired-rollouts-weights revision 42dd6d51, droplet retained and reused for t1-c4-independent-s1), t1-c4-independent-s1 | 46 | 92 |
| D | own | t1-c4-independent-s2, t1-c4-paired-s2 (the seed pairs run in opposite arm order on C and D) | 46 | 92 |
| E | own | t1-c0-clean-s0, then t1-c0-clean-s1 if B runs one job | 23.5 or 46 | 47 or 92 |

Each seed pair runs both arms on one droplet, in opposite order on the two droplets, so that neither arm is always first. The own account carries 231 to 276 USD of chains and is topped up above that before C, D and E are created. Every chain leaves credit for collection with weights and verification; there is no archive droplet: the laptop copy verified by `verify_run --weights` plus the Hugging Face upload of the trainer checkpoints and the final adapter are the durable copies, and a droplet is destroyed as soon as its last run is collected and verified. Which droplet ran which seed is in the launch registers and the manifests (`hostname`).

## Before a droplet is destroyed

```
python scripts/collect_remote.py --register registers/launches/<utc>_do-<name>.json --with-weights
python scripts/verify_run.py outputs/runs/<run_id> --weights --threshold 0.617
python scripts/build_run_register.py --runs outputs/runs/<run_id> --threshold 0.617
python scripts/upload_weights.py outputs/runs/<run_id> --repo <user>/<name>
```

All four must pass for every run on the droplet: the register, episodes, groups, every `attempt<n>/`, the five trainer checkpoints and `adapter_final` are then on the laptop, `verify_run` has accepted them with the weights, the run register is written, and `upload_weights` (which refuses a run that `verify_run --weights` would not accept) has put every checkpoint file and the final adapter into the project's Hugging Face model repository in one commit and written `registers/weights/<run_id>.json` with the revision and the sha256 of every file. Only then is the droplet destroyed, and the table above records the date.

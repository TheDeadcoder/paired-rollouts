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

## Running sweep jobs

Jobs run through the same body as the Modal function (`pairedrl.ops.job.run_job`) via `scripts/run_local.py`, so the run directory (manifest with `provider`, episodes, groups, trainer checkpoints, `attempt<n>/` on relaunch) is identical. From the laptop, with a clean committed tree:

```
python scripts/launch_remote.py --host root@<ip> --specs configs/a.json,configs/b.json --label do-<name> [--preregistration v1]
python scripts/collect_remote.py --register registers/launches/<utc>_do-<name>.json [--no-download]
```

`launch_remote` moves the droplet's checkout to the laptop's commit, reinstalls the package in the container, and starts one detached shell that runs the specs one after another (one job per droplet at a time; more droplets for more concurrency), logging to `/root/work/runs/<spec>.log`; it appends `LAUNCHED` ledger rows with call id `<host>:<run_id>`. `collect_remote` rsyncs each run directory without weights into `outputs/runs/`, prints the state, flags a manifest that has not progressed for two hours as STALE, and fills in the ledger. A job that died is relaunched with the same spec: the previous attempt moves to `attempt<n>/` and training resumes from the newest checkpoint.

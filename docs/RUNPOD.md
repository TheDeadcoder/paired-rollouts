# RunPod

The third provider, used from 2026-09-16 for the tier-1 runs that the DigitalOcean droplets stopped carrying (DEVIATIONS 2026-09-16). A pod is one GPU host reached over SSH, so the DigitalOcean tooling applies unchanged with `--container none`: the job runs on the pod itself, detached, and nothing depends on the laptop.

## Pod

GPU Droplet equivalent: Secure Cloud, On-Demand (never Spot, which can be reclaimed mid-run), H100 SXM 80 GB at the rate the console shows (3.49 USD/h on 2026-09-16), CUDA filter 13.0 (the Modal runs recorded torch 2.13.0 on CUDA 13.0; the same PyPI wheel must load), the official Runpod PyTorch template the console offers for the host (on 2026-09-16 `Runpod PyTorch 2.4.0`, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`; neither its torch nor its CUDA toolkit is used: the venv installs the CUDA 13.0 wheels from PyPI, which need only the host driver, 580 or newer, so the console's "potential CUDA version mismatch" warning about the template's toolkit does not apply once the filter is pinned; the venv's Python is the template's `python3`, 3.11 there against 3.12 in the Modal image, recorded by the stack check), container disk 50 GB, volume disk `/workspace` 100 GB, SSH over a public IP with TCP 22 exposed, the public key stored in the account before the pod is created. Each pod gets an alias in `~/.ssh/config` (`Host runpod-a`, `HostName`, `Port`, `User root`, `IdentityFile`) so that `ssh`, `scp` and `rsync` reach it without options and the alias is the `--host` of the launcher and the collector.

## Bootstrapping

```
bash scripts/bootstrap_pod.sh runpod-a
```

Clones the repository at the laptop's HEAD into `/workspace/paired-rollouts`, creates `/workspace/venv` with the Modal image's pins from PyPI (one resolution, as `scripts/run_modal.py`), puts the venv first on PATH for SSH commands, installs the package, downloads the model into `/workspace/hf` and prints the versions; the script stops before installing anything when the host driver is older than 580; the last lines show which `python` an SSH command sees, which must be the venv's. Idempotent. The stack check follows on the first pod of the provider (`scripts/stack_check_local.py --provider runpod`, register `registers/stack_check_runpod_qwen3.5-2b.json`).

## Running sweep jobs

```
python scripts/launch_remote.py --host runpod-a --specs configs/<spec>.json --label runpod-a --provider runpod --usd-per-hour 3.49 --container none --remote-repo /workspace/paired-rollouts --runs-dir /workspace/runs --host-runs-dir /workspace/runs --hf-home /workspace/hf --preregistration v1
python scripts/collect_remote.py --register registers/launches/<utc>_runpod-a.json [--no-download] [--with-weights]
```

The launcher moves the pod's checkout to the launch commit, reinstalls the package, and starts one detached shell (`nohup`) that runs the specs one after another with the retry policy of the DigitalOcean chains; the SSH session ends and the pod keeps running the job. One chain per pod. Collection, verification, registration and archiving are exactly the DigitalOcean procedure (docs/DIGITALOCEAN.md, "Before a droplet is destroyed"); a pod is stopped only after `verify_run --weights` accepted its run and `upload_weights` archived it.

## Sizing

An H100 SXM runs a tier-1 C4 spec in about 17 h (100 steps at about 480 s plus 3 to 4 h of evaluation phases; the C2 runs on the Modal H100 took 14.1 to 17.4 h) and a C0 spec in about 16 h: about 59 and 56 USD at 3.49 USD/h, plus about 2 USD of bootstrap per pod. Stock is thin (one H100 SXM per pod, low availability), so runs are placed one per pod as pods become available, whole seed pairs on one provider.

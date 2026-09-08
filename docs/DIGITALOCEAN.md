# DigitalOcean MI300X

Second provider for the sweep (AMD MI300X, 192 GB, 1.99 USD/h). A condition is never split across providers; a run's manifest records the provider, the GPU and every package version.

## Droplet

GPU Droplet, 1 x AMD MI300X, image "AMD AI/ML Ready" (Ubuntu, ROCm, Docker preinstalled), SSH key login as root. Billing runs while the droplet exists; destroy it when it is not needed.

## Container

The training stack runs inside a vLLM ROCm image so that vLLM, Triton and torch come prebuilt for ROCm. Preferred image: the vLLM release image matching the Modal stack (`vllm/vllm-openai-rocm:v0.27.1`); fallback: `rocm/vllm:latest`, whose vLLM version is then recorded. The remaining packages are installed on top at the Modal pins (`trl==1.12.0`, `transformers==5.16.1`, `peft==0.20.0`, `accelerate==1.14.0`, `datasets==5.0.1`, `flash-linear-attention[rocm]==0.5.2`) after a `pip install --dry-run` shows that neither torch nor vllm would be replaced; if the dry run wants to replace either, the check stops there and the image choice is revisited.

```
docker run -d --name pairedrl --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 32g \
  -v /root/work:/work --entrypoint bash <image> -c "sleep infinity"
docker exec pairedrl bash -c "cd /work/paired-rollouts && pip install -e . && HF_HOME=/work/hf python scripts/stack_check_local.py --provider digitalocean --model Qwen/Qwen3.5-2B --real-eval 32"
```

## Stack check

`scripts/stack_check_local.py` runs the day-1 toy checks (row fields reach reset, tool calls parsed, tool mask applied with no sentinel among model tokens, three training steps, step time) and, with `--real-eval N`, one final evaluation of the real back-office runner on N test tasks whose per-set seconds and episodes per second are comparable with the Modal speed checks (H100, 384 episodes in 777 to 831 s, about 0.47 episodes per second; the v3 calibration, 4200 episodes in 5857 s, about 0.72). The register `registers/stack_check_digitalocean_qwen3.5-2b.json` is copied back and committed; its ratio to the H100 throughput is the number the budget uses for MI300X hours.

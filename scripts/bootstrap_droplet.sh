#!/usr/bin/env bash
# Prepare a fresh DigitalOcean MI300X droplet (quick-start package "vLLM 0.27.1") for sweep jobs, from the laptop:
#
#   bash scripts/bootstrap_droplet.sh root@<ip> <vllm docker image repository:tag> [model]
#
# Copies a small remote script to the droplet and runs it: clone the repository at this tree's HEAD into
# /root/work/paired-rollouts, stop whatever container the package started on boot so the GPU is free, start the
# `pairedrl` container on the package's vLLM image with /root/work mounted at /work, install the training stack at
# the Modal pins on top of the image's torch and vLLM (stopping if pip would replace either), install the package,
# download the model into /work/hf and print the versions. Idempotent: a second run refreshes the checkout and
# reinstalls. The same steps as docs/DIGITALOCEAN.md, which the first droplet followed by hand.
set -euo pipefail

HOST="${1:?ssh target, for instance root@203.0.113.5}"
IMAGE="${2:?docker image of the vLLM package, repository:tag}"
MODEL="${3:-Qwen/Qwen3.5-2B}"
COMMIT="$(git rev-parse HEAD)"
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$HOST")

"${SSH[@]}" "mkdir -p /root/work && cat > /root/work/bootstrap_remote.sh" <<'REMOTE'
set -euo pipefail
IMAGE="$1"; COMMIT="$2"; MODEL="$3"
PINS=(trl==1.12.0 transformers==5.16.1 peft==0.20.0 accelerate==1.14.0 datasets==5.0.1 'flash-linear-attention[rocm]==0.5.2')

echo "== host"
rocm-smi --showproductname | grep -i 'card series' | head -1 || true
docker --version
df -h /root | tail -1

echo "== checkout $COMMIT"
mkdir -p /root/work/runs
cd /root/work
if [ -d paired-rollouts/.git ]; then
  (cd paired-rollouts && git fetch -q origin)
else
  git clone -q https://github.com/TheDeadcoder/paired-rollouts.git
fi
cd paired-rollouts && git checkout -q "$COMMIT" && git rev-parse HEAD

echo "== container"
for c in $(docker ps --format '{{.Names}}' | grep -vx pairedrl || true); do
  docker update --restart=no "$c" > /dev/null
  docker stop "$c" > /dev/null
  echo "stopped $c"
done
if ! docker ps -a --format '{{.Names}}' | grep -qx pairedrl; then
  docker run -d --name pairedrl --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 32g \
    -v /root/work:/work --entrypoint bash "$IMAGE" -c 'sleep infinity' > /dev/null
fi
docker start pairedrl > /dev/null
docker exec pairedrl python -c 'import torch, vllm; print(torch.__version__, torch.version.hip, vllm.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'

echo "== pip dry run (torch and vllm must not be replaced)"
DRY="$(docker exec pairedrl pip install --dry-run "${PINS[@]}" 2>&1 | grep -iE '^Would install' || true)"
echo "${DRY:-nothing to install}"
if echo "$DRY" | grep -qE '(^| )(torch|vllm)-'; then
  echo "pip would replace torch or vllm; stopping"; exit 1
fi

echo "== install"
docker exec pairedrl pip install -q "${PINS[@]}"
docker exec pairedrl bash -c 'cd /work/paired-rollouts && pip install -q -e .'
docker exec pairedrl python -c 'import torch, vllm, trl, transformers, peft, fla; print(torch.__version__, vllm.__version__, trl.__version__, transformers.__version__, peft.__version__)'

echo "== model $MODEL into /work/hf"
docker exec -e HF_HOME=/work/hf pairedrl python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"

echo "== ready at $COMMIT"
REMOTE

"${SSH[@]}" "bash /root/work/bootstrap_remote.sh '$IMAGE' '$COMMIT' '$MODEL'"

#!/usr/bin/env bash
# Prepare a RunPod GPU pod (an official Runpod PyTorch template with SSH over a public IP, on a CUDA 13 host) for
# sweep jobs, from the laptop:
#
#   bash scripts/bootstrap_pod.sh <ssh host or alias> [model]
#
# Copies a small remote script to the pod and runs it: clone the repository at this tree's HEAD into
# /workspace/paired-rollouts, create /workspace/venv and install the Modal image's pins into it from PyPI (the
# same resolution as scripts/run_modal.py, so the template's own torch is never used), put the venv first on PATH
# for every SSH command (the pod's shell reads ~/.bashrc for remote commands; the line goes above its interactive
# guard), install the package, download the model into /workspace/hf and print the versions. Everything lives on
# /workspace, the volume disk, so a pod restart keeps it. Idempotent: a second run refreshes the checkout and
# reinstalls. Jobs are then launched with scripts/launch_remote.py --container none.
set -euo pipefail

HOST="${1:?ssh target or ~/.ssh/config alias, for instance runpod-a}"
MODEL="${2:-Qwen/Qwen3.5-2B}"
COMMIT="$(git rev-parse HEAD)"
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$HOST")

"${SSH[@]}" "mkdir -p /workspace && cat > /workspace/bootstrap_remote.sh" <<'REMOTE'
set -euo pipefail
COMMIT="$1"; MODEL="$2"
PINS=(vllm==0.27.1 'trl[vllm,peft]==1.12.0' torch==2.13.0 transformers==5.16.1 peft==0.20.0 accelerate==1.14.0 datasets==5.0.1 flash-linear-attention==0.5.2 'numpy>=1.26' 'pydantic>=2.7' 'pyyaml>=6.0')

echo "== host"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
if [ "${DRIVER%%.*}" -lt 580 ]; then
  echo "driver $DRIVER is older than 580: the torch 2.13.0 CUDA 13.0 wheel cannot initialize on this host; recreate the pod with the CUDA version filter set to 13.0"
  exit 2
fi
python3 --version
df -h /workspace | tail -1

echo "== checkout $COMMIT"
mkdir -p /workspace/runs
cd /workspace
if [ -d paired-rollouts/.git ]; then
  (cd paired-rollouts && git fetch -q origin)
else
  git clone -q https://github.com/TheDeadcoder/paired-rollouts.git
fi
cd paired-rollouts && git checkout -q -f "$COMMIT" && git rev-parse HEAD

echo "== venv (Python 3.12: flashinfer, a vLLM dependency, has array.array[int] type hints that need 3.12; the template ships 3.11)"
if ! command -v python3.12 >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -y -q python3.12 python3.12-venv python3.12-dev
fi
if [ ! -x /workspace/venv/bin/python ] || ! /workspace/venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; then
  rm -rf /workspace/venv
  python3.12 -m venv /workspace/venv
fi
touch /root/.bashrc
grep -q 'workspace/venv/bin' /root/.bashrc || sed -i '1i export PATH=/workspace/venv/bin:$PATH' /root/.bashrc
/workspace/venv/bin/pip install -q --upgrade pip

echo "== install"
/workspace/venv/bin/pip install -q "${PINS[@]}"
/workspace/venv/bin/pip install -q -e /workspace/paired-rollouts
/workspace/venv/bin/python -c 'import torch, vllm, trl, transformers, peft, fla; print(torch.__version__, torch.version.cuda, vllm.__version__, trl.__version__, transformers.__version__, peft.__version__, torch.cuda.get_device_name(0))'

echo "== model $MODEL into /workspace/hf"
HF_HOME=/workspace/hf /workspace/venv/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"

echo "== ready at $COMMIT"
REMOTE

"${SSH[@]}" "bash /workspace/bootstrap_remote.sh '$COMMIT' '$MODEL'"

echo "== python seen by SSH commands"
"${SSH[@]}" 'which python; which pip; python -c "import sys, torch; print(sys.executable, torch.__version__, torch.version.cuda)"'

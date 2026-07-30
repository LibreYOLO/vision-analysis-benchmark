#!/bin/bash
# RF100-VL campaign box bootstrap for vast.ai.
#
# Runs on top of pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime, the tag whose
# torch build (2.11.0+cu128) matches the local validation machine, so sm_120
# (RTX 5090) and sm_89 (RTX 4090) both work without a torch reinstall.
#
# Stdout lands in /var/log/onstart.log (read it with `vast_job.py logs <id>`).
# The harness itself is uploaded separately with scp; this script only prepares
# the environment and writes /root/SETUP_DONE when it is safe to start a job.
#
# Once this bootstrap is stable it becomes the Dockerfile in ../Dockerfile and
# boxes launch ready-to-run instead of installing for two minutes.
set -uo pipefail

LIBREYOLO_REF="${LIBREYOLO_REF:-19b9321ab11450f9d9569ca059e2346267a51aca}"

# The image's interpreter is Ubuntu 24.04's system python, which PEP 668 marks
# externally-managed: pip refuses to install into it and the whole bootstrap
# dies with ModuleNotFoundError several steps later. A rented container IS the
# environment, so opting out is correct here; a venv would only add a PATH
# layer that every later `exec` would have to remember to activate.
export PIP_BREAK_SYSTEM_PACKAGES=1

echo "SETUP_START $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# git for the pip-from-source install, rsync for harness uploads, aria2 for
# fast dataset staging, and the libGL/xcb set that cv2 imports need.
apt-get install -y -qq \
  git rsync aria2 unzip \
  libgl1 libglib2.0-0 libxcb1 libxrender1 libxext6 libsm6 2>&1 | tail -2

# Pin the image's torch so no transitive dependency can swap in a CPU wheel or
# a CUDA build that does not match the host driver.
python - <<'PY' > /root/constraints.txt
import torch
import torchvision
print(f"torch=={torch.__version__.split('+')[0]}")
print(f"torchvision=={torchvision.__version__.split('+')[0]}")
PY
echo "--- torch constraints ---"
cat /root/constraints.txt

pip install -q -c /root/constraints.txt \
  "libreyolo[rfdetr] @ git+https://github.com/LibreYOLO/libreyolo@${LIBREYOLO_REF}" 2>&1 | tail -5
pip install -q -c /root/constraints.txt pycocotools rf100vl huggingface_hub 2>&1 | tail -3

echo "--- runtime check ---"
python - <<'PY'
import torch

import libreyolo
from libreyolo.validation.config import ValidationConfig

print("libreyolo", libreyolo.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
    a = torch.randn(2048, 2048, device="cuda")
    print("matmul ok", float((a @ a).sum()) == float((a @ a).sum()))

# The protocol needs both knobs; a build without them would silently produce
# AP at maxDets 100 instead of 500.
fields = getattr(ValidationConfig, "__dataclass_fields__", {})
assert "eval_max_det" in fields, "libreyolo build predates eval_max_det: wrong ref"
print("eval_max_det: OK")
PY
rc=$?

if [ $rc -ne 0 ]; then
  echo "SETUP_FAILED $(date -u +%FT%TZ)"
  exit 1
fi

mkdir -p /root/rf100-vl /root/rf100vl-weights /root/out
echo "SETUP_DONE $(date -u +%FT%TZ)" | tee /root/SETUP_DONE

#!/bin/bash
# Host acceptance test: run this FIRST on any freshly rented box, before
# installing anything. It takes about a minute and answers the only question
# that matters at that moment: is this host worth the next twenty minutes?
#
#   scp -P <PORT> deploy/vast/accept-box.sh root@<HOST>:/root/
#   ssh -p <PORT> root@<HOST> bash /root/accept-box.sh
#
# Exit code 0 means keep the box. Anything else means destroy it and rent
# another; the marketplace has plenty, and a dud costs about five cents if you
# catch it here instead of after staging 43 GB.
#
# Every check below corresponds to a real host we were billed for:
#   - GPUs present but the container could not start (broken NVIDIA runtime)
#   - torch without kernels for the installed cards
#   - huggingface.co resolving to IPv6 only, with no IPv6 egress
#   - huggingface.co blocked outright while PyPI and GitHub worked fine
#     (snapshot_download just hangs, silently, forever)
set -u

fails=0
pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; fails=$((fails + 1)); }

echo "=== host acceptance test ==="

# 1. GPUs visible to the driver.
if gpus=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) \
   && [ -n "$gpus" ]; then
  pass "gpus: $(echo "$gpus" | wc -l) x $(echo "$gpus" | head -1)"
else
  fail "gpus: nvidia-smi returned nothing (broken container runtime)"
fi

# 2. torch actually has kernels for them. A cu124 image on a Blackwell card
#    imports fine and then cannot launch a single kernel.
torch_report=$(python - <<'PY' 2>&1
try:
    import torch
except Exception as exc:  # noqa: BLE001 - report anything, do not raise
    print(f"FAIL torch import: {exc}")
else:
    archs = set(torch.cuda.get_arch_list())
    if not torch.cuda.is_available():
        print("FAIL torch.cuda.is_available() is False")
    else:
        missing = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            sm = f"sm_{props.major}{props.minor}"
            if sm not in archs:
                missing.append(f"device {index} is {sm}")
        if missing:
            print("FAIL no kernels for " + ", ".join(missing))
        else:
            print(f"OK torch {torch.__version__}, {torch.cuda.device_count()} device(s)")
PY
)
case "$torch_report" in
  OK*) pass "torch: ${torch_report#OK }" ;;
  *)   fail "torch: ${torch_report#FAIL }" ;;
esac

# 3. Real compute, not just enumeration.
matmul=$(python - <<'PY' 2>&1
try:
    import torch
    x = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    print(f"OK matmul {(x @ x).sum().item():.0f}")
except Exception as exc:  # noqa: BLE001
    print(f"FAIL {type(exc).__name__}: {exc}")
PY
)
case "$matmul" in
  OK*) pass "compute: a real matmul ran on the GPU" ;;
  *)   fail "compute: ${matmul#FAIL }" ;;
esac

# 4. The artifact hub, which is where the dataset comes from and where the
#    results go. This is the check that would have saved us the most time.
hub_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 \
  https://huggingface.co/api/datasets/LibreYOLO/rf100-vl 2>/dev/null)
if [ "$hub_code" = "000" ] || [ -z "$hub_code" ]; then
  fail "network: huggingface.co unreachable (staging would hang silently)"
else
  pass "network: huggingface.co reachable (HTTP $hub_code)"
fi

# 5. Package indexes, needed to install anything at all.
pypi_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://pypi.org/simple/ 2>/dev/null)
[ "$pypi_code" = "200" ] && pass "network: pypi reachable" || fail "network: pypi unreachable ($pypi_code)"

# 6. Enough disk for the dataset plus checkpoints.
free_gb=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$free_gb" ] && [ "$free_gb" -ge 120 ]; then
  pass "disk: ${free_gb}GB free"
else
  fail "disk: only ${free_gb:-?}GB free, want >= 120GB"
fi

echo "==="
if [ "$fails" -eq 0 ]; then
  echo "ACCEPT: keep this box"
  exit 0
fi
echo "REJECT: destroy this box and rent another ($fails check(s) failed)"
exit 1

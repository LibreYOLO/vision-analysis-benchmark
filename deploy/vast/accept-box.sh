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

# Pull the verdict line out of a captured report.
#
# The probes below capture stderr as well as stdout, deliberately, so that an
# import traceback is visible rather than swallowed. That means anything else
# python writes to stderr is also captured, and a DeprecationWarning from an
# unrelated dependency lands AHEAD of the verdict. Matching the whole capture
# against `OK*` therefore fails on a completely healthy box: observed on
# 2026-08-01, where a `pynvml` FutureWarning emitted by torch made an 8x5090
# that was mid-campaign report "REJECT: destroy this box".
#
# A false REJECT is the expensive direction to be wrong in, since it throws
# away a good host and sends you back into the rental lottery. So scan for the
# verdict line instead of assuming it comes first.
verdict_of() { printf '%s\n' "$1" | grep -E '^(OK|FAIL)' | tail -1; }

# Find an interpreter before using one. Vast's own images keep their
# environment in a venv that a non-interactive `ssh host command` does not have
# on PATH, so a bare `python` is not found and every torch check below reports
# FAIL on a perfectly healthy box. That is the worst possible direction for an
# acceptance test to be wrong in: it rejects good hosts and sends you back into
# the rental lottery. Prefer the venv interpreter, then the usual names, and
# allow an override for images that put it somewhere else entirely.
PY_BIN=""
for _cand in "${PYTHON:-}" /venv/main/bin/python /opt/conda/bin/python python3 python; do
  [ -n "$_cand" ] || continue
  if command -v "$_cand" >/dev/null 2>&1; then PY_BIN="$_cand"; break; fi
done

echo "=== host acceptance test ==="

if [ -z "$PY_BIN" ]; then
  fail "python: no interpreter found (tried \$PYTHON, /venv/main/bin/python, /opt/conda/bin/python, python3, python)"
else
  pass "python: $PY_BIN"
fi

# 1. GPUs visible to the driver.
if gpus=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) \
   && [ -n "$gpus" ]; then
  pass "gpus: $(echo "$gpus" | wc -l) x $(echo "$gpus" | head -1)"
else
  fail "gpus: nvidia-smi returned nothing (broken container runtime)"
fi

# 2. torch actually has kernels for them. A cu124 image on a Blackwell card
#    imports fine and then cannot launch a single kernel.
torch_report=$("${PY_BIN:-python}" - <<'PY' 2>&1
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
torch_verdict=$(verdict_of "$torch_report")
case "$torch_verdict" in
  OK*)   pass "torch: ${torch_verdict#OK }" ;;
  FAIL*) fail "torch: ${torch_verdict#FAIL }" ;;
  *)     fail "torch: no verdict from the probe: $(printf '%s' "$torch_report" | tr '\n' ' ' | cut -c1-200)" ;;
esac

# 3. Real compute, not just enumeration.
matmul=$("${PY_BIN:-python}" - <<'PY' 2>&1
try:
    import torch
    x = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    print(f"OK matmul {(x @ x).sum().item():.0f}")
except Exception as exc:  # noqa: BLE001
    print(f"FAIL {type(exc).__name__}: {exc}")
PY
)
matmul_verdict=$(verdict_of "$matmul")
case "$matmul_verdict" in
  OK*)   pass "compute: a real matmul ran on the GPU" ;;
  FAIL*) fail "compute: ${matmul_verdict#FAIL }" ;;
  *)     fail "compute: no verdict from the probe: $(printf '%s' "$matmul" | tr '\n' ' ' | cut -c1-200)" ;;
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

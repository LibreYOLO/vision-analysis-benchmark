# RF100-VL campaign image.
#
# Build context is the harness repo root:
#   docker build -f deploy/Dockerfile -t ghcr.io/libreyolo/rf100vl-bench:dev .
#
# Base tag choice: torch 2.11.0+cu128 is the exact build the protocol is
# validated against locally, and cu128 is REQUIRED for the primary GPU pool
# (RTX 5090 is sm_120; a cu124 image cannot launch a kernel on one at all).
# The same image also covers RTX 4090 (sm_89).
#
# Datasets are deliberately NOT baked in. RF100-VL is ~40 GB, which would turn
# a ~30s cold start into a multi-minute image pull on every uncached host, for
# data that stages from a CDN at ~100 MB/s anyway.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    # The base image's interpreter is Ubuntu 24.04's system python, which
    # PEP 668 marks externally-managed, so pip refuses to install into it.
    # An image is a single-purpose environment; a venv would only add a PATH
    # layer every downstream command would have to activate.
    PIP_BREAK_SYSTEM_PACKAGES=1

# git for source installs, rsync for artifact sync, aria2 for fast dataset
# staging, and the libGL/xcb set that cv2 needs but the runtime image omits.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git rsync aria2 unzip ca-certificates \
        libgl1 libglib2.0-0 libxcb1 libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/freeze_torch.py /opt/freeze_torch.py
RUN python /opt/freeze_torch.py /opt/constraints.txt

# Pinned by commit, never by branch: a moving ref would make two boxes in the
# same campaign disagree about what they measured.
ARG LIBREYOLO_REF=19b9321ab11450f9d9569ca059e2346267a51aca
ENV LIBREYOLO_REF=${LIBREYOLO_REF}
RUN pip install -c /opt/constraints.txt \
        "libreyolo[rfdetr] @ git+https://github.com/LibreYOLO/libreyolo@${LIBREYOLO_REF}"
RUN pip install -c /opt/constraints.txt pycocotools rf100vl huggingface_hub

COPY . /opt/vision-analysis-benchmark
RUN pip install -c /opt/constraints.txt -e /opt/vision-analysis-benchmark

# Fail the BUILD on a wrong libreyolo ref rather than the campaign.
RUN python /opt/vision-analysis-benchmark/deploy/verify_image.py

RUN mkdir -p /workspace/rf100-vl /workspace/weights /workspace/results
WORKDIR /workspace

# On a real box, rerun with the GPU assertions active:
#   python /opt/vision-analysis-benchmark/deploy/verify_image.py --require-gpu
CMD ["bash"]

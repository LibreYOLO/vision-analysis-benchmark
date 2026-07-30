"""Emit a pip constraints file pinning the base image's torch stack.

libreyolo's install would otherwise be free to resolve a different torch, and a
CPU-only or wrong-CUDA wheel turns into a silent fallback that only shows up as
a mysteriously slow campaign. Pinning to what the image already ships means pip
treats those two as already satisfied and never touches them.
"""

import sys

import torch
import torchvision

lines = [
    f"torch=={torch.__version__.split('+')[0]}",
    f"torchvision=={torchvision.__version__.split('+')[0]}",
]
with open(sys.argv[1] if len(sys.argv) > 1 else "/opt/constraints.txt", "w") as handle:
    handle.write("\n".join(lines) + "\n")
print("\n".join(lines))

#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_MAX_JOBS="${VLLM_MAX_JOBS:-4}"
VLLM_COMMIT="88d34c6409e9fb3c7b8ca0c04756f061d2099eb1"

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        "The paper runtime used Python 3.10; activate a fresh Python 3.10 "
        "environment before running this installer."
    )
PY

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/lora-kv-cache-vllm.XXXXXX")"
trap 'rm -rf "${VLLM_BUILD_ROOT}"' EXIT

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128
"${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/requirements.txt"

git clone --branch v0.20.0 --depth 1 \
    https://github.com/vllm-project/vllm.git "${VLLM_BUILD_ROOT}/vllm"
if [[ "$(git -C "${VLLM_BUILD_ROOT}/vllm" rev-parse HEAD)" != "${VLLM_COMMIT}" ]]; then
    echo "Unexpected vLLM v0.20.0 commit; refusing to build." >&2
    exit 1
fi

(
    cd "${VLLM_BUILD_ROOT}/vllm"
    "${PYTHON_BIN}" use_existing_torch.py
    "${PYTHON_BIN}" -m pip install -r requirements/build/cuda.txt
    MAX_JOBS="${VLLM_MAX_JOBS}" \
        "${PYTHON_BIN}" -m pip install --no-build-isolation .
)

# KVPress 0.5.3 metadata caps Transformers below 5.3. The paper runtime uses
# Transformers 5.7.0, so keep the verified combination without dependency resolution.
"${PYTHON_BIN}" -m pip install --no-deps kvpress==0.5.3

"${PYTHON_BIN}" - <<'PY'
from importlib.metadata import version

import torch

expected = {
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "torchaudio": "2.11.0+cu128",
    "vllm": "0.20.0+cu128",
    "jsonlines": "4.0.0",
    "transformers": "5.7.0",
    "kvpress": "0.5.3",
}
actual = {name: version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"Runtime version mismatch: expected={expected!r}, actual={actual!r}")
if torch.version.cuda != "12.8":
    raise SystemExit(f"Expected torch CUDA 12.8, found {torch.version.cuda!r}")

from jsonlines import open as jsonlines_open  # noqa: F401,E402
from kvpress import CompactorPress  # noqa: F401,E402
from vllm import SamplingParams  # noqa: F401,E402

print("LoRA-KV-Cache CUDA 12.8 runtime verification: PASS")
PY

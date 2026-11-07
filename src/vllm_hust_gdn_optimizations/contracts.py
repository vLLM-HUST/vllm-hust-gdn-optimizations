"""Pinned current-Host contract for the admitted GDN mechanism."""

from __future__ import annotations

import hashlib
from pathlib import Path

HOST_REPOSITORY = "vLLM-HUST/vllm-ascend-hust"
HOST_REVISION = "7602797593170da4c531f1d8882a7b82dce8a1bc"
FILES = {
    "vllm_ascend/device/device_op.py": "0488cb2578df5075a73245a0d540ee8de93ea71972ae8b636cc7cdef83823b8e",
    "vllm_ascend/ops/gdn.py": "0cd1fc830637a7ef3b392ffbf72221aa60a35546a7ee50611981dca3ddc2fc65",
    "vllm_ascend/ops/triton/fused_gdn_gating.py": "5587bf3440e64eda125899f5de2d1ab609061e2b5336f847af101ada51d81058",
    "vllm_ascend/worker/worker.py": "10ef420b01c7e0355e27c455808e84a2b0c3b2f66e50e495ad7c178cebaabdac",
}


class AdmissionError(RuntimeError):
    """The requested mechanism is not admitted for this process."""


def digest(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_file(path: str | Path, relative: str) -> dict[str, str]:
    actual = digest(path)
    expected = FILES[relative]
    if actual != expected:
        raise AdmissionError(
            f"Host source mismatch for {relative}: expected {expected}, got {actual}"
        )
    return {"path": relative, "sha256": actual}


def verify_host(root: str | Path) -> list[dict[str, str]]:
    root = Path(root)
    return [verify_file(root / relative, relative) for relative in FILES]

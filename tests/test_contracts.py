import os
from pathlib import Path

import pytest

from vllm_hust_gdn_optimizations.contracts import AdmissionError, verify_host


def test_current_host_contract_if_requested():
    root = os.getenv("VLLM_ASCEND_HOST_ROOT")
    if not root:
        pytest.skip("set VLLM_ASCEND_HOST_ROOT to check the current Host")
    assert len(verify_host(root)) == 4


def test_changed_host_fails_closed(tmp_path: Path):
    for relative in (
        "vllm_ascend/device/device_op.py",
        "vllm_ascend/ops/gdn.py",
        "vllm_ascend/ops/triton/fused_gdn_gating.py",
        "vllm_ascend/worker/worker.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("changed\n")
    with pytest.raises(AdmissionError):
        verify_host(tmp_path)

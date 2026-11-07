from types import SimpleNamespace

import pytest

from vllm_hust_gdn_optimizations import static_decay


class _StopAfterOrderCheck(Exception):
    pass


def test_worker_initializes_before_device_modules(monkeypatch):
    original_import = static_decay.importlib.import_module
    worker_imported = False

    def guarded_import(name):
        nonlocal worker_imported
        if name == "vllm_ascend.worker.worker":
            worker_imported = True
            return SimpleNamespace(NPUWorker=object)
        if name == "vllm_ascend.device.device_op":
            assert worker_imported
            raise _StopAfterOrderCheck
        return original_import(name)

    monkeypatch.setattr(static_decay.importlib, "import_module", guarded_import)
    monkeypatch.setattr(static_decay, "_STATE", None)

    with pytest.raises(_StopAfterOrderCheck):
        static_decay.install()

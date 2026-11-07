import importlib

import pytest


def fresh_module():
    import vllm_hust_gdn_optimizations as module

    return importlib.reload(module)


def test_default_load_is_inert(monkeypatch):
    monkeypatch.delenv("VLLM_HUST_GDN_OPTIMIZATIONS", raising=False)
    module = fresh_module()
    module.load()
    assert module.receipt()["installed"] is False


def test_native_delayed_is_explicitly_blocked(monkeypatch):
    monkeypatch.setenv("VLLM_HUST_GDN_OPTIMIZATIONS", "native-delayed")
    module = fresh_module()
    with pytest.raises(module.AdmissionError, match="blocked"):
        module.load()


def test_unknown_selector_fails_closed(monkeypatch):
    monkeypatch.setenv("VLLM_HUST_GDN_OPTIMIZATIONS", "unknown")
    module = fresh_module()
    with pytest.raises(module.AdmissionError, match="unknown"):
        module.load()

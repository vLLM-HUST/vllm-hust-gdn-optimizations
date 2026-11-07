import importlib
import json
import logging
import os

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


def test_static_decay_receipt_is_structured_and_emitted_once(caplog):
    from vllm_hust_gdn_optimizations import static_decay

    state = {
        "pid": os.getpid(),
        "host_revision": "host",
        "cache": None,
        "worker": None,
        "candidate_calls": 0,
    }
    static_decay._EMITTED_EVENTS.clear()
    with caplog.at_level(logging.WARNING):
        static_decay._emit_receipt_once("hooks-installed", state)
        static_decay._emit_receipt_once("hooks-installed", state)

    records = [
        record.message
        for record in caplog.records
        if record.message.startswith("VLLM_HUST_GDN_RECEIPT ")
    ]
    assert len(records) == 1
    payload = json.loads(records[0].split(" ", 1)[1])
    assert payload["event"] == "hooks-installed"
    assert payload["selector"] == "static-decay"
    assert payload["installed"] is True

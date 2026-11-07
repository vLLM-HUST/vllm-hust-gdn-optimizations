"""Current-Host binding for the immutable GDN decay coefficient cache."""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import logging
import os

from .cache import DecayCache
from .contracts import HOST_REVISION, AdmissionError, verify_file

_STATE = None
_EMITTED_EVENTS: set[tuple[int, str]] = set()
_LOGGER = logging.getLogger(__name__)


def _state_receipt(state: dict) -> dict:
    cache = state["cache"]
    return {
        key: value
        for key, value in state.items()
        if key not in {"cache", "worker", "pid"}
    } | {"installed": True, "cache": cache.receipt() if cache else None}


def _emit_receipt_once(event: str, state: dict) -> None:
    """Emit one machine-readable custody record per worker and event."""
    key = (os.getpid(), event)
    if key in _EMITTED_EVENTS:
        return
    _EMITTED_EVENTS.add(key)
    payload = _state_receipt(state)
    payload.update({"event": event, "pid": os.getpid(), "selector": "static-decay"})
    _LOGGER.warning("VLLM_HUST_GDN_RECEIPT %s", json.dumps(payload, sort_keys=True))


def _positive_int(name: str) -> int:
    value = os.getenv(name, "")
    try:
        parsed = int(value)
    except ValueError as error:
        raise AdmissionError(f"{name} must be a positive integer") from error
    if parsed <= 0:
        raise AdmissionError(f"{name} must be a positive integer")
    return parsed


def install() -> dict:
    global _STATE
    if _STATE is not None:
        if _STATE["pid"] != os.getpid():
            raise AdmissionError("fork inherited GDN hooks; use spawned workers")
        return _STATE

    import torch

    # Completing the worker module first follows the Host's initialization
    # order and avoids device_op <-> fused_moe circular imports at CLI startup.
    NPUWorker = importlib.import_module("vllm_ascend.worker.worker").NPUWorker
    device_op = importlib.import_module("vllm_ascend.device.device_op")
    AscendGatedDeltaNetAttention = importlib.import_module(
        "vllm_ascend.ops.gdn"
    ).AscendGatedDeltaNetAttention
    native = importlib.import_module("vllm_ascend.ops.triton.fused_gdn_gating")

    from . import kernels

    bindings = (
        (device_op, "vllm_ascend/device/device_op.py"),
        (AscendGatedDeltaNetAttention, "vllm_ascend/ops/gdn.py"),
        (native, "vllm_ascend/ops/triton/fused_gdn_gating.py"),
        (NPUWorker, "vllm_ascend/worker/worker.py"),
    )
    sources = [
        verify_file(inspect.getsourcefile(value) or "", relative)
        for value, relative in bindings
    ]
    expected_layers = _positive_int("VLLM_HUST_GDN_STATIC_DECAY_EXPECTED_LAYERS")
    heads_per_rank = _positive_int("VLLM_HUST_GDN_STATIC_DECAY_HEADS_PER_RANK")
    try:
        candidate_tokens = tuple(
            int(item)
            for item in os.getenv("VLLM_HUST_GDN_STATIC_DECAY_TOKENS", "1,16").split(
                ","
            )
        )
    except ValueError as error:
        raise AdmissionError("candidate token counts must be integers") from error
    if not candidate_tokens or any(value <= 0 for value in candidate_tokens):
        raise AdmissionError("candidate token counts must be positive")
    if device_op.fused_gdn_gating_patch is not native.fused_gdn_gating_patch:
        raise AdmissionError("GDN gate already has a different owner")

    original_gate = device_op.fused_gdn_gating_patch
    original_load = NPUWorker.load_model
    original_execute = NPUWorker.execute_model
    state = {
        "pid": os.getpid(),
        "host_revision": HOST_REVISION,
        "sources": sources,
        "cache": None,
        "worker": None,
        "candidate_calls": 0,
        "native_calls": 0,
        "unsupported_native_calls": 0,
        "worker_entries": 0,
        "expected_layers": expected_layers,
        "heads_per_rank": heads_per_rank,
        "candidate_tokens": candidate_tokens,
    }

    def gate(a_log, a, b, bias, beta=1.0, threshold=20.0):
        tokens, heads = a.shape
        eligible = bool(
            tokens in candidate_tokens
            and heads == heads_per_rank
            and tuple(b.shape) == tuple(a.shape)
            and a.is_contiguous()
            and b.is_contiguous()
        )
        if eligible:
            cache = state["cache"]
            if cache is None:
                raise AdmissionError("GDN gate ran before coefficient preparation")
            state["candidate_calls"] += 1
            _emit_receipt_once("candidate-exercised", state)
            return kernels.fused_gdn_gating(
                cache.lookup(a_log), a, b, bias, beta, threshold
            )
        state["native_calls"] += 1
        state["unsupported_native_calls"] += 1
        return original_gate(a_log, a, b, bias, beta, threshold)

    @functools.wraps(original_load)
    def load_model(worker, *args, **kwargs):
        if state["worker"] is not None:
            raise AdmissionError("model reload needs a fresh GDN worker")
        config = worker.vllm_config
        if (
            config.model_config.enable_sleep_mode
            or config.weight_transfer_config is not None
            or config.speculative_config is not None
        ):
            raise AdmissionError(
                "sleep, weight transfer, and speculative decoding are unqualified"
            )
        result = original_load(worker, *args, **kwargs)
        cache = DecayCache()

        def prepare(a_log):
            if (
                str(a_log.dtype) != "torch.float32"
                or tuple(a_log.shape) != (heads_per_rank,)
                or not a_log.is_contiguous()
                or a_log.device.type != "npu"
            ):
                raise AdmissionError("unsupported A_log dtype/device/shape/stride")
            coefficient = torch.empty_like(a_log)
            if heads_per_rank % 8:
                raise AdmissionError("heads per rank must be divisible by 8")
            kernels.prepare_decay[(heads_per_rank // 8,)](a_log, coefficient)
            return coefficient

        modules = [
            module
            for module in worker.model_runner.model.modules()
            if getattr(module, "A_log", None) is not None
        ]
        cache.prepare(modules, prepare, expected_layers)
        torch.npu.synchronize()
        state["cache"] = cache
        state["worker"] = worker
        _emit_receipt_once("cache-prepared", state)
        return result

    @functools.wraps(original_execute)
    def execute_model(worker, *args, **kwargs):
        if state["worker"] is not worker:
            raise AdmissionError("unowned GDN worker execution")
        if device_op.fused_gdn_gating_patch is not gate:
            raise AdmissionError("GDN gate ownership changed")
        state["cache"].validate()
        state["worker_entries"] += 1
        return original_execute(worker, *args, **kwargs)

    def immutable(*args, **kwargs):
        raise AdmissionError("GDN graph epoch is immutable; restart the worker")

    device_op.fused_gdn_gating_patch = gate
    NPUWorker.load_model = load_model
    NPUWorker.execute_model = execute_model
    for name in (
        "reload_weights",
        "start_weight_update",
        "update_weights",
        "finish_weight_update",
        "sleep",
        "wake_up",
    ):
        if hasattr(NPUWorker, name):
            setattr(NPUWorker, name, immutable)
    _STATE = state
    _emit_receipt_once("hooks-installed", state)
    return state


def receipt() -> dict:
    if _STATE is None:
        return {"installed": False, "host_revision": HOST_REVISION}
    return _state_receipt(_STATE)

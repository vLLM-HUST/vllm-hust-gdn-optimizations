"""Mountable, default-off GDN optimization family."""

from __future__ import annotations

import os

from .contracts import HOST_REVISION, AdmissionError

_SELECTED = None


def load() -> None:
    global _SELECTED
    selected = os.getenv("VLLM_HUST_GDN_OPTIMIZATIONS", "").strip()
    if _SELECTED is not None:
        if selected != _SELECTED:
            raise AdmissionError("GDN selector changed inside a live process")
        return
    if selected == "":
        _SELECTED = selected
        return
    if selected == "native-delayed":
        raise AdmissionError(
            "native-delayed is blocked until its current-runner lifecycle rewrite is qualified"
        )
    if selected != "static-decay":
        raise AdmissionError(f"unknown GDN optimization: {selected}")
    from .static_decay import install

    install()
    _SELECTED = selected


def receipt() -> dict:
    if _SELECTED != "static-decay":
        return {
            "host_revision": HOST_REVISION,
            "selected": _SELECTED or "",
            "installed": False,
        }
    from .static_decay import receipt as static_receipt

    return static_receipt() | {"selected": _SELECTED}

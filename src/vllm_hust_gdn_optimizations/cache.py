"""Own and validate immutable precomputed GDN decay coefficients."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import AdmissionError


def signature(tensor):
    return (
        id(tensor),
        tensor.data_ptr(),
        tensor._version,
        str(tensor.device),
        str(tensor.dtype),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


@dataclass
class Entry:
    module: object
    weight: object
    coefficient: object
    weight_signature: tuple
    coefficient_signature: tuple


class DecayCache:
    def __init__(self):
        self.entries: list[Entry] = []
        self.by_pointer: dict[int, Entry] = {}
        self.prepared = False

    def prepare(self, modules, make_coefficient, expected_layers: int) -> None:
        if self.prepared or self.entries:
            raise AdmissionError("cache cannot be rebuilt in a live graph epoch")
        entries = []
        for module in modules:
            weight = module.A_log
            coefficient = make_coefficient(weight)
            entries.append(
                Entry(
                    module,
                    weight,
                    coefficient,
                    signature(weight),
                    signature(coefficient),
                )
            )
        if len(entries) != expected_layers:
            raise AdmissionError(
                f"expected {expected_layers} GDN layers, got {len(entries)}"
            )
        pointers = [entry.weight.data_ptr() for entry in entries]
        if len(set(pointers)) != len(pointers):
            raise AdmissionError("aliased layer weights need a separate contract")
        self.entries = entries
        self.by_pointer = {entry.weight.data_ptr(): entry for entry in entries}
        self.prepared = True

    def validate(self) -> None:
        if not self.prepared:
            raise AdmissionError("decay cache is not prepared")
        for entry in self.entries:
            if (
                entry.module.A_log is not entry.weight
                or signature(entry.weight) != entry.weight_signature
            ):
                raise AdmissionError("GDN weight changed; restart and recapture")
            if signature(entry.coefficient) != entry.coefficient_signature:
                raise AdmissionError("decay coefficient changed; restart and recapture")

    def lookup(self, weight):
        entry = self.by_pointer.get(weight.data_ptr())
        if (
            entry is None
            or entry.weight is not weight
            or signature(weight) != entry.weight_signature
        ):
            raise AdmissionError("missing or stale loaded-weight coefficient")
        return entry.coefficient

    def receipt(self) -> dict:
        return {
            "layers": len(self.entries),
            "bytes": sum(
                entry.coefficient.numel() * entry.coefficient.element_size()
                for entry in self.entries
            ),
            "prepared": self.prepared,
        }

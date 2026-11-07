from types import SimpleNamespace

import pytest
import torch

from vllm_hust_gdn_optimizations.cache import DecayCache
from vllm_hust_gdn_optimizations.contracts import AdmissionError


def prepared():
    module = SimpleNamespace(A_log=torch.tensor([0.0, 1.0]))
    cache = DecayCache()
    cache.prepare([module], lambda value: -torch.exp(value), 1)
    return module, cache


def test_cache_keeps_coefficient_storage_and_value():
    module, cache = prepared()
    coefficient = cache.lookup(module.A_log)
    pointer = coefficient.data_ptr()
    cache.validate()
    assert cache.lookup(module.A_log).data_ptr() == pointer
    torch.testing.assert_close(coefficient, -torch.exp(module.A_log))


@pytest.mark.parametrize("mutation", ["inplace", "replacement", "storage"])
def test_cache_rejects_weight_mutation(mutation):
    module, cache = prepared()
    if mutation == "inplace":
        module.A_log.add_(1)
    elif mutation == "replacement":
        module.A_log = module.A_log.clone()
    else:
        module.A_log.data = module.A_log.clone()
    with pytest.raises(AdmissionError):
        cache.validate()


def test_cache_rejects_coefficient_mutation():
    _, cache = prepared()
    cache.entries[0].coefficient.zero_()
    with pytest.raises(AdmissionError):
        cache.validate()


def test_cache_rejects_layer_count_alias_and_rebuild():
    module = SimpleNamespace(A_log=torch.tensor([0.0, 1.0]))
    with pytest.raises(AdmissionError):
        DecayCache().prepare([module], lambda value: value.clone(), 2)
    with pytest.raises(AdmissionError):
        DecayCache().prepare([module, module], lambda value: value.clone(), 2)
    cache = DecayCache()
    cache.prepare([module], lambda value: value.clone(), 1)
    with pytest.raises(AdmissionError):
        cache.prepare([module], lambda value: value.clone(), 1)

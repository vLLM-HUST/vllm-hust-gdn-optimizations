# vLLM-HUST GDN Optimizations

Default-off GDN execution optimization MOD family for vLLM-HUST on Ascend.
Loading the package is inert unless one reviewed mechanism is selected
explicitly.

## Scope

GDN normalization, gating, layout, recurrent-value, QKV and state-execution optimizations.

This repository is the canonical integration and evidence home for the family. Every optimization is opt-in and disabled by default until its own compatibility and performance qualification is complete.

## Current status

| Mechanism | Selector | Status |
| --- | --- | --- |
| Static decay coefficient cache | `static-decay` | Ported to current Host; NPU qualification pending |
| Native delayed state update | `native-delayed` | Blocked from activation; runner/lifecycle rewrite required |
| Legacy Native copy compatibility bridge | — | Retired; current Host/Ascend no longer needs it |

The default plugin path is a no-op. To request the admitted mechanism, provide
both the selector and the model-specific shape contract:

```bash
VLLM_HUST_GDN_OPTIMIZATIONS=static-decay \
VLLM_HUST_GDN_STATIC_DECAY_EXPECTED_LAYERS=30 \
VLLM_HUST_GDN_STATIC_DECAY_HEADS_PER_RANK=16 \
  vllm serve ...
```

The layer/head values above are examples, not a qualification. They must match
the fixed model and tensor-parallel profile recorded by the benchmark run.
Unknown selectors, the blocked Native delayed selector, changed Host sources,
and changed weight/state storage all fail closed.

Selected workers emit `VLLM_HUST_GDN_RECEIPT` JSON records when hooks install,
the immutable coefficient cache is prepared, and the candidate kernel first
executes. Benchmark custody must retain these records separately for every
rank.

## Compatibility check

The current source contract is pinned to `vLLM-HUST/vllm-ascend-hust` commit
`7602797593170da4c531f1d8882a7b82dce8a1bc`:

```bash
python tools/check_host_contract.py /path/to/vllm-ascend-hust
pytest -q
```

This proves source/lifecycle compatibility only. Website performance numbers
remain `retest_pending` until the benchmark protocol is completed on NPU.

## Ownership

- Maintainer: [ShuhaoZhangTony](https://github.com/ShuhaoZhangTony)
- Advisors: none recorded
- Organization: [vLLM-HUST](https://github.com/vLLM-HUST)

## Evidence

Measured single-MOD, composition, and compatibility results are published through [vllm-hust-benchmark](https://github.com/vLLM-HUST/vllm-hust-benchmark) and presented with the corresponding MOD in the [MOD catalog](https://vllm-hust.sage.org.ai/plugins.html).

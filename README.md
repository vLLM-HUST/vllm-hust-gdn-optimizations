# vLLM-HUST GDN Optimizations

Default-off GDN execution optimization MOD family for vLLM-HUST on Ascend.

## Scope

GDN normalization, gating, layout, recurrent-value, QKV and state-execution optimizations.

This repository is the canonical integration and evidence home for the family. Every optimization is opt-in and disabled by default until its own compatibility and performance qualification is complete.

## Ownership

- Maintainer: [ShuhaoZhangTony](https://github.com/ShuhaoZhangTony)
- Advisors: none recorded
- Organization: [vLLM-HUST](https://github.com/vLLM-HUST)

## Evidence

Measured single-MOD, composition, and compatibility results are published through [vllm-hust-benchmark](https://github.com/vLLM-HUST/vllm-hust-benchmark) and presented with the corresponding MOD in the [MOD catalog](https://vllm-hust.sage.org.ai/plugins.html).

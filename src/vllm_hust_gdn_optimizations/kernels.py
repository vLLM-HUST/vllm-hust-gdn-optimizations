# Adapted from vLLM Ascend's fused_gdn_gating implementation.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static-decay variant of the current Ascend GDN gating kernel."""

import torch
from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(
    do_not_specialize=[
        "seq_len",
        "NUM_HEADS",
        "NUM_BATCHES",
        "beta",
        "threshold",
        "ROW_ITER",
    ]
)
def fused_gdn_gating_kernel(
    g,
    beta_output,
    decay,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS,
    NUM_BATCHES,
    beta,
    threshold,
    BLK_HEADS: tl.constexpr,
    BLK_BATCHES: tl.constexpr,
    ROW_ITER,
):
    i_b, i_s = tl.program_id(0), tl.program_id(1)
    col_iter = tl.cdiv(NUM_HEADS, BLK_HEADS)
    for row_idx in range(ROW_ITER):
        batch_off = (
            i_b * ROW_ITER * BLK_BATCHES
            + row_idx * BLK_BATCHES
            + tl.arange(0, BLK_BATCHES)
        )
        for col_idx in range(col_iter):
            head_off = col_idx * BLK_HEADS + tl.arange(0, BLK_HEADS)
            off = (
                batch_off[:, None] * seq_len * NUM_HEADS
                + i_s * NUM_HEADS
                + head_off[None, :]
            )
            head_mask = head_off < NUM_HEADS
            mask = head_mask[None, :] & (batch_off[:, None] < NUM_BATCHES)
            static_decay = tl.load(decay + head_off, mask=head_mask)
            value_a = tl.load(a + off, mask=mask)
            value_b = tl.load(b + off, mask=mask)
            bias = tl.load(dt_bias + head_off, mask=head_mask)
            x = value_a.to(tl.float32) + bias.to(tl.float32)[None, :]
            softplus = tl.where(
                beta * x <= threshold,
                (1 / beta) * tl.log(1 + tl.exp(beta * x)),
                x,
            )
            value_g = static_decay.to(tl.float32) * softplus
            tl.store(g + off, value_g.to(g.dtype.element_ty), mask=mask)
            sigmoid_b = tl.sigmoid(value_b.to(tl.float32))
            tl.store(
                beta_output + off,
                sigmoid_b.to(beta_output.dtype.element_ty),
                mask=mask,
            )


def fused_gdn_gating(
    decay: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, num_heads = a.shape
    num_cores = get_vectorcore_num()
    row_per_core = triton.cdiv(batch, num_cores)
    row_iter = triton.cdiv(row_per_core, 64)
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[(num_cores, 1)](
        g,
        beta_output,
        decay,
        a,
        b,
        dt_bias,
        1,
        num_heads,
        batch,
        beta,
        threshold,
        BLK_HEADS=8,
        BLK_BATCHES=64,
        ROW_ITER=row_iter,
    )
    return g, beta_output


@triton.jit
def prepare_decay(a_log, decay):
    head = tl.program_id(0) * 8 + tl.arange(0, 8)
    value = tl.load(a_log + head)
    tl.store(decay + head, -tl.exp(value.to(tl.float32)))

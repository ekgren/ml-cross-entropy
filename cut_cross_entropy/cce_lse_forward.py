# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Literal, overload

import torch
import triton
import triton.language as tl

from cut_cross_entropy.tl_autotune import cce_forward_autotune
from cut_cross_entropy.tl_utils import b_bin_fn, tl_logaddexp, tl_softcapping


def _cce_lse_forward_kernel(
    E,
    C,
    Bias,
    LSE,
    MaxLogits, # New output tensor for max logits
    LA,
    Locks,
    Valids,
    softcap,
    B,
    V,
    D,
    BMax,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_biasv,
    stride_lse_b, # Assuming MaxLogits will use the same stride logic
    stride_vb,
    num_locks,
    # Meta-parameters
    B_BIN,
    HAS_BIAS: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,  #
    GROUP_B: tl.constexpr,  #
    EVEN_D: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_LA: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_b = tl.cdiv(B, BLOCK_B)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_in_group = GROUP_B * num_pid_v
    group_id = pid // num_pid_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_pid_b - first_pid_b, GROUP_B)
    pid_b = (first_pid_b + ((pid % num_pid_in_group) % group_size_b)).to(tl.int64)
    pid_v = ((pid % num_pid_in_group) // group_size_b).to(tl.int64)

    offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b, mask=offs_b < B, other=BMax).to(tl.int64)

    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        e_mask = offs_b[:, None] < BMax
        if not EVEN_D:
            e_mask = e_mask & (offs_d[None, :] < (D - d * BLOCK_D))

        e = tl.load(e_ptrs, mask=e_mask, other=0.0)

        c_mask = offs_v[None, :] < V
        if not EVEN_D:
            c_mask = c_mask & (offs_d[:, None] < (D - d * BLOCK_D))

        c = tl.load(c_ptrs, mask=c_mask, other=0.0)

        accum = tl.dot(e, c, accum, input_precision=DOT_PRECISION)

        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    tl.debug_barrier()

    if HAS_BIAS:
        bias = tl.load(Bias + offs_v * stride_biasv, mask=offs_v < V, other=0.0)
        bias = bias.to(dtype=accum.dtype)
        accum += bias[None, :]

    logits = tl.where(offs_v[None, :] < V, accum, -float("inf"))
    if HAS_SOFTCAP:
        logits = tl_softcapping(logits, softcap)

    if HAS_LA:
        this_avg_logit = tl.sum(logits, 0) / B # Note: sum over B dim of logits block
        tl.atomic_add(LA + offs_v, this_avg_logit, mask=offs_v < V)

    this_mx = tl.max(logits, axis=1) # Shape: (BLOCK_B,) max logit in this vocab block for each token
    e_exp = tl.exp(logits - this_mx[:, None]) # Renamed 'e' to 'e_exp' to avoid conflict
    this_lse = this_mx + tl.log(tl.sum(e_exp, axis=1)) # Shape: (BLOCK_B,) LSE for this vocab block

    # Use original offs_b for output pointers, not the potentially modified one if HAS_VALIDS
    output_offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
    o_mask = output_offs_b < B

    lse_ptrs = LSE + (stride_lse_b * output_offs_b)
    max_logits_ptrs = MaxLogits + (stride_lse_b * output_offs_b) # Using same stride as LSE

    this_locks = Locks + (pid_b // tl.cdiv(B, BLOCK_B * num_locks))
    while tl.atomic_cas(this_locks, 0, 1) == 1:
        pass

    # Update LSE
    lse_val = tl.load(lse_ptrs, mask=o_mask, other=0.0, eviction_policy="evict_last") # other should be -inf for LSE accumulation start
    lse_val = tl_logaddexp(lse_val, this_lse)
    tl.store(lse_ptrs, lse_val, mask=o_mask, eviction_policy="evict_last")

    # Update MaxLogits using atomic_max
    # this_mx is float32, MaxLogits tensor is float32.
    # atomic_max should be fine.
    tl.atomic_max(max_logits_ptrs, this_mx, mask=o_mask)

    tl.debug_barrier()
    tl.atomic_xchg(this_locks, 0)


_cce_lse_forward_kernel = triton.jit(_cce_lse_forward_kernel)
_cce_lse_forward_kernel = triton.heuristics(  # type: ignore
    {
        "EVEN_D": lambda args: args["D"] % args["BLOCK_D"] == 0,
        "HAS_BIAS": lambda args: args["Bias"] is not None,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
        "HAS_LA": lambda args: args["LA"] is not None,
        "GROUP_B": lambda args: 8,
        "DOT_PRECISION": lambda args: "tf32"
        if torch.get_float32_matmul_precision() == "high"
        else "ieee",
    }
)(_cce_lse_forward_kernel)
_cce_lse_forward_kernel = cce_forward_autotune()(_cce_lse_forward_kernel)  # type: ignore


# Update overloads and function signature
@overload
def cce_lse_forward_kernel(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None = None,
    valids: torch.Tensor | None = None,
    softcap: float | None = None,
    return_logit_avg: Literal[False] = False,
) -> tuple[torch.Tensor, torch.Tensor]: ... # Returns (LSE, MaxLogits)


@overload
def cce_lse_forward_kernel(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None = None,
    valids: torch.Tensor | None = None,
    softcap: float | None = None,
    return_logit_avg: Literal[True] = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ... # Returns (LSE, MaxLogits, LA)


@overload
def cce_lse_forward_kernel(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None = None,
    valids: torch.Tensor | None = None,
    softcap: float | None = None,
    return_logit_avg: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...


def cce_lse_forward_kernel(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None = None,
    valids: torch.Tensor | None = None,
    softcap: float | None = None,
    return_logit_avg: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
    # Check constraints.
    assert e.shape[1] == c.shape[1], "Incompatible dimensions"
    assert e.is_contiguous(), "Matrix A must be contiguous"
    if valids is not None:
        assert valids.ndim == 1
        B = valids.numel()
    else:
        B, _ = e.shape

    if bias is not None:
        assert bias.ndim == 1
        assert c.shape[0] == bias.shape[0]

    V, D = c.shape
    # Allocates output.
    # LSE should be initialized to -inf for logaddexp accumulation
    lse = e.new_full((B,), -float("inf"), dtype=torch.float32)
    # MaxLogits should also be initialized to -inf for max accumulation
    max_logits = e.new_full((B,), -float("inf"), dtype=torch.float32)
    locks = e.new_full(
        (triton.cdiv(B, 128),),
        0,
        dtype=torch.int32, # Triton docs suggest torch.int32 for locks
    )

    if return_logit_avg:
        logit_avg = e.new_full((V,), 0.0, dtype=torch.float32)
    else:
        logit_avg = None

    # 1D launch kernel where each block gets its own program.
    def grid(META) -> tuple[int]:
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(V, META["BLOCK_V"]),)

    _cce_lse_forward_kernel[grid](
        e,
        c,
        bias,
        lse,
        max_logits, # Pass the new tensor
        logit_avg,
        locks,
        valids,
        softcap,
        B,
        V,
        D,  #
        e.size(0),
        e.stride(0),
        e.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        1 if bias is None else bias.stride(0),
        lse.stride(0), # Assuming max_logits uses same stride logic
        1 if valids is None else valids.stride(0),
        num_locks=locks.size(0),
        B_BIN=b_bin_fn(B),
    )

    if return_logit_avg:
        assert logit_avg is not None
        return lse, max_logits, logit_avg
    else:
        return lse, max_logits
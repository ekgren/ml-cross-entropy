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
    MaxLogits,
    ArgMaxLogits, # New output tensor for argmax of logits
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
    stride_lse_b, 
    stride_vb,
    num_locks,
    # Meta-parameters
    B_BIN,
    HAS_BIAS: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
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

    offs_v_block = tl.arange(0, BLOCK_V) # Local offsets within the vocab block
    offs_v_global = (pid_v * BLOCK_V + offs_v_block).to(tl.int64) # Global vocab indices

    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v_global[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d_iter in range(0, tl.cdiv(D, BLOCK_D)): # Renamed d to d_iter
        e_mask = offs_b[:, None] < BMax
        if not EVEN_D:
            e_mask = e_mask & (offs_d[None, :] < (D - d_iter * BLOCK_D))

        e_val = tl.load(e_ptrs, mask=e_mask, other=0.0) # Renamed e to e_val

        c_mask = offs_v_global[None, :] < V
        if not EVEN_D:
            c_mask = c_mask & (offs_d[:, None] < (D - d_iter * BLOCK_D))

        c_val = tl.load(c_ptrs, mask=c_mask, other=0.0) # Renamed c to c_val

        accum = tl.dot(e_val, c_val, accum, input_precision=DOT_PRECISION)

        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    tl.debug_barrier()

    if HAS_BIAS:
        bias_val = tl.load(Bias + offs_v_global * stride_biasv, mask=offs_v_global < V, other=0.0) # Renamed bias
        bias_val = bias_val.to(dtype=accum.dtype)
        accum += bias_val[None, :]

    logits = tl.where(offs_v_global[None, :] < V, accum, -float("inf"))
    if HAS_SOFTCAP:
        logits = tl_softcapping(logits, softcap)

    if HAS_LA:
        this_avg_logit = tl.sum(logits, 0) / B
        tl.atomic_add(LA + offs_v_global, this_avg_logit, mask=offs_v_global < V)

    # Calculate max and argmax for the current block of logits
    # block_max_logits_val shape: (BLOCK_B,)
    # block_argmax_indices_local shape: (BLOCK_B,) - indices are local to the current BLOCK_V
    block_max_logits_val = tl.max(logits, axis=1)
    block_argmax_indices_local = tl.argmax(logits, axis=1)
    # Convert local argmax to global vocabulary index
    block_argmax_indices_global = pid_v * BLOCK_V + block_argmax_indices_local

    # For LSE calculation
    e_exp = tl.exp(logits - block_max_logits_val[:, None])
    this_lse = block_max_logits_val + tl.log(tl.sum(e_exp, axis=1))

    output_offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
    o_mask = output_offs_b < B

    lse_ptrs = LSE + (stride_lse_b * output_offs_b)
    max_logits_ptrs = MaxLogits + (stride_lse_b * output_offs_b)
    arg_max_logits_ptrs = ArgMaxLogits + (stride_lse_b * output_offs_b) # Assuming same stride

    this_locks = Locks + (pid_b // tl.cdiv(B, BLOCK_B * num_locks))
    while tl.atomic_cas(this_locks, 0, 1) == 1:
        pass

    # --- Critical section under lock ---
    # Update LSE
    lse_val = tl.load(lse_ptrs, mask=o_mask, other=-float("inf"), eviction_policy="evict_last") # Start LSE accum from -inf
    lse_val = tl_logaddexp(lse_val, this_lse)
    tl.store(lse_ptrs, lse_val, mask=o_mask, eviction_policy="evict_last")

    # Update MaxLogits and ArgMaxLogits
    # We need to do a read-compare-write for both max_logit and its corresponding argmax
    current_max_val = tl.load(max_logits_ptrs, mask=o_mask, other=-float("inf"), eviction_policy="evict_first")
    
    # Create a mask for tokens where the new block_max_logits_val is greater
    needs_update_mask = o_mask & (block_max_logits_val > current_max_val)

    # Update MaxLogits for those tokens
    tl.store(max_logits_ptrs, block_max_logits_val, mask=needs_update_mask, eviction_policy="evict_last")
    # Update ArgMaxLogits for the same tokens
    tl.store(arg_max_logits_ptrs, block_argmax_indices_global, mask=needs_update_mask, eviction_policy="evict_last")
    # --- End Critical section ---

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


@overload
def cce_lse_forward_kernel(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None = None,
    valids: torch.Tensor | None = None,
    softcap: float | None = None,
    return_logit_avg: Literal[False] = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ... # LSE, MaxLogits, ArgMaxLogits


@overload
def cce_lse_forward_kernel(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None = None,
    valids: torch.Tensor | None = None,
    softcap: float | None = None,
    return_logit_avg: Literal[True] = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ... # LSE, MaxLogits, ArgMaxLogits, LA


@overload
def cce_lse_forward_kernel(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None = None,
    valids: torch.Tensor | None = None,
    softcap: float | None = None,
    return_logit_avg: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


def cce_lse_forward_kernel(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None = None,
    valids: torch.Tensor | None = None,
    softcap: float | None = None,
    return_logit_avg: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    lse = e.new_full((B,), -float("inf"), dtype=torch.float32)
    max_logits = e.new_full((B,), -float("inf"), dtype=torch.float32)
    # Initialize ArgMaxLogits with a placeholder, e.g., -1 or 0. Dtype should be int (e.g., int32 or int64)
    arg_max_logits = e.new_full((B,), -1, dtype=torch.int32) # Using int32 for indices
    locks = e.new_full(
        (triton.cdiv(B, 128),),
        0,
        dtype=torch.int32,
    )

    if return_logit_avg:
        logit_avg = e.new_full((V,), 0.0, dtype=torch.float32)
    else:
        logit_avg = None

    def grid(META) -> tuple[int]:
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(V, META["BLOCK_V"]),)

    _cce_lse_forward_kernel[grid](
        e,
        c,
        bias,
        lse,
        max_logits,
        arg_max_logits, # Pass the new tensor
        logit_avg,
        locks,
        valids,
        softcap,
        B,
        V,
        D,
        e.size(0),
        e.stride(0),
        e.stride(1),
        c.stride(0),
        c.stride(1),
        1 if bias is None else bias.stride(0),
        lse.stride(0), 
        1 if valids is None else valids.stride(0),
        num_locks=locks.size(0),
        B_BIN=b_bin_fn(B),
    )

    if return_logit_avg:
        assert logit_avg is not None
        return lse, max_logits, arg_max_logits, logit_avg
    else:
        return lse, max_logits, arg_max_logits
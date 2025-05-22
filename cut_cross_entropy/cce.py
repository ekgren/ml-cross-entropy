# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from dataclasses import dataclass
from typing import cast

import torch

from cut_cross_entropy.cce_backward import cce_backward_kernel
from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.doc import CCE_OPTS_DOC, LINEAR_CROSS_ENTROPY_DOC, add_doc_start
from cut_cross_entropy.indexed_dot import indexed_neg_dot_forward_kernel
from cut_cross_entropy.utils import (
    _build_flat_valids,
    _handle_eps,
    handle_reduction_none,
)
from cut_cross_entropy.vocab_parallel.utils import (
    VocabParallelOptions,
    vp_reduce_correct_logit,
    vp_reduce_lse,
    # We might need a vp_reduce_max for max_logits if it's not globally maxed by kernel
    # However, the atomic_max in the kernel should handle this across vocab blocks.
    # If vocab_parallel_options means sharding C, then LSE/MaxLogits reduction is needed.
)


@dataclass
class CCEParams:
    targets: torch.Tensor
    valids: torch.Tensor | None
    softcap: float | None
    reduction: str
    filter_eps: float | None
    shift: int
    batch_shape: torch.Size
    accum_e_fp32: bool
    accum_c_fp32: bool
    filter_e_grad: bool
    filter_c_grad: bool
    vocab_parallel_options: VocabParallelOptions | None


@torch.compile(fullgraph=True)
def sort_logit_avg(logit_avg: torch.Tensor) -> torch.Tensor:
    return torch.argsort(logit_avg).to(torch.int32)


class LinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    # Update return type annotation
    def forward(
        ctx,
        e: torch.Tensor,
        c: torch.Tensor,
        bias: torch.Tensor | None,
        params: CCEParams,
    ) -> tuple[torch.Tensor, torch.Tensor]: # Returns (loss, max_softmax_values)
        needs_grad = e.requires_grad or c.requires_grad
        return_logit_avg = needs_grad and params.filter_eps is not None

        # --- Modified unpacking ---
        ret_from_lse_fwd = cce_lse_forward_kernel(
            e=e,
            c=c,
            bias=bias,
            valids=params.valids,
            softcap=params.softcap,
            return_logit_avg=return_logit_avg,
        )
        if return_logit_avg:
            # cce_lse_forward_kernel now returns (lse, max_logits, logit_avg)
            assert isinstance(ret_from_lse_fwd, tuple) and len(ret_from_lse_fwd) == 3
            lse, max_logits, logit_avg = ret_from_lse_fwd
        else:
            # cce_lse_forward_kernel now returns (lse, max_logits)
            assert isinstance(ret_from_lse_fwd, tuple) and len(ret_from_lse_fwd) == 2
            lse, max_logits = ret_from_lse_fwd
            logit_avg = None
        # --- End Modified unpacking ---

        if (vp_opts := params.vocab_parallel_options) is not None:
            lse = vp_reduce_lse(lse, pg=vp_opts.group)
            # We also need to reduce max_logits if it's vocabulary parallel
            # Assuming vp_reduce_lse can be adapted or a new vp_reduce_max is available.
            # For now, let's assume atomic_max in kernel handles the full vocab range correctly
            # IF `C` passed to kernel is the full `C`. If `C` is sharded, then `max_logits`
            # from kernel is local to that shard and needs reduction (e.g., torch.max across group).
            # Given `vp_reduce_lse` exists, likely `max_logits` also needs reduction.
            # Let's create a hypothetical vp_reduce_max_val (similar to LSE reduction)
            # if vp_opts.group.size() > 1:
            #    all_max_logits = [torch.empty_like(max_logits) for _ in range(vp_opts.group.size())]
            #    torch.distributed.all_gather(all_max_logits, max_logits, group=vp_opts.group)
            #    max_logits = torch.stack(all_max_logits, dim=0).max(dim=0)[0]
            # This part is complex with VocabParallel; for simplicity, assuming kernel + atomics
            # provide global max_logits if C is full, or this needs a proper reduction.
            # The CCE paper mentions sharding C along vocab. If so, this reduction is essential.
            # For now, I'll assume `max_logits` from `cce_lse_forward_kernel` is global or
            # `vp_reduce_lse` logic could be adapted. If C is sharded for cce_lse_forward_kernel,
            # `max_logits` must be reduced across ranks. A simple max reduction:
            torch.distributed.all_reduce(max_logits, op=torch.distributed.ReduceOp.MAX, group=vp_opts.group)


            if params.valids is not None:
                targets = params.targets[params.valids + params.shift]
            else:
                targets = params.targets

            vp_valids = (
                ((targets >= vp_opts.start) & (targets < vp_opts.stop)).nonzero().to(torch.int32)
            )
            assert vp_valids.size(1) == 1
            vp_valids = vp_valids.squeeze(-1)

            if params.valids is not None:
                neg_dot_valids = params.valids[vp_valids]
            else:
                neg_dot_valids = vp_valids

            neg_dot_targets = params.targets - vp_opts.start
        else:
            neg_dot_valids = params.valids
            neg_dot_targets = params.targets
            vp_valids = None

        neg_dot = indexed_neg_dot_forward_kernel(
            e=e,
            c=c,
            inds=neg_dot_targets,
            bias=bias,
            shift=params.shift,
            valids=neg_dot_valids,
            softcap=params.softcap,
            out_dtype=lse.dtype,
        )

        if params.vocab_parallel_options is not None:
            global_neg_dot = neg_dot.new_zeros(lse.size())
            assert vp_valids is not None
            global_neg_dot[vp_valids] = neg_dot

            neg_dot = vp_reduce_correct_logit(
                global_neg_dot, pg=params.vocab_parallel_options.group, dtype=lse.dtype
            )

        nll = neg_dot.add_(lse) # nll per token (flat valid tokens)

        # --- New: Calculate max_log_softmax per token ---
        # lse and max_logits should be of the same shape as nll (num_valid_tokens,)
        max_log_softmax_flat = max_logits - lse
        # --- End New ---

        reduction = params.reduction
        if reduction == "mean":
            loss = nll.mean()
            # max_log_softmax_reduced would typically be per-token, not mean reduced.
            # So we keep max_log_softmax_flat for shaping.
            max_log_softmax_reduced = handle_reduction_none(params.batch_shape, params.valids, params.shift, max_log_softmax_flat)

        elif reduction == "sum":
            loss = nll.sum()
            max_log_softmax_reduced = handle_reduction_none(params.batch_shape, params.valids, params.shift, max_log_softmax_flat)
        elif reduction == "none":
            loss = handle_reduction_none(params.batch_shape, params.valids, params.shift, nll)
            max_log_softmax_reduced = handle_reduction_none(params.batch_shape, params.valids, params.shift, max_log_softmax_flat)
        else:
            raise ValueError(f"Unknown reduction {reduction}")

        ctx.save_for_backward(e, c, bias, lse, params.targets, params.valids, logit_avg)
        ctx.params = params

        # Exponentiate to get final max_softmax_values
        max_softmax_values_final = torch.exp(max_log_softmax_reduced)

        return loss, max_softmax_values_final

    @staticmethod
    # grad_out corresponds to loss, second grad corresponds to max_softmax_values (likely None)
    def backward(
        ctx, grad_out_loss: torch.Tensor, grad_out_max_softmax: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        # grad_out_max_softmax is ignored as it's an observational output.
        # If it were part of a differentiable objective, this would change.
        e, c, bias, lse, targets, valids, logit_avg = ctx.saved_tensors

        if logit_avg is not None:
            vocab_ordering = sort_logit_avg(logit_avg)
        else:
            vocab_ordering = None

        params = cast(CCEParams, ctx.params)
        reduction = params.reduction
        if reduction == "mean":
            # If reduction is mean, grad_out_loss is scalar.
            # Need to scale it appropriately for the per-token gradients.
            num_elements = lse.numel() # Number of valid tokens processed
            if valids is not None: # if valids were used, lse.numel() is already correct
                 pass # lse.numel() is already num_valid_tokens
            else: # if no valids, lse.numel() is B (total tokens before shift/ignore)
                 # This needs to be robust to how lse is shaped/sized
                 # Let's use the count of elements contributing to the mean
                 actual_elements_in_mean = nll.numel() # Assuming nll was (num_valid_tokens,) before reduction
                 if actual_elements_in_mean > 0 :
                     grad_scale = 1.0 / actual_elements_in_mean # Use nll's numel before reduction
                 else:
                     grad_scale = 0.0 # Avoid division by zero
            # Correct scaling for mean reduction:
            # The grad_out (scalar) needs to be distributed.
            # The cce_backward_kernel's grad_scale is applied to d_out,
            # where d_out is the per-item gradient.
            # So if grad_out_loss is scalar, d_out should be grad_out_loss / num_items
            # And cce_backward_kernel's grad_scale should be 1.0
            # OR d_out is grad_out_loss (scalar) and cce_backward_kernel's grad_scale is 1/num_items
            # The current code uses grad_scale = 1 / lse.numel(). This seems fine if lse.numel() is correct.
            # Let's use grad_out_loss.numel() to determine if it's scalar (reduction=mean/sum) or tensor (reduction=none)
            if grad_out_loss.numel() == 1: # mean or sum
                grad_d_out = grad_out_loss # Pass scalar
                grad_scale_for_kernel = (1.0 / lse.numel()) if lse.numel() > 0 else 0.0
                if reduction == "sum":
                    grad_scale_for_kernel = 1.0
            else: # none
                grad_d_out = grad_out_loss.view(-1) # Flatten if reduction was none
                grad_scale_for_kernel = 1.0

        elif reduction == "sum":
            grad_d_out = grad_out_loss # Pass scalar
            grad_scale_for_kernel = 1.0
        elif reduction == "none":
            grad_d_out = grad_out_loss.view(-1)
            grad_scale_for_kernel = 1.0
        else:
            raise ValueError(f"Unknown reduction {reduction}")


        if (vp_opts := params.vocab_parallel_options) is not None:
            is_my_target = (targets >= vp_opts.start) & (targets < vp_opts.stop)
            targets_for_kernel = torch.where( # Renamed to avoid conflict with saved targets
                is_my_target,
                targets - vp_opts.start,
                targets.new_full((), c.size(0) + 1),
            )
        else:
            targets_for_kernel = targets


        de, dc, dbias = cce_backward_kernel(
            do=grad_d_out, # Use the appropriately shaped/scaled grad_out
            e=e,
            c=c,
            bias=bias,
            lse=lse,
            valids=valids,
            softcap=params.softcap,
            filter_eps=params.filter_eps,
            targets=targets_for_kernel, # Use the modified targets
            shift=params.shift,
            vocab_ordering=vocab_ordering,
            grad_scale=grad_scale_for_kernel, # Use the calculated scale
            accum_e_fp32=params.accum_e_fp32,
            accum_c_fp32=params.accum_c_fp32,
            filter_e_grad=params.filter_e_grad,
            filter_c_grad=params.filter_c_grad,
        )

        return de, dc, dbias, None


def linear_cross_entropy_apply(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None,
    params: CCEParams,
) -> tuple[torch.Tensor, torch.Tensor]: # Update return type
    # Loss and max_softmax_values are returned by apply now
    loss, max_softmax_values = LinearCrossEntropyFunction.apply(e, c, bias, params)
    # Ensure they are Tensors
    assert isinstance(loss, torch.Tensor)
    assert isinstance(max_softmax_values, torch.Tensor)


    if params.shift != 0 and params.reduction == "none":
        # This slicing assumes loss and max_softmax_values are (B, S_full_from_batch_shape)
        # and we want (B, S_effective_after_shift)
        loss_final = loss[..., params.shift :]
        max_softmax_values_final = max_softmax_values[..., params.shift :]
    else:
        loss_final = loss
        max_softmax_values_final = max_softmax_values

    return loss_final, max_softmax_values_final


@add_doc_start(LINEAR_CROSS_ENTROPY_DOC)
@add_doc_start(*(doc_str + "\n" for doc_str in CCE_OPTS_DOC))
def cce_linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool | int = 0,
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    vocab_parallel_options: VocabParallelOptions | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: # Update return type
    assert e.size()[0:-1] == targets.size()
    assert e.size(-1) == c.size(1)
    if not torch.cuda.is_bf16_supported(): # Check Triton support as well if possible
        raise RuntimeError(
            "Cut Cross Entropy requires an ampere GPU or newer. "
            "Consider using torch_compile_linear_cross_entropy for scenarios where one is not available."
        )

    batch_shape = targets.size()

    e_orig_ndim = e.ndim # Store original ndim for potential reshaping of outputs
    e = e.contiguous()
    targets = targets.contiguous()

    shift = int(shift)
    valids = _build_flat_valids(targets, ignore_index, shift)

    # Flatten e and targets for kernel processing
    # Note: linear_cross_entropy_apply handles the final shift slicing if reduction is 'none'
    # The 'e' passed to kernels is already shifted effectively by 'valids' and how 'neg_dot' uses 'shift'
    if e_orig_ndim > 2 : # If e was (B, S, D)
        # Pass unflattened e to linear_cross_entropy_apply, which will flatten it
        # Or, flatten here and ensure unflattening logic is robust for both loss and max_softmax
        pass # CCEParams will get the original batch_shape


    flat_e = e.flatten(0, -2)
    flat_targets = targets.flatten()


    if (flat_targets.data_ptr() % 16) != 0:
        # This padding ensures aligned memory access for targets in kernels
        # It's a bit unusual to modify targets like this in place or by reassigning.
        # Original code uses: targets = torch.nn.functional.pad(targets, (0, 1))[:-1]
        # This ensures data_ptr alignment without changing content for aligned inputs.
        # For flat_targets:
        padded_flat_targets = torch.nn.functional.pad(flat_targets, (0, 1))
        flat_targets_for_cce = padded_flat_targets[:-1]
        if (flat_targets_for_cce.data_ptr() % 16) != 0 :
             # If still not aligned (e.g. original was odd length and padding didn't fix due to initial offset)
             # this indicates a deeper issue or a more complex alignment strategy is needed.
             # For now, assume this strategy works as in original.
             pass

    else:
        flat_targets_for_cce = flat_targets


    assert (flat_targets_for_cce.data_ptr() % 16) == 0, \
        f"flat_targets_for_cce not 16-byte aligned: {flat_targets_for_cce.data_ptr() % 16}"

    cce_params = CCEParams(
        targets=flat_targets_for_cce, # Pass aligned flat targets
        valids=valids,
        softcap=softcap,
        reduction=reduction,
        filter_eps=_handle_eps(filter_eps, flat_e.dtype),
        shift=shift, # Pass the original shift value
        batch_shape=batch_shape, # Original batch_shape for unflattening
        accum_e_fp32=accum_e_fp32,
        accum_c_fp32=accum_c_fp32,
        filter_e_grad=filter_e_grad and filter_eps is not None,
        filter_c_grad=filter_c_grad and filter_eps is not None,
        vocab_parallel_options=vocab_parallel_options,
    )
    # Pass flat_e to apply function
    return linear_cross_entropy_apply(flat_e, c, bias, cce_params)
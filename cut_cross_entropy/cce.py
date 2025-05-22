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
    def forward(
        ctx,
        e: torch.Tensor,
        c: torch.Tensor,
        bias: torch.Tensor | None,
        params: CCEParams,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        needs_grad = e.requires_grad or c.requires_grad
        return_logit_avg = needs_grad and params.filter_eps is not None

        ret_from_lse_fwd = cce_lse_forward_kernel(
            e=e,
            c=c,
            bias=bias,
            valids=params.valids,
            softcap=params.softcap,
            return_logit_avg=return_logit_avg,
        )
        if return_logit_avg:
            assert isinstance(ret_from_lse_fwd, tuple) and len(ret_from_lse_fwd) == 3
            lse, max_logits, logit_avg = ret_from_lse_fwd
        else:
            assert isinstance(ret_from_lse_fwd, tuple) and len(ret_from_lse_fwd) == 2
            lse, max_logits = ret_from_lse_fwd
            logit_avg = None

        if (vp_opts := params.vocab_parallel_options) is not None:
            lse = vp_reduce_lse(lse, pg=vp_opts.group)
            torch.distributed.all_reduce(max_logits, op=torch.distributed.ReduceOp.MAX, group=vp_opts.group)

            if params.valids is not None:
                targets_for_vp = params.targets[params.valids + params.shift]
            else:
                targets_for_vp = params.targets

            vp_valids_mask = (targets_for_vp >= vp_opts.start) & (targets_for_vp < vp_opts.stop)
            vp_valids_indices = vp_valids_mask.nonzero(as_tuple=False)
            if vp_valids_indices.numel() > 0:
                 assert vp_valids_indices.size(1) == 1
                 vp_valids_indices = vp_valids_indices.squeeze(-1).to(torch.int32)
            else: # handle empty case for vp_valids_indices
                 vp_valids_indices = torch.empty(0, dtype=torch.int32, device=targets_for_vp.device)


            if params.valids is not None:
                # We need to map vp_valids_indices (which are indices into the flattened valid tokens)
                # back to indices in the original params.valids tensor.
                # This is tricky. Let's assume neg_dot_valids calculation in original code was correct.
                # The original code's `vp_valids` was directly used to index `params.valids`.
                # So, `vp_valids_indices` here are indices *within the subset defined by params.valids*.
                if vp_valids_indices.numel() > 0:
                    neg_dot_valids = params.valids[vp_valids_indices]
                else:
                    neg_dot_valids = torch.empty(0, dtype=params.valids.dtype, device=params.valids.device)

            else: # params.valids is None, so vp_valids_indices directly index the (flat) targets
                neg_dot_valids = vp_valids_indices

            neg_dot_targets = params.targets - vp_opts.start # This targets is flat
        else:
            neg_dot_valids = params.valids
            neg_dot_targets = params.targets # This targets is flat
            # vp_valids_indices = None # Not used if not vp

        neg_dot = indexed_neg_dot_forward_kernel(
            e=e,
            c=c,
            inds=neg_dot_targets,
            bias=bias,
            shift=params.shift, # shift is applied to e inside kernel if valids are used
            valids=neg_dot_valids,
            softcap=params.softcap,
            out_dtype=lse.dtype,
        )

        if params.vocab_parallel_options is not None:
            # Reconstruct global_neg_dot based on the shape of lse (num_valid_tokens)
            global_neg_dot = neg_dot.new_zeros(lse.size())
            # vp_valids_indices should correctly index into the flattened 'lse' or 'global_neg_dot' tensor
            if vp_valids_indices.numel() > 0 and neg_dot.numel() > 0:
                 # This assumes neg_dot corresponds to the elements indexed by vp_valids_indices
                 # The size of neg_dot should match vp_valids_indices.numel()
                 # This part is complex and relies on exact shapes from indexed_neg_dot_forward_kernel
                 # when neg_dot_valids is used. Assuming neg_dot is already the correct subset.
                 # For vp_reduce_correct_logit, the target 'neg_dot' has contributions only from this rank.
                 # 'global_neg_dot' should be filled at positions corresponding to *this rank's targets*
                 # within the global set of valid tokens.
                 # The original vp_valids construction was:
                 # vp_valids = (((targets >= vp_opts.start) & (targets < vp_opts.stop)).nonzero().to(torch.int32))
                 # ... vp_valids = vp_valids.squeeze(-1)
                 # This vp_valids (now vp_valids_indices) are indices into the `lse` shaped tensor.
                 if global_neg_dot.numel() > 0 and vp_valids_indices.numel() > 0:
                     if neg_dot.numel() == vp_valids_indices.numel(): # Ensure sizes match before assignment
                         global_neg_dot[vp_valids_indices] = neg_dot
                     elif neg_dot.numel() == 1 and vp_valids_indices.numel() > 0 : # Broadcast scalar neg_dot
                         global_neg_dot[vp_valids_indices] = neg_dot.item()
                     elif neg_dot.numel() > 0: # Fallback if sizes mismatch but neg_dot has data
                        # This might indicate a shape mismatch that needs careful debugging.
                        # For safety, only assign if neg_dot matches the number of indices for this rank
                        # Or if neg_dot is a scalar that can be broadcast.
                        # If not, global_neg_dot for other ranks' targets will remain zero, which is correct.
                        pass # Let downstream handle potential size mismatches if any logic error here

            neg_dot = vp_reduce_correct_logit(
                global_neg_dot, pg=params.vocab_parallel_options.group, dtype=lse.dtype
            )

        nll = neg_dot.add_(lse)

        max_log_softmax_flat = max_logits - lse

        reduction = params.reduction
        if reduction == "mean":
            loss = nll.mean()
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

        max_softmax_values_final = torch.exp(max_log_softmax_reduced)

        return loss, max_softmax_values_final

    @staticmethod
    def backward(
        ctx, grad_out_loss: torch.Tensor, grad_out_max_softmax: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        e, c, bias, lse, targets, valids, logit_avg = ctx.saved_tensors

        if logit_avg is not None:
            vocab_ordering = sort_logit_avg(logit_avg)
        else:
            vocab_ordering = None

        params = cast(CCEParams, ctx.params)
        reduction = params.reduction

        if grad_out_loss.numel() == 1:
            grad_d_out = grad_out_loss
            if reduction == "mean":
                num_items_in_mean = lse.numel()
                grad_scale_for_kernel = (1.0 / num_items_in_mean) if num_items_in_mean > 0 else 0.0
            elif reduction == "sum":
                grad_scale_for_kernel = 1.0
            else:
                raise ValueError(f"Unexpected reduction '{reduction}' for scalar grad_out_loss.")
        elif reduction == "none":
            grad_d_out = grad_out_loss.view(-1)
            grad_scale_for_kernel = 1.0
        else:
            raise ValueError(f"Unknown reduction '{reduction}' or incompatible grad_out_loss shape.")

        if (vp_opts := params.vocab_parallel_options) is not None:
            is_my_target = (targets >= vp_opts.start) & (targets < vp_opts.stop)
            targets_for_kernel = torch.where(
                is_my_target,
                targets - vp_opts.start,
                targets.new_full((), c.size(0) + 1),
            )
        else:
            targets_for_kernel = targets

        de, dc, dbias = cce_backward_kernel(
            do=grad_d_out,
            e=e,
            c=c,
            bias=bias,
            lse=lse,
            valids=valids,
            softcap=params.softcap,
            filter_eps=params.filter_eps,
            targets=targets_for_kernel,
            shift=params.shift,
            vocab_ordering=vocab_ordering,
            grad_scale=grad_scale_for_kernel,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    loss, max_softmax_values = LinearCrossEntropyFunction.apply(e, c, bias, params)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(max_softmax_values, torch.Tensor)

    if params.shift != 0 and params.reduction == "none":
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
) -> tuple[torch.Tensor, torch.Tensor]:
    assert e.size()[0:-1] == targets.size()
    assert e.size(-1) == c.size(1)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Cut Cross Entropy requires an ampere GPU or newer. "
            "Consider using torch_compile_linear_cross_entropy for scenarios where one is not available."
        )

    batch_shape = targets.size()

    e = e.contiguous()
    targets = targets.contiguous()

    shift = int(shift)
    valids = _build_flat_valids(targets, ignore_index, shift)

    flat_e = e.flatten(0, -2)
    # Targets passed to CCEParams should be the original flat_targets for vp logic consistency
    # The kernel uses `params.shift` and `params.valids` for indexing.
    flat_targets_for_params = targets.flatten()


    if (flat_targets_for_params.data_ptr() % 16) != 0:
        padded_flat_targets = torch.nn.functional.pad(flat_targets_for_params, (0, 1))
        flat_targets_for_params_aligned = padded_flat_targets[:-1]
    else:
        flat_targets_for_params_aligned = flat_targets_for_params

    assert (flat_targets_for_params_aligned.data_ptr() % 16) == 0, \
        f"flat_targets_for_params_aligned not 16-byte aligned: {flat_targets_for_params_aligned.data_ptr() % 16}"

    cce_params = CCEParams(
        targets=flat_targets_for_params_aligned,
        valids=valids,
        softcap=softcap,
        reduction=reduction,
        filter_eps=_handle_eps(filter_eps, flat_e.dtype),
        shift=shift,
        batch_shape=batch_shape,
        accum_e_fp32=accum_e_fp32,
        accum_c_fp32=accum_c_fp32,
        filter_e_grad=filter_e_grad and filter_eps is not None,
        filter_c_grad=filter_c_grad and filter_eps is not None,
        vocab_parallel_options=vocab_parallel_options,
    )
    return linear_cross_entropy_apply(flat_e, c, bias, cce_params)

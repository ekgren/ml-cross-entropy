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
    # Update return type annotation for three outputs
    def forward(
        ctx,
        e: torch.Tensor,
        c: torch.Tensor,
        bias: torch.Tensor | None,
        params: CCEParams,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # Returns (loss, max_softmax_values, argmax_indices)
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
            assert isinstance(ret_from_lse_fwd, tuple) and len(ret_from_lse_fwd) == 4
            lse, max_logits, arg_max_logits_flat, logit_avg = ret_from_lse_fwd
        else:
            assert isinstance(ret_from_lse_fwd, tuple) and len(ret_from_lse_fwd) == 3
            lse, max_logits, arg_max_logits_flat = ret_from_lse_fwd
            logit_avg = None

        if (vp_opts := params.vocab_parallel_options) is not None:
            lse = vp_reduce_lse(lse, pg=vp_opts.group)
            
            # For max_logits and arg_max_logits with vocab parallelism:
            # We need to gather all max_logits and their corresponding arg_max_logits from each rank,
            # then find the true global max and its argmax.
            if vp_opts.group.size() > 1:
                # Gather max_logits from all ranks
                all_max_logits_list = [torch.empty_like(max_logits) for _ in range(vp_opts.group.size())]
                torch.distributed.all_gather(all_max_logits_list, max_logits, group=vp_opts.group)
                all_max_logits_tensor = torch.stack(all_max_logits_list, dim=0) # Shape: (world_size, num_tokens)

                # Gather arg_max_logits_flat from all ranks
                # These argmax are local to each shard's vocab range.
                all_arg_max_logits_flat_list = [torch.empty_like(arg_max_logits_flat) for _ in range(vp_opts.group.size())]
                torch.distributed.all_gather(all_arg_max_logits_flat_list, arg_max_logits_flat, group=vp_opts.group)
                all_arg_max_logits_flat_tensor = torch.stack(all_arg_max_logits_flat_list, dim=0) # Shape: (world_size, num_tokens)

                # Find the rank that has the true maximum logit for each token
                global_max_logits, best_rank_indices = torch.max(all_max_logits_tensor, dim=0) # global_max_logits: (num_tokens,), best_rank_indices: (num_tokens,)
                
                # Use best_rank_indices to pick the argmax from the correct rank
                # The arg_max_logits from all_arg_max_logits_flat_tensor are global vocab indices already
                # because the Triton kernel calculates `block_argmax_indices_global`.
                # So, we just need to select from the correct rank.
                # Create an expanded best_rank_indices for gather
                # best_rank_indices_expanded = best_rank_indices.unsqueeze(0) # Shape: (1, num_tokens)
                # global_arg_max_logits_flat = torch.gather(all_arg_max_logits_flat_tensor, 0, best_rank_indices_expanded).squeeze(0)
                
                # Simpler way to gather:
                # For each token, we have its best_rank_index.
                # We need to select the argmax from that rank's all_arg_max_logits_flat_tensor row.
                # This is effectively: global_arg_max_logits_flat[i] = all_arg_max_logits_flat_tensor[best_rank_indices[i], i]
                # A more direct way to implement this gather:
                num_tokens = best_rank_indices.size(0)
                global_arg_max_logits_flat = all_arg_max_logits_flat_tensor[best_rank_indices, torch.arange(num_tokens, device=best_rank_indices.device)]


                max_logits = global_max_logits
                arg_max_logits_flat = global_arg_max_logits_flat
            # If world_size is 1, no reduction needed, max_logits and arg_max_logits_flat are already global.


            if params.valids is not None:
                targets_for_vp = params.targets[params.valids + params.shift]
                targets_for_vp = params.targets[params.valids + params.shift]
            else:
                targets_for_vp = params.targets
                targets_for_vp = params.targets

            vp_valids_mask = (targets_for_vp >= vp_opts.start) & (targets_for_vp < vp_opts.stop)
            vp_valids_indices = vp_valids_mask.nonzero(as_tuple=False)
            if vp_valids_indices.numel() > 0:
                 assert vp_valids_indices.size(1) == 1
                 vp_valids_indices = vp_valids_indices.squeeze(-1).to(torch.int32)
            else: 
                 vp_valids_indices = torch.empty(0, dtype=torch.int32, device=targets_for_vp.device)

            if params.valids is not None:
                if vp_valids_indices.numel() > 0:
                    neg_dot_valids = params.valids[vp_valids_indices]
                else:
                    neg_dot_valids = torch.empty(0, dtype=params.valids.dtype, device=params.valids.device)
            else: 
                neg_dot_valids = vp_valids_indices
            neg_dot_targets = params.targets - vp_opts.start 
        else:
            neg_dot_valids = params.valids
            neg_dot_targets = params.targets
        
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
            # Reconstruct global_neg_dot based on the shape of lse (num_valid_tokens)
            global_neg_dot = neg_dot.new_zeros(lse.size())
            if vp_valids_indices.numel() > 0 and neg_dot.numel() > 0:
                 if global_neg_dot.numel() > 0 and vp_valids_indices.numel() > 0:
                     if neg_dot.numel() == vp_valids_indices.numel(): 
                         global_neg_dot[vp_valids_indices] = neg_dot
                     elif neg_dot.numel() == 1 and vp_valids_indices.numel() > 0 : 
                         global_neg_dot[vp_valids_indices] = neg_dot.item()
            neg_dot = vp_reduce_correct_logit(
                global_neg_dot, pg=params.vocab_parallel_options.group, dtype=lse.dtype
            )

        nll = neg_dot.add_(lse)
        max_log_softmax_flat = max_logits - lse

        reduction = params.reduction
        if reduction == "mean":
            loss = nll.mean()
            max_log_softmax_reduced = handle_reduction_none(params.batch_shape, params.valids, params.shift, max_log_softmax_flat)
            arg_max_logits_reduced = handle_reduction_none(params.batch_shape, params.valids, params.shift, arg_max_logits_flat.to(max_log_softmax_reduced.dtype)) # Ensure dtype consistency if needed
        elif reduction == "sum":
            loss = nll.sum()
            max_log_softmax_reduced = handle_reduction_none(params.batch_shape, params.valids, params.shift, max_log_softmax_flat)
            arg_max_logits_reduced = handle_reduction_none(params.batch_shape, params.valids, params.shift, arg_max_logits_flat.to(max_log_softmax_reduced.dtype))
        elif reduction == "none":
            loss = handle_reduction_none(params.batch_shape, params.valids, params.shift, nll)
            max_log_softmax_reduced = handle_reduction_none(params.batch_shape, params.valids, params.shift, max_log_softmax_flat)
            arg_max_logits_reduced = handle_reduction_none(params.batch_shape, params.valids, params.shift, arg_max_logits_flat) # Dtype should be int
        else:
            raise ValueError(f"Unknown reduction {reduction}")

        ctx.save_for_backward(e, c, bias, lse, params.targets, params.valids, logit_avg)
        ctx.params = params

        max_softmax_values_final = torch.exp(max_log_softmax_reduced)
        # Ensure arg_max_logits_reduced is integer type
        arg_max_indices_final = arg_max_logits_reduced.to(torch.int32)


        return loss, max_softmax_values_final, arg_max_indices_final

    @staticmethod
    # Update for three outputs from forward, backward receives grads for first two (loss, max_softmax)
    def backward(
        ctx, grad_out_loss: torch.Tensor, grad_out_max_softmax: torch.Tensor | None, grad_out_argmax: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        # grad_out_max_softmax and grad_out_argmax are ignored
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

        if grad_out_loss.numel() == 1:
            grad_d_out = grad_out_loss
            if reduction == "mean":
                num_items_in_mean = lse.numel()
                grad_scale_for_kernel = (1.0 / num_items_in_mean) if num_items_in_mean > 0 else 0.0
            elif reduction == "sum":
                grad_scale_for_kernel = 1.0
            else:
                raise ValueError(f"Unexpected reduction '{reduction}' for scalar grad_out_loss.")
            else:
                raise ValueError(f"Unexpected reduction '{reduction}' for scalar grad_out_loss.")
        elif reduction == "none":
            grad_d_out = grad_out_loss.view(-1)
            grad_scale_for_kernel = 1.0
        else:
            raise ValueError(f"Unknown reduction '{reduction}' or incompatible grad_out_loss shape.")
            raise ValueError(f"Unknown reduction '{reduction}' or incompatible grad_out_loss shape.")

        if (vp_opts := params.vocab_parallel_options) is not None:
            is_my_target = (targets >= vp_opts.start) & (targets < vp_opts.stop)
            targets_for_kernel = torch.where(
            targets_for_kernel = torch.where(
                is_my_target,
                targets - vp_opts.start,
                targets.new_full((), c.size(0) + 1),
            )
        else:
            targets_for_kernel = targets

        de, dc, dbias = cce_backward_kernel(
            do=grad_d_out,
            do=grad_d_out,
            e=e,
            c=c,
            bias=bias,
            lse=lse,
            valids=valids,
            softcap=params.softcap,
            filter_eps=params.filter_eps,
            targets=targets_for_kernel,
            targets=targets_for_kernel,
            shift=params.shift,
            vocab_ordering=vocab_ordering,
            grad_scale=grad_scale_for_kernel,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # Update return type
    loss, max_softmax_values, arg_max_indices = LinearCrossEntropyFunction.apply(e, c, bias, params)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(max_softmax_values, torch.Tensor)
    assert isinstance(arg_max_indices, torch.Tensor)


    if params.shift != 0 and params.reduction == "none":
        loss_final = loss[..., params.shift :]
        max_softmax_values_final = max_softmax_values[..., params.shift :]
        arg_max_indices_final = arg_max_indices[..., params.shift :]

    else:
        loss_final = loss
        max_softmax_values_final = max_softmax_values
        arg_max_indices_final = arg_max_indices


    return loss_final, max_softmax_values_final, arg_max_indices_final


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # Update return type
    assert e.size()[0:-1] == targets.size()
    assert e.size(-1) == c.size(1)
    if not torch.cuda.is_bf16_supported():
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
    flat_targets_for_params = targets.flatten()

    if (flat_targets_for_params.data_ptr() % 16) != 0:
        padded_flat_targets = torch.nn.functional.pad(flat_targets_for_params, (0, 1))
        flat_targets_for_params_aligned = padded_flat_targets[:-1]
    else:
        flat_targets_for_params_aligned = flat_targets_for_params
        flat_targets_for_params_aligned = flat_targets_for_params

    assert (flat_targets_for_params_aligned.data_ptr() % 16) == 0, \
        f"flat_targets_for_params_aligned not 16-byte aligned: {flat_targets_for_params_aligned.data_ptr() % 16}"
    assert (flat_targets_for_params_aligned.data_ptr() % 16) == 0, \
        f"flat_targets_for_params_aligned not 16-byte aligned: {flat_targets_for_params_aligned.data_ptr() % 16}"

    cce_params = CCEParams(
        targets=flat_targets_for_params_aligned,
        targets=flat_targets_for_params_aligned,
        valids=valids,
        softcap=softcap,
        reduction=reduction,
        filter_eps=_handle_eps(filter_eps, flat_e.dtype),
        shift=shift,
        batch_shape=batch_shape,
        shift=shift,
        batch_shape=batch_shape,
        accum_e_fp32=accum_e_fp32,
        accum_c_fp32=accum_c_fp32,
        filter_e_grad=filter_e_grad and filter_eps is not None,
        filter_c_grad=filter_c_grad and filter_eps is not None,
        vocab_parallel_options=vocab_parallel_options,
    )
    return linear_cross_entropy_apply(flat_e, c, bias, cce_params)

"""Pointing a live model's parameters at arena bytes — the half of the load with no
sglang in it.

This was private machinery inside :mod:`flashboot.loaders.arena_loader_base` until a
SECOND front end appeared. ModelExpress drives exactly the same sequence from its own
load strategy — drop the device-init storage, allocate one arena, rebind every saved
tensor onto its slice, mark the state the arena already holds as final — and it must not
have to come in through sglang's ``BaseModelLoader`` to do it. Both front ends now call
these four functions, so the two paths cannot drift into binding the model differently.

Nothing here reads a checkpoint or moves a byte: it takes placements (from
:mod:`flashboot.utils.checkpoint_layout`) and an already-allocated arena, and only ever changes
where tensors POINT. Filling the arena is somebody else's job — disk on the seed
(:mod:`flashboot.arena_fill`), a peer's arena on a clone.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn

from flashboot.arena import ArenaHandle, ArenaRebinder
from flashboot.utils.checkpoint_layout import row_major_stride


def rebinder_for(model: nn.Module, tensor_placements: Sequence) -> ArenaRebinder:
    """Build the rebinder and check UP FRONT that every saved tensor has a slot.

    The check runs before any storage is dropped: a checkpoint that names a tensor this
    build does not have is a mismatched export, and finding that out after the model's
    parameters have been emptied would leave nothing to fall back with.
    """
    rebinder = ArenaRebinder(model)
    for placement in tensor_placements:
        if (placement.name not in rebinder.parameter_owner
                and placement.name not in rebinder.buffer_owner):
            raise RuntimeError(
                f"[flashboot] saved tensor {placement.name!r} is neither a parameter "
                f"nor a buffer of the model — the checkpoint does not match this build")
    return rebinder


def drop_saved_parameter_storage(model: nn.Module, tensor_placements: Sequence,
                                 device: torch.device, have_cuda: bool = True) -> None:
    """Release the device-init storage of every parameter the arena is about to own.

    Called BEFORE the arena is allocated so peak memory stays at ~1x the weights instead
    of holding the init copy and the arena at the same time. The Parameter objects
    survive (empty storage), so anything that captured them keeps working until the
    rebind gives them real bytes.
    """
    parameters = dict(model.named_parameters())
    for placement in tensor_placements:
        parameter = parameters.get(placement.name)
        if parameter is not None:
            parameter.data = torch.empty(0, dtype=parameter.dtype, device=device)
    if have_cuda:
        torch.cuda.empty_cache()


def bind_placements(model: nn.Module, rebinder: ArenaRebinder,
                    arena_handle: ArenaHandle, tensor_placements: Sequence) -> None:
    """Point every saved param/buffer at its arena address and record the layout.

    ``set_`` keeps the Parameter identity (an O(1) metadata operation), so extra
    attributes and ``.data`` aliases captured at init survive; the redirect scan then
    fixes the plain-attribute snapshots that did not go through ``.data``. dtype and
    shape both come from the placement — i.e. from the checkpoint header — because by
    now the parameter's own storage is gone and nothing can be fished back out of it.
    """
    for placement in tensor_placements:
        view = rebinder.build_view(arena_handle.byte_view, placement.arena_offset,
                                   placement.nbytes, placement.dtype,
                                   placement.shape, row_major_stride(placement.shape))
        if (placement.name in rebinder.buffer_owner
                and placement.name not in rebinder.parameter_owner):
            rebinder.assign_buffer_view(placement.name, view)
        else:
            rebinder.assign_parameter_view(placement.name, view)
        arena_handle.layout[placement.name] = (placement.arena_offset, placement.nbytes)
    rebinder.redirect_stale_references()
    # Two families of "the arena already holds the final post-load state" fixups, kept
    # apart by concern: generic quantization skip flags, and deepseek-lineage state.
    mark_quant_weights_finalized(model)
    fixup_deepseek_arena_state(model)
    model._flashboot_arena = arena_handle    # keepalive: the arena outlives every view


def mark_quant_weights_finalized(model: nn.Module) -> None:
    """Quantization: mark quantized weights/scales as ALREADY in their final layout so
    ``process_weights_after_loading`` does not re-quant / re-transform the arena bytes.

    The arena holds the checkpoint's post-``process_weights_after_loading`` bytes, so
    these per-param/per-module flags must read "done". Leaving one unset does not cost
    time — it corrupts weights, deterministically, because none of the transforms they
    gate is idempotent. They are set on the Parameter object, which ``set_`` rebinding
    preserves, so this is idempotent itself. Generic across quantized fused-MoE / fp8
    checkpoints; a no-op for models without these params.

      * ``_fast_format_expert_loaded`` — skip the fused-MoE expert shuffle/requant.
      * ``format_ue8m0`` — gate block-scale requant (fp8.py / linear.py).
      * ``_fb_sf_transformed`` — gate the non-idempotent MoE scale-layout transform.
    """
    for _, module in model.named_modules():
        if hasattr(module, "w13_weight") and hasattr(module, "w2_weight"):
            module._fast_format_expert_loaded = True
        for attribute in ("weight_scale_inv", "w13_weight_scale_inv",
                          "w2_weight_scale_inv"):
            parameter = getattr(module, attribute, None)
            if isinstance(parameter, torch.nn.Parameter):
                parameter.format_ue8m0 = True
                parameter._fb_sf_transformed = True


def fixup_deepseek_arena_state(model: nn.Module) -> int:
    """DeepSeek (v2 lineage, incl. DSV4 / GLM MoE) arena state that the rebind cannot
    carry — a no-op for other architectures. Returns the compressors marked.

      * Attention compressor ``ape_converted`` (DSV4-only): ``apply_ape_hotfix``
        permutes ``ape`` in place (chunk+cat, NOT idempotent). The saved ``ape`` is
        already permuted, and ``ape_converted`` is a runtime flag the arena binding does
        not restore; without marking it done, post-load permutes ``ape`` AGAIN ->
        deterministic systematic attention drift, byte-identical across runs, which
        reads like a model quality problem rather than a loader bug.
      * ``FusedMoE.correction_bias`` re-point: deepseek_v2 captures
        ``self.correction_bias = gate.e_score_correction_bias.data`` at ``__init__`` — a
        standalone ref to the OLD ``.data``. The redirect scan catches it for the main
        model but not the NEXTN draft, so re-point explicitly at the gate's CURRENT
        (arena-bound) parameter data so both are correct.
    """
    marked_compressors = 0
    for _, module in model.named_modules():
        if getattr(module, "ape_converted", None) is False:
            module.ape_converted = True
            marked_compressors += 1
        gate = getattr(module, "gate", None)
        if (hasattr(module, "correction_bias")
                and gate is not None
                and getattr(gate, "e_score_correction_bias", None) is not None):
            module.correction_bias = gate.e_score_correction_bias.data
    if marked_compressors:
        print(f"[flashboot] marked {marked_compressors} attention compressor(s) "
              f"ape_converted=True (arena already holds the post-load hotfixed ape)",
              flush=True)
    return marked_compressors


def assert_bound_tensors_live_in_the_arena(model: nn.Module, arena_handle: ArenaHandle,
                                           tag: str) -> List[str]:
    """Every bound param/buffer's address must lie inside the arena. Returns the names
    checked; raises on the first escapee — a post-load step that reallocated a weight out
    of the contiguous block, which would silently leave it out of the registered region.
    """
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    checked: List[str] = []
    for name in arena_handle.layout:
        tensor = parameters.get(name)
        if tensor is None:
            tensor = buffers.get(name)
        if tensor is None or tensor.numel() == 0:
            continue
        if not (arena_handle.base <= tensor.data_ptr()
                < arena_handle.base + arena_handle.size):
            raise RuntimeError(
                f"[flashboot] arena contiguity {tag}: {name} ptr={tensor.data_ptr()} "
                f"outside [{arena_handle.base},{arena_handle.base + arena_handle.size})")
        checked.append(name)
    return checked

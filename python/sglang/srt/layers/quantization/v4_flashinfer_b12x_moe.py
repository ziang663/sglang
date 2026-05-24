"""Experimental DeepSeek-V4 MXFP4 MoE path using FlashInfer B12x on SM120.

This path keeps the checkpoint's packed FP4 weights on GPU and converts only
the block-scale layout expected by FlashInfer's SM120 W4A16 kernels:

* DeepSeek-V4 checkpoint: E8M0 scale, one scale per 32 K values.
* FlashInfer B12x W4A16: E4M3 scale storage, one scale per 16 K values, in
  the swizzled storage layout wrapped by convert_sf_to_mma_layout.

The conversion duplicates each group-32 scale into two group-16 entries and
casts the power-of-two E8M0 value to E4M3. Packed FP4 weights are not
dequantized.

This is intentionally opt-in. FlashInfer B12x 0.6.11 does not expose the
DeepSeek-V4 asymmetric SwiGLU clamp, so callers must reject clamped models
unless they are doing a performance-only experiment.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def use_v4_b12x_moe() -> bool:
    return os.environ.get("SGLANG_V4_USE_B12X_MOE") == "1"


def ignore_b12x_swiglu_limit() -> bool:
    return os.environ.get("SGLANG_V4_B12X_IGNORE_SWIGLU_LIMIT") == "1"


def b12x_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() in {(12, 0), (12, 1)}


def _ensure_cuda_headers_for_flashinfer_jit() -> None:
    """Make FlashInfer JIT host compilation see CUDA runtime headers.

    In the local CUDA-13 venv, nvcc is available under the venv but some
    FlashInfer C++ JIT commands only add $VENV/include, not
    $VENV/targets/x86_64-linux/include. CPLUS_INCLUDE_PATH fixes the host C++
    compile without changing FlashInfer.
    """

    candidates = []
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidates.extend(
            [
                os.path.join(venv, "targets", "x86_64-linux", "include"),
                os.path.join(venv, "lib", "python3.10", "site-packages", "nvidia", "cu13", "include"),
                os.path.join(venv, "lib", "python3.10", "site-packages", "nvidia", "cuda_runtime", "include"),
            ]
        )
    candidates.extend(
        [
            "/usr/local/cuda/targets/x86_64-linux/include",
            "/usr/local/cuda/include",
        ]
    )
    existing = [p for p in candidates if os.path.exists(os.path.join(p, "cuda_fp16.h"))]
    if not existing:
        return
    for env_name in ("CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
        cur = os.environ.get(env_name, "")
        parts = [p for p in cur.split(":") if p]
        merged = existing + [p for p in parts if p not in existing]
        os.environ[env_name] = ":".join(merged)


def _swizzle_scale_storage(unswizzled: torch.Tensor) -> torch.Tensor:
    """Inverse of FlashInfer's unswizzle_block_scale for one expert batch.

    Input/output shape is [E, rows_padded, cols_padded]. The physical order is
    what convert_sf_to_mma_layout later wraps as a 6D strided MMA view.
    """

    if unswizzled.ndim != 3:
        raise ValueError(f"expected [E, rows, cols] scale tensor, got {unswizzled.shape}")
    e, rows_padded, cols_padded = unswizzled.shape
    if rows_padded % 128 != 0 or cols_padded % 4 != 0:
        raise ValueError(
            "scale tensor must be padded to rows%128==0 and cols%4==0, "
            f"got {unswizzled.shape}"
        )
    return (
        unswizzled.view(e, rows_padded // 128, 4, 32, cols_padded // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .view(e, rows_padded, cols_padded)
    )


def _v4_scale_to_b12x_mma(
    scale_e8m0: torch.Tensor,
    *,
    rows: int,
    cols: int,
    num_experts: int,
) -> torch.Tensor:
    """Convert V4 [E, rows, cols/32] E8M0 scale to B12x 6D MMA scale view."""

    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    if scale_e8m0.dtype != torch.float8_e8m0fnu:
        if scale_e8m0.dtype == torch.uint8:
            scale_e8m0 = scale_e8m0.view(torch.float8_e8m0fnu)
        else:
            scale_e8m0 = scale_e8m0.to(torch.float8_e8m0fnu)

    if tuple(scale_e8m0.shape) != (num_experts, rows, cols // 32):
        raise ValueError(
            f"expected V4 scale shape {(num_experts, rows, cols // 32)}, "
            f"got {tuple(scale_e8m0.shape)}"
        )

    rows_padded = ((rows + 127) // 128) * 128
    cols16 = cols // 16
    cols_padded = ((cols16 + 3) // 4) * 4

    # group32 -> group16. Converting through fp32 preserves E8M0 powers of two
    # in the E4M3 range used by the B12x W4A16 scale loader.
    scale16 = scale_e8m0.to(torch.float32).repeat_interleave(2, dim=2)
    scale16 = scale16.to(torch.float8_e4m3fn)

    storage = torch.empty(
        (num_experts, rows_padded, cols_padded),
        device=scale16.device,
        dtype=torch.float8_e4m3fn,
    )
    storage.fill_(1.0)
    storage[:, :rows, :cols16] = scale16
    storage = _swizzle_scale_storage(storage)

    return convert_sf_to_mma_layout(
        storage.reshape(num_experts * rows_padded, cols_padded),
        m=rows,
        k=cols,
        num_groups=num_experts,
        sf_vec_size=16,
    )


def convert_v4_weights_to_b12x(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _ensure_cuda_headers_for_flashinfer_jit()

    if w13.dtype != torch.uint8:
        w13 = w13.view(torch.uint8)
    if w2.dtype != torch.uint8:
        w2 = w2.view(torch.uint8)

    num_experts = w13.shape[0]
    hidden_size = w13.shape[2] * 2
    intermediate_size = w2.shape[2] * 2
    w13_rows = 2 * intermediate_size

    w13_mma_scale = _v4_scale_to_b12x_mma(
        w13_scale,
        rows=w13_rows,
        cols=hidden_size,
        num_experts=num_experts,
    )
    w2_mma_scale = _v4_scale_to_b12x_mma(
        w2_scale,
        rows=hidden_size,
        cols=intermediate_size,
        num_experts=num_experts,
    )
    w1_alpha = torch.ones(num_experts, device=w13.device, dtype=torch.float32)
    w2_alpha = torch.ones(num_experts, device=w13.device, dtype=torch.float32)
    return w13.contiguous(), w13_mma_scale, w1_alpha, w2.contiguous(), w2_mma_scale, w2_alpha


def apply_v4_b12x_moe(
    *,
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale,
    w1_alpha: torch.Tensor,
    w2: torch.Tensor,
    w2_scale,
    w2_alpha: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    _ensure_cuda_headers_for_flashinfer_jit()
    from flashinfer import b12x_fused_moe

    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.to(torch.float32)

    return b12x_fused_moe(
        x=hidden_states,
        w1_weight=w13,
        w1_weight_sf=w13_scale,
        w1_alpha=w1_alpha,
        w2_weight=w2,
        w2_weight_sf=w2_scale,
        w2_alpha=w2_alpha,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        num_experts=num_experts,
        top_k=topk_ids.shape[1],
        activation_precision="bf16",
    )

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import (
    get_device_sm,
    is_blackwell_supported,
    is_cuda,
    is_musa,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_musa = is_musa()

# DeepGEMM kernels are valid on Hopper and data-center Blackwell paths, but
# not consumer Blackwell SM120. Importing deep_gemm can still succeed there,
# so gate by capability before any kernel path is selected.
DEEPGEMM_CAPS = {(9, 0), (10, 0), (10, 3)}


def _compute_enable_deep_gemm():
    if torch.cuda.is_available() and torch.cuda.get_device_capability() not in DEEPGEMM_CAPS:
        return False

    sm_version = get_device_sm()
    if (_is_cuda and sm_version < 90) or (_is_musa and sm_version < 31):
        return False
    if not (_is_cuda or _is_musa):
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_blackwell_supported()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
DEEPGEMM_NEED_TMA_ALIGNED_SCALES = not (DEEPGEMM_SCALE_UE8M0 or _is_musa)

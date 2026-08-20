"""Device arena binding + rebinding a model's parameters onto zero-copy arena views.

The arena is ONE contiguous ``cudaMalloc`` region per rank (``flashboot._C.DeviceArena``)
holding the checkpoint's raw bytes in file layout. After the fill (disk on the seed,
an NVLink pull on the clone) every saved parameter/buffer is REBOUND onto a typed view of its
arena bytes, so the model serves zero-copy (1x weight memory) and the whole image can be
pulled by a clone in one NVLink copy.

Two pieces:

  * :class:`ArenaHandle` — the allocated arena plus its addressing info, passed
    explicitly between the load steps (allocate -> fill -> rebind -> serve) instead of
    living as loose loader attributes.
  * :class:`ArenaRebinder` — swaps a device-initialized model's params/buffers for arena
    views and then redirects any stale plain-attribute references to the old tensors.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Tuple

import torch

from flashboot import _C


@dataclasses.dataclass
class ArenaHandle:
    """One rank's allocated device arena. Owns the memory: ``arena`` is the native DeviceArena
    whose destructor frees the cudaMalloc, so this object must stay alive as long as any
    model view into it (the loader stashes it on the model as ``_flashboot_arena``)."""
    arena: object                        # flashboot._C.DeviceArena
    base: int                            # device address of byte 0
    size: int                            # arena bytes
    byte_view: torch.Tensor              # uint8 tensor aliasing the whole arena
    layout: Dict[str, Tuple[int, int]]   # tensor name -> (arena_offset, nbytes)


def allocate_device_arena(total_bytes: int, device: int,
                          fabric_exportable: bool = False) -> ArenaHandle:
    """Allocate ONE ``total_bytes`` arena on ``device`` and wrap it as an ArenaHandle
    (with an aliasing uint8 view for offset math). The layout starts empty and is filled
    by the rebind step. ``fabric_exportable`` switches the allocation from cudaMalloc to
    fabric-exportable CUDA VMM memory — needed ONLY by a seed that serves its arena over
    the fabric transport (ipc exports plain cudaMalloc memory)."""
    arena = _C.DeviceArena()
    arena.create(int(max(total_bytes, 256)), int(device), bool(fabric_exportable))
    base = int(arena.ptr(0))
    template = torch.empty(0, dtype=torch.uint8, device=torch.device("cuda", device))
    byte_view = _C.arena_view(base, [int(max(total_bytes, 1))], [1], template)
    return ArenaHandle(arena=arena, base=base, size=int(total_bytes),
                        byte_view=byte_view, layout={})


class ArenaRebinder:
    """Replaces a device-initialized model's parameters/buffers with zero-copy views onto
    the loaded GPU arena, then redirects stale plain-attribute references that still point
    at the old tensors.

    Why a class: the assign and redirect steps share mutable state — the
    ``id(old) -> new_view`` map and a keepalive list of the old tensors. The keepalive
    matters: without it, freed tensors' ids get recycled onto fresh views and the redirect
    scan would corrupt healthy tensors.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        # name -> (owning_module, attribute) for every param and buffer.
        self.parameter_owner = {}
        self.buffer_owner = {}
        for module_name, module in model.named_modules():
            prefix = module_name + "." if module_name else ""
            for parameter_name in module._parameters.keys():
                self.parameter_owner[prefix + parameter_name] = (module, parameter_name)
            for buffer_name in module._buffers.keys():
                self.buffer_owner[prefix + buffer_name] = (module, buffer_name)
        self.replaced_tensor_views = {}   # id(old_tensor) -> new_cuda_tensor
        self._keepalive = []              # strong refs to old tensors (id-recycle guard)
        self.n_redirected = 0
        self.unresolved_meta_references = []

    @staticmethod
    def build_view(arena_bytes: torch.Tensor, offset: int, nbytes: int,
                   torch_dtype: torch.dtype, shape: List[int],
                   stride: List[int]) -> torch.Tensor:
        """Reinterpret ``arena_bytes[offset : offset+nbytes]`` as a typed, shaped tensor.
        ``as_strided`` preserves non-row-major layouts (e.g. MN-major TMA-aligned ue8m0
        scales) including any storage padding between logical rows."""
        typed_1d = arena_bytes[offset:offset + nbytes].view(torch_dtype)
        if not shape:
            return typed_1d.as_strided((), (), typed_1d.storage_offset())  # 0-d scalar
        if not stride:
            return typed_1d.reshape(shape)  # no stride metadata: row-major
        return typed_1d.as_strided(tuple(shape), tuple(stride), typed_1d.storage_offset())

    def assign_parameter_view(self, name: str, view: torch.Tensor) -> bool:
        """Rebind parameter ``name`` onto ``view``: ``set_`` retargets the existing
        Parameter in place, an O(1) metadata operation that preserves both the Parameter
        object and its TensorImpl — so extra attributes (``format_ue8m0`` etc.), external
        references to the Parameter, AND ``.data``-level aliases captured at init (e.g.
        deepseek_v2's ``correction_bias = gate.e_score_correction_bias.data``) all keep
        working without any redirect.

        ``set_`` succeeds exactly when the checkpoint tensor matches the device-init
        parameter (same dtype/shape extent — a mismatch overruns the non-resizable
        from_blob arena storage and raises). An incompatible checkpoint therefore fails
        loudly right here instead of serving a mis-bound model."""
        owner = self.parameter_owner.get(name)
        if owner is None:
            return False
        module, attribute = owner
        old_parameter = module._parameters[attribute]
        if old_parameter is None:
            return False
        with torch.no_grad():
            old_parameter.set_(view.untyped_storage(), view.storage_offset(),
                               view.shape, view.stride())
        # Same object before and after — record self->self so the redirect scan no-ops.
        self._keepalive.append(old_parameter)
        self.replaced_tensor_views[id(old_parameter)] = old_parameter
        return True

    def assign_buffer_view(self, name: str, view: torch.Tensor) -> bool:
        """Install ``view`` as buffer ``name`` (direct ``_buffers`` dict swap, recording
        the old->view mapping for the redirect scan)."""
        owner = self.buffer_owner.get(name)
        if owner is None:
            return False
        module, attribute = owner
        old_buffer = module._buffers[attribute]
        module._buffers[attribute] = view
        if old_buffer is not None:
            self._keepalive.append(old_buffer)
            self.replaced_tensor_views[id(old_buffer)] = view
        return True

    def redirect_stale_references(self) -> None:
        """Walk every module's ``__dict__`` (plus dataclass-like sub-objects and tensor
        lists/tuples/dicts) and rewrite any plain-attribute reference that still points at
        a replaced tensor. Some sub-modules snapshot a Parameter/buffer at init as a plain
        attribute (e.g. TopKConfig.correction_bias); those stale references crash DLPack
        at the first forward unless redirected here. Records the redirect count and any
        tensors left on the meta device."""
        replaced = self.replaced_tensor_views
        n_redirected = 0
        unresolved = []

        def redirect(value: object) -> object:
            if isinstance(value, torch.Tensor) and id(value) in replaced:
                return replaced[id(value)]
            return value

        def scan(obj: object, path: str, depth: int = 0) -> None:
            nonlocal n_redirected
            if depth > 6:
                return
            attributes = getattr(obj, "__dict__", None)
            if attributes is None:
                return
            for key in list(attributes.keys()):
                value = attributes[key]
                if isinstance(value, torch.Tensor):
                    if id(value) in replaced:
                        setattr(obj, key, replaced[id(value)])
                        n_redirected += 1
                    elif value.device.type == "meta":
                        unresolved.append(f"{path}.{key}")
                elif isinstance(value, (list, tuple)):
                    if any(isinstance(item, torch.Tensor) for item in value):
                        new_sequence = [redirect(item) if isinstance(item, torch.Tensor)
                                        else item for item in value]
                        n_redirected += sum(1 for a, b in zip(value, new_sequence)
                                            if a is not b)
                        setattr(obj, key, type(value)(new_sequence))
                        for i, item in enumerate(value):
                            if (isinstance(item, torch.Tensor)
                                    and item.device.type == "meta"
                                    and id(item) not in replaced):
                                unresolved.append(f"{path}.{key}[{i}]")
                elif isinstance(value, dict):
                    for sub_key, sub_value in list(value.items()):  # shallow dict scan
                        if isinstance(sub_value, torch.Tensor):
                            if id(sub_value) in replaced:
                                value[sub_key] = replaced[id(sub_value)]
                                n_redirected += 1
                            elif sub_value.device.type == "meta":
                                unresolved.append(f"{path}.{key}[{sub_key!r}]")
                else:
                    # Recurse into dataclass-like / namespaced config objects only.
                    if (not isinstance(value, torch.nn.Module)
                            and not isinstance(value, (int, float, str, bool,
                                                       type(None), bytes))
                            and hasattr(value, "__dict__")):
                        cls = type(value)
                        module_of = getattr(cls, "__module__", "") or ""
                        if module_of.startswith(("sglang", "torch.")) or "Config" in cls.__name__:
                            scan(value, f"{path}.{key}", depth + 1)

        for module_name, module in self.model.named_modules():
            scan(module, module_name or "<root>", depth=0)
        self.n_redirected = n_redirected
        self.unresolved_meta_references = unresolved

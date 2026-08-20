"""safetensors header parsing + dtype mapping (pure CPU, torch imported lazily).

A safetensors file is ``<u64 little-endian header_len><header_len bytes of JSON><data>``.
The JSON header maps every tensor name to its dtype / shape / byte range inside the data
region; parsing it costs one small read, no tensor data is touched. This is all the
checkpoint knowledge the flashboot loaders need — the actual bytes are moved by
:mod:`flashboot.arena_fill` (seed) or an NVLink pull (clone).
"""
from __future__ import annotations

import json
import struct
from typing import Tuple

_SAFETENSORS_DTYPE_TO_TORCH_NAME = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
    "U8": "uint8", "BOOL": "bool",
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
    "F8_E8M0": "float8_e8m0fnu",  # mxfp4 block scales (DeepSeek-V4 experts)
}


def safetensors_dtype_to_torch(dtype_str: str):
    """Map a safetensors dtype string (e.g. ``"F8_E4M3"``) to the torch dtype."""
    import torch

    name = _SAFETENSORS_DTYPE_TO_TORCH_NAME.get(dtype_str)
    if name is None:
        raise KeyError(f"[flashboot] unsupported safetensors dtype {dtype_str!r}")
    return getattr(torch, name)


def read_safetensors_header(path: str) -> Tuple[dict, int]:
    """Return ``(header_without_metadata_key, data_region_offset)`` for one file.

    On-disk layout::

        [ 8 bytes: u64 LE header_len ][ header_len bytes: UTF-8 JSON ][ data region ]

    The JSON parses into a flat dict keyed by tensor name, plus one optional reserved
    ``"__metadata__"`` key. A typical parsed header looks like::

        {
            "__metadata__": {"format": "pt", ...},          # optional free-form str->str map
            "model.layers.0.mlp.gate.weight": {             # one entry per tensor
                "dtype": "F8_E4M3",                         # safetensors dtype string
                "shape": [4096, 4096],                      # logical shape (row-major)
                "data_offsets": [0, 16777216],              # [start, end) BYTES in the data region
            },
            "model.layers.0.input_layernorm.weight": {
                "dtype": "BF16", "shape": [4096], "data_offsets": [16777216, 16785408],
            },
            ...
        }

    So each tensor entry carries exactly three fields:
      * ``dtype``        — safetensors code (map via :func:`safetensors_dtype_to_torch`).
      * ``shape``        — list of ints; ``[]`` for a 0-d scalar; product x itemsize == nbytes.
      * ``data_offsets`` — ``[start, end)`` byte range **relative to the data region**, i.e.
        the tensor's bytes are the file bytes ``[data_region_offset + start,
        data_region_offset + end)``. Entries are contiguous and cover the whole data region.

    We drop ``"__metadata__"`` (bookkeeping only, not a tensor) so callers can iterate the
    returned dict as "name -> tensor spec" without special-casing it. ``data_region_offset``
    (``8 + header_len``) is where the data region begins, added to each entry's offsets to get
    absolute file positions.
    """
    with open(path, "rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"[flashboot] {path}: truncated safetensors header length")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_bytes = fh.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"[flashboot] {path}: truncated safetensors header body")
    header = json.loads(header_bytes)
    header.pop("__metadata__", None)
    return header, 8 + header_len

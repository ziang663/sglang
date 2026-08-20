#!/usr/bin/env python3
"""Check the aligned sharded_state export, on CPU, in a couple of seconds.

    python3 scripts/flashboot/verify_pad_alignment.py

No GPU, no model, no checkpoint. It builds a state dict shaped like the thing that
breaks alignment in practice -- fp32 scalars and fp8 blocks mixed in among bf16
weights -- writes it both ways, and checks the result.

Three of the four cases are regressions. Each one produces a file that saves without
complaint and loads without complaint, and is silently wrong:

  1. Fillers of the wrong dtype. safetensors orders the data region by (dtype, name),
     so a uint8 filler sorts to the end of the file and aligns nothing.
  2. Fillers reaching the loader. Stock load_model indexes state_dict by every key in
     the file and then asserts nothing is left over, so a filler is a KeyError.
  3. Padding that changes the weights. The bytes a reader gets for a real tensor must
     be identical either way.

Exit status is 0 only if every case passes.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile

import torch
from safetensors.torch import save_file

ALIGNMENT = 256
PAD_KEY_SUFFIX = "!__pad__"


def pad_state_dict(part: dict, alignment: int) -> dict:
    """The reference implementation -- same logic as the integrated save_model."""
    if alignment <= 0:
        return part
    padded = dict(part)
    for key, tensor in part.items():
        size = tensor.nelement() * tensor.element_size()
        gap = (-size) % alignment
        if gap:
            padded[key + PAD_KEY_SUFFIX] = torch.zeros(
                gap // tensor.element_size(), dtype=tensor.dtype, device=tensor.device
            )
    return padded


def read_header(path: str) -> dict:
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length))


def build_state_dict() -> dict:
    torch.manual_seed(0)
    return {
        # Big weights, the common case.
        "layers.0.mlp.w1": torch.randn(1024, 512, dtype=torch.bfloat16),
        "layers.1.mlp.w1": torch.randn(777, 33, dtype=torch.bfloat16),
        "layers.0.attn.qkv": torch.randn(300, 129, dtype=torch.bfloat16),
        # The fp32 scalars that knock everything after them out of alignment.
        "layers.0.attn.k_scale": torch.randn(1, dtype=torch.float32),
        "layers.1.hc_attn_scale": torch.randn(3, dtype=torch.float32),
        # A third dtype, so the (dtype, name) ordering has something to sort.
        "layers.1.w_fp8": torch.zeros(101, dtype=torch.float8_e4m3fn),
    }


def main() -> int:
    part = build_state_dict()
    workdir = tempfile.mkdtemp(prefix="fbpad-")
    failures = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    print("unpadded export (stock behaviour)")
    plain = os.path.join(workdir, "plain.safetensors")
    save_file(dict(part), plain)
    header = read_header(plain)
    real = {k: v for k, v in header.items() if k != "__metadata__"}
    misaligned = [k for k, v in real.items() if v["data_offsets"][0] % ALIGNMENT]
    check(
        "stock file is NOT aligned (this is what the patch exists for)",
        len(misaligned) > 0,
        f"{len(misaligned)}/{len(real)} tensors off a {ALIGNMENT}B boundary",
    )

    print("padded export")
    aligned = os.path.join(workdir, "aligned.safetensors")
    save_file(pad_state_dict(dict(part), ALIGNMENT), aligned)
    header = read_header(aligned)
    reals = {
        k: v
        for k, v in header.items()
        if k != "__metadata__" and not k.endswith(PAD_KEY_SUFFIX)
    }
    fillers = [k for k in header if k.endswith(PAD_KEY_SUFFIX)]
    off = [k for k, v in reals.items() if v["data_offsets"][0] % ALIGNMENT]
    check(
        "every real tensor starts on a boundary",
        not off,
        f"{len(reals)} tensors, {len(fillers)} fillers" + (f", off: {off[:3]}" if off else ""),
    )

    grew = os.path.getsize(aligned) - os.path.getsize(plain)
    check(
        "the file grows by a negligible amount",
        grew < os.path.getsize(plain) * 0.02,
        f"+{grew} bytes ({grew / os.path.getsize(plain) * 100:.3f}%)",
    )

    # Regression 1: fillers must carry the dtype of the tensor they follow.
    wrong = dict(part)
    for key, tensor in part.items():
        gap = (-(tensor.nelement() * tensor.element_size())) % ALIGNMENT
        if gap:
            wrong[key + PAD_KEY_SUFFIX] = torch.zeros(gap, dtype=torch.uint8)
    bad_path = os.path.join(workdir, "wrongdtype.safetensors")
    save_file(wrong, bad_path)
    header = read_header(bad_path)
    reals = {
        k: v
        for k, v in header.items()
        if k != "__metadata__" and not k.endswith(PAD_KEY_SUFFIX)
    }
    still_off = [k for k, v in reals.items() if v["data_offsets"][0] % ALIGNMENT]
    check(
        "uint8 fillers do NOT align anything (saves fine, aligns nothing)",
        len(still_off) > 0,
        f"{len(still_off)}/{len(reals)} still misaligned -- fillers sorted away",
    )

    # Regression 2: a loader that does not skip fillers breaks.
    model_keys = set(part)
    file_keys = {k for k in read_header(aligned) if k != "__metadata__"}
    check(
        "fillers are present in the file and absent from the model",
        file_keys - model_keys == set(fillers) and model_keys <= file_keys,
        f"{len(fillers)} keys a loader must skip",
    )

    # Regression 3: the weights themselves are untouched.
    from safetensors.torch import load_file

    a, b = load_file(plain), load_file(aligned)
    same = all(torch.equal(a[k], b[k]) for k in part)
    check("real tensors are byte-identical either way", same)

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

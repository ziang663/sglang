"""sglang build hooks.

SGLANG_BUILD_RUST_EXTS controls which Rust extensions are built:
  - unset or "all": build every declared Rust extension (the default).
  - "none": build no Rust extensions.
  - comma-separated names: build only extensions whose target matches one of the
    given (case-insensitive) substrings, e.g. "grpc" matches
    "sglang.srt.grpc._core".

This is a build-time environment variable, so it is read directly from
os.environ instead of sglang.srt.environ, which is not available until after the
package has been built.
"""

import os

from setuptools import setup
from setuptools.command.build_ext import build_ext as build_ext_orig

try:
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    CUDA_HOME = None
    BuildExtension = None
    CUDAExtension = None

try:
    from setuptools_rust import build_rust
except ModuleNotFoundError as exc:
    if exc.name != "setuptools_rust":
        raise
    # Alternate platform pyprojects do not declare Rust extensions.
    build_rust = None

_BUILD_RUST_EXTS_ENV = "SGLANG_BUILD_RUST_EXTS"
_BUILD_FLASHBOOT_ENV = "SGLANG_BUILD_FLASHBOOT"


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _flashboot_rdma_enabled() -> bool:
    requested = os.environ.get("FB_BUILD_RDMA", "auto").strip().lower()
    if requested in ("1", "true", "yes", "on"):
        return True
    if requested in ("0", "false", "no", "off"):
        return False
    if requested not in ("", "auto"):
        raise ValueError(
            f"FB_BUILD_RDMA={requested!r} is not supported: expected auto, 1 or 0"
        )

    include_dirs = ["/usr/include", "/usr/local/include"]
    include_dirs += os.environ.get("CPATH", "").split(os.pathsep)
    include_dirs += os.environ.get("C_INCLUDE_PATH", "").split(os.pathsep)
    include_dirs += os.environ.get("CPLUS_INCLUDE_PATH", "").split(os.pathsep)
    for prefix_var in ("CONDA_PREFIX", "PREFIX"):
        prefix = os.environ.get(prefix_var, "")
        if prefix:
            include_dirs.append(os.path.join(prefix, "include"))
    return any(
        directory
        and os.path.exists(os.path.join(directory, "infiniband", "verbs.h"))
        for directory in include_dirs
    )


def _flashboot_extension():
    if not _truthy_env(_BUILD_FLASHBOOT_ENV):
        return None
    if CUDAExtension is None or BuildExtension is None:
        raise RuntimeError(
            f"{_BUILD_FLASHBOOT_ENV}=1 requires torch to be installed at build time"
        )

    build_rdma = _flashboot_rdma_enabled()
    sources = [
        "../csrc/flashboot/python_bindings.cc",
        "../csrc/flashboot/device_arena.cu",
        "../csrc/flashboot/pinned_stage.cpp",
        "../csrc/flashboot/peer_arena_import.cu",
        "../csrc/flashboot/imex_check.cc",
        "../csrc/flashboot/chain_broadcast.cu",
    ]
    if build_rdma:
        sources.append("../csrc/flashboot/rdma_read.cpp")

    arches = os.environ.get("FLASHBOOT_CUDA_ARCH", "9.0;10.0").split(";")
    nvcc_arch = []
    for arch in arches:
        arch = arch.strip().replace(".", "")
        if arch:
            nvcc_arch += [f"-gencode=arch=compute_{arch},code=sm_{arch}"]

    rdma_macros = [] if build_rdma else ["-DFB_NO_RDMA"]
    driver_stub_dirs = (
        [os.path.join(CUDA_HOME, "lib64", "stubs")] if CUDA_HOME else []
    )
    return CUDAExtension(
        name="flashboot._C",
        sources=sources,
        include_dirs=["../include"],
        libraries=(["ibverbs"] if build_rdma else []) + ["cuda"],
        library_dirs=driver_stub_dirs,
        extra_compile_args={
            "cxx": [
                "-O3",
                "-std=c++17",
                "-fvisibility=default",
                "-pthread",
            ]
            + rdma_macros,
            "nvcc": ["-O3", "-std=c++17"] + rdma_macros + nvcc_arch,
        },
    )


def _selected_rust_extensions(declared):
    """Return the Rust extensions selected by SGLANG_BUILD_RUST_EXTS.

    `ext.name` is the fully-qualified target (e.g. "sglang.srt.grpc._core") for
    the string-target declarations in pyproject.toml, so comma-separated names
    are matched as case-insensitive substrings of it.
    """
    declared = list(declared)
    raw = os.environ.get(_BUILD_RUST_EXTS_ENV)
    if raw is None:
        return declared

    spec = raw.strip().lower()
    # An empty or whitespace-only value is treated as unset (build everything).
    if not spec or spec == "all":
        return declared
    if spec == "none":
        return []

    tokens = [token.strip() for token in spec.split(",")]
    if not all(tokens):
        raise ValueError(
            f"{_BUILD_RUST_EXTS_ENV}={raw!r} has an empty item; unset it or use "
            "'all', 'none', or a comma-separated list of extension names"
        )

    matched = set()
    unmatched = []
    for token in tokens:
        hits = {ext.name for ext in declared if token in ext.name.lower()}
        if hits:
            matched |= hits
        else:
            unmatched.append(token)
    if unmatched:
        declared_names = sorted(ext.name for ext in declared)
        raise ValueError(
            f"{_BUILD_RUST_EXTS_ENV} matched no declared Rust extension for: "
            f"{unmatched}; declared extensions are {declared_names}"
        )

    return [ext for ext in declared if ext.name in matched]


if build_rust is not None:

    class BuildRust(build_rust):
        """Build only the Rust extensions selected by SGLANG_BUILD_RUST_EXTS."""

        def run(self) -> None:
            rust_extensions = _selected_rust_extensions(self.extensions or [])
            self.extensions = rust_extensions
            self.distribution.rust_extensions = rust_extensions
            if not rust_extensions:
                return
            super().run()

    _cmdclass = {"build_rust": BuildRust}
else:
    _cmdclass = {}

flashboot_ext = _flashboot_extension()
ext_modules = [flashboot_ext] if flashboot_ext is not None else []
if flashboot_ext is not None:
    _cmdclass["build_ext"] = BuildExtension
else:
    _cmdclass.setdefault("build_ext", build_ext_orig)

setup(cmdclass=_cmdclass, ext_modules=ext_modules)

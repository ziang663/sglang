FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      git-lfs \
      libgomp1 \
      libnuma1 \
      libopenmpi3t64 \
      libibverbs1 \
      librdmacm1 \
      libnl-3-200 \
      libnl-route-3-200 \
      netcat-openbsd \
      openssh-client \
      pciutils \
      procps \
      rdma-core \
      wget \
    && rm -rf /var/lib/apt/lists/*

# Vendored image: copy the already validated local environment into the same
# absolute paths used by the conda/venv shebangs and editable installs.
# Build with /hisys/alan as the context:
#   DOCKER_BUILDKIT=1 docker build \
#     -f sglang/docker/dsv4-5090-vendored.Dockerfile \
#     -t sglang-dsv4-5090:vendored /hisys/alan
WORKDIR /hisys/alan/sglang
COPY sglang /hisys/alan/sglang
COPY flashinfer-sm120-sparse-mla /hisys/alan/flashinfer-sm120-sparse-mla

ENV VENV=/hisys/alan/sglang/.venv-dsv4-sm120 \
    CUDA_HOME=/hisys/alan/sglang/.venv-dsv4-sm120 \
    CUDA_PATH=/hisys/alan/sglang/.venv-dsv4-sm120 \
    TORCH_CUDA_ARCH_LIST=12.0a \
    DG_JIT_NVCC_COMPILER=/hisys/alan/sglang/.venv-dsv4-sm120/bin/nvcc \
    SGLANG_ENABLE_JIT_DEEPGEMM=1 \
    SGLANG_OPT_DEEPGEMM_HC_PRENORM=1 \
    SGLANG_OPT_USE_JIT_INDEXER_METADATA=0 \
    SGLANG_V4_TRITON_SM120_TILE=auto \
    SGLANG_DSV4_FLASHINFER_SM120=1 \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
    PATH=/hisys/alan/sglang/.venv-dsv4-sm120/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/hisys/alan/sglang/.venv-dsv4-sm120/lib:/hisys/alan/sglang/.venv-dsv4-sm120/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

COPY sglang/docker/entrypoint-dsv4-5090-vendored.sh /usr/local/bin/entrypoint-dsv4-5090-vendored.sh
RUN chmod +x /usr/local/bin/entrypoint-dsv4-5090-vendored.sh \
    && test -x /hisys/alan/sglang/.venv-dsv4-sm120/bin/python \
    && test -x /hisys/alan/sglang/.venv-dsv4-sm120/bin/sglang

EXPOSE 30000
ENTRYPOINT ["/usr/local/bin/entrypoint-dsv4-5090-vendored.sh"]

FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ARG DEEPGEMM_PR318_COMMIT=7a7a41a1bac7dacabe74057e7600e59f98f85bce
ARG FLASHINFER_SM120_COMMIT=c9ce444351052385828d607e57d78dfb65ac868e

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      git \
      git-lfs \
      libgomp1 \
      libnuma1 \
      libopenmpi-dev \
      libibverbs-dev \
      librdmacm-dev \
      libnl-3-dev \
      libnl-route-3-dev \
      netcat-openbsd \
      ninja-build \
      openssh-client \
      pciutils \
      procps \
      python3.12-full \
      python3.12-dev \
      python3-pip \
      python3-packaging \
      rdma-core \
      wget \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/sglang
COPY . /workspace/sglang

RUN python3 -m venv /opt/venv

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=12.0a \
    DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc \
    SGLANG_ENABLE_JIT_DEEPGEMM=1 \
    SGLANG_OPT_DEEPGEMM_HC_PRENORM=1 \
    SGLANG_OPT_USE_JIT_INDEXER_METADATA=0 \
    SGLANG_V4_TRITON_SM120_TILE=auto \
    SGLANG_DSV4_FLASHINFER_SM120=1 \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/root/.local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e /workspace/sglang/python

RUN git clone --recursive https://github.com/lucifer1004/flashinfer.git /tmp/flashinfer \
    && cd /tmp/flashinfer \
    && git checkout "${FLASHINFER_SM120_COMMIT}" \
    && git apply /workspace/sglang/docker/flashinfer-sm120-dsv4-pbs256.patch \
    && python -m pip install --no-build-isolation --force-reinstall --no-deps . \
    && rm -rf /tmp/flashinfer /root/.cache/pip

RUN git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /tmp/DeepGEMM \
    && cd /tmp/DeepGEMM \
    && git fetch origin pull/318/head \
    && git checkout "${DEEPGEMM_PR318_COMMIT}" \
    && git submodule update --init --recursive \
    && git apply /workspace/sglang/docker/deepgemm-pr318-metadata-dynamic-smem.patch \
    && DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=0 TORCH_CUDA_ARCH_LIST=12.0a \
       python -m pip install --no-build-isolation --force-reinstall --no-deps . \
    && rm -rf /tmp/DeepGEMM /root/.cache/pip

COPY docker/entrypoint-dsv4-5090.sh /usr/local/bin/entrypoint-dsv4-5090.sh
RUN chmod +x /usr/local/bin/entrypoint-dsv4-5090.sh

EXPOSE 30000
ENTRYPOINT ["/usr/local/bin/entrypoint-dsv4-5090.sh"]

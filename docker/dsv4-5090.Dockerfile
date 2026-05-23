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
      rdma-core \
      wget \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && python3 -m pip config set global.break-system-packages true \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/sglang
COPY . /workspace/sglang

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=12.0a \
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
    PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:/root/.local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -e /workspace/sglang/python

COPY docker/entrypoint-dsv4-5090.sh /usr/local/bin/entrypoint-dsv4-5090.sh
RUN chmod +x /usr/local/bin/entrypoint-dsv4-5090.sh

EXPOSE 30000
ENTRYPOINT ["/usr/local/bin/entrypoint-dsv4-5090.sh"]

# DeepSeek V4 Flash on 8x RTX 5090

This branch contains the RTX 5090 / SM120 path used to validate
DeepSeek-V4-Flash with SGLang.

## Validated Host Command

The host validation used the local CUDA 13.2 conda environment because the host
`/usr/local/cuda` symlink pointed to CUDA 12.3. The two required environment
settings are:

```bash
export CUDA_HOME=/path/to/cuda-13.2
export CUDA_PATH=$CUDA_HOME
export TORCH_CUDA_ARCH_LIST=12.0a
```

The working launch command was:

```bash
sglang serve \
  --model-path /models/preset/deepseek-ai/DeepSeek-V4-Flash/v1.0 \
  --trust-remote-code \
  --tp 8 \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.82 \
  --max-running-requests 8 \
  --chunked-prefill-size 2048 \
  --disable-flashinfer-autotune \
  --reasoning-parser deepseek-v4 \
  --served-model-name DeepSeek-V4-Flash-local \
  --cuda-graph-bs 1 2 4 8
```

The server automatically selected:

- `attention_backend=dsv4`
- `moe_runner_backend=flashinfer_mxfp4`
- GPU-only MXFP4 MoE on SM120

## Build Custom Image

```bash
docker build -f docker/dsv4-5090.Dockerfile -t sglang-dsv4-5090:sm120 .
```

## Run

```bash
docker run --rm --gpus all --ipc=host --shm-size 32g \
  -p 30000:30000 \
  -v /models:/models:ro \
  sglang-dsv4-5090:sm120
```

## Smoke Test

```bash
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "DeepSeek-V4-Flash-local",
    "messages": [{"role": "user", "content": "用一句话回答：1+1等于几？"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

## Failure Modes Fixed

- CUDA 12.3 `nvcc` cannot compile `sm_120a`; use CUDA 13.2.
- PyTorch JIT must be constrained with `TORCH_CUDA_ARCH_LIST=12.0a`; otherwise
  CUDA 13.2 rejects legacy targets such as `compute_52`.

# DeepSeek V4 Flash on 8x RTX 5090

This branch contains the RTX 5090 / SM120 path used to validate
DeepSeek-V4-Flash with SGLang. The image is built from source and does not
depend on the validation host's local virtualenv.

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
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 32768 \
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

Clone this branch on a machine with Docker/BuildKit and CUDA-capable NVIDIA
runtime support:

```bash
git clone -b latest_5090 https://github.com/ziang663/sglang.git
cd sglang
DOCKER_BUILDKIT=1 \
docker build -f docker/dsv4-5090.Dockerfile -t sglang-dsv4-5090:sm120 .
```

The Dockerfile:

- starts from `nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04`
- installs this SGLang branch from source
- fetches DeepGEMM PR318 at commit `7a7a41a1bac7dacabe74057e7600e59f98f85bce`
- applies `docker/deepgemm-pr318-metadata-dynamic-smem.patch`
- builds and installs the patched DeepGEMM package in the image

## Run

```bash
docker run --rm --gpus all --network host --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /models:/models:ro \
  sglang-dsv4-5090:sm120
```

Optional runtime overrides:

```bash
docker run --rm --gpus all --network host --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /models:/models:ro \
  -e MODEL_PATH=/models/preset/deepseek-ai/DeepSeek-V4-Flash/v1.0 \
  -e TP=8 \
  -e MEM_FRACTION_STATIC=0.82 \
  -e CHUNKED_PREFILL_SIZE=8192 \
  -e MAX_PREFILL_TOKENS=32768 \
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
- SGLang's DeepGEMM capability gate must allow `(12, 0)` so the DSv4 path can
  use the PR318 SM120 kernels.
- DeepGEMM PR318's paged MQA metadata kernel needed dynamic shared memory for
  larger SGLang chunks. The included patch validated `chunked_prefill_size=8192`
  on RTX 5090; much larger chunks around `12288+` can still exceed per-block
  shared-memory capacity.

## Bench Notes

Do not compare default `bench_serving` warmup numbers with cold TTFT. With
`num_prompts=1`, the default warmup uses the same random prompt and can populate
the prefix/radix cache.

Cold measurement:

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random \
  --random-input-len 50000 \
  --random-output-len 100 \
  --random-range-ratio 1.0 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --model /models/preset/deepseek-ai/DeepSeek-V4-Flash/v1.0 \
  --served-model-name DeepSeek-V4-Flash-local \
  --tokenizer /models/preset/deepseek-ai/DeepSeek-V4-Flash/v1.0 \
  --warmup-requests 0 \
  --flush-cache \
  --disable-tqdm
```

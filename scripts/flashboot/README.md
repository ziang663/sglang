# FlashBoot RDMA clone

FlashBoot adds two SGLang load formats:

- `flashload`: load a `sharded_state` checkpoint into one contiguous GPU arena per rank and publish it.
- `flashclone`: derive the same arena layout from safetensors headers and pull weight bytes from a running seed instance over CUDA IPC, NVLink fabric, or RDMA.

The FlashBoot Python package is included in `python/flashboot`. The native extension is optional and is not built during normal SGLang installs. Enable it explicitly:

```bash
cd python
SGLANG_BUILD_FLASHBOOT=1 FB_BUILD_RDMA=1 pip install -e .
```

`FB_BUILD_RDMA=auto` probes for `infiniband/verbs.h`; set it to `1` to fail loudly when the raw ibverbs backend cannot be built, or `0` to build only the CUDA IPC / fabric transports.

Typical flow:

```bash
MODEL_PATH=/path/to/hf SHARDS=/path/to/sharded_state \
  bash scripts/flashboot/01_save_shard_state.sh

bash scripts/flashboot/02_run_seed.sh

SEED_IPS=<seed-host> bash scripts/flashboot/03_run_clone.sh
```

Common settings live in `scripts/flashboot/env.sh`. Export `FB_NNODES`, `FB_NODE_RANK`, and `FB_MASTER` for multi-node runs. The `FB_` prefix is intentional: many cluster launchers export unrelated `NNODES` and `NODE_RANK` values.

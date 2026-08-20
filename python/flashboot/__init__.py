"""flashboot — fast sglang startup for ShardedStateLoader checkpoints.

Loads a sglang ShardedStateLoader checkpoint into ONE contiguous GPU arena (zero-copy,
1x weight memory) and lets a second instance (a *clone*) pull that whole image straight
from the seed's GPU memory instead of re-reading the checkpoint from storage — over
NVLink fabric (CUDA fabric handles + IMEX, GB300 NVL72 — crosses nodes inside the
IMEX domain), CUDA IPC (same node), or one-sided RDMA READ (raw ibverbs + GPUDirect,
H100 InfiniBand — crosses nodes with no IMEX domain); single-instance pull or
chunk-pipelined chain broadcast for K instances starting together, plus an
all-gather on every transport for the flashload_dp pattern (every rank loads 1/K,
the collective makes everyone whole).

Used through sglang's ``--load-format flashload``:

    seed  : --model-loader-extra-config '{"role":"sharded"}'
    clone : --model-loader-extra-config '{"role":"sharded_clone","seed_ip":"<ip0,ip1>","seed_port":<N>}'
    dp    : --model-loader-extra-config '{"role":"sharded_dp","dp_world":K,"dp_rank":d,"seed_ip":"<ip>","seed_port":<N>}'

sglang dispatches to the loader classes via :mod:`flashboot.sglang_plugin`.

Package map, top layer first — every dependency points DOWN this list, and a test
(tests/test_layering.py) fails if one ever points up again:

    sglang_plugin         the one entry point sglang calls (role -> loader class)
    sharded_loader        compatibility shell: sglang hard-codes this path

    loaders/              the roles — who reads the checkpoint, who takes an arena
      seed                  fills its arena from its own shard, then offers it
      clone                 reads no checkpoint — takes a seed's arena
      data_parallel         K replicas read 1/K each and gather the rest
      arena_loader_base     what all three do the same way (build, allocate, rebind,
                            release a failed attempt, serve)
      coordination          this instance's place, and where it meets the others
      process_topology      where this process sits; which shard is therefore its own
      split_fill            bind to a placeholder now, read the weights last

    collectives/          the ordering a transfer follows, whatever carries it
      all_gather            join, rendezvous, connect, gather — one implementation

    transport/            how the bytes of one arena reach another; NOTHING above this
                          package names a transport
      contracts             what a transport promises above it: arena requirements, a
                            publisher (seed side), an opened arena and a participant
                            (puller side)
      selection             resolves FLASHBOOT_TRANSPORT (fabric / rdma / ipc; auto
                            prefers fabric where the IMEX preflight passes, then rdma)
      nvlink_arena          ipc/fabric: export a handle, import it, copy-engine D2D —
                            including the fall-through to a seed's rdma offer
      rdma_arena            rdma: register the arena, exchange cards, one-sided READ
      rdma_data_plane       the one-sided-READ engine: registrations, cards, the
                            progress-counter pipeline, the all-gather modes
      nvlink_all_gather_engine  import each owner's handle, copy its segment, release
      nvlink_chain_engine       flag-gated chunk pipeline (_C.ChainReceiver)
      rdma_engine           which one-sided-READ engine runs here, and its endpoint
      nixl_endpoint         a NIXL agent wearing the native endpoint's method surface
      infiniband            GPU -> HCA positional map + queue-pair card pairing

    rendezvous/           processes finding each other and swapping the credential to
                          read each other's memory. Small JSON over TCP, no CUDA, and
                          every credential forwarded as an opaque dict
      seed_server           a seed publishes, clones take — and the chain rendezvous
      dp_gather_server      the DP gather: everybody arrives, everybody leaves
                            with the full membership
      collective_round      the round both symmetric servers run
      joining               the other side of all three: what a clone or member calls
      addressing            which port a shard serves on; which host a clone dials
      arena_advertisement   what a seed says about its arena, and the version guard
      message_framing       length-prefixed JSON, capped against a stray peer

    arena                 the GPU arena, and rebinding model params onto views of it
    arena_fill            disk/tmpfs -> pinned host -> GPU double-buffered fill pump
    model_binding         pointing a live model's parameters at arena bytes
    weight_preload        background storage -> tmpfs shard prestager (cold start)

    utils/                pure math and parsing: no arena, no socket, no torch at
                          module scope, which is why every layer above may reach into
                          it and why none of them has to reach through each other
      checkpoint_layout     safetensors part files -> arena placement math (pure CPU)
      safetensors_metadata  safetensors header parsing + dtype mapping
      arena_partitioning    which ranks connect, which bytes each one owns, and the
                            chunk a pipelined broadcast moves and signals in
      launch_arguments      argv parsing + rank topology — stdlib only

    preflight             CLI node check: python -m flashboot.preflight

This ``__init__`` deliberately imports NOTHING: importing ``flashboot`` must stay
stdlib-cheap so lightweight submodules (weight_preload, utils.launch_arguments) remain
usable before any heavy torch/sglang import.
"""

__version__ = "0.2.0"

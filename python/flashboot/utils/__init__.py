"""The bottom of the package: pure math and parsing that every layer may reach for.

What sits here is decided by a property, not by a topic. These modules compute and they
parse; none of them allocates a GPU arena, opens a socket, touches a file descriptor or
imports torch at module scope. That is why a loader, a transport, a collective and the
rendezvous may all import from here without any of them importing each other, and why
the dependency arrows in this package only ever point downwards into it.

    checkpoint_layout     safetensors part files -> aligned arena placement math
    safetensors_metadata  safetensors header parsing + dtype mapping
    arena_partitioning    which ranks connect, which arena bytes each one owns, and
                          the chunk a pipelined broadcast moves and signals in
    launch_arguments      argv parsing + rank topology, stdlib only

``launch_arguments`` and ``arena_partitioning`` import nothing from flashboot at all,
and ``checkpoint_layout`` reaches only for ``safetensors_metadata`` beside it, so this
package stays importable before any heavy dependency is available — which is what lets
``weight_preload`` run as a prestager long before torch is loaded.

Nothing is re-exported at this level on purpose: a caller that wants
``rank_part_files`` should say which module it came from, because the name alone does
not tell you whether you are about to do integer math or read a file header.
"""

"""Measurement harness for the lazy CUDA-graph experiment (stdlib only).

client      HTTP client for SGLang's native /generate API: readiness, streaming
            TTFT/TPOT, trace replay, throughput.
logparse    Parse SGLang startup logs (capture phases, memory) and the lazy
            capture controller's step lines.
arm         Run one experiment arm end to end: start the server, measure,
            write a JSON record. Used on Modal and by the local dry run.
mock_server A CPU-only stand-in for `python -m sglang.launch_server`, used by
            the dry run to exercise every code path without a GPU.
"""

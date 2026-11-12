"""Tools to measure and cut the Python start-up cost of LLM serving engines.

Modules:
    pycheck       audit .pyc validity (missing / stale / timestamp / hash) for a tree or an import set
    importprof    run an import in a fresh interpreter and record what it costs (time, modules,
                  compiles, bytes read, mapped shared objects)
    spawnbench    child-ready latency under spawn, forkserver(+preload) and fork
    prewarm       record a file manifest and read it back in parallel to warm the page cache
    residency     page-cache residency (mincore) and eviction (posix_fadvise), Linux only
    cudafacts     NVML probe and the guarded torch.cuda stand-ins the zygote uses while preloading
    zygote        parent-side API of the forkserver zygote
    zygote_preload  the code that runs inside the forkserver process (preload + invariants)
    sglang_launch wrapper that runs `sglang.launch_server` with the zygote, no SGLang source change
    profile_startup  the command-line profiler that ties the studies together

Everything here is standard library only, so it can be copied into an engine image as is.
"""

__all__ = []
__version__ = "0.1.0"

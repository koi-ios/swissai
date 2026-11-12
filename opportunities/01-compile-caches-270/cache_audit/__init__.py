"""cache-audit: find, map and gate the on-disk JIT/compile caches an engine writes.

Run an engine (or any warmup command) cold in a fresh HOME/TMPDIR sandbox, record
every file it writes with an unprivileged tracer, cluster the files into cache
roots, map each root to the variable that relocates it (knowledge base in
``kb/caches.toml``), verify the mapping by re-running with the variables set,
and emit a table, a model-launch env snippet and a CI-gate exit code.

Pure standard library (Python >= 3.10, ``tomllib`` or ``tomli``), so it can run
inside the engine images themselves.
"""

__version__ = "0.1.0"

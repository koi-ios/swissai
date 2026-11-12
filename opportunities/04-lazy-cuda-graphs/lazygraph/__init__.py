"""Trace-driven policy and simulator for lazy CUDA-graph capture.

Modules:
  buckets   - SGLang 0.5.20 default capture lists (decode bs, prefill tokens)
  costs     - per-bucket capture cost and eager penalty models
  trace     - per-model traffic profiles from the Swiss AI serving trace
  policy    - expected-value ranking, startup set, --cuda-graph-config JSON
  simulate  - overlay deferred capture steps on the trace's idle periods
  cli       - command line entry point (python -m lazygraph.cli --help)
"""

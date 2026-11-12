# Known-bad check

Put here one small change that is certainly wrong for this project, as a patch or a config: a weight shard zeroed, a cache directory pointed at the wrong place, a tolerance set to infinity, a benchmark that returns before the server is ready.

The oracle and the harness must reject it. Run it:

- when the harness is frozen;
- before each sweep;
- whenever the model or the agent tool changes.

Record each run in `VERIFY.md`. A harness that passes a known-bad change cannot be used to accept a good one.

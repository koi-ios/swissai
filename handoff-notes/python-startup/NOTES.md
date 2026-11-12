# Working notes: idea 4, the other half of the post-servekit cold start

Running log of verified findings, written as I go (context may be compacted).
Paths: SHARED=/private/tmp/claude-501/-Users-koiios--claude-spares-spare-qHWvhZ/85cbb828-b415-43aa-adc6-b088d62505ae/scratchpad/shared
Scratch: /private/tmp/claude-501/-Users-koiios--claude-spares-spare-qHWvhZ/85cbb828-b415-43aa-adc6-b088d62505ae/scratchpad/python-startup

Note: a parallel coordinator's session also works on this idea in
/Users/koiios/Documents/Projects/swissai/opportunities/05-python-startup (read-only for me).
Its results/log_timeline.md has a per-experiment table (spawn -> server_args, etc.).

## F1. The SyntaxWarning: what it is and what it proves (verified 11 Nov)

- Exactly one warning text, in every log that has one:
  `/usr/local/lib/python3.12/dist-packages/torchao/quantization/quant_api.py:1731: SyntaxWarning: invalid escape sequence '\.'`
  117 occurrences in total.
- 44 of 72 `.out` files contain it. All 44 are servekit-eval (v0.5.16 image) or servekit-jit-caching-eval.
  The 28 without are v0.5.10 runs (cpu-sweep, lustre-loading-exp) and one non-SGLang job.
  Absence in v0.5.10 proves nothing: that torchao version may not have the docstring.
- Per log: 2 (single-node TP4 servekit-eval), 5 (GLM-4.7, 4 nodes), 1 (jit-caching-eval, and one apertus mmap run).
- Image in these runs: `lmsysorg/sglang:v0.5.16` (EDF `experiments/servekit-eval/scripts/sglang.toml`), Python 3.12 dist-packages,
  NOT the platform's sglang_base (conda).
- A SyntaxWarning for an invalid escape is emitted by the compiler. A process that loads a valid .pyc never emits it.
  So: each job compiled quant_api.py from source at least once => the image has no valid pyc for it.
- Which processes compile: Youssef's servekit-eval sbatch puts `patches/sitecustomize.py` on PYTHONPATH.
  sitecustomize imports `sglang.srt.model_executor.model_runner_components.load_model_utils` in EVERY python process
  (servekit wrapper, servekit bench, SGLang main + children, servekit verify). servekit itself never imports torch/sglang
  (`servekit/prepare.py:36`: "`import sglang` costs tens of seconds"; `_shim/__init__.py` keeps sglang imports in function bodies).
- Log apertus8b-servekit-3183677.out: lines 9-16: two pynvml FutureWarnings + two SyntaxWarnings = wrapper and bench,
  started concurrently. Line 24 (SGLang main), lines 32-43 (6 children), line 457 (servekit verify, much later) show the
  pynvml FutureWarning (they import torch) but NO SyntaxWarning.
  => the pyc written by the first process was reused by every later process in the same container instance.
  => the container root FS is writable during the job, and recompilation is once per job per node, not once per process.
- GLM-4.7, 4 nodes: 5 = head (wrapper + bench race) 2 + 3 other nodes x 1. Consistent: once per node per job.
- One run shows 1 instead of 2: the race resolved (one process wrote the pyc before the other imported it).

## F2. Youssef's experiment artifact: the sitecustomize pre-import hides cold I/O from servekit's clock

- In servekit-eval, the wrapper (servekit launch/profile) and bench processes import torch+sglang via sitecustomize
  BEFORE servekit Popen's SGLang. So SGLang's own main process starts with a warm page cache.
- Other session's table (opportunities/05-python-startup/results/log_timeline.md), spawn -> `server_args=`:
  - Bristen jit-cache A/B (v0.5.16, no wrapper pre-import, n=2): 56.8 [49.1-64.4] s
  - Bristen servekit-eval (v0.5.16, wrapper pre-imports, n=12): 13.8 [12.3-26.3] s
  - Clariden servekit-eval (n=18): 11.3 [9.9-60.8] s
  - Bristen v0.5.10 lustre-loading (no wrapper, n=18): 29.9 [22.0-37.3] s
  TODO: verify with my own parser.
- Consequence: EVALUATION.md's 24.7-26.2 s "process start + TP worker spawn" (Clariden servekit runs) is a LOWER bound for a
  platform launch; the cold first import is partly outside servekit's clock in those runs.

## F3. Platform images (model-launch @02b74ce, images/*/Dockerfile)

- sglang_0.5.20 (= sglang_base): conda Python 3.12; everything via `pip install --no-cache-dir` (pip compiles pyc by default).
  Single-stage build from nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04.
- vLLM images install torch + vLLM with **uv** (`uv pip install`): vllm_cuda13 (l.35-36), vllm_0.30.0 (l.37-38),
  vllm_cuda13_v2 (l.41-43, 80-107), vllm_apertus_1.5_release (l.50-115). uv does NOT compile bytecode unless
  `--compile-bytecode` / `UV_COMPILE_BYTECODE=1`. No Dockerfile sets it. => the vLLM images likely ship torch, vllm,
  transformers, ... with no .pyc at all. (To verify: uv docs; vllm upstream Dockerfile for vllm_nightly.)
- vllm_apertus_1.5_release sets `PYTHONPATH=/workspace/vllm` => vLLM imported from a git checkout (no __pycache__).
- EDFs (assets/envs/*.toml): image = /capstor/.../container-images/ci/<name>-{arch}.sqsh; no `writable` key; no PYTHON* vars.
- vllm_kisti.toml documents image creation by `enroot import -o ....sqsh docker://...` (~4 min).

## F4. SGLang spawn

- servekit `_shim/__init__.py`: "SGLang forces the `spawn` start method: the schedulers ... are fresh interpreters".
  Each TP scheduler re-imports everything. Log shows 6 children importing torch for TP4 (4 schedulers + detokenizer + 1).

## Log timeline example (Clariden, Apertus-8B TP4, servekit serial, job 3183677)
- 12:24:13 SGLang Popen (run.json started_at 1787653453.06); process_startup 0.42 s; tp_worker_spawn 25.92 s
- 12:24:25 server_args= (main process imported + parsed: ~11 s after Popen)
- 12:24:28 main tokenizer; 12:24:39 TP0 tokenizer; 12:24:40 Init torch distributed begin (all 4)
- 12:24:41 "sglang is using nccl==2.28.9"; 12:24:50 "All Reduce config"; 12:24:56 dist init ends (elapsed 16.0-16.8 s)

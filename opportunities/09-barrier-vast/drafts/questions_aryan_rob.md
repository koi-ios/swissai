# Draft: model-launch-images#22 (post-load barrier at 8 nodes / TP32)

Nothing here has been sent. These are drafts for the user to adapt.

## To Aryan (one message)

Hi Aryan, I looked into the post-load barrier failures from #271 (GLM-5/5.1 and Kimi-K3 at 8 nodes, TP32). A few facts first, then three questions.

**Facts**
- **The timeout is fixed.** It is a hard-coded 480 s in every SGLang release, from 0.5.10.post1 through today's main (`load_model_utils.py:51` in 0.5.20). Neither `--dist-timeout` nor any environment variable changes it. The factory TOML's `SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S` is read by nothing.
- **gloo checks the ranks one by one.** I reproduced the barrier on CPU with gloo. Rank 0 waits up to 480 s for each lagging rank in turn.
- **What the error list means.** If the error lists all 31 ranks, rank 0 sat in the barrier for about 31 × 480 s ≈ 4 h. The other ranks were then dead, hung or hours late. A load that is merely 10–15 min slower lists only one rank, and the rest pass.

**Questions**
1. Could you share rank 0's lines `[Rank 0]: Rank k failed to pass monitoredBarrier in 480000 ms` with their timestamps, and one non-zero rank's log tail?
   - Did any non-zero rank print `Load weight end`?
   - Did any show an OOM kill (`sacct`, `dmesg`, or the cgroup OOM counter)?
2. How long does GLM-5.1 BF16 take to load on the old `sglang.toml` image at 8 nodes? Does it come up on the first try?
3. Would a no-rebuild workaround be acceptable for now? It is a small `sitecustomize.py` on `PYTHONPATH` that honours `SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S` for the SGLang images. I can also open the upstream SGLang change that makes this variable real.

## To Rob (one message)

Hi Rob, three quick image questions for #22.
1. Inside `ci/sglang_cuda13.sqsh` and the `sglang_base` image, what do these print?
   ```
   python -c "import sglang, torch, safetensors; print(sglang.__version__, torch.__version__, safetensors.__version__)"
   ```
   The Dockerfile pins 0.5.10.post1 since 17 Apr. Before that it was unpinned.
2. Was `SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S=3600` in the factory's Kimi-K3.1 TOML ever seen to take effect? No SGLang release reads it, so if the run passed, something else made the difference.
3. Could you run `lfs getstripe -c -S` on one GLM-5.1 shard under `hf_models/models/zai-org/`? Stripe count matters for the read-time estimate.

## How to read the barrier logs (from `repro/barrier_gloo.py`)

| what rank 0 logs | what it means |
|---|---|
| `Ranks 1 failed…` after ~480 s, then nothing | One rank, or a whole node, was more than 480 s late. The others arrived inside their own windows. |
| `Rank k failed…` once every ~480 s, k = 1, 2, 3, … | The other ranks are not arriving at all: dead, hung, or hours late. |
| the error comes within seconds of rank 0's `Load weight end` | Not explained by gloo semantics; check the earlier gloo/TCPStore errors. |

Ranks that did reach the barrier keep waiting after rank 0 dies, up to the gloo group timeout (120 min in SGLang), unless the launcher kills them.

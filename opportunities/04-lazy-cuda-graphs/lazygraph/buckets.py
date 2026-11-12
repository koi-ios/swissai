"""SGLang 0.5.20 default CUDA-graph bucket lists.

Mirrors python/sglang/srt/arg_groups/cuda_graph_hook.py at v0.5.20:
  generate_prefill_cuda_graph_batch_sizes (lines 557-574)
  generate_decode_cuda_graph_batch_sizes  (lines 577-609, non-speculative,
                                           padding enabled)
Defaults on H100/GH200-class GPUs (arg_groups/memory_hook.py): prefill max =
chunked_prefill_size = 8192 tokens; decode max_bs = 256 for TP < 4, 512 for
TP >= 4 (the lab's 0.5.16 logs show 36 decode sizes up to 256).
"""

from __future__ import annotations

import bisect
from typing import List, Optional, Sequence


def prefill_buckets(max_tokens: int = 8192) -> List[int]:
    sizes = (
        list(range(4, 33, 4))
        + list(range(48, 257, 16))
        + list(range(288, 513, 32))
        + list(range(576, 1024 + 1, 64))
        + list(range(1280, 4096 + 1, 256))
        + list(range(4608, max_tokens + 1, 512))
    )
    return [s for s in sizes if s <= max_tokens]


def decode_buckets(max_bs: int = 256) -> List[int]:
    sizes = (
        [1, 2, 4, 8, 12]
        + list(range(16, 257, 8))
        + list(range(272, 512, 16))
        + list(range(512, max_bs + 1, 32))
    )
    sizes = [s for s in sizes if s <= max_bs]
    if max_bs not in sizes:
        sizes.append(max_bs)
    return sizes


def bucket_of(value: int, buckets: Sequence[int]) -> Optional[int]:
    """Smallest bucket >= value (SGLang's _pad_to_bucket), None above the max."""
    i = bisect.bisect_left(buckets, value)
    return buckets[i] if i < len(buckets) else None

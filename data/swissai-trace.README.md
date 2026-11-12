---
pretty_name: DataSys LLM Serving Trace
license: other
language:
  - en
tags:
  - llm-serving
  - inference
  - request-trace
  - token-reuse
  - qwen3
size_categories:
  - 10M<n<100M
---

# LLM Serving Trace

This dataset contains anonymized LLM serving request metadata prepared for
systems research on request patterns, token accounting, latency, and reusable
input-token buckets. It is packaged as newline-delimited JSON.

## Dataset Files

| File | Rows | Size | Description |
| --- | ---: | ---: | --- |
| `trace.jsonl` | 16,329,237 | 6.98 GB | Request-level trace records with timestamps, status, model name, reported token counts, and generation parameters. |
| `qwen3-32b-buckets.jsonl` | 3,994,435 | 4.62 GB | Qwen/Qwen3-32B subset augmented with token-bucket information for token reuse analysis. |

## Loading

The files are distributed as raw newline-delimited JSON. The `model_parameters`
field is heterogeneous across requests (mixed types and occasional extra keys),
so the schema is not friendly to `datasets.load_dataset`'s automatic Arrow
inference. Download the files with `huggingface_hub` and read them directly.

```python
import json
from pathlib import Path

from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="eth-easl/swissai-serving-trace",
    repo_type="dataset",
    allow_patterns=["trace.jsonl"],  # or ["qwen3-32b-buckets.jsonl"], or both
)

with (Path(local_dir) / "trace.jsonl").open("r", encoding="utf-8") as handle:
    for line in handle:
        record = json.loads(line)
        # process record
```

To grab a single file without materialising the whole snapshot tree, use
`hf_hub_download`:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="eth-easl/swissai-serving-trace",
    filename="qwen3-32b-buckets.jsonl",
    repo_type="dataset",
)
```

If you prefer a dataframe and have enough memory, pandas can read the JSONL
directly (`trace.jsonl` is ~7 GB, so consider `chunksize=`):

```python
import pandas as pd

frames = pd.read_json(path, lines=True, chunksize=100_000)
for chunk in frames:
    ...  # process chunk
```

## Schema

### `trace`

Each row contains:

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Request identifier. |
| `status` | string | Request status, for example `DEFAULT` or `ERROR`. |
| `created_at` | string | Request creation timestamp in ISO-8601 format. |
| `finished_at` | string | Request completion timestamp in ISO-8601 format. |
| `model` | string | Model identifier used for the request. |
| `model_parameters` | object | Generation parameters supplied with the request. |
| `reported_token_input` | int64 | Reported input-token count, or `-1` when unavailable. |
| `reported_token_output` | int64 | Reported output-token count, or `-1` when unavailable. |

`model_parameters` is the raw user-supplied generation config and is
deliberately not normalised. Most rows contain the keys below, but values are
mixed JSON types because they reflect what the client sent:

| Field | Observed JSON types |
| --- | --- |
| `temperature` | number, `"null"` string, missing |
| `max_tokens` | number, string, `null` |
| `top_p` | number, `"null"` string, missing |
| `frequency_penalty` | number, string, list, `null`, missing |
| `presence_penalty` | number, string, bool, `null`, missing |
| `seed` | integer, missing |

A small number of rows also carry an extra `n` integer key. Coerce these to
your preferred schema at read time.

### `qwen3-32b-buckets`

This config contains the same request metadata fields as `trace`, plus:

| Field | Type | Description |
| --- | --- | --- |
| `token_count` | int64 | Number of Qwen/Qwen3-32B input tokens retained for the request. |
| `bucket_ids` | list[int64] | Deterministic token-bucket identifiers for the request input. Buckets were generated with 16-token buckets and right padding. |

## Intended Uses

This dataset is intended for research on LLM serving workloads, including:

- request arrival and completion patterns;
- latency and status analysis;
- input/output token accounting;
- cache locality and token reuse over bucketized input-token sequences.

The bucketized Qwen subset is useful when the raw token IDs should not be
distributed or when experiments only need stable identifiers for repeated token
chunks.

## Limitations

The trace records contain serving metadata rather than original prompts or model
outputs. Reported token counts may be unavailable for failed or incomplete
requests, represented as `-1`. The `bucket_ids` are model- and preprocessing-
specific and should not be interpreted as vocabulary token IDs.

## Citation

If you found this dataset useful to your work, please consider citing our work:

```
@inproceedings {318577,
  author = {Xiaozhe Yao and Youhe Jiang and Ilia Badanin and Qinghao Hu and Robert Matthew Smith and Binhang Yuan and Imanol Schlag and Eiko Yoneki and Ana Klimovic},
  title = {{OpenTela}: Unifying Decentralized Computing Resources for Heterogeneous {LLM} Serving (Operational Systems)},
  booktitle = {20th USENIX Symposium on Operating Systems Design and Implementation (OSDI 26)},
  year = {2026},
  isbn = {978-1-939133-55-7},
  address = {Seattle, WA},
  pages = {1821--1838},
  url = {https://www.usenix.org/conference/osdi26/presentation/yao},
  publisher = {USENIX Association},
  month = jul
}
```
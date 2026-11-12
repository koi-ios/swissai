# RUNBOOK: `bench_storage.py` on capstor, iopsstor and VAST

The goal is one comparable table per filesystem: how fast one node (and then 8 nodes at once) reads model weights through each read path SGLang and servekit use. Each run takes about 10–15 min on one node. Nothing here needs root, a GPU or a container, and nothing is written except the result files.

## What it measures

| method | stands for |
|---|---|
| `mmap` | SGLang/vLLM default safetensors loading: page faults on a shared mapping |
| `mmap-stride` | one TP rank's slice of a row-split weight. 128 B of every 4 KiB row is GLM-5's expert `down_proj` at TP32: every page is faulted for 1/32 of the bytes. |
| `pread` | `--weight-loader-disable-mmap`, servekit's stager, fastsafetensors without GDS |
| `odirect` | `dd iflag=direct` in Youssef's OST experiment; O_DIRECT bypasses the client cache |

**Cold reads.** Before every run the script drops the client page cache for the files it will read (`posix_fadvise(DONTNEED)`). It then records the fraction still resident (`mincore`) as `resident_before`. Each run reads a new region of the file set, so storage-side caches are not re-hit; keep the total read below the file-set size (the script warns otherwise).

## 0. Setup (any node with Python ≥ 3.8; numpy only for `mmap-stride`)

```bash
cp bench_storage.py $SCRATCH/ && cd $SCRATCH
python3 bench_storage.py --dry-run /capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7 --glob '*.safetensors'
```

## 1. One node on capstor (HDD Lustre), against real shards

```bash
srun -A <account> -p normal -N1 --exclusive -t 00:30:00 \
  python3 bench_storage.py /capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7 \
    --glob '*.safetensors' --methods mmap,mmap-stride,pread,odirect \
    --readers 1,4,16,64 --block-sizes 1M,16M --bytes-per-run 4G --repeats 3
```

- 24 points × 3 repeats = 72 runs, 288 GB read. The 717 GB file set is larger, so no region is read twice.
- Expected time is about 12 min. The mmap points dominate: about 0.2 GB/s per node if Youssef's numbers hold.
- Record the Lustre client settings next to the results:
  ```bash
  lctl get_param llite.*.max_read_ahead_mb llite.*.max_read_ahead_per_file_mb llite.*.max_read_ahead_whole_mb
  lfs getstripe -c -S <one shard>
  ```

## 2. The same on iopsstor (flash Lustre) and on VAST

- Run the same command against a directory on the target filesystem. If no model is there yet, either:
  - copy one large model into it (preferred: real data has realistic compressibility); or
  - create random test data, larger than the planned reads:
    ```bash
    python3 bench_storage.py --create /<vast-mount>/$USER/bench --create-size 300G --create-files 60
    ```
- **VAST only.** Record what decides NFS read behaviour:
  ```bash
  findmnt -T <path>                                   # proto=rdma|tcp, nconnect=, rsize=, vers=
  nfsstat -m
  cat /sys/class/bdi/$(mountpoint -d $(findmnt -no TARGET -T <path>))/read_ahead_kb
  ```
  The script also stores the mount options and `read_ahead_kb` in the first JSON line.
- **VAST only.** Two cheap readahead variants for mmap. Neither needs root. NFS read-around for page faults is sized by `read_ahead_kb`.
  ```bash
  python3 bench_storage.py <path> --methods mmap --readers 1,4,16 --madvise sequential --bytes-per-run 4G
  python3 bench_storage.py <path> --methods mmap --readers 1,4,16 --madvise willneed  --bytes-per-run 4G
  ```

## 3. Eight nodes at once (the TP32 situation)

```bash
srun -A <account> -p normal -N8 --ntasks-per-node=1 --exclusive -t 00:30:00 \
  python3 bench_storage.py <same path> --methods mmap,pread --readers 4,16 --block-sizes 16M \
    --bytes-per-run 8G --repeats 3 --out results/storage-8n-$(hostname)-$SLURM_PROCID.jsonl
```

- Every node reads the same regions at the same time, as 32 TP ranks reading one checkpoint do.
- Compare GB/s across the 8 files. The **node-to-node spread** is what the 480 s post-load barrier is sensitive to. Youssef's GLM-4.7 runs showed 83 s and 132 s across 4 nodes on loads of about 15 min.

## 4. Send back

```bash
tar czf storage-results-$(date +%F).tgz results/storage-*.jsonl results/storage-*.summary.md
```

- Each `*.summary.md` has the median and min–max over 3 runs, the median reader spread, and the environment (mount, fstype, options, readahead).
- To compare two filesystems, put the summary tables side by side; no extra tooling is needed.

## Reading the results

| question | answer from |
|---|---|
| Does servekit's staging (parallel pread → `/dev/shm`) still pay on VAST? | `pread`/`odirect` at 16–64 readers vs `mmap` at 1–4 readers. If they are within about 2×, SGLang's default loader is fine there. |
| How much parallelism does the storage need? | GB/s vs readers for `odirect` 16M. capstor kept climbing to 64 readers (6.7 GB/s on one OST). |
| Does a TP slice pay for whole pages? | `mmap-stride` vs `mmap` in GB/s swept: if they are equal, a TP rank pays the full page cost for its 1/32. |
| Was it really cold? | `resident_before` should be about 0%. If it is not, the run measured the page cache. |
| Stragglers | `reader spread`, and the per-node GB/s in step 3 |

## Limits

- `posix_fadvise(DONTNEED)` drops only this client's clean pages. Storage-side caches (Lustre OSS, VAST) are avoided by reading new regions, not by dropping them.
- On macOS there is no `posix_fadvise` and no O_DIRECT. The script uses fresh regions and `F_NOCACHE`, and `resident_before` shows whether that worked. The local test showed 0–6%.
- One process per reader. A GPU loader adds host-to-device copies that this benchmark does not model.

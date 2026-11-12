# Sooryas-MacBook-Pro.local  /private/tmp/claude-501/-Users-koiios-Documents-Projects-swissai/3b089b18-f5c1-4768-afb6-7c5a4295ab4c/scratchpad/benchdata/bench-000.bin

| method | readers | block | median GB/s | min-max GB/s | reader spread (median) | resident before (max) | n |
|---|---|---|---|---|---|---|---|
| mmap | 1 | 1M | 0.91 | 0.85-1.02 | 0.00 s | 0% | 3 |
| mmap | 4 | 1M | 0.99 | 0.93-1.64 | 0.00 s | 0% | 3 |
| mmap | 16 | 1M | 2.59 | 2.31-2.73 | 0.00 s | 0% | 3 |
| mmap-stride | 1 | 16M | 0.73 | 0.72-0.75 | 0.00 s | 0% | 3 |
| mmap-stride | 4 | 16M | 0.79 | 0.70-1.22 | 0.00 s | 0% | 3 |
| mmap-stride | 16 | 16M | 1.07 | 0.67-1.37 | 0.17 s | 0% | 3 |
| pread | 1 | 1M | 4.47 | 2.40-4.74 | 0.00 s | 0% | 3 |
| pread | 4 | 1M | 6.25 | 4.33-6.30 | 0.05 s | 5% | 3 |
| pread | 16 | 1M | 5.44 | 4.35-6.13 | 0.05 s | 6% | 3 |
| pread | 1 | 16M | 6.34 | 4.59-6.51 | 0.00 s | 4% | 3 |
| pread | 4 | 16M | 4.50 | 4.09-5.92 | 0.03 s | 4% | 3 |
| pread | 16 | 16M | 4.04 | 3.72-5.35 | 0.08 s | 0% | 3 |
| odirect | 1 | 1M | 2.92 | 2.58-3.28 | 0.00 s | 0% | 3 |
| odirect | 4 | 1M | 4.25 | 4.18-5.92 | 0.00 s | 0% | 3 |
| odirect | 16 | 1M | 4.30 | 3.92-5.97 | 0.01 s | 0% | 3 |
| odirect | 1 | 16M | 4.34 | 4.20-6.01 | 0.00 s | 0% | 3 |
| odirect | 4 | 16M | 4.51 | 4.49-6.21 | 0.01 s | 0% | 3 |
| odirect | 16 | 16M | 4.66 | 4.08-5.98 | 0.08 s | 0% | 3 |

```
{
 "host": "Sooryas-MacBook-Pro.local",
 "platform": "macOS-15.6-arm64-arm-64bit",
 "python": "3.12.13",
 "cpus": 14,
 "page_size": 16384,
 "path": "/private/tmp/claude-501/-Users-koiios-Documents-Projects-swissai/3b089b18-f5c1-4768-afb6-7c5a4295ab4c/scratchpad/benchdata/bench-000.bin",
 "mount": "/dev/disk3s5 on /System/Volumes/Data (apfs, local, journaled, nobrowse, protect, root data)",
 "mem_total_kb": 50331648,
 "files": 10,
 "file_bytes": 21474836480,
 "cold": true,
 "rotate": true,
 "drop_cache_supported": false
}
```

PREDICTIONS (see docstring for the basis)

| case | GB per node, page-exact | with read-around | vs 480 GB node DRAM | cold mmap load per node | P(fail) node spread CV 4% / 6% / 10% |
|---|---|---|---|---|---|
| GLM-5.1 BF16, TP32, 8 nodes (fails) | 622 | 637 | 1.30x | 52 min (48-58) | 0% / 4% / 24% |
| GLM-5.1 BF16, TP32 + EP32, 8 nodes (proposal) | 194 | 208 | 0.40x | 16 min (15-18) | 0% / 0% / 0% |
| GLM-5.1-FP8, TP16, 4 nodes (passes) | 374 | 381 | 0.78x | 31 min (29-35) | 0% / 0% / 4% |

check: GLM-4.7 whole model at TP4 on one node = 706 GB (checkpoint 716.7 GB incl. one MTP layer); a PP4 stage is ~176 GB

"""Per-node read volume and cold mmap load time for TP-sharded MoE checkpoints,
and the chance that node-to-node spread alone fails SGLang's post-load barrier.

All outputs are PREDICTIONS built on:
  * the model config (Hugging Face config.json) and SGLang's TP slicing:
      replicated: q_a_proj, kv_a_proj_with_mqa, DSA indexer, norms, router
      dim-0 split (column parallel): q_b_proj, kv_b_proj, gate/up (experts, shared, dense),
                                     embed_tokens, lm_head
      dim-1 split (row parallel):    o_proj, down (experts, shared, dense)
    with ranks placed 4 per node in rank order (SGLang's default);
  * page-granular mmap faults (4 KiB pages), with and without read-around
    (read-around >= one row makes every page of a row-split tensor read);
  * the measured per-node mmap rate of GLM-4.7 TP4*PP4 on 4 nodes (Youssef
    Boughizane's servekit-eval logs): 177 GB per node in 820-983 s;
  * the measured node-to-node coefficient of variation in those runs (4.0 % on
    Bristen, 5.9 % on Clariden; one 4-node job each).

Run: python estimate/tp_load_model.py   (reads vendor/hf/*.config.json)
"""
from __future__ import annotations

import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
HF = os.path.join(HERE, "..", "vendor", "hf")
PAGE = 4096
RANKS_PER_NODE = 4
NODE_RATE_GBPS = (0.180, 0.200, 0.216)  # min / median / max, measured (see docstring)
NODE_DRAM_GB = 480  # GH200 node: 4 x 120 GB LPDDR5X (HBM excluded)


def pages_row_split(rows: int, row_bytes: int, tp: int, node: int, readaround: bool) -> int:
    """Bytes of whole pages a node faults for a dim-1 split [rows, cols] tensor."""
    if readaround:
        return rows * row_bytes
    chunk = row_bytes * RANKS_PER_NODE // tp  # node-local ranks are adjacent
    start_in_row = node * chunk
    period = math.lcm(row_bytes, PAGE) // row_bytes  # rows until the page pattern repeats
    touched = set()
    for r in range(min(period, rows)):
        a = r * row_bytes + start_in_row
        b = a + chunk - 1
        touched.update(range(a // PAGE, b // PAGE + 1))
    per_period = len(touched) * PAGE
    return min(rows * row_bytes, per_period * rows // min(period, rows))


def node_bytes(cfg: dict, tp: int, elem: float, node: int = 0, readaround: bool = False,
               load_mtp: bool = True, ep: int = 1) -> dict:
    """Bytes node `node` faults from the checkpoint, by tensor family."""
    c = cfg.get("text_config", cfg)
    H = c["hidden_size"]
    L = c["num_hidden_layers"]
    dense = c.get("first_k_dense_replace", 0)
    E = c.get("n_routed_experts", 0) + c.get("n_shared_experts", 0)
    I = c["moe_intermediate_size"]
    Id = c["intermediate_size"]
    V = c["vocab_size"]
    nh = c["num_attention_heads"]
    out = {"replicated": 0.0, "col": 0.0, "row": 0.0}
    frac_col = RANKS_PER_NODE / tp

    def rep(n):
        out["replicated"] += n * elem

    def col(rows, cols):
        out["col"] += rows * cols * elem * frac_col

    def row(rows, cols):
        out["row"] += pages_row_split(rows, int(cols * elem), tp, node, readaround)

    layers = L + (c.get("num_nextn_predict_layers", 0) if load_mtp else 0)
    for layer in range(layers):
        if "q_lora_rank" in c and c["q_lora_rank"]:  # MLA (GLM-5, DeepSeek, Kimi)
            ql, kl, rope = c["q_lora_rank"], c["kv_lora_rank"], c["qk_rope_head_dim"]
            nope, v = c["qk_nope_head_dim"], c["v_head_dim"]
            rep(ql * H + (kl + rope) * H)  # fused_qkv_a_proj_with_mqa is replicated
            col(nh * (nope + rope), ql)  # q_b_proj
            col(nh * (nope + v), kl)  # kv_b_proj
            row(H, nh * v)  # o_proj
            if c.get("index_n_heads"):  # DSA indexer, replicated
                rep(c["index_n_heads"] * c["index_head_dim"] * ql + c["index_head_dim"] * H
                    + c["index_n_heads"] * H)
        else:  # GQA (GLM-4.7)
            hd = c.get("head_dim") or H // nh
            col(nh * hd + 2 * c["num_key_value_heads"] * hd, H)
            row(H, nh * hd)
        if layer < dense:
            col(2 * Id, H)
            row(H, Id)
        else:
            routed = c.get("n_routed_experts", 0)
            if ep > 1:  # --ep-size: whole routed experts per rank, no slicing
                out["ep_experts"] = out.get("ep_experts", 0.0) + 3 * I * H * elem * routed * RANKS_PER_NODE / ep
                col(2 * I * (E - routed), H)  # shared expert stays TP-split
                row(H * (E - routed), I)
            else:
                col(2 * I * E, H)  # gate + up of every expert (shared included)
                row(H * E, I)  # down of every expert, stacked rows
            rep(routed * H)  # router
        if layer >= L:  # MTP / NextN extra projection (replicated)
            rep(2 * H * H)
    col(V, H)  # embed_tokens
    col(V, H)  # lm_head
    out["total"] = out["replicated"] + out["col"] + out["row"] + out.get("ep_experts", 0.0)
    return out


def barrier_fail_probability(nodes: int, mean_s: float, cv: float, timeout_s: float = 480,
                             rank_cv: float = 0.002, trials: int = 20000, seed: int = 0) -> float:
    """Monte Carlo of gloo monitored_barrier(wait_all_ranks=True) as reproduced in
    repro/barrier_gloo.py: rank 0 checks ranks 1..N-1 in order; rank k fails if
    it arrives after (when rank 0 started waiting on it) + timeout."""
    rnd = random.Random(seed)
    fails = 0
    n = nodes * RANKS_PER_NODE
    for _ in range(trials):
        node_t = [rnd.gauss(mean_s, cv * mean_s) for _ in range(nodes)]
        t = [node_t[r // RANKS_PER_NODE] * (1 + rnd.gauss(0, rank_cv)) for r in range(n)]
        clock = t[0]
        failed = False
        for k in range(1, n):
            deadline = clock + timeout_s
            if t[k] > deadline:
                failed = True
                clock = deadline
            else:
                clock = max(clock, t[k])
        fails += failed
    return fails / trials


def main():
    def load(name):
        with open(os.path.join(HF, name)) as f:
            return json.load(f)

    cases = [
        ("GLM-5.1 BF16, TP32, 8 nodes (fails)", load("zai-org_GLM-5.1.config.json"), 32, 2.0, 8, 1),
        ("GLM-5.1 BF16, TP32 + EP32, 8 nodes (proposal)", load("zai-org_GLM-5.1.config.json"), 32, 2.0, 8, 32),
        ("GLM-5.1-FP8, TP16, 4 nodes (passes)", load("zai-org_GLM-5.1-FP8.config.json"), 16, 1.0, 4, 1),
    ]
    print("PREDICTIONS (see docstring for the basis)\n")
    print("| case | GB per node, page-exact | with read-around | vs 480 GB node DRAM | cold mmap load per node | P(fail) node spread CV 4% / 6% / 10% |")
    print("|---|---|---|---|---|---|")
    for name, cfg, tp, elem, nodes, ep in cases:
        exact = node_bytes(cfg, tp, elem, ep=ep)["total"] / 1e9
        ra = node_bytes(cfg, tp, elem, readaround=True, ep=ep)["total"] / 1e9
        lo, med, hi = (exact / r for r in reversed(NODE_RATE_GBPS))
        p = [barrier_fail_probability(nodes, exact / NODE_RATE_GBPS[1], cv) for cv in (0.04, 0.06, 0.10)]
        print(f"| {name} | {exact:.0f} | {ra:.0f} | {exact / NODE_DRAM_GB:.2f}x | "
              f"{med / 60:.0f} min ({lo / 60:.0f}-{hi / 60:.0f}) | "
              + " / ".join(f"{x * 100:.0f}%" for x in p) + " |")
    # sanity check of the byte model on the calibration case: GLM-4.7 at TP4 on one node
    g47 = load("zai-org_GLM-4.7.config.json")
    full = node_bytes(g47, 4, 2.0, load_mtp=False)["total"] / 1e9
    print(f"\ncheck: GLM-4.7 whole model at TP4 on one node = {full:.0f} GB "
          f"(checkpoint 716.7 GB incl. one MTP layer); a PP4 stage is ~{full / 4:.0f} GB")


if __name__ == "__main__":
    main()

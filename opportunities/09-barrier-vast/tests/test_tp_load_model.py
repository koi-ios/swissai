"""Sanity checks for estimate/tp_load_model.py (pure arithmetic, <1 s)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "estimate"))
import tp_load_model as m  # noqa: E402

HF = os.path.join(os.path.dirname(__file__), "..", "vendor", "hf")


def cfg(name):
    with open(os.path.join(HF, name)) as f:
        return json.load(f)


def test_byte_model_matches_checkpoint_when_one_node_holds_all_slices():
    # TP4 on one node reads every byte once: model ~= checkpoint (716.7 GB incl. ~1 MTP layer)
    got = m.node_bytes(cfg("zai-org_GLM-4.7.config.json"), 4, 2.0, load_mtp=True)["total"] / 1e9
    assert 700 < got < 730


def test_row_split_reads_whole_pages():
    # GLM-5 expert down_proj rows are 4 KiB: a TP32 node faults every page for 1/8 of the bytes
    assert m.pages_row_split(6144, 4096, 32, 0, readaround=False) == 6144 * 4096
    # a 32 KiB row (o_proj at TP32): 4 KiB of 32 KiB per row
    assert m.pages_row_split(6144, 32768, 32, 0, readaround=False) == 6144 * 4096


def test_ep_reduces_node_bytes():
    c = cfg("zai-org_GLM-5.1.config.json")
    tp = m.node_bytes(c, 32, 2.0)["total"]
    ep = m.node_bytes(c, 32, 2.0, ep=32)["total"]
    assert 2.5 < tp / ep < 4


def test_barrier_failure_grows_with_spread():
    p = [m.barrier_fail_probability(8, 3000, cv, trials=4000) for cv in (0.02, 0.06, 0.15)]
    assert p[0] <= p[1] <= p[2] and p[2] > 0.3

"""The sharded_state round-trip bugs, reproduced on CPU with SGLang v0.5.20's own code.

Each case builds a tiny model with the structure SGLang gives the real one,
loads it the default way, then round-trips it through ShardedStateLoader's
save_model/load_model as v0.5.20 ships them (extracted from the vendored
source, not re-implemented), and finally through servekit's raw preshard.

  case                     | SGLang code that runs                         | sharded_state      | raw
  MLA kv_b_proj alias      | deepseek_v2.py:1920, :2070-2071               | KeyError on load   | exact
  MXFP4 triton experts     | mxfp4.py:1016-1043 (triton branch)            | silent, wrong      | exact
  GemmaRMSNorm.gemma_weight| layernorm.py:1068-1089 (__init__, loader)     | silent, wrong      | exact

Run: ../.venv/bin/python -m pytest repro -q   (from the opportunity folder)
Writes results/cpu_repro.json.
"""
import dataclasses
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SERVEKIT = ROOT / "vendor" / "servekit"
sys.path.insert(0, str(SERVEKIT / "src"))
sys.path.insert(0, str(HERE))

import sglang_src as S  # noqa: E402
from servekit import deepstate  # noqa: E402
from servekit.rawshard.store import RawIndex, RecordSession  # noqa: E402

RESULTS = {}


def teardown_module(module):
    out = ROOT / "results" / "cpu_repro.json"
    out.parent.mkdir(exist_ok=True)
    RESULTS["sglang_source_extracted"] = sorted(set(S.EXTRACTED))
    RESULTS["torch"] = torch.__version__
    out.write_text(json.dumps(RESULTS, indent=2) + "\n")


def _out(d):
    """ShardedStateLoader.save_model writes into an existing directory (servekit's script mkdirs it)."""
    os.makedirs(d, exist_ok=True)
    return str(d)


def _save_ck(ck, d):
    from safetensors.torch import save_file

    os.makedirs(d, exist_ok=True)
    save_file({k: v.contiguous() for k, v in ck.items()}, os.path.join(d, "model.safetensors"))
    return [os.path.join(d, "model.safetensors")]


def _iterate(paths):
    from safetensors import safe_open

    for p in paths:
        with safe_open(p, framework="pt") as f:
            for k in f.keys():  # noqa: SIM118
                yield k, f.get_tensor(k)


def _file_keys(d):
    from safetensors import safe_open

    keys, nbytes = [], 0
    for name in sorted(os.listdir(d)):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(d, name), framework="pt") as f:
                for k in f.keys():  # noqa: SIM118
                    keys.append(k)
                    t = f.get_tensor(k)
                    nbytes += t.numel() * t.element_size()
    return keys, nbytes


def _raw_round_trip(make_model, load_default, paths, tmp, finish=None):
    """Record one default load, replay it into a fresh model the default way."""
    session = RecordSession(str(tmp / "raw"), 0)
    with deepstate.deterministic_empty():
        recorded = load_default(make_model(), session.stream("target/0", _iterate(paths)))
    stats = session.finish(recorded)
    index = RawIndex(str(tmp / "raw"), 0)
    with deepstate.deterministic_empty():
        replayed = load_default(make_model(), index.replay(0, verify=True))
    assert index.verdict(replayed) == []
    if finish:
        finish(recorded)
        finish(replayed)
    return recorded, replayed, stats


# ===================================================================== MLA


def _mla_statements(deepseek_path: Path):
    init = S.statements(deepseek_path, "DeepseekV2AttentionMLA.__init__",
                        lambda s: "attn_mha" in s and "kv_b_proj" in s and "\n" not in s)
    forward = S.statements(deepseek_path, "DeepseekV2AttentionMLA.forward_prepare",
                           lambda s: s.startswith("if self.attn_mha.kv_b_proj is None"))
    return init, forward


class RadixAttention(nn.Module):  # stand-in: holds no parameters
    pass


def make_mla_classes(init_stmt: str, forward_stmt: str):
    class SelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.kv_b_proj = nn.Linear(8, 32, bias=False)  # ColumnParallelLinear at TP1
            self.o_proj = nn.Linear(16, 16, bias=False)
            self.attn_mha = RadixAttention()
            exec(init_stmt, {"object": object}, {"self": self})  # deepseek_v2.py:1920 (or the backport)
            self.w_kc = None
            self.w_vc = None

        def forward_prepare(self):
            if forward_stmt:
                exec(forward_stmt, {"hasattr": hasattr}, {"self": self})  # deepseek_v2.py:2070-2071

        def post_load_weights(self):  # the absorption deepseek_v2 does after loading
            w = self.kv_b_proj.weight.unflatten(0, (-1, 8))
            self.w_kc, self.w_vc = (t.contiguous() for t in w.split([4, 4], dim=1))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = SelfAttn()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Layer() for _ in range(2)])

        def load_weights(self, weights):
            params = dict(self.named_parameters())
            for name, w in weights:
                params[name].data.copy_(w)
            self.post_load_weights()
            return self

        def post_load_weights(self):
            for layer in self.model.layers:
                layer.self_attn.post_load_weights()

        def forward_prepare(self):
            for layer in self.model.layers:
                layer.self_attn.forward_prepare()

    return Model


def _mla_checkpoint():
    g = torch.Generator().manual_seed(1)
    ck = {}
    for i in range(2):
        ck[f"model.layers.{i}.self_attn.kv_b_proj.weight"] = torch.randn(32, 8, generator=g)
        ck[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(16, 16, generator=g)
    return ck


def test_mla_alias_breaks_the_sharded_state_round_trip(tmp_path):
    init, forward = _mla_statements(S.DEEPSEEK)
    assert init == "self.attn_mha.kv_b_proj = None"
    Model = make_mla_classes(init, forward)
    paths = _save_ck(_mla_checkpoint(), tmp_path / "ck")
    Loader = S.sharded_state_loader(Model)

    served = Model().load_weights(_iterate(paths))
    served.forward_prepare()  # one forward: the alias is now a registered submodule
    Loader.save_model(served, _out(tmp_path / "sharded"))
    keys, _ = _file_keys(tmp_path / "sharded")
    assert "model.layers.0.self_attn.attn_mha.kv_b_proj.weight" in keys
    assert "model.layers.0.self_attn.kv_b_proj.weight" not in keys  # deduped away by _filter_subtensors

    with pytest.raises(KeyError, match="attn_mha.kv_b_proj.weight"):
        S.load_with(Loader, str(tmp_path / "sharded"))

    # control from sglang#35702: saved before any forward, it loads
    unserved = Model().load_weights(_iterate(paths))
    Loader.save_model(unserved, _out(tmp_path / "sharded-noforward"))
    S.load_with(Loader, str(tmp_path / "sharded-noforward"))
    RESULTS["mla_alias"] = {
        "sharded_state_after_a_forward": "KeyError: model.layers.0.self_attn.attn_mha.kv_b_proj.weight",
        "sharded_state_before_any_forward": "loads",
    }


def _patched_deepseek(tmp_path: Path, patch_source: str) -> Path:
    pkg = tmp_path / "site" / "sglang" / "srt" / "models"
    pkg.mkdir(parents=True)
    shutil.copy(S.DEEPSEEK, pkg / "deepseek_v2.py")
    spec = importlib.util.spec_from_loader("patch_under_test", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(patch_source, "sglang_35715.py", "exec"), module.__dict__)  # servekit's own patch file
    saved = list(sys.path)
    sys.path[:] = [str(tmp_path / "site")] + [p for p in sys.path if "sglang" not in p]
    try:
        rc = module.main()
    finally:
        sys.path[:] = saved
    return pkg / "deepseek_v2.py" if rc == 0 else None


def test_servekit_main_backport_no_longer_applies_to_0_5_20(tmp_path, capsys):
    old = subprocess.run(["git", "-C", str(SERVEKIT), "show", "cd8f0b8:src/servekit/_patches/sglang_35715.py"],
                         capture_output=True, text=True, check=True).stdout
    assert _patched_deepseek(tmp_path, old) is None
    err = capsys.readouterr().err
    assert "the forward_prepare fill-in is not in" in err
    RESULTS["servekit_cd8f0b8_backport_on_0_5_20"] = "refused: " + err.strip().splitlines()[0][:160]


def test_fixed_backport_makes_mla_round_trip(tmp_path):
    fixed = (SERVEKIT / "src" / "servekit" / "_patches" / "sglang_35715.py").read_text()
    patched = _patched_deepseek(tmp_path, fixed)
    assert patched is not None
    text = patched.read_text()
    init = 'object.__setattr__(self.attn_mha, "kv_b_proj", self.kv_b_proj)'
    assert text.count(init) == 1 and "self.attn_mha.kv_b_proj = self.kv_b_proj" not in text
    Model = make_mla_classes(init, "")
    paths = _save_ck(_mla_checkpoint(), tmp_path / "ck")
    Loader = S.sharded_state_loader(Model)
    with deepstate.deterministic_empty():
        served = Model().load_weights(_iterate(paths))
        served.forward_prepare()
        Loader.save_model(served, _out(tmp_path / "sharded"))
        reloaded = S.load_with(Loader, str(tmp_path / "sharded"))
    report = deepstate.compare(deepstate.fingerprint(served), deepstate.fingerprint(reloaded))
    assert report.ok, report.summary()
    RESULTS["mla_alias"]["sharded_state_with_fixed_backport"] = "exact (deep-state equal)"


def test_raw_preshard_round_trips_mla_without_any_patch(tmp_path):
    init, forward = _mla_statements(S.DEEPSEEK)
    Model = make_mla_classes(init, forward)
    paths = _save_ck(_mla_checkpoint(), tmp_path / "ck")
    recorded, replayed, _ = _raw_round_trip(Model, lambda m, w: m.load_weights(w), paths, tmp_path,
                                            finish=lambda m: m.forward_prepare())
    report = deepstate.compare(deepstate.fingerprint(recorded), deepstate.fingerprint(replayed))
    assert report.ok, report.summary()
    RESULTS["mla_alias"]["raw_preshard_unpatched_sglang"] = "exact (deep-state equal)"


# =================================================================== MXFP4


@dataclasses.dataclass
class _Storage:  # triton_kernels.tensor.Storage
    data: torch.Tensor


@dataclasses.dataclass
class _TritonTensor:  # triton_kernels.tensor.Tensor
    storage: _Storage


def _swizzle_stand_in(quant_tensor, scale, num_warps):
    """Stand-in for mxfp4.py:230 _swizzle_mxfp4 (needs triton_kernels): a layout change."""
    return (_TritonTensor(_Storage(quant_tensor.data.transpose(-2, -1).contiguous())), None,
            _TritonTensor(_Storage(scale.data.transpose(-2, -1).contiguous())))


@pytest.fixture
def triton_kernels_stub(monkeypatch):
    """`from triton_kernels.matmul import FlexCtx, PrecisionConfig` needs a module to import from."""
    matmul = types.ModuleType("triton_kernels.matmul")

    @dataclasses.dataclass
    class FlexCtx:
        rhs_data: object = None

    @dataclasses.dataclass
    class PrecisionConfig:
        b_mx_scale: object = None
        flex_ctx: object = None

    matmul.FlexCtx, matmul.PrecisionConfig = FlexCtx, PrecisionConfig
    pkg = types.ModuleType("triton_kernels")
    pkg.matmul = matmul
    monkeypatch.setitem(sys.modules, "triton_kernels", pkg)
    monkeypatch.setitem(sys.modules, "triton_kernels.matmul", matmul)


E, I2, H = 4, 32, 64  # experts, 2*intermediate, hidden


def make_mxfp4_classes():
    body = S.if_body(S.MXFP4, "Mxfp4MoEMethod.process_weights_after_loading", "self.use_triton_kernels")
    source = "def process_weights_after_loading(self, layer):\n" + textwrap.indent(body, "    ") + "\n"
    ns = {"torch": torch, "Parameter": nn.Parameter, "_swizzle_mxfp4": _swizzle_stand_in}
    pwal = S.compile_into(source, ns, "process_weights_after_loading")

    class Mxfp4MoEMethod:  # what the triton path needs of the real class
        use_triton_kernels = True
        process_weights_after_loading = pwal

    def p(*shape, dtype):
        return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

    class FusedMoE(nn.Module):
        """The parameters Mxfp4MoEMethod.create_weights registers (mxfp4.py:426), small."""

        def __init__(self):
            super().__init__()
            self.w13_weight = p(E, I2, H // 2, dtype=torch.uint8)
            self.w13_weight_scale = p(E, I2, H // 32, dtype=torch.uint8)
            self.w13_weight_bias = p(E, I2, dtype=torch.bfloat16)
            self.w2_weight = p(E, H, I2 // 4, dtype=torch.uint8)
            self.w2_weight_scale = p(E, H, max(I2 // 64, 1), dtype=torch.uint8)
            self.w2_weight_bias = p(E, H, dtype=torch.bfloat16)
            self.quant_method = Mxfp4MoEMethod()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = FusedMoE()
            self.norm = nn.Parameter(torch.empty(H, dtype=torch.bfloat16), requires_grad=False)

        def load_weights(self, weights):
            params = dict(self.named_parameters())
            for name, w in weights:
                params[name].data.copy_(w)
            for _, m in self.named_modules():  # DefaultModelLoader.postprocess_weights
                if getattr(m, "quant_method", None) is not None:
                    m.quant_method.process_weights_after_loading(m)
            return self

        def forward(self, x):  # what apply() reads: the parked triton tensors
            qm = self.experts.quant_method
            w13 = qm.w13_weight_triton_tensor.storage.data.float()
            return x * self.norm.float() + w13.mean() + qm.w13_precision_config.b_mx_scale.storage.data.float().mean()

    return Model


def _mxfp4_checkpoint():
    g = torch.Generator().manual_seed(2)
    u8 = lambda *s: torch.randint(0, 256, s, generator=g, dtype=torch.uint8)  # noqa: E731
    return {
        "experts.w13_weight": u8(E, I2, H // 2), "experts.w13_weight_scale": u8(E, I2, H // 32),
        "experts.w13_weight_bias": torch.randn(E, I2, generator=g).bfloat16(),
        "experts.w2_weight": u8(E, H, I2 // 4), "experts.w2_weight_scale": u8(E, H, max(I2 // 64, 1)),
        "experts.w2_weight_bias": torch.randn(E, H, generator=g).bfloat16(),
        "norm": torch.randn(H, generator=g).bfloat16(),
    }


def test_mxfp4_triton_experts_are_silently_dropped(tmp_path, triton_kernels_stub):
    Model = make_mxfp4_classes()
    ck = _mxfp4_checkpoint()
    paths = _save_ck(ck, tmp_path / "ck")
    Loader = S.sharded_state_loader(Model)
    with deepstate.deterministic_empty():
        served = Model().load_weights(_iterate(paths))
        Loader.save_model(served, _out(tmp_path / "sharded"))
        reloaded = S.load_with(Loader, str(tmp_path / "sharded"))  # no error raised
    keys, saved = _file_keys(tmp_path / "sharded")
    assert "experts.w13_weight" not in keys and "experts.w2_weight" not in keys
    total = sum(t.numel() * t.element_size() for t in ck.values())

    ref, got = deepstate.fingerprint(served), deepstate.fingerprint(reloaded)
    registered = deepstate.compare(ref, got, only_kinds=("param", "buffer"))
    full = deepstate.compare(ref, got)
    assert registered.ok, "a state_dict-level check passes"
    assert "experts.quant_method.w13_weight_triton_tensor.storage.data" in full.differ
    assert "experts.quant_method.w13_precision_config.b_mx_scale.storage.data" in full.differ
    x = torch.ones(H)
    assert not torch.equal(served(x), reloaded(x))
    RESULTS["mxfp4_triton"] = {
        "sharded_state_load": "no error",
        "checkpoint_bytes": total, "sharded_state_bytes": saved,
        "dropped_fraction": round(1 - saved / total, 3),
        "state_dict_check": "passes (wrongly)",
        "deepstate_check": {"differ": full.differ, "missing": full.missing},
        "outputs_equal": False,
    }


def test_raw_preshard_round_trips_mxfp4_triton(tmp_path, triton_kernels_stub):
    Model = make_mxfp4_classes()
    paths = _save_ck(_mxfp4_checkpoint(), tmp_path / "ck")
    recorded, replayed, stats = _raw_round_trip(Model, lambda m, w: m.load_weights(w), paths, tmp_path)
    report = deepstate.compare(deepstate.fingerprint(recorded), deepstate.fingerprint(replayed))
    assert report.ok, report.summary()
    x = torch.ones(H)
    assert torch.equal(recorded(x), replayed(x))
    RESULTS["mxfp4_triton"]["raw_preshard"] = {"deepstate": "exact", "outputs_equal": True,
                                               "stored_bytes": stats["stored_bytes"]}


# ================================================================ GemmaRMSNorm


def make_gemma_classes():
    init, _ = S.function(S.LAYERNORM, "GemmaRMSNorm.__init__")
    loader, _ = S.function(S.LAYERNORM, "GemmaRMSNorm._weight_loader")
    source = "class GemmaRMSNorm(nn.Module):\n" + textwrap.indent(init, "    ") + "\n" + textwrap.indent(loader, "    ")
    GemmaRMSNorm = S.compile_into(source, {"nn": nn, "torch": torch, "Optional": None}, "GemmaRMSNorm")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = GemmaRMSNorm(16)

        def load_weights(self, weights):
            for name, w in weights:
                param = dict(self.named_parameters())[name]
                param.weight_loader(param, w)
            return self

    return Model


def test_gemma_weight_goes_stale_after_a_sharded_state_load(tmp_path):
    Model = make_gemma_classes()
    paths = _save_ck({"norm.weight": torch.randn(16, generator=torch.Generator().manual_seed(3))}, tmp_path / "ck")
    Loader = S.sharded_state_loader(Model)
    served = Model().load_weights(_iterate(paths))
    assert torch.equal(served.norm.gemma_weight, served.norm.weight + 1)
    Loader.save_model(served, _out(tmp_path / "sharded"))
    reloaded = S.load_with(Loader, str(tmp_path / "sharded"))
    assert torch.equal(reloaded.norm.weight, served.norm.weight)  # the parameter round-trips...
    assert torch.equal(reloaded.norm.gemma_weight, torch.ones(16))  # ...the buffer the fused paths read does not
    full = deepstate.compare(deepstate.fingerprint(served), deepstate.fingerprint(reloaded))
    assert full.differ == ["norm.gemma_weight"] and full.kinds["norm.gemma_weight"] == "nonpersistent_buffer"
    RESULTS["gemma_rmsnorm"] = {
        "sharded_state": "loads; norm.gemma_weight stays 1.0 (as if the checkpoint norm were 0)",
        "readers_in_v0_5_20": "layernorm.py forward_with_allreduce_fusion, ROCm and NPU paths; "
                              "qwen35_flashinfer_fusion.py; minimax_m3.py:1102 (q_norm.gemma_weight)",
    }


def test_raw_preshard_round_trips_gemma_weight(tmp_path):
    Model = make_gemma_classes()
    paths = _save_ck({"norm.weight": torch.randn(16, generator=torch.Generator().manual_seed(3))}, tmp_path / "ck")
    recorded, replayed, _ = _raw_round_trip(Model, lambda m, w: m.load_weights(w), paths, tmp_path)
    assert deepstate.compare(deepstate.fingerprint(recorded), deepstate.fingerprint(replayed)).ok
    RESULTS["gemma_rmsnorm"]["raw_preshard"] = "exact"


# ============================================================ the guard gap


def test_servekit_main_guard_lets_a_bare_gpt_oss_launch_through(tmp_path, monkeypatch):
    old = subprocess.run(["git", "-C", str(SERVEKIT), "show", "cd8f0b8:src/servekit/quant_guard.py"],
                         capture_output=True, text=True, check=True).stdout
    from servekit import engine_args

    module = types.ModuleType("servekit.quant_guard_cd8f0b8")
    module.__dict__["__package__"] = "servekit"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(old, "quant_guard@cd8f0b8.py", "exec"), module.__dict__)
    src = tmp_path / "gpt-oss-20b"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"architectures": ["GptOssForCausalLM"],
                                                 "quantization_config": {"quant_method": "mxfp4"}}))
    bare = ["python", "-m", "sglang.launch_server", "--model-path", str(src)]
    assert module.check_command(src, bare) == []  # main: not refused
    from servekit import quant_guard

    assert quant_guard.check_command(src, bare)  # this branch: refused for sharded_state
    RESULTS["quant_guard_gpt_oss_default_backend"] = {
        "servekit_cd8f0b8": "not refused", "this_branch": "refused for sharded_state, allowed as raw",
        "why": "SGLang 0.5.20 resolves --moe-runner-backend auto to triton_kernel for gpt-oss below SM100 "
               "(arg_groups/model_overrides/gpt_oss.py:106-125)",
    }

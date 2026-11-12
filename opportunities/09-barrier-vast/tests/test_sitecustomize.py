"""The post-import hook patches the constant in a fake SGLang package laid out
like 0.5.10.post1 and like 0.5.20, and leaves it alone when the env var is unset."""
import os
import subprocess
import sys
import textwrap

PATCHES = os.path.join(os.path.dirname(__file__), "..", "patches")


def _fake_sglang(tmp_path):
    for mod in ("model_runner", "model_runner_components/load_model_utils"):
        f = tmp_path / "sglang" / "srt" / "model_executor" / f"{mod}.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480\n"
                     "def timeout():\n    return UNBALANCED_MODEL_LOADING_TIMEOUT_S\n")
    for d in ("sglang", "sglang/srt", "sglang/srt/model_executor",
              "sglang/srt/model_executor/model_runner_components"):
        (tmp_path / d / "__init__.py").touch()


def _run(tmp_path, env_value):
    code = textwrap.dedent("""
        import sglang.srt.model_executor.model_runner as old
        import sglang.srt.model_executor.model_runner_components.load_model_utils as new
        print(old.timeout(), new.timeout())
    """)
    env = {k: v for k, v in os.environ.items() if k != "SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S"}
    env["PYTHONPATH"] = os.pathsep.join([os.path.abspath(PATCHES), str(tmp_path)])
    if env_value is not None:
        env["SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S"] = env_value
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    return out.stdout.split()


def test_hook_patches_both_layouts(tmp_path):
    _fake_sglang(tmp_path)
    assert _run(tmp_path, "3600") == ["3600", "3600"]


def test_no_env_no_change(tmp_path):
    _fake_sglang(tmp_path)
    assert _run(tmp_path, None) == ["480", "480"]


def test_garbage_env_ignored(tmp_path):
    _fake_sglang(tmp_path)
    assert _run(tmp_path, "one hour") == ["480", "480"]

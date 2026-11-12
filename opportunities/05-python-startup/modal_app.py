"""Modal experiments for opportunity 05 (Python start-up before torch.distributed init).

Run by the user later (no Modal account is configured where this was written). See MODAL.md.

    python modal_app.py --dry-run                          # local: validate, print plan + cost
    modal run modal_app.py::smoke                          # ~2 min CPU, checks both images
    modal run modal_app.py::cpu_study --engine sglang      # ~25 min CPU: imports, pycs, spawn, zygote
    modal run modal_app.py::cpu_study --engine vllm        # ~20 min CPU
    modal run modal_app.py::prewarm_study                  # 9 fresh CPU containers, cold vs prewarmed
    PYSTARTUP_GPU=L40S:2 modal run modal_app.py::gpu_e2e   # ~30 min on 2 GPUs, end-to-end TP2
    modal volume get pystartup-results / results/modal/    # fetch everything

Images are the exact upstream engine images. Arms without bytecode, or with timestamp or
unchecked-hash bytecode, are derived inside the container (PYTHONPYCACHEPREFIX), so no image
is rebuilt and the platform's pip-built (pycs) and uv-built (no pycs) images are both covered.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
REMOTE_SRC = "/opt/pystartup-src"
RESULTS = "/results"
HF = "/hf"

SGLANG_TAG = os.environ.get("PYSTARTUP_SGLANG_IMAGE", "lmsysorg/sglang:v0.5.20")
VLLM_TAG = os.environ.get("PYSTARTUP_VLLM_IMAGE", "vllm/vllm-openai:v0.33.0")
GPU = os.environ.get("PYSTARTUP_GPU", "L40S:2")
MODEL = os.environ.get("PYSTARTUP_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

PLAN = [
    # name, resources, expected minutes, approx USD (modal.com/pricing, Nov 2026; verify)
    ("smoke", "2 CPU containers x 2 vCPU", 3, 0.02),
    ("cpu_study --engine sglang", "1 container, 8 vCPU / 16 GiB", 30, 0.30),
    ("cpu_study --engine vllm", "1 container, 8 vCPU / 16 GiB", 25, 0.25),
    ("prewarm_study", "9 single-use containers, 8 vCPU / 16 GiB", 20, 0.25),
    (f"gpu_e2e ({GPU})", "1 container, 2 GPUs, 16 vCPU", 35, 5.0 if "H100" in GPU else 2.5),
]


def plan_text() -> str:
    lines = [f"{'experiment':34s} {'resources':42s} {'min':>4s} {'USD':>6s}"]
    for n, r, m, c in PLAN:
        lines.append(f"{n:34s} {r:42s} {m:4d} {c:6.2f}")
    lines.append(f"{'total':34s} {'':42s} {sum(p[2] for p in PLAN):4d} {sum(p[3] for p in PLAN):6.2f}")
    lines.append("Budget cap for this opportunity: USD 15 (one GPU retry fits). Split: account A runs "
                 "smoke + cpu_study x2 + prewarm_study (~USD 1), account B runs gpu_e2e (~USD 2.5-5).")
    return "\n".join(lines)


if "--dry-run" in sys.argv:
    # Local validation: no Modal calls. Check the pieces this app ships into the containers.
    sys.path.insert(0, SRC)
    import importlib

    for m in ("pystartup.profile_startup", "pystartup.zygote", "pystartup.zygote_preload", "pystartup.e2e",
              "pystartup.sglang_launch", "pystartup.prewarm", "pystartup.spawnbench", "pystartup._importprobe"):
        importlib.import_module(m)
    import modal  # noqa: F401  (the app definition below must also import cleanly)

    print("dry run: pystartup modules import; modal", modal.__version__)
    print(f"images: {SGLANG_TAG}, {VLLM_TAG}; GPU: {GPU}; model: {MODEL}")
    print(plan_text())

import modal  # noqa: E402

app = modal.App("pystartup-05-python-startup")
results = modal.Volume.from_name("pystartup-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


def _engine_image(tag: str) -> modal.Image:
    # The engine images bring their own Python 3.12 on PATH; nothing is pip-installed into
    # them, so their bytecode state stays as shipped. Our code is mounted, not copied.
    return (modal.Image.from_registry(tag)
            .entrypoint([])
            .env({"PYTHONPATH": REMOTE_SRC, "HF_HOME": HF, "PYTHONUNBUFFERED": "1"})
            .add_local_dir(SRC, remote_path=REMOTE_SRC, copy=False,
                           ignore=["**/__pycache__/**", "**/*.pyc"]))


sglang_image = _engine_image(SGLANG_TAG)
vllm_image = _engine_image(VLLM_TAG)


def _save(name: str, obj) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    results.commit()
    return path


def _container_info() -> dict:
    import platform

    info = {"hostname": platform.node(), "python": sys.version, "cpus": os.cpu_count(), "t": time.time()}
    try:
        info["dev_shm_bytes"] = os.statvfs("/dev/shm").f_bavail * os.statvfs("/dev/shm").f_frsize
    except OSError:
        pass
    try:
        info["kernel"] = open("/proc/version").read().strip()
    except OSError:
        pass
    return info


# ------------------------------------------------------------------------------------- smoke
@app.function(image=sglang_image, cpu=2, memory=4096, timeout=900, volumes={RESULTS: results})
def smoke_sglang() -> dict:
    from pystartup import profile_startup as ps

    rep = {"container": _container_info(), "env": ps.env_report("sglang", audit=False)}
    rep["stub_dir"] = ps.cuda_stub_dir()
    r = __import__("pystartup.importprof", fromlist=["x"]).run_probe(
        ["sglang.launch_server"], "asis", fake_cuda=True, env_extra=ps.fake_env())
    rep["probe_ok"] = r.get("returncode") == 0
    rep["probe_import_s"] = r.get("import_s")
    rep["probe_errors"] = r.get("errors") or r.get("stderr_tail")
    _save("smoke_sglang.json", rep)
    return rep


@app.function(image=vllm_image, cpu=2, memory=4096, timeout=900, volumes={RESULTS: results})
def smoke_vllm() -> dict:
    from pystartup import profile_startup as ps

    rep = {"container": _container_info(), "env": ps.env_report("vllm", audit=False)}
    _save("smoke_vllm.json", rep)
    return rep


@app.local_entrypoint()
def smoke():
    a = smoke_sglang.spawn()
    b = smoke_vllm.spawn()
    for name, fc in (("sglang", a), ("vllm", b)):
        r = fc.get()
        print(name, json.dumps({"python": r["container"]["python"].split()[0], "cpus": r["container"]["cpus"],
                                "versions": r["env"]["versions"], "roots": r["env"]["roots"],
                                "probe_ok": r.get("probe_ok"), "probe_import_s": r.get("probe_import_s"),
                                "probe_errors": r.get("probe_errors")}, indent=1))


# --------------------------------------------------------------------------------- CPU study
def _cpu_study(engine: str, repeats: int, workers: int) -> dict:
    from pystartup import profile_startup as ps
    from pystartup import spawnbench, zygote

    t = time.time()
    out = {"container": _container_info(), "engine": engine}
    out["env"] = ps.env_report(engine, audit=True)          # bytecode state of the image as shipped
    _save(f"cpu_study_{engine}.json", out)
    roles = list(ps.ROLES[engine])
    out["imports"] = {}
    for role in roles:
        out["imports"][role] = ps.imports_study(engine, role, ["asis", "nopyc", "prefix"], repeats, fake=True)
        _save(f"cpu_study_{engine}.json", out)
    heavy = "scheduler" if engine == "sglang" else "worker"
    files = [f for f in out["imports"][heavy].get("manifest", []) if f.endswith(".py")]
    out["compileall"] = ps.compileall_study(engine, heavy, files, repeats, fake=True)
    # fix (a) for the whole image, as an image-build step would do it: time and size
    pdir = "/tmp/pystartup-image-pycs"
    from pystartup import importprof

    out["compileall_image"] = importprof.compileall_prefix(out["env"]["roots"], pdir, invalidation="unchecked-hash")
    _save(f"cpu_study_{engine}.json", out)
    # prewarm manifest for prewarm_study (first-access order of the heavy role's import)
    man = os.path.join(RESULTS, f"manifest_{engine}_{heavy}.txt")
    from pystartup import prewarm

    out["manifest_files"] = prewarm.write_manifest(out["imports"][heavy].get("manifest", []), man)
    # worker start: spawn vs zygote vs fork, with the heavy role's import set
    os.environ.update(ps.fake_env())
    facts = json.loads(os.environ["PYSTARTUP_FAKE_CUDA"])
    out["spawn"] = spawnbench.bench(ps.ROLES[engine][heavy], workers, repeats, ["spawn", "zygote", "fork"],
                                    facts=facts)
    # how much of the heavy import set a zygote can preload under the guard (emulated facts)
    cands = out["imports"][heavy].get("module_order", [])
    try:
        cal = zygote.calibrate(cands, facts=facts)
        out["calibration"] = {"candidates": len(cands), "preloadable": len(cal["modules"]),
                              "trips": cal["trips"][:50], "targets_not_ok":
                              {k: v for k, v in cal["targets"].items() if v != "ok"},
                              "seconds": cal.get("calibration_s")}
    except Exception as e:  # noqa: BLE001
        out["calibration"] = {"error": repr(e)[:2000]}
    out["study_s"] = round(time.time() - t, 1)
    _save(f"cpu_study_{engine}.json", out)
    for role in roles:  # drop the bulky lists from the returned copy
        out["imports"][role].pop("manifest", None)
        out["imports"][role].pop("module_order", None)
    return out


@app.function(image=sglang_image, cpu=8, memory=16384, timeout=5400, volumes={RESULTS: results},
              single_use_containers=True)
def cpu_study_sglang(repeats: int = 3, workers: int = 4) -> dict:
    return _cpu_study("sglang", repeats, workers)


@app.function(image=vllm_image, cpu=8, memory=16384, timeout=5400, volumes={RESULTS: results},
              single_use_containers=True)
def cpu_study_vllm(repeats: int = 3, workers: int = 4) -> dict:
    return _cpu_study("vllm", repeats, workers)


@app.local_entrypoint()
def cpu_study(engine: str = "sglang", repeats: int = 3, workers: int = 4):
    fn = cpu_study_sglang if engine == "sglang" else cpu_study_vllm
    r = fn.remote(repeats, workers)
    for role, s in r["imports"].items():
        print(role, json.dumps(s["summary"], indent=1)[:3000])
    print("compileall:", json.dumps(r.get("compileall"), indent=1)[:2000])
    print("spawn:", json.dumps(r.get("spawn", {}).get("summary"), indent=1))
    print("calibration:", json.dumps(r.get("calibration"), indent=1)[:2000])


# ----------------------------------------------------------------------------- prewarm study
@app.function(image=sglang_image, cpu=8, memory=16384, timeout=1800, volumes={RESULTS: results},
              single_use_containers=True)
def prewarm_one(arm: str, rep: int, threads: int = 32) -> dict:
    """One fresh container: is the image cold here, and what does a parallel prewarm buy?"""
    import threading

    from pystartup import importprof, prewarm
    from pystartup import profile_startup as ps

    man = os.path.join(RESULTS, "manifest_sglang_scheduler.txt")
    paths = prewarm.load_manifest(man)
    rec = {"arm": arm, "rep": rep, "container": _container_info(), "manifest_files": len(paths),
           "residency_at_start": prewarm.residency(paths[:400])}
    targets = ps.ROLES["sglang"]["scheduler"]
    t0 = time.time()
    if arm == "prewarm-then-import":
        rec["prewarm"] = prewarm.prewarm(paths, threads=threads)
    elif arm == "prewarm-concurrent":
        th = threading.Thread(target=lambda: rec.__setitem__("prewarm", prewarm.prewarm(paths, threads=threads)))
        th.start()
    r = importprof.run_probe(targets, "asis", fake_cuda=True, env_extra=ps.fake_env())
    rec["import_wall_s"] = r["wall_s"]
    rec["import_s"] = r.get("import_s")
    rec["breakdown_s"] = r.get("breakdown_s")
    rec["t0_to_imported_s"] = round(time.time() - t0, 3)
    if arm == "prewarm-concurrent":
        th.join()
    rec["residency_after"] = prewarm.residency(paths[:400])
    r2 = importprof.run_probe(targets, "asis", fake_cuda=True, env_extra=ps.fake_env())
    rec["second_import_wall_s"] = r2["wall_s"]  # same container, now warm
    _save(f"prewarm_{arm}_r{rep}.json", rec)
    return rec


@app.local_entrypoint()
def prewarm_study(repeats: int = 3, threads: int = 32):
    arms = ["cold", "prewarm-then-import", "prewarm-concurrent"]
    calls = [prewarm_one.spawn(a, r, threads) for r in range(repeats) for a in arms]
    rows = [c.get() for c in calls]
    for a in arms:
        xs = [r for r in rows if r["arm"] == a]
        print(a, json.dumps({"t0_to_imported_s": [x["t0_to_imported_s"] for x in xs],
                             "second_import_wall_s": [x["second_import_wall_s"] for x in xs],
                             "prewarm": [x.get("prewarm", {}).get("seconds") for x in xs],
                             "residency_at_start": [x["residency_at_start"].get("resident_fraction") for x in xs]}))


# -------------------------------------------------------------------------------- GPU e2e TP2
@app.function(image=sglang_image, gpu=GPU, cpu=16, memory=65536, timeout=3600,
              volumes={RESULTS: results, HF: hf_cache})
def gpu_e2e_run(repeats: int = 3, arms: str = "spawn,zygote,zygote-cal,spawn-nopyc,zygote-cal-nopyc",
                tp: int = 2) -> dict:
    import subprocess

    from huggingface_hub import snapshot_download

    info = _container_info()
    info["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                         "--format=csv,noheader"], capture_output=True, text=True).stdout
    model = snapshot_download(MODEL, cache_dir=os.path.join(HF, "hub"))
    hf_cache.commit()
    out_dir = os.path.join(RESULTS, f"e2e_{GPU.replace(':', 'x')}_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "container.json"), "w") as f:
        json.dump(info, f, indent=1)
    # 1. prove the zygote never initialises CUDA, with real NVML facts on this node
    proof = subprocess.run([sys.executable, "-m", "pystartup.zygote", "smoke", "--modules",
                            "torch,triton,transformers,sgl_kernel,flashinfer", "--facts", "probe"],
                           capture_output=True, text=True, timeout=900)
    with open(os.path.join(out_dir, "zygote_smoke.txt"), "w") as f:
        f.write(proof.stdout + "\n--- stderr ---\n" + proof.stderr[-20000:])
    # 2. the end-to-end arms
    r = subprocess.run([sys.executable, "-m", "pystartup.e2e", "--model", model, "--tp", str(tp),
                        "--repeats", str(repeats), "--arms", arms, "--out", out_dir],
                       capture_output=True, text=True, timeout=3300)
    with open(os.path.join(out_dir, "e2e_stdout.txt"), "w") as f:
        f.write(r.stdout + "\n--- stderr ---\n" + r.stderr[-50000:])
    results.commit()
    summary = {}
    try:
        with open(os.path.join(out_dir, "e2e.json")) as f:
            summary = json.load(f).get("summary", {})
    except (OSError, ValueError):
        pass
    return {"out_dir": out_dir, "container": info, "zygote_smoke_rc": proof.returncode,
            "zygote_smoke_tail": proof.stdout[-3000:], "e2e_rc": r.returncode, "summary": summary,
            "stdout_tail": r.stdout[-4000:]}


@app.local_entrypoint()
def gpu_e2e(repeats: int = 3, arms: str = "spawn,zygote,zygote-cal,spawn-nopyc,zygote-cal-nopyc", tp: int = 2):
    r = gpu_e2e_run.remote(repeats, arms, tp)
    print(json.dumps({k: r[k] for k in ("out_dir", "zygote_smoke_rc", "e2e_rc")}, indent=1))
    print(r["container"].get("nvidia_smi"))
    print(r["zygote_smoke_tail"])
    print(json.dumps(r["summary"], indent=1))

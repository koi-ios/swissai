"""Turn the study's captures into false-alarm rates, detection power and ROC curves.

Reads results/<dir>/<variant>.jsonl.gz (written by cpu/study.py, or by
modal_app.py for real engines -- same record format), fits `servekit verify`'s
noise model on the calibration runs exactly as `--reference` would, scores
every held-out run with it, and writes:

  runs.csv     one row per run: family, fault, statistics, decisions, effect sizes
  summary.md   the tables the write-up quotes
  roc.csv      false-alarm rate vs detection rate per fault class, sweeping a
               multiplier on every calibrated threshold

    python -I cpu/report.py results/cpu
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "servekit" / "src"))

from servekit import verify_stats as vs  # noqa: E402

ALPHA = 0.01
FIXED_TOLS = (1e-6, 5e-2, 1e-1)  # servekit today; SGLang CI's prefill tolerance; the lab's cross-hardware drift


def load(path: Path):
    recs = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _max_abs(ref, cap) -> float:
    if cap is None:
        return math.inf
    al = vs.align(ref, cap)
    vals = [abs(g - r) if math.isfinite(g) else math.inf for p in al.prompts for r, g in zip(p["ref"], p["got"])]
    return max(vals) if vals else 0.0


def _greedy_equal(ref, cap) -> bool:
    if cap is None:
        return False
    by = {p["key"]: p for p in cap["prompts"]}
    return all(p.get("greedy_tokens") == by[p["key"]].get("greedy_tokens") for p in ref["prompts"])


def fired(stats, model) -> list:
    out = [n for n in vs.CONTINUOUS if stats.get(n) > model.thresholds.get(n, 0.0)]
    if stats.flips:
        out.append("flips")
    if stats.greedy_bad:
        out.append("greedy")
    return out


def score(stats, model) -> float:
    """How far past its threshold the most extreme statistic is (> 1 alarms)."""
    s = max((stats.get(n) / model.thresholds[n]) if model.thresholds.get(n, 0) > 0 else
            (math.inf if stats.get(n) > 0 else 0.0) for n in vs.CONTINUOUS)
    if stats.flips or stats.greedy_bad:
        s = max(s, math.inf)
    return s


def analyse(variant: str, recs: list) -> dict:
    by_role = defaultdict(list)
    ref = rep = None
    for r in recs:
        sp = r["spec"]
        if sp["family"] == "reference":
            ref = r
        elif sp["family"] == "repeat":
            rep = r
        else:
            by_role[sp.get("role")].append(r)
    if ref is None:
        raise SystemExit(f"{variant}: no reference run yet")
    reference = ref["capture"]
    calib = sorted(by_role["calibration"], key=lambda r: r["spec"]["seed"])
    held = by_role["heldout"]
    flt = by_role["fault"]
    repeat_cap = dict(rep["capture"], schedule={"kind": "repeat"}) if rep else None

    def model_with(k, extra=()):
        caps = ([repeat_cap] if repeat_cap else []) + [dict(c["capture"], schedule={"kind": "perturbed"}) for c in calib[:k]]
        caps += [dict(c["capture"], schedule={"kind": "external"}) for c in extra]
        return vs.fit(reference, caps, alpha=ALPHA)

    ks = [k for k in (4, 8, 16, 32) if k <= len(calib)]
    models = {f"K={k}": model_with(k) for k in ks}
    main_key = f"K={ks[-1]}"
    main = models[main_key]
    tp2 = sorted([r for r in held if r["spec"]["family"] == "tp2"], key=lambda r: r["spec"]["seed"])
    if len(tp2) >= 6:
        models["K=16+tp2x5"] = model_with(ks[-1], tp2[:5])
    # More calibration: the first 16 held-out request-level runs join the 16 calibration runs;
    # this model is only scored on runs it did not see (see `excluded` below).
    req = sorted([r for r in held if r["spec"]["family"] == "request"], key=lambda r: r["spec"]["seed"])
    excluded = set()
    if len(req) >= 32 and 16 <= len(calib) < 32:
        caps = ([repeat_cap] if repeat_cap else []) + [dict(c["capture"], schedule={"kind": "perturbed"})
                                                      for c in calib[:16] + req[:16]]
        models["K=32"] = vs.fit(reference, caps, alpha=ALPHA)
        excluded = {r["spec"]["id"] for r in req[:16]}

    rows = []
    for r in held + flt + by_role["fault-load"]:
        sp, cap = r["spec"], r["capture"]
        if cap is None:  # the server would not start on this checkpoint: a loud failure, detected by anything
            row = {"variant": variant, "id": sp["id"], "family": sp["family"], "role": sp.get("role"),
                   "class": sp.get("class", "benign"), "magnitude": sp.get("magnitude", ""), "server_failed": True,
                   "kl_vs_ref": None, "kl_fault_only": None, "heldout_nll_delta": None,
                   "tokens_changed_ref_schedule": None, "max_abs_delta": math.inf, "greedy_equal": False,
                   "exact:alarm": True, "weights:alarm": None}
            for name in models:
                row.update({f"{name}:alarm": True, f"{name}:score": math.inf, f"{name}:fired": "server-failed"})
            for tol in FIXED_TOLS:
                row[f"tol{tol:g}:alarm"] = True
            rows.append(row)
            continue
        row = {"variant": variant, "id": sp["id"], "family": sp["family"], "role": sp.get("role"),
               "class": sp.get("class", "benign"), "magnitude": sp.get("magnitude", ""),
               "kl_vs_ref": r["effect"]["kl_vs_ref"], "kl_fault_only": r["effect"].get("kl_fault_only"),
               "heldout_nll_delta": r["effect"].get("heldout_nll_delta"),
               "tokens_changed_ref_schedule": r["effect"].get("tokens_changed_ref_schedule"),
               "max_abs_delta": _max_abs(reference, cap), "greedy_equal": _greedy_equal(reference, cap),
               "server_failed": False}
        if "weights" in cap or "weights" in reference:
            from servekit.verify import compare_weights
            eq, _ = compare_weights(reference.get("weights"), cap.get("weights"))
            row["weights:alarm"] = None if eq is None else (not eq)
        for name, m in models.items():
            if name == "K=32" and sp["id"] in excluded:
                row[f"{name}:alarm"], row[f"{name}:score"], row[f"{name}:fired"] = None, None, "used to calibrate"
                continue
            d = vs.decide_statistical(reference, cap, m)
            st = d.stats
            row[f"{name}:alarm"] = not d.passed
            row[f"{name}:score"] = score(st, m)
            row[f"{name}:fired"] = "+".join(fired(st, m))
            if name == main_key:
                row.update({"max_z": st.max_z, "rms_z": st.rms_z, "shift": st.shift, "flips": st.flips,
                            "greedy_bad": st.greedy_bad, "greedy_explained": st.greedy_explained})
        for tol in FIXED_TOLS:
            row[f"tol{tol:g}:alarm"] = row["max_abs_delta"] > tol or not row["greedy_equal"]
        # Exact mode compares the candidate captured on the reference schedule: the fault alone.
        if sp.get("role") == "fault":
            changed = row["tokens_changed_ref_schedule"]
            row["exact:alarm"] = (changed or 0) > 0 or not row["greedy_equal"] if changed is not None else None
        rows.append(row)

    # calibration self-consistency: the noise model's view of its own runs
    calib_rows = []
    for c in calib:
        calib_rows.append({"seed": c["spec"]["seed"], "max_abs_delta": _max_abs(reference, c["capture"]),
                           "kl_vs_ref": c["effect"]["kl_vs_ref"]})
    return {"variant": variant, "rows": rows, "models": models, "main": main_key, "calib_rows": calib_rows,
            "repeat_bitwise": repeat_cap is not None and _max_abs(reference, repeat_cap) == 0.0
            and _greedy_equal(reference, repeat_cap)}


def _rate(xs):
    xs = list(xs)
    return (sum(xs) / len(xs), len(xs)) if xs else (float("nan"), 0)


def _fmt_rate(x):
    r, n = x
    return "-" if n == 0 else f"{sum_hits(r, n)}/{n}"


def sum_hits(r, n):
    return int(round(r * n))


def roc(rows, model_key, benign_family="request"):
    benign = [r[f"{model_key}:score"] for r in rows if r["role"] == "heldout" and r["family"] == benign_family]
    classes = sorted({r["class"] for r in rows if r["role"] == "fault"})
    grid = [10 ** (e / 20) for e in range(-40, 61)]
    out = []
    for c in grid:
        fpr = sum(s > c for s in benign) / max(1, len(benign))
        for cls in classes:
            sc = [r[f"{model_key}:score"] for r in rows if r["class"] == cls and r["role"] == "fault"]
            out.append({"multiplier": c, "class": cls, "fpr": fpr, "tpr": sum(s > c for s in sc) / max(1, len(sc))})
    return out


def auc(points):
    pts = sorted({(p["fpr"], p["tpr"]) for p in points} | {(0.0, 0.0), (1.0, 1.0)})
    # upper envelope: best tpr at each fpr
    best = {}
    for f, t in pts:
        best[f] = max(best.get(f, 0.0), t)
    xs = sorted(best)
    area, prev_x, prev_y = 0.0, 0.0, 0.0
    run = 0.0
    for x in xs:
        run = max(run, best[x])
        area += (x - prev_x) * prev_y
        prev_x, prev_y = x, run
    area += (1.0 - prev_x) * prev_y
    return area


def write(results_dir: Path):
    variants = [p.name[: -len(".jsonl.gz")] for p in sorted(results_dir.glob("*.jsonl.gz"))]
    analyses = [analyse(v, load(results_dir / f"{v}.jsonl.gz")) for v in variants]
    all_rows = [r for a in analyses for r in a["rows"]]
    keys = sorted({k for r in all_rows for k in r})
    with open(results_dir / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    md = ["# CPU study: noise floor, false alarms, detection power", "",
          f"Generated by `cpu/report.py` from `{results_dir.name}/*.jsonl.gz`. Model: Qwen3-0.6B on the CPU "
          f"engine (`cpu/engine.py`); alpha = {ALPHA} (Bonferroni over max_z, rms_z, shift). Counts are runs.", ""]
    roc_rows = []
    for a in analyses:
        rows, mk = a["rows"], a["main"]
        m = a["models"][mk]
        md += [f"## {a['variant']}", ""]
        md.append(f"- Reference schedule reproduces bitwise (repeat run): **{a['repeat_bitwise']}** -> `--mode auto` "
                  f"picks **{'exact' if a['repeat_bitwise'] else 'statistical'}**.")
        md.append(f"- Noise model ({mk}): scale s = {m.scale:.4g}; margin_tau = {m.margin_tau:.3g}; calibration "
                  f"max |delta logprob| = {m.max_abs_delta:.3g}; thresholds " +
                  ", ".join(f"{n} {m.thresholds[n]:.3g}" for n in vs.CONTINUOUS) + ".")
        kls = [c["kl_vs_ref"] for c in a["calib_rows"] if c["kl_vs_ref"] is not None]
        if kls:
            md.append(f"- Benign KL(P_ref || P_run) on calibration runs: median {statistics.median(kls):.2e} "
                      f"(min {min(kls):.2e}, max {max(kls):.2e}).")
        md.append("")
        md += ["### False alarms on held-out benign runs", "",
               "| family | runs | " + " | ".join(f"statistical {k}" for k in a["models"]) +
               " | " + " | ".join(f"max abs delta > {t:g}" for t in FIXED_TOLS) + " |",
               "|---|---|" + "---|" * (len(a["models"]) + len(FIXED_TOLS))]
        fams = sorted({r["family"] for r in rows if r["role"] == "heldout"}, key=lambda f: (f != "request", f))
        for fam in fams:
            sub = [r for r in rows if r["role"] == "heldout" and r["family"] == fam]
            if not sub:
                continue
            cells = [f"{sum(bool(r[f'{k}:alarm']) for r in sub if r[f'{k}:alarm'] is not None)}/"
                     f"{sum(r[f'{k}:alarm'] is not None for r in sub)}" for k in a["models"]]
            cells += [f"{sum(r[f'tol{t:g}:alarm'] for r in sub)}/{len(sub)}" for t in FIXED_TOLS]
            md.append(f"| {fam} | {len(sub)} | " + " | ".join(cells) + " |")
        md.append("")
        md += ["### Detection by fault class and magnitude", "",
               "Each cell: runs alarmed / runs. `exact` = the candidate captured on the reference schedule differs "
               "in any logprob (what `--mode exact` sees on a deterministic engine). KL is the fault's own effect, "
               "mean KL(P_clean || P_faulty) per position on the reference schedule; benign request-level KL is "
               + (f"~{statistics.median(kls):.1e}." if kls else "not measured here (no full distributions)."), "",
               f"| class | magnitude | statistical {mk} | which fired | exact | max abs delta > 0.05 | weight checksums | median KL | median held-out dNLL |",
               "|---|---|---|---|---|---|---|---|---|"]
        groups = defaultdict(list)
        for r in rows:
            if r["role"] == "fault":
                groups[(r["class"], r["magnitude"])].append(r)
        for (cls, mag), sub in groups.items():
            fired_counts = defaultdict(int)
            for r in sub:
                for f in (r[f"{mk}:fired"] or "none").split("+"):
                    fired_counts[f] += 1
            fired_s = ", ".join(f"{k} {v}" for k, v in sorted(fired_counts.items(), key=lambda x: -x[1]))
            kls_ = [r["kl_fault_only"] for r in sub if r["kl_fault_only"] is not None]
            dns = [r["heldout_nll_delta"] for r in sub if r["heldout_nll_delta"] is not None]
            kl = f"{statistics.median(kls_):.2e}" if kls_ else "n/a"
            dn = f"{statistics.median(dns):+.2e}" if dns else "n/a"
            ex = [r["exact:alarm"] for r in sub if r.get("exact:alarm") is not None]
            wt = [r["weights:alarm"] for r in sub if r.get("weights:alarm") is not None]
            md.append(f"| {cls} | {mag} | {sum(r[f'{mk}:alarm'] for r in sub)}/{len(sub)} | {fired_s} | "
                      f"{sum(ex)}/{len(ex)} | {sum(r['tol0.05:alarm'] for r in sub)}/{len(sub)} | "
                      f"{(str(sum(wt)) + '/' + str(len(wt))) if wt else 'n/a'} | {kl} | {dn} |")
        md.append("")
        loaded = [r for r in rows if r["role"] == "fault-load"]
        if loaded:
            md += ["### The same faults, captured under load", "",
                   "A candidate that is itself nondeterministic: each faulty server captured once more with "
                   "perturbed concurrency and filler traffic.", "",
                   f"| class | magnitude | statistical {mk} | which fired |", "|---|---|---|---|"]
            lg = defaultdict(list)
            for r in loaded:
                lg[(r["class"], r["magnitude"])].append(r)
            for (cls, mag), sub in lg.items():
                fired_s = ", ".join(sorted({x for r in sub for x in (r[f"{mk}:fired"] or "none").split("+")}))
                md.append(f"| {cls} | {mag} | {sum(r[f'{mk}:alarm'] for r in sub)}/{len(sub)} | {fired_s} |")
            md.append("")
        # detection vs effect size, relative to the benign KL
        kls = [k for k in kls if k is not None]
        if kls and any(r.get("kl_fault_only") is not None for r in rows):
            base = statistics.median(kls)
            bins = [(0, 0.3), (0.3, 1), (1, 3), (3, 10), (10, 100), (100, math.inf)]
            md += ["### Detection vs effect size", "",
                   "Faults binned by their own KL divided by the median benign KL of request-level noise.", "",
                   "| KL / benign KL | runs | statistical | exact |", "|---|---|---|---|"]
            for lo, hi in bins:
                def ratio(r):
                    k = r["kl_fault_only"]
                    return k / base if k == k and math.isfinite(k) else math.inf  # NaN/inf logits: off the scale
                sub = [r for r in rows if r["role"] == "fault" and r.get("kl_fault_only") is not None
                       and (lo <= ratio(r) < hi or (hi == math.inf and ratio(r) == math.inf))]
                if sub:
                    md.append(f"| [{lo:g}, {hi:g}) | {len(sub)} | {sum(r[f'{mk}:alarm'] for r in sub)}/{len(sub)} | "
                              f"{sum(r['exact:alarm'] for r in sub)}/{len(sub)} |")
            md.append("")
        pts = roc(rows, mk)
        for p in pts:
            p["variant"] = a["variant"]
        roc_rows += pts
        md += ["### ROC (thresholds scaled by a common multiplier; false alarms on held-out request-level runs)", "",
               "| class | AUC | detection at multiplier 1 |", "|---|---|---|"]
        for cls in sorted({p["class"] for p in pts}):
            cp = [p for p in pts if p["class"] == cls]
            at1 = min(cp, key=lambda p: abs(math.log(p["multiplier"])))
            md.append(f"| {cls} | {auc(cp):.3f} | {at1['tpr']:.2f} at FPR {at1['fpr']:.2f} |")
        md.append("")
    with open(results_dir / "roc.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant", "class", "multiplier", "fpr", "tpr"])
        w.writeheader()
        for p in roc_rows:
            w.writerow(p)
    (results_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    write(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "cpu")

"""Convert benchmark CSV files into LaTeX tables for the report.

    python report/make_tables.py   (reads benchmarks/output, examples/output; writes report/tables)
"""

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "output"
EX = ROOT / "examples" / "output"
OUT = ROOT / "report" / "tables"


def read(p):
    with open(p) as fh:
        return list(csv.DictReader(fh))


def sci(x, digits=1):
    x = float(x)
    if x == 0 or not math.isfinite(x):
        return "--" if not math.isfinite(x) else "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return f"${m:.{digits}f}\\cdot 10^{{{e}}}$"


def fix(x, d=2):
    x = float(x)
    return "--" if not math.isfinite(x) else f"{x:.{d}f}"


def table_N():
    rows = read(BENCH / "N_accuracy.csv")
    ratios = sorted({int(r["r/h"]) for r in rows})
    methods = [("planar (ours)", "tricubic"), ("planar (ours)", "trilinear"), ("heaviside", "tricubic"),
               ("heaviside", "trilinear"), ("binary", "tricubic"),
               ("baseline (student-like)", "trilinear"), ("baseline (student-like)", "nearest node")]
    names = {("planar (ours)", "tricubic"): "planes + tricubic (used)",
             ("planar (ours)", "trilinear"): "planes + trilinear",
             ("heaviside", "tricubic"): "Heaviside + tricubic",
             ("heaviside", "trilinear"): "Heaviside + trilinear",
             ("binary", "tricubic"): "binary + tricubic",
             ("baseline (student-like)", "trilinear"): "student-like + trilinear",
             ("baseline (student-like)", "nearest node"): "student-like, nodal"}
    lines = [r"\begin{tabular}{l" + "c" * len(ratios) + "}", r"\toprule",
             r"method $\backslash$ $r/h$ & " + " & ".join(f"${q}$" for q in ratios) + r" \\", r"\midrule"]
    for m in methods:
        vals = []
        for q in ratios:
            r = [x for x in rows if int(x["r/h"]) == q and (x["occupancy"], x["interp"]) == m]
            vals.append(sci(r[0]["max_err"]) if r else "--")
        lines.append(names[m] + " & " + " & ".join(vals) + r" \\")
    lines.append(r"\midrule")
    vals = []
    for q in ratios:
        r = [x for x in rows if int(x["r/h"]) == q and (x["occupancy"], x["interp"]) == methods[0]]
        vals.append(sci(r[0]["mean_err"]))
    lines.append(r"used: mean signed error & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "N_accuracy.tex").write_text("\n".join(lines) + "\n")


def table_sdf():
    rows = read(BENCH / "sdf_accuracy.csv")
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"mesh & grid & $\max|\phi-d|/h$ & sign errors & time [s] \\", r"\midrule"]
    for r in rows:
        se = r["sign_errors"] if int(r["sign_errors"]) >= 0 else "--"
        lines.append(f"{r['mesh']} & {r['grid']} & {sci(r['max_err/h'])} & {se} & {fix(r['seconds'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "sdf_accuracy.tex").write_text("\n".join(lines) + "\n")


def table_reinit():
    rows = read(BENCH / "reinit_accuracy.csv")
    lines = [r"\begin{tabular}{cccc}", r"\toprule",
             r"$h$ & $\max|\phi-d|/h$, $|d|<1.5h$ & $\max|\phi-d|/h$, $|d|<5h$ & relative volume change \\",
             r"\midrule"]
    for r in rows:
        lines.append(f"{float(r['h']):g} & {sci(r['max_err_near/h'])} & {sci(r['max_err_band/h'])} & "
                     f"{sci(r['rel_volume_change'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "reinit_accuracy.tex").write_text("\n".join(lines) + "\n")


def table_sphere():
    import numpy as np
    rows = read(BENCH / "sphere_convergence.csv")
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"law & $h$ & steps & $|\bar R - R_{\rm exact}|$ & spread $\max R-\min R$ & fitted order \\",
             r"\midrule"]
    laws = []
    for r in rows:
        if r["law"] not in laws:
            laws.append(r["law"])
    for i, law in enumerate(laws):
        rr = [r for r in rows if r["law"] == law]
        h = np.array([float(r["h"]) for r in rr])
        e = np.array([float(r["abs_err"]) for r in rr])
        slope = np.polyfit(np.log(h), np.log(e), 1)[0]
        if i:
            lines.append(r"\midrule")
        for j, r in enumerate(rr):
            name = law.replace("^", r"\^{}") if j == 0 else ""
            order = f"{slope:.2f}" if j == len(rr) - 1 else ""
            lines.append(f"{name} & {float(r['h']):.4f} & {r['steps']} & {sci(r['abs_err'])} & "
                         f"{sci(r['spread'])} & {order} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "sphere_convergence.tex").write_text("\n".join(lines) + "\n")


def table_cube():
    rows = read(BENCH / "cube_self_convergence.csv")
    lines = [r"\begin{tabular}{cccccc}", r"\toprule",
             r"resolution & $h$ & steps & mean dist. to $256^3$ & Hausdorff & order \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['resolution']} & {float(r['h']):.5f} & {r['steps']} & {sci(r['mean_dist_to_256'])} & "
                     f"{sci(r['hausdorff_to_256'])} & {fix(r['order_mean'], 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "cube_self_convergence.tex").write_text("\n".join(lines) + "\n")


def table_perf():
    rows = read(BENCH / "performance.csv")
    lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
             r"device & grid & nodes & steps & SDF [s] & evolution [s] & ms/step & meshing [s] & total [s] \\",
             r"\midrule"]
    for r in rows:
        dev = "GPU" if r["device"] == "cuda" else "CPU"
        lines.append(f"{dev} & {r['grid']} & {float(r['nodes_M']):.1f}M & {r['steps']} & "
                     f"{fix(r['sdf_s'], 1)} & {fix(r['evolve_s'], 1)} & {fix(r['per_step_ms'], 0)} & "
                     f"{fix(r['mesh_s'], 1)} & {fix(r['total_s'], 1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "performance.tex").write_text("\n".join(lines) + "\n")


def table_toy():
    rows = read(EX / "toy_summary.csv")
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"solid & law & steps & mean depth & max depth & wall [s] \\", r"\midrule"]
    last = None
    seen = set()
    for r in rows:
        key = (r["shape"], r["label"])
        if key in seen or r["label"].startswith("SVI "):   # comparison repeats a family run
            continue
        seen.add(key)
        if last is not None and r["shape"] != last:
            lines.append(r"\midrule")
        last = r["shape"]
        lab = r["label"].replace("^", r"\^{}")
        flag = "" if r["stop"] == "target_volume" else f" ({r['stop']}, $V/V_0={float(r['fraction']):.3f}$)"
        lines.append(f"{r['shape'].replace('_', ' ')} & {lab}{flag} & {r['steps']} & {float(r['depth_mean']):.4f} & "
                     f"{float(r['depth_max']):.4f} & {fix(r['wall'], 1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "toy_summary.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (table_N, table_sdf, table_reinit, table_sphere, table_cube, table_perf, table_toy):
        try:
            fn()
            print("ok", fn.__name__)
        except FileNotFoundError as e:
            print("skip", fn.__name__, e)

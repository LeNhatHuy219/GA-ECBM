"""CUB-200-2011: aggregate GA-ECBM vs ECBM (official) over training seeds for the paper (Table 2, Fig. 2, Sect. 5.1).

Usage: python3 cub_three_seeds.py <out.json> <results.json seed A> [<results.json seed B> ...]
Each results.json is the unchanged output of a Kaggle score run (keys D_val, D_test, A0_val, A0_test, gate);
D = GA-ECBM, A0 = ECBM (official code and inference).
Reports mean and sample s.d. over seeds, per-seed values, in how many seeds GA-ECBM is better and, with three or more
seeds, paired t-tests over seeds with a Holm adjustment over the five metrics of Table 2. Harm/help are % of all
images; the "average over levels" uses the ten levels 10%..100%.
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1])
FILES = [Path(p) for p in sys.argv[2:]]
R = [json.loads(p.read_text()) for p in FILES]
LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]          # rows of the per-level table


def level(curve, ratio):
    row = [c for c in curve if abs(c["ratio"] - ratio) < 1e-9]
    assert len(row) == 1, ratio
    return row[0]


def summary(run, split):
    curve = run[split]["curve"]
    assert [round(c["ratio"], 1) for c in curve] == [round(0.1 * i, 1) for i in range(11)]
    rest = [c for c in curve if c["ratio"] > 0]
    u = run[split]["unintervened"]
    out = {"class": u["class_accuracy"], "concept": u["concept_accuracy"], "overall": u["overall_concept_accuracy"],
           "int_auc": run[split]["intervention_auc"],
           "harm_avg": 100 * float(np.mean([c["harm_rate"] for c in rest])),
           "help_avg": 100 * float(np.mean([c["help_rate"] for c in rest])),
           "seconds": run[split]["seconds"]}
    out["net_avg"] = out["help_avg"] - out["harm_avg"]
    unv = [c["unverified_concept_accuracy"] for c in rest if c["ratio"] < 1 and c.get("unverified_concept_accuracy") is not None]
    out["unverified_min"], out["unverified_max"] = float(min(unv)), float(max(unv))
    if "direct_energy" in run[split]:
        out["image_energy_class"] = run[split]["direct_energy"]["class_accuracy"]
    if run[split].get("inference_steps"):
        out["steps_min"], out["steps_max"] = run[split]["inference_steps"]["min"], run[split]["inference_steps"]["max"]
    for r in LEVELS:
        c = level(curve, r)
        out[f"L{int(r * 100)}"] = {"class": c["class_accuracy"], "help": 100 * c["help_rate"], "harm": 100 * c["harm_rate"],
                                    "unverified": c.get("unverified_concept_accuracy")}
    return out


def ms(values):
    v = np.array(values, float)
    return [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else None]


res = {"files": [str(p) for p in FILES], "n_seeds": len(R), "splits": {}}
for split in ("val", "test"):
    per = {m: [summary(run, f"{m}_{split}") for run in R] for m in ("A0", "D")}
    agg = {}
    for m in ("A0", "D"):
        keys = [k for k in per[m][0] if not k.startswith("L") and k in per[m][-1]]
        agg[m] = {k: ms([p[k] for p in per[m]]) for k in keys}
        agg[m]["levels"] = {f"L{int(r * 100)}": {q: (ms([p[f'L{int(r * 100)}'][q] for p in per[m]])
                                                  if per[m][0][f"L{int(r * 100)}"][q] is not None else None)
                                              for q in ("class", "help", "harm", "unverified")} for r in LEVELS}
    better = {}
    for k, higher in (("class", True), ("concept", True), ("overall", True), ("int_auc", True), ("harm_avg", False),
                      ("net_avg", True), ("help_avg", True)):
        d = np.array([g[k] - e[k] for g, e in zip(per["D"], per["A0"])])
        better[k] = {"diff_mean": float(d.mean()), "diff_min": float(d.min()), "diff_max": float(d.max()),
                     "ga_better_seeds": int((d > 0).sum() if higher else (d < 0).sum())}
    res["splits"][split] = {"per_seed": per, "mean_sd": agg, "ga_vs_ecbm": better}
if len(R) > 2:
    # Paired t-tests over seeds (GA-ECBM vs ECBM), Holm-adjusted over the five metrics of the CUB table.
    from scipy import stats
    TABLE = ["class", "concept", "overall", "int_auc", "harm_avg"]
    for split in ("val", "test"):
        per = res["splits"][split]["per_seed"]
        pv = {k: float(stats.ttest_rel([p[k] for p in per["D"]], [p[k] for p in per["A0"]]).pvalue)
              for k in TABLE + ["help_avg", "net_avg"]}
        order = sorted(TABLE, key=lambda k: pv[k])
        holm, running = {}, 0.0
        for i, k in enumerate(order):
            running = max(running, min(1.0, (len(TABLE) - i) * pv[k]))
            holm[k] = running
        res["splits"][split]["paired_t"] = {k: {"p": pv[k], **({"p_holm": holm[k]} if k in holm else {})} for k in pv}
res["gate_w28"] = ms([run["gate"][28] for run in R])
res["gate_w28_per_seed"] = [run["gate"][28] for run in R]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1) + "\n")

# ---------- console summary ----------
f = lambda x, d=3: "--" if x is None else (f"{x[0]:.{d}f}" + ("" if x[1] is None else f" ± {x[1]:.{d}f}"))
for split in ("val", "test"):
    a = res["splits"][split]["mean_sd"]
    print(f"\n[{split}] n_seeds={len(R)}")
    for m, name in (("A0", "ECBM"), ("D", "GA-ECBM")):
        print(f"  {name:8s} class {f(a[m]['class'])} | concept {f(a[m]['concept'])} | overall {f(a[m]['overall'])} | "
              f"int AUC {f(a[m]['int_auc'])} | harm {f(a[m]['harm_avg'], 2)} | help {f(a[m]['help_avg'], 2)} | "
              f"net {f(a[m]['net_avg'], 2)} | unverified {a[m]['unverified_min'][0]:.3f}..{a[m]['unverified_max'][0]:.3f} | "
              f"seconds {f(a[m]['seconds'], 1)}")
    if "image_energy_class" in a["D"]:
        print(f"  GA image-label energy alone: class {f(a['D']['image_energy_class'])}")
    for k, v in res["splits"][split]["ga_vs_ecbm"].items():
        print(f"  GA - ECBM {k:9s} mean {v['diff_mean']:+.4f} [{v['diff_min']:+.4f}, {v['diff_max']:+.4f}]  GA better {v['ga_better_seeds']}/{len(R)}")
print("\n[test] per level  (E = ECBM, GA)")
lv = {m: res["splits"]["test"]["mean_sd"][m]["levels"] for m in ("A0", "D")}
for r in LEVELS:
    key = f"L{int(r * 100)}"
    print(f"  {int(r * 100):3d}%  class {f(lv['A0'][key]['class'])} / {f(lv['D'][key]['class'])}  "
          f"help {f(lv['A0'][key]['help'], 2)} / {f(lv['D'][key]['help'], 2)}  harm {f(lv['A0'][key]['harm'], 2)} / {f(lv['D'][key]['harm'], 2)}  "
          f"unverified {f(lv['A0'][key]['unverified'])} / {f(lv['D'][key]['unverified'])}")
print(f"\ngate w(28): {f(res['gate_w28'])}  per seed {[round(x, 4) for x in res['gate_w28_per_seed']]}")
print("saved", OUT)

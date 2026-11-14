"""Paired comparisons over the six seeds: GA-ECBM vs the ECBM interpolation at a given alpha (fixed k=0 thresholds)."""
import json
import numpy as np
from scipy import stats
from mix_tradeoff import OUT, RUNS, SEEDS, labelled, logit, metrics

rows = {"ga": [], 0.10: [], 0.40: [], 0.50: []}
for s in SEEDS:
    z = np.load(labelled(f"per_case_seed{s}.npz"))
    y = z["A_joint_test_y"]
    LJ, LA = logit(z["A_joint_test_scores_all_k"]), logit(z["A_author_test_scores_all_k"])
    tJ = logit(RUNS[s]["A_joint"]["thresholds_test_from_val"]["0"])
    rows["ga"].append(metrics(y, logit(z["D_test_scores_all_k"]), logit(RUNS[s]["D"]["thresholds_test_from_val"]["0"])))
    for a in (0.10, 0.40, 0.50):
        rows[a].append(metrics(y, (1 - a) * LJ + a * LA, tJ))
res = {}
for a in (0.10, 0.40, 0.50):
    res[str(a)] = {}
    print(f"--- GA vs ECBM interpolation alpha={a}  (mean difference GA - mix, paired t over 6 seeds)")
    for k in ("harm13", "help13", "auc13", "auc7", "f1_13", "f1_7"):
        g = np.array([r[k] for r in rows["ga"]]); m = np.array([r[k] for r in rows[a]])
        if k in ("harm13", "help13"):
            net_g = np.array([r["help13"] - r["harm13"] for r in rows["ga"]])
        d = g - m; p = stats.ttest_rel(g, m).pvalue
        res[str(a)][k] = {"ga": float(g.mean()), "mix": float(m.mean()), "diff": float(d.mean()), "p": float(p),
                          "ga_better_seeds": int((d < 0).sum() if k == "harm13" else (d > 0).sum())}
        scale = 100 if k in ("harm13", "help13") else 1
        print(f"  {k:7s} GA {g.mean()*scale:7.3f}  mix {m.mean()*scale:7.3f}  diff {d.mean()*scale:+7.3f}  p={p:.3f}  GA better in {res[str(a)][k]['ga_better_seeds']}/6 seeds")
    ng = np.array([r["help13"] - r["harm13"] for r in rows["ga"]]); nm = np.array([r["help13"] - r["harm13"] for r in rows[a]])
    res[str(a)]["net13"] = {"ga": float(ng.mean()), "mix": float(nm.mean()), "p": float(stats.ttest_rel(ng, nm).pvalue)}
    print(f"  net13   GA {ng.mean()*100:7.3f}  mix {nm.mean()*100:7.3f}  diff {(ng-nm).mean()*100:+7.3f}  p={res[str(a)]['net13']['p']:.3f}")
OUT.mkdir(parents=True, exist_ok=True)
json.dump(res, open(OUT / "paired.json", "w"), indent=1)

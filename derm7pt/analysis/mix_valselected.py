"""Paper Sect. 5.3 and Fig. 4: ECBM interpolation with alpha chosen on
validation, under the paper's main protocol (per-k thresholds selected on validation).

Interpolation: z_alpha = (1 - alpha) * logit(p_ECBM 1:1:0.01) + alpha * logit(p_ECBM concept-only), alpha = 0, 0.05, ..., 1.
Scores: validation (all 128 subsets) recomputed from the paper checkpoints by extract_val_scores.py (reproduces the
paper's validation numbers to ~1e-6); test = per-case scores saved by the paper notebook. No model is retrained or
re-run here.

Selection rules, fixed before any test metric is computed (validation, two-fold cross-fitted thresholds as in the paper,
means over the six seeds; one alpha for all seeds):
  alpha_match = the alpha whose validation harm (k = 1-3) is closest to GA-ECBM's validation harm (k = 1-3);
  alpha_net   = the alpha with the highest validation net benefit (help - harm, k = 1-3).
Test: thresholds per k are selected on the full validation set and applied unchanged (paper rule, same for GA-ECBM).
"""
import json, sys
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import OUT, labelled, notebook_functions, runs  # noqa: E402
best_threshold, val_halves = notebook_functions()   # verbatim from the paper notebook
RUNS = runs()
SEEDS = ["42", "123", "2026", "11", "222", "3333"]
G = 7
KS = np.array([0] + [len(s) for k in range(1, G + 1) for s in combinations(range(G), k)])
ALPHAS = [round(a, 2) for a in np.linspace(0, 1, 21)]
logit = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7)) - np.log(np.clip(1 - p, 1e-7, 1 - 1e-7))
sigmoid = lambda z: 1 / (1 + np.exp(-z))
mix = lambda J, A, a: sigmoid((1 - a) * logit(J.astype(float)) + a * logit(A.astype(float)))


def thresholds(y, S):
    return {k: best_threshold(np.tile(y, int((KS == k).sum())), S[KS == k].ravel()) for k in range(G + 1)}


def predict(S, t):
    pred = np.zeros(S.shape, int)
    for k in range(G + 1):
        pred[KS == k] = (S[KS == k] >= t[k]).astype(int)
    return pred


def crossfit_pred(y, S):
    half, pred = val_halves(y), np.zeros(S.shape, int)
    for h in (0, 1):
        t = thresholds(y[half != h], S[:, half != h])
        pred[:, half == h] = predict(S[:, half == h], t)
    return pred


def metrics(y, S, pred):
    base = pred[0]
    by_k, case_net = {}, np.zeros(len(y))
    for k in range(1, G + 1):
        rows = np.where(KS == k)[0]
        help_c = np.mean([(base != y) & (pred[r] == y) for r in rows], 0)
        harm_c = np.mean([(base == y) & (pred[r] != y) for r in rows], 0)
        by_k[k] = {"auc": np.mean([roc_auc_score(y, S[r]) for r in rows]),
                   "f1": np.mean([f1_score(y, pred[r], average="macro", zero_division=0) for r in rows]),
                   "help": help_c.mean(), "harm": harm_c.mean(), "help_c": help_c, "harm_c": harm_c}
    m = lambda key, kk: float(np.mean([by_k[k][key] for k in kk]))
    out = {"help13": m("help", (1, 2, 3)), "harm13": m("harm", (1, 2, 3)), "auc13": m("auc", (1, 2, 3)),
           "f1_13": m("f1", (1, 2, 3)), "auc7": float(by_k[7]["auc"]), "f1_7": float(by_k[7]["f1"]),
           "help_all": m("help", range(1, 8)), "harm_all": m("harm", range(1, 8))}
    out["net13"] = out["help13"] - out["harm13"]
    cases = {"net13": np.mean([by_k[k]["help_c"] - by_k[k]["harm_c"] for k in (1, 2, 3)], 0),
             "harm13": np.mean([by_k[k]["harm_c"] for k in (1, 2, 3)], 0),
             "help13": np.mean([by_k[k]["help_c"] for k in (1, 2, 3)], 0)}
    return out, cases


KEYS = ["help13", "harm13", "net13", "auc13", "f1_13", "auc7", "f1_7", "help_all", "harm_all"]
mean = lambda rows, k: float(np.mean([r[k] for r in rows]))

# ---------------- 1. validation only: choose alpha ----------------
val = {"ga": [], **{a: [] for a in ALPHAS}}
for s in SEEDS:
    v = np.load(labelled(f"val_scores_seed{s}.npz"))
    e, g = {"y": v["y"], "J": v["ecbm_joint"], "A": v["ecbm_concept_only"]}, {"S": v["ga_ecbm"]}
    y = e["y"]
    val["ga"].append(metrics(y, g["S"], crossfit_pred(y, g["S"]))[0])
    for a in ALPHAS:
        S = mix(e["J"], e["A"], a)
        val[a].append(metrics(y, S, crossfit_pred(y, S))[0])
ga_harm = mean(val["ga"], "harm13")
alpha_match = min(ALPHAS, key=lambda a: abs(mean(val[a], "harm13") - ga_harm))
alpha_net = max(ALPHAS, key=lambda a: mean(val[a], "net13"))
print(f"VALIDATION: GA harm13 {ga_harm*100:.2f}% net13 {mean(val['ga'], 'net13')*100:.2f} | "
      f"alpha_match = {alpha_match} (harm {mean(val[alpha_match], 'harm13')*100:.2f}%) | "
      f"alpha_net = {alpha_net} (net {mean(val[alpha_net], 'net13')*100:.2f}, harm {mean(val[alpha_net], 'harm13')*100:.2f}%)")

# ---------------- 2. test, thresholds from the full validation set ----------------
test, cases = {"ga": [], **{a: [] for a in ALPHAS}}, {"ga": [], **{a: [] for a in ALPHAS}}
check = []
for s in SEEDS:
    v, z = np.load(labelled(f"val_scores_seed{s}.npz")), np.load(labelled(f"per_case_seed{s}.npz"))
    e, g = {"y": v["y"], "J": v["ecbm_joint"], "A": v["ecbm_concept_only"]}, {"S": v["ga_ecbm"]}
    yv, yt = e["y"], z["D_test_y"]
    assert (yt == z["A_joint_test_y"]).all()
    tg = thresholds(yv, g["S"])
    check.append(max(abs(tg[k] - RUNS[s]["D"]["thresholds_test_from_val"][str(k)]) for k in range(G + 1)))
    r, c = metrics(yt, z["D_test_scores_all_k"], predict(z["D_test_scores_all_k"], tg))
    test["ga"].append(r); cases["ga"].append(c)
    for a in ALPHAS:
        Sv, St = mix(e["J"], e["A"], a), mix(z["A_joint_test_scores_all_k"], z["A_author_test_scores_all_k"], a)
        r, c = metrics(yt, St, predict(St, thresholds(yv, Sv)))
        test[a].append(r); cases[a].append(c)
print(f"check: max |GA threshold recomputed - paper| = {max(check):.1e}")
paper = {"help13": "help_k1_3", "harm13": "harm_k1_3", "auc13": "auc_k1_3", "f1_13": "f1_k1_3", "auc7": "auc_k7", "f1_7": "f1_k7"}
for name, v in (("ga", "D"), (0.0, "A_joint"), (1.0, "A_author")):
    dev = max(abs(mean(test[name], k) - np.mean([RUNS[s][v]["test"][pk] for s in SEEDS])) for k, pk in paper.items())
    print(f"check: {v} test means vs paper Table 4, max |diff| = {dev:.1e}")

rng = np.random.RandomState(0)
boot_idx = rng.randint(0, len(cases["ga"][0]["net13"]), size=(5000, len(cases["ga"][0]["net13"])))
result = {"rule": __doc__, "alpha_match": alpha_match, "alpha_net": alpha_net,
          "val": {str(k): {m: mean(v, m) for m in KEYS} for k, v in val.items()},
          "test": {str(k): {m: [mean(v, m), float(np.std([r[m] for r in v], ddof=1))] for m in KEYS} for k, v in test.items()},
          "compare": {}}
print("\nTEST (means over six seeds; per-k thresholds from validation)")
print("model          " + " ".join(f"{k:>8s}" for k in KEYS))
for name in ["ga"] + ALPHAS:
    tag = "GA-ECBM" if name == "ga" else f"mix a={name}" + (" *" if name in (alpha_match, alpha_net) else "")
    print(f"{tag:14s} " + " ".join(f"{mean(test[name], k):8.4f}" for k in KEYS))
for label, a in (("alpha_match", alpha_match), ("alpha_net", alpha_net)):
    comp = {}
    print(f"\nGA-ECBM vs ECBM interpolation at {label} = {a} (test; paired t over seeds; case bootstrap 95% CI)")
    for k in KEYS:
        gv, mv = np.array([r[k] for r in test["ga"]]), np.array([r[k] for r in test[a]])
        d = gv - mv
        better = int((d < 0).sum()) if k.startswith("harm") else int((d > 0).sum())
        comp[k] = {"ga": float(gv.mean()), "mix": float(mv.mean()), "diff": float(d.mean()),
                   "p": float(stats.ttest_rel(gv, mv).pvalue), "ga_better_seeds": better}
        if k in ("net13", "harm13", "help13"):
            diff_case = np.mean([cases["ga"][i][k] - cases[a][i][k] for i in range(len(SEEDS))], 0)
            bs = diff_case[boot_idx].mean(1)
            comp[k]["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        sc = 100 if k.startswith(("help", "harm", "net")) else 1
        ci = f"  CI [{comp[k]['ci95'][0]*sc:+.2f}, {comp[k]['ci95'][1]*sc:+.2f}]" if "ci95" in comp[k] else ""
        print(f"  {k:8s} GA {gv.mean()*sc:7.3f} | mix {mv.mean()*sc:7.3f} | diff {d.mean()*sc:+7.3f} | p={comp[k]['p']:.3g} | "
              f"GA better {better}/6{ci}")
    result["compare"][label] = comp
OUT.mkdir(parents=True, exist_ok=True)
json.dump(result, open(OUT / "mix_valselected.json", "w"), indent=1)

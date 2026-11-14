"""Interpolated ECBM procedures (paper Sect. 5.3, fixed k = 0 threshold variant): help-harm trade-off of ECBM when its two intervention procedures are mixed.

Uses only the saved per-case test outputs of the paper notebook (no retraining, no new model evaluation).
z_alpha = (1 - alpha) * logit(p_ECBM 1:1:0.01) + alpha * logit(p_ECBM concept-only); alpha = 0 / 1 reproduce the two
reported ECBM procedures. All procedures share the same k = 0 scores, so one fixed threshold (the validation-selected
k = 0 threshold, as in the paper's fixed-threshold sensitivity analysis) is used for every alpha and every k.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELLED, OUT, runs, labelled  # noqa: E402

D = LABELLED          # per-case outputs with labels re-attached (attach_labels.py)
RUNS = runs()
SEEDS = list(RUNS)
SUBSETS = [()] + [c for k in range(1, 8) for c in combinations(range(7), k)]
KS = np.array([len(s) for s in SUBSETS])
logit = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7)) - np.log(np.clip(1 - p, 1e-7, 1 - 1e-7))


def metrics(y, Z, t):
    """Z: [128, N] scores (logit or prob, threshold t in the same units). Means over subsets of each k, then over k."""
    base = Z[0] >= t
    by_k = {}
    for k in range(1, 8):
        rows = np.where(KS == k)[0]
        pr = [Z[r] >= t for r in rows]
        by_k[k] = {"auc": np.mean([roc_auc_score(y, Z[r]) for r in rows]),
                   "f1": np.mean([f1_score(y, p, average="macro", zero_division=0) for p in pr]),
                   "harm": np.mean([((base == y) & (p != y)).mean() for p in pr]),
                   "help": np.mean([((base != y) & (p == y)).mean() for p in pr])}
    m = lambda key, kk: float(np.mean([by_k[k][key] for k in kk]))
    return {"harm13": m("harm", (1, 2, 3)), "help13": m("help", (1, 2, 3)), "auc13": m("auc", (1, 2, 3)),
            "f1_13": m("f1", (1, 2, 3)), "auc7": by_k[7]["auc"], "f1_7": by_k[7]["f1"],
            "harm_all": m("harm", range(1, 8)), "help_all": m("help", range(1, 8))}


if __name__ == "__main__":
    out = {"alpha": {}, "ga_fixed": [], "check": []}
    alphas = np.round(np.linspace(0, 1, 21), 2)
    per_seed = {a: [] for a in alphas}
    for s in SEEDS:
        z = np.load(labelled(f"per_case_seed{s}.npz"))
        y = z["A_joint_test_y"]
        J, A, G = z["A_joint_test_scores_all_k"], z["A_author_test_scores_all_k"], z["D_test_scores_all_k"]
        assert np.allclose(J[0], A[0]) and J.shape == (128, len(y))
        # sanity: subset order reproduces the reported per-k test AUCs
        for v, S in (("A_joint", J), ("A_author", A), ("D", G)):
            for k in (1, 7):
                rep = RUNS[s][v]["test"]["intervention_by_k"][str(k)]["roc_auc"]
                got = np.mean([roc_auc_score(y, S[r]) for r in np.where(KS == k)[0]])
                out["check"].append(abs(rep - got))
        tJ = logit(RUNS[s]["A_joint"]["thresholds_test_from_val"]["0"])
        tG = logit(RUNS[s]["D"]["thresholds_test_from_val"]["0"])
        LJ, LA = logit(J), logit(A)
        for a in alphas:
            per_seed[a].append(metrics(y, (1 - a) * LJ + a * LA, tJ))
        out["ga_fixed"].append(metrics(y, logit(G), tG))
    agg = lambda rows: {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1))]
                        for k in rows[0]}
    out["alpha"] = {str(a): agg(per_seed[a]) for a in alphas}
    out["ga_fixed"] = agg(out["ga_fixed"])
    out["ga_paper_thresholds"] = {k: float(np.mean([RUNS[s]["D"]["test"][k] for s in SEEDS]))
                                  for k in ("harm_k1_3", "help_k1_3", "auc_k1_3", "auc_k7", "f1_k1_3", "f1_k7")}
    out["max_abs_check_diff"] = float(max(out.pop("check")))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "mix_tradeoff.json", "w"), indent=1)
    print("max |reported - recomputed| AUC:", out["max_abs_check_diff"])
    print(" alpha | harm13  help13  net13 | AUC13  AUC7 | F1_13  F1_7 | harm_all")
    for a in alphas:
        r = out["alpha"][str(a)]
        print(f" {a:5.2f} | {r['harm13'][0]*100:5.2f}  {r['help13'][0]*100:5.2f}  {(r['help13'][0]-r['harm13'][0])*100:5.2f} |"
              f" {r['auc13'][0]:.3f}  {r['auc7'][0]:.3f} | {r['f1_13'][0]:.3f}  {r['f1_7'][0]:.3f} | {r['harm_all'][0]*100:5.2f}")
    g = out["ga_fixed"]
    print(f"  GA   | {g['harm13'][0]*100:5.2f}  {g['help13'][0]*100:5.2f}  {(g['help13'][0]-g['harm13'][0])*100:5.2f} |"
          f" {g['auc13'][0]:.3f}  {g['auc7'][0]:.3f} | {g['f1_13'][0]:.3f}  {g['f1_7'][0]:.3f} | {g['harm_all'][0]*100:5.2f}  (fixed k=0 threshold)")

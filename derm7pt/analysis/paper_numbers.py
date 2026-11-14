"""Recompute every Derm7pt number of the paper (Tables 3-4, Sect. 5.2 and the Derm7pt part of Sect. 5.4) and compare
it with the value printed in the paper.

Inputs: results/derm7pt/runs.json and summary.json (written by the paper notebook) and the per-case outputs with
labels re-attached (attach_labels.py). The empirical log-odds baseline of Sect. 5.4 also needs the training labels, so
it is computed only when the Derm7pt meta directory is given.
Usage: python derm7pt/analysis/paper_numbers.py [<path to release_v0/meta>]
Model names: A_joint = ECBM (1:1:0.01), A_author = ECBM (concept-only), D = GA-ECBM.
Writes results/derm7pt/analysis/paper_numbers.json; exits with an error if a printed value does not match.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, RESULTS, SEEDS, labelled, runs  # noqa: E402

R = runs()
SUMMARY = json.loads((RESULTS / "summary.json").read_text())
MAPPING = json.loads((RESULTS / "label_mapping.json").read_text())
G = 7
KS = np.array([0] + [len(s) for k in range(1, G + 1) for s in combinations(range(G), k)])
# Seven-point checklist weights (Argenziano et al. 1998): 2 = major, 1 = minor criterion, 0 = not a criterion.
CHECKLIST = {("pigment_network", "atypical"): 2, ("blue_whitish_veil", "present"): 2,
             ("vascular_structures", "linear irregular"): 2, ("vascular_structures", "dotted"): 2,
             ("pigmentation", "diffuse irregular"): 1, ("pigmentation", "localized irregular"): 1,
             ("streaks", "irregular"): 1, ("dots_and_globules", "irregular"): 1,
             ("regression_structures", "blue areas"): 1, ("regression_structures", "white areas"): 1,
             ("regression_structures", "combinations"): 1}
CHECKS, RESULT = [], {}


def fmt_p(p):
    """Rounding used for p-values in the paper."""
    if p >= 0.1:
        return f"{p:.2f}"
    if p >= 0.01:
        return f"{p:.3f}"
    return f"{p:.1g}"


def check(name, value, printed, how=None):
    shown = how(value) if how else printed.__class__(value)
    CHECKS.append((name, shown, printed, shown == printed))
    RESULT[name] = value


def seeds(model, split, key, sub="unintervened"):
    return np.array([R[s][model][split][sub][key] if sub else R[s][model][split][key] for s in SEEDS])


f3 = lambda v: f"{v:.3f}"
ms = lambda v, d=3, scale=1: f"{np.mean(v) * scale:.{d}f} ± {np.std(v, ddof=1) * scale:.{d}f}"

# ---------------- Table 3: test set without intervention (ECBM = A_joint) ----------------
TABLE3 = [("accuracy", "0.787 ± 0.018", "0.790 ± 0.023", "0.86"), ("balanced_accuracy", "0.737 ± 0.017", "0.752 ± 0.032", "0.25"),
          ("f1_macro", "0.729 ± 0.015", "0.738 ± 0.027", "0.56"), ("sensitivity", "0.634 ± 0.055", "0.673 ± 0.081", "0.27"),
          ("specificity", "0.840 ± 0.037", "0.830 ± 0.038", "0.74"), ("roc_auc", "0.837 ± 0.012", "0.839 ± 0.019", "0.87"),
          ("concept_accuracy", "0.679 ± 0.010", "0.665 ± 0.018", "0.19"), ("concept_f1_macro", "0.362 ± 0.036", "0.362 ± 0.057", "1.00"),
          ("exact_match", "0.072 ± 0.010", "0.081 ± 0.011", "0.0501")]
for key, pa, pd_, pp in TABLE3:
    a, d = seeds("A_joint", "test", key), seeds("D", "test", key)
    assert np.allclose(a, seeds("A_author", "test", key))      # both ECBM procedures share the unintervened model
    check(f"T3 {key} ECBM", a, pa, ms)
    check(f"T3 {key} GA-ECBM", d, pd_, ms)
    p = stats.ttest_rel(d, a).pvalue
    check(f"T3 {key} p", p, pp, (lambda v: f"{v:.4f}") if key == "exact_match" else fmt_p)

# ---------------- Table 4: test set under intervention; p = GA-ECBM vs concept-only, Holm over 8 ----------------
TABLE4 = [("f1_k1_3", 1, 3, ("0.729 ± 0.014", "0.756 ± 0.011", "0.765 ± 0.024"), "0.39", "0.50"),
          ("auc_k1_3", 1, 3, ("0.838 ± 0.013", "0.854 ± 0.012", "0.869 ± 0.019"), "0.25", "0.50"),
          ("help_k1_3", 100, 1, ("0.1 ± 0.2", "9.0 ± 1.9", "4.5 ± 1.3"), "0.005", "0.025"),
          ("harm_k1_3", 100, 1, ("0.2 ± 0.5", "6.9 ± 1.6", "1.9 ± 0.7"), "0.0003", "0.003"),
          ("f1_k7", 1, 3, ("0.729 ± 0.016", "0.817 ± 0.012", "0.849 ± 0.011"), "0.008", "0.033"),
          ("auc_k7", 1, 3, ("0.840 ± 0.014", "0.917 ± 0.005", "0.934 ± 0.007"), "0.002", "0.013"),
          ("help_all", 100, 1, ("0.3 ± 0.3", "11.8 ± 1.9", "8.2 ± 1.4"), "0.022", "0.067"),
          ("harm_all", 100, 1, ("0.4 ± 0.9", "7.7 ± 1.4", "2.8 ± 0.9"), "0.0006", "0.005")]
raw = {}
for key, scale, dec, printed, pp, _ in TABLE4:
    for model, pr in zip(("A_joint", "A_author", "D"), printed):
        check(f"T4 {key} {model}", seeds(model, "test", key, None), pr, lambda v, d=dec, s=scale: ms(v, d, s))
    raw[key] = stats.ttest_rel(seeds("D", "test", key, None), seeds("A_author", "test", key, None)).pvalue
    assert abs(raw[key] - SUMMARY["tests_D_vs_A_author"]["test"][key]["paired_t_p"]) < 1e-12
    check(f"T4 {key} p", raw[key], pp, fmt_p)
holm, running = {}, 0.0
for i, key in enumerate(sorted(raw, key=raw.get)):
    running = max(running, min(1.0, (len(raw) - i) * raw[key]))
    holm[key] = running
for key, *_, ph in TABLE4:
    check(f"T4 {key} p_Holm", holm[key], ph, fmt_p)

# ---------------- Sect. 5.2, without intervention ----------------
d_auc = seeds("D", "test", "roc_auc") - seeds("A_joint", "test", "roc_auc")
half = stats.t.ppf(0.975, len(SEEDS) - 1) * d_auc.std(ddof=1) / np.sqrt(len(SEEDS))
check("ROC-AUC seed-wise difference, 95% t-interval", [d_auc.mean() - half, d_auc.mean() + half], "-0.022 to +0.025",
      lambda v: f"{v[0]:.3f} to {v[1]:+.3f}")
dl = SUMMARY["tests"]["delong_test"]
check("DeLong per-seed difference range", [min(dl[s]["delta"] for s in SEEDS), max(dl[s]["delta"] for s in SEEDS)],
      "-0.029 to +0.032", lambda v: f"{v[0]:.3f} to {v[1]:+.3f}")
check("DeLong seeds with p < 0.05", sum(dl[s]["p"] < 0.05 for s in SEEDS), "0", str)
d_c = seeds("D", "test", "concept_accuracy") - seeds("A_joint", "test", "concept_accuracy")
check("concept accuracy difference (points)", 100 * d_c.mean(), "-1.4", lambda v: f"{v:.1f}")
check("validation ROC-AUC, GA-ECBM / ECBM", [seeds("D", "val", "roc_auc").mean(), seeds("A_joint", "val", "roc_auc").mean()],
      "0.828 / 0.802", lambda v: f"{v[0]:.3f} / {v[1]:.3f}")
check("validation ROC-AUC p", stats.ttest_rel(seeds("D", "val", "roc_auc"), seeds("A_joint", "val", "roc_auc")).pvalue, "0.045", fmt_p)

# ---------------- Sect. 5.2, under intervention ----------------
by_k = lambda model, k, key: np.array([R[s][model]["test"]["intervention_by_k"][str(k)][key] for s in SEEDS])
check("ECBM (1:1:0.01) ROC-AUC, k = 7", by_k("A_joint", 7, "roc_auc").mean(), "0.840", f3)
check("concept-only ROC-AUC / harm %, k = 1", [by_k("A_author", 1, "roc_auc").mean(), 100 * by_k("A_author", 1, "harm_rate").mean()],
      "0.832 / 6.6", lambda v: f"{v[0]:.3f} / {v[1]:.1f}")
check("GA-ECBM ROC-AUC / harm %, k = 1", [by_k("D", 1, "roc_auc").mean(), 100 * by_k("D", 1, "harm_rate").mean()],
      "0.853 / 1.5", lambda v: f"{v[0]:.3f} / {v[1]:.1f}")
check("harm % at k = 7, GA-ECBM / concept-only", [100 * by_k("D", 7, "harm_rate").mean(), 100 * by_k("A_author", 7, "harm_rate").mean()],
      "4.5 / 9.2", lambda v: f"{v[0]:.1f} / {v[1]:.1f}")
for key, printed in (("harm_k1_3", "6/6"), ("f1_k7", "6/6"), ("auc_k7", "6/6")):
    d = seeds("D", "test", key, None) - seeds("A_author", "test", key, None)
    check(f"seeds in which GA-ECBM is better, {key}", int((d < 0).sum() if key.startswith("harm") else (d > 0).sum()), printed,
          lambda v: f"{v}/6")
net = lambda m: seeds(m, "test", "help_k1_3", None) - seeds(m, "test", "harm_k1_3", None)
check("net benefit k = 1-3 (points), GA-ECBM / concept-only", [100 * net("D").mean(), 100 * net("A_author").mean()],
      "2.66 / 2.15", lambda v: f"{v[0]:.2f} / {v[1]:.2f}")
check("net benefit k = 1-3 p", stats.ttest_rel(net("D"), net("A_author")).pvalue, "0.63", fmt_p)
unv = lambda m, k: by_k(m, k, "unverified_concept_accuracy").mean()
check("unverified concept accuracy, ECBM, k = 1..6", [unv("A_author", k) for k in range(1, 7)] + [unv("A_joint", k) for k in range(1, 7)],
      "0.679", lambda v: f"{min(v):.3f}" if f"{min(v):.3f}" == f"{max(v):.3f}" else f"{min(v):.3f}-{max(v):.3f}")
check("unverified concept accuracy, GA-ECBM, k = 6", unv("D", 6), "0.676", f3)
gate = np.array([R[s]["D"]["gate"][7] for s in SEEDS])
check("gate w(7)", gate, "0.247 ± 0.003", ms)
check("gate w(7) at initialization", float(np.exp(-7 * np.log1p(np.exp(-1.5)))), "0.244", f3)


# ---------------- case-level quantities (labels needed) ----------------
def predict(S, thresholds):
    pred = np.zeros(S.shape, int)
    for k in range(G + 1):
        pred[KS == k] = (S[KS == k] >= thresholds[str(k)]).astype(int)
    return pred


def case_rates(y, pred):
    """Per-case help and harm, averaged over the subsets of each k and then over k = 1, 2, 3."""
    base = pred[0]
    help_k = [np.mean([(base != y) & (pred[r] == y) for r in np.where(KS == k)[0]], 0) for k in (1, 2, 3)]
    harm_k = [np.mean([(base == y) & (pred[r] != y) for r in np.where(KS == k)[0]], 0) for k in (1, 2, 3)]
    return np.mean(help_k, 0), np.mean(harm_k, 0)


cases = {m: [] for m in ("A_author", "D")}
fixed_harm = {m: [] for m in ("A_author", "D")}
below = {m: [] for m in ("A_author", "D")}
for s in SEEDS:
    z = np.load(labelled(f"per_case_seed{s}.npz"))
    y = z["D_test_y"]
    for m in ("A_author", "D"):
        S = z[f"{m}_test_scores_all_k"].astype(float)
        th = R[s][m]["thresholds_test_from_val"]
        help_c, harm_c = case_rates(y, predict(S, th))
        assert abs(harm_c.mean() - R[s][m]["test"]["harm_k1_3"]) < 1e-12 and abs(help_c.mean() - R[s][m]["test"]["help_k1_3"]) < 1e-12
        cases[m].append({"help": help_c, "harm": harm_c, "y": y, "S7": S[KS == 7][0]})
        fixed_harm[m].append(case_rates(y, predict(S, {str(k): th["0"] for k in range(G + 1)}))[1].mean())
        auc0 = roc_auc_score(y, S[0])
        below[m].append(int(sum(roc_auc_score(y, S[r]) < auc0 for r in range(1, len(S)))))
fa, fd = np.array(fixed_harm["A_author"]), np.array(fixed_harm["D"])
check("harm % k = 1-3 with the k = 0 threshold, GA-ECBM / concept-only", [100 * fd.mean(), 100 * fa.mean()], "2.4 / 13.2",
      lambda v: f"{v[0]:.1f} / {v[1]:.1f}")
check("harm with the k = 0 threshold, p", stats.ttest_rel(fd, fa).pvalue, "0.0004", fmt_p)
check("subsets lowering test ROC-AUC below k = 0, GA-ECBM (per seed)", below["D"], "0 in every seed",
      lambda v: "0 in every seed" if max(v) == 0 else str(v))
check("subsets lowering test ROC-AUC below k = 0, concept-only (per seed)", below["A_author"], "2-11",
      lambda v: f"{min(v)}-{max(v)}")

# Paired bootstrap over test cases (5,000 resamples; all six seeds kept in every resample), conditional on the fitted
# models, the thresholds and the test split.
N = len(cases["D"][0]["y"])
boot = np.random.default_rng(0).integers(0, N, size=(5000, N))   # the generator used for the paper
ci = lambda b: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
for key, printed in (("harm", "-6.0 to -4.0"), ("net", "-1.2 to +2.2")):
    get = (lambda c: c["help"] - c["harm"]) if key == "net" else (lambda c: c["harm"])
    diff = np.mean([get(g) - get(a) for g, a in zip(cases["D"], cases["A_author"])], 0)
    check(f"bootstrap 95% interval, {key} k = 1-3 (points)", [100 * x for x in ci(diff[boot].mean(1))], printed,
          lambda v: f"{v[0]:.1f} to {v[1]:+.1f}" if v[1] > 0 else f"{v[0]:.1f} to {v[1]:.1f}")
auc_diff = []
for idx in boot:
    auc_diff.append(np.mean([roc_auc_score(g["y"][idx], g["S7"][idx]) - roc_auc_score(a["y"][idx], a["S7"][idx])
                             for g, a in zip(cases["D"], cases["A_author"])]))
check("bootstrap 95% interval, ROC-AUC k = 7", ci(auc_diff), "+0.004 to +0.031", lambda v: f"{v[0]:+.3f} to {v[1]:+.3f}")

# ---------------- Sect. 5.4: learned evidence vs the seven-point checklist ----------------
states, evidence, weights = [], [], []
for g, (concept, mapping) in enumerate(MAPPING.items()):
    th = np.mean([np.array(R[s]["D"]["theta"][g]) for s in SEEDS], 0)     # [non-melanoma, melanoma] x states
    d = th[1] - th[0]
    d = d - d[0]                                                           # relative to "absent"
    inv = {i: name for name, i in mapping.items()}
    for i in range(1, len(mapping)):
        states.append(f"{concept}: {inv[i]}"); evidence.append(float(d[i])); weights.append(CHECKLIST.get((concept, inv[i]), 0))
rho = stats.spearmanr(evidence, weights)
assert abs(rho.correlation - SUMMARY["theta_checklist_spearman"]["rho"]) < 1e-9
check("number of non-absent states", len(states), "21", str)
check("Spearman rho, learned evidence vs checklist", rho.correlation, "0.66", lambda v: f"{v:.2f}")
check("Spearman p", rho.pvalue, "0.001", fmt_p)
RESULT["evidence_by_state"] = dict(sorted(zip(states, evidence), key=lambda t: -t[1]))
if len(sys.argv) > 1:
    import pandas as pd
    meta_dir = Path(sys.argv[1])
    meta = pd.read_csv(meta_dir / "meta.csv")
    train = meta.loc[pd.read_csv(meta_dir / "train_indexes.csv")["indexes"]]
    y = train["diagnosis"].astype(str).str.lower().str.startswith("melanoma").astype(int)
    log_odds = []
    for concept, mapping in MAPPING.items():
        v = train[concept].astype(str).str.strip().str.lower().map(lambda x: mapping.get(x, 0))
        for i in range(1, len(mapping)):       # state vs absent, Haldane correction +0.5
            a, b = ((v == i) & (y == 1)).sum() + .5, ((v == i) & (y == 0)).sum() + .5
            c, e = ((v == 0) & (y == 1)).sum() + .5, ((v == 0) & (y == 0)).sum() + .5
            log_odds.append(np.log(a / b) - np.log(c / e))
    check("Spearman rho, training-set log-odds vs checklist", stats.spearmanr(log_odds, weights).correlation, "0.69",
          lambda v: f"{v:.2f}")

# ---------------- report ----------------
width = max(len(c[0]) for c in CHECKS)
for name, shown, printed, ok in CHECKS:
    print(f"{'OK  ' if ok else 'DIFF'} {name:<{width}}  computed {shown:<18} paper {printed}")
OUT.mkdir(parents=True, exist_ok=True)
to_json = lambda v: v.tolist() if isinstance(v, np.ndarray) else (float(v) if isinstance(v, np.floating) else v)
(OUT / "paper_numbers.json").write_text(json.dumps({k: to_json(v) for k, v in RESULT.items()}, indent=1) + "\n")
bad = [c for c in CHECKS if not c[3]]
print(f"\n{len(CHECKS) - len(bad)}/{len(CHECKS)} values match the paper" + ("" if len(sys.argv) > 1 else
      " (log-odds baseline skipped: pass the Derm7pt meta directory)"))
sys.exit(1 if bad else 0)

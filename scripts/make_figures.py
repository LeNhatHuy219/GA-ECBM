"""Figures for the GA-ECBM paper (English), drawn from the saved experiment outputs.

Usage: python3 make_figures.py <derm7pt runs.json> <cub results.json[,results.json,...]> <output dir> [cub split: val|test]
Several comma-separated CUB results.json files (one per training seed) are drawn as mean +/- 1 s.d. bands.
All four figures share the same printed width, fonts, colours and legend position.
Colours: categorical slots 1-3 of the dataviz reference palette (validated, light mode, all pairs).
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DERM, OUT = Path(sys.argv[1]), Path(sys.argv[3])
CUB_FILES = [Path(p) for p in sys.argv[2].split(",")]
CUB_SPLIT = sys.argv[4] if len(sys.argv) > 4 else "val"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Printed width: 7.16 in (two-column/single-column article) by default; FIG_WIDTH_IN=4.8 gives the LNCS text width
# (12.2 cm) with fonts sized for that width instead of scaling the wide figure down.
W = float(os.environ.get("FIG_WIDTH_IN", "7.16"))
NARROW = W < 6
FS = {"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
      "legend.fontsize": 6.5} if NARROW else {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                                              "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5}
H3 = 1.72 if NARROW else 2.15      # height of the one-row, three-panel figures
plt.rcParams.update({"font.family": "DejaVu Sans", **FS, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": AXIS,
                     "axes.linewidth": 0.8, "pdf.fonttype": 42})


def style(ax):
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0, pad=3)


def intervention_axis(ax, title):
    style(ax)
    ax.set_title(title, loc="left", pad=7)
    ax.set_xlabel("Verified groups (%)" if NARROW else "Verified concept groups (%)", labelpad=5)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlim(-2, 102)


def curve_style(key):
    if key == "D":
        return dict(marker="o", linestyle="-")
    if key == "A_joint":
        return dict(marker="^", linestyle=":")
    return dict(marker="s", linestyle="--")


def finish_figure(fig, ax, ncol, panels):
    if NARROW:
        fig.subplots_adjust(left=0.085, right=0.975, bottom=0.25, top=0.76,
                            wspace=0.42 if panels == 3 else 0.3)
    else:
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.23, top=0.77,
                            wspace=0.35 if panels == 3 else 0.25)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=ncol,
               handlelength=2.2, columnspacing=1.4 if NARROW else 2, bbox_to_anchor=(0.52, 0.995))


# ---------- Derm7pt: intervention curves (test, 6 seeds) ----------
R = json.loads(DERM.read_text())
seeds = list(R)
assert len(seeds) == 6, "The manuscript reports six Derm7pt seeds."
audit = {"Derm7pt": {}, "CUB": {}, "harm_unit": "percent of all cases",
         "x_unit": "percent of verified groups",
         "CUB_uncertainty_band": ("sample standard deviation across %d seeds" % len(CUB_FILES)) if len(CUB_FILES) > 1 else None,
         "Derm7pt_uncertainty_band": "sample standard deviation across six seeds"}
VARIANTS = [("A_joint", "ECBM (1:1:0.01)", BLUE), ("A_author", "ECBM (concept-only)", AQUA), ("D", "GA-ECBM", ORANGE)]
fig, axes = plt.subplots(1, 3, figsize=(W, H3))
for v, label, color in VARIANTS:
    audit["Derm7pt"][v] = {}
    for ax, key in zip(axes, ("f1_macro", "roc_auc", "harm_rate")):
        curves = []
        for s in seeds:
            r = R[s][v]["test"]
            k0 = 0.0 if key == "harm_rate" else r["unintervened"][key]
            curves.append([k0] + [r["intervention_by_k"][str(k)][key] for k in range(1, 8)])
        c = np.array(curves); mu, sd = c.mean(0), c.std(0, ddof=1)
        audit["Derm7pt"][v][key] = {"mean_raw": mu.tolist(), "sd_raw": sd.tolist()}
        x = np.arange(8) / 7 * 100
        scale = 100 if key == "harm_rate" else 1
        ax.fill_between(x, (mu - sd)*scale, (mu + sd)*scale, color=color, alpha=0.14, lw=0)
        ax.plot(x, mu*scale, color=color, lw=1.6, **curve_style(v), ms=3.5, mec="white", mew=0.5, label=label)
for ax, title in zip(axes, ("(a) Macro F1", "(b) ROC-AUC", "(c) Harm (%)")):
    intervention_axis(ax, title)
axes[0].set_ylim(0.70, 0.90)
axes[1].set_ylim(0.80, 0.96)
axes[2].set_ylim(-0.4, 15); axes[2].set_yticks([0, 5, 10, 15])
finish_figure(fig, axes[0], 3, 3)
fig.savefig(OUT / "derm7pt_intervention.pdf"); fig.savefig(OUT / "derm7pt_intervention.png", dpi=200); plt.close(fig)

# ---------- Derm7pt: learned additive table vs 7-point checklist ----------
SEVEN_POINT = {("pigment_network", "atypical"): 2, ("blue_whitish_veil", "present"): 2,
               ("vascular_structures", "linear irregular"): 2, ("vascular_structures", "dotted"): 2,
               ("pigmentation", "diffuse irregular"): 1, ("pigmentation", "localized irregular"): 1,
               ("streaks", "irregular"): 1, ("dots_and_globules", "irregular"): 1,
               ("regression_structures", "blue areas"): 1, ("regression_structures", "white areas"): 1,
               ("regression_structures", "combinations"): 1}
PRETTY = {"pigment_network": "Pigment network", "streaks": "Streaks", "pigmentation": "Pigmentation",
          "regression_structures": "Regression", "dots_and_globules": "Dots/globules", "blue_whitish_veil": "Blue-whitish veil",
          "vascular_structures": "Vascular"}
GROUP_ORDER = ["pigment_network", "streaks", "pigmentation", "regression_structures", "dots_and_globules",
               "blue_whitish_veil", "vascular_structures"]
summary_mapping = json.loads((DERM.parent / "label_mapping.json").read_text())  # Required: never invent state labels.
labels, dtheta, weights = [], [], []
for g, name in enumerate(GROUP_ORDER):
    th = np.mean([np.array(R[s]["D"]["theta"][g]) for s in seeds], axis=0)   # [2, K_g]
    d = th[1] - th[0]; d = d - d[0]
    mapping = summary_mapping[name] if summary_mapping else None
    for i in range(1, th.shape[1]):
        state = ({v: k for k, v in mapping.items()}[i] if mapping else None)
        labels.append(f"{PRETTY[name]}: {state}"); dtheta.append(float(d[i])); weights.append(SEVEN_POINT.get((name, state), 0))
rho = stats.spearmanr(dtheta, weights)
order = np.argsort(dtheta)
W_COL = {2: ORANGE, 1: "#f2b48f", 0: "#b8b8b0"}
fig, ax = plt.subplots(figsize=(W, 2.5 if NARROW else 3.35))
ax.barh(range(len(order)), [dtheta[i] for i in order], height=0.62, color=[W_COL[weights[i]] for i in order])
ax.set_yticks(range(len(order))); ax.set_yticklabels([labels[i] for i in order], fontsize=5.6 if NARROW else 7.5)
ax.axvline(0, color=INK2, lw=0.7)
style(ax); ax.grid(axis="y", visible=False)
ax.set_xlabel(r"$\Delta\theta$ (melanoma $-$ non-melanoma), relative to absent", labelpad=6)
handles = [plt.Rectangle((0, 0), 1, 1, color=W_COL[w]) for w in (2, 1, 0)]
fig.legend(handles, ["Checklist weight 2", "Checklist weight 1", "Not in checklist"], frameon=False,
           loc="upper center", ncol=3, bbox_to_anchor=(0.6 if NARROW else 0.53, 0.995), handlelength=1.4,
           columnspacing=1.4 if NARROW else 2)
fig.subplots_adjust(left=0.43 if NARROW else 0.39, right=0.985, bottom=0.165 if NARROW else 0.16, top=0.9 if NARROW else 0.90)
fig.savefig(OUT / "derm7pt_theta_checklist.pdf"); fig.savefig(OUT / "derm7pt_theta_checklist.png", dpi=200); plt.close(fig)
print(f"Spearman rho = {rho.correlation:.3f}, p = {rho.pvalue:.4f}, n = {len(dtheta)}")

# ---------- CUB: intervention curves ----------
CS = [json.loads(p.read_text()) for p in CUB_FILES]
fig, axes = plt.subplots(1, 3, figsize=(W, H3))
for key, label, color in (("A0", "ECBM (official)", AQUA), ("D", "GA-ECBM", ORANGE)):
    curves = [C[f"{key}_{CUB_SPLIT}"]["curve"] for C in CS]
    x = [c["ratio"] * 100 for c in curves[0]]
    assert all([c["ratio"] for c in cv] == [c["ratio"] for c in curves[0]] for cv in curves)
    audit["CUB"][key] = {}
    for ax, metric in zip(axes, ("class_accuracy", "help_rate", "harm_rate")):
        scale = 1 if metric == "class_accuracy" else 100
        v = np.array([[c[metric] for c in cv] for cv in curves]) * scale
        mu = v.mean(0)
        audit["CUB"][key][metric] = {"mean": mu.tolist(), "per_seed": v.tolist()}
        if len(CS) > 1:
            sd = v.std(0, ddof=1)
            ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.14, lw=0)
        ax.plot(x, mu, color=color, lw=1.6, **curve_style(key), ms=3.5, mec="white", mew=0.5, label=label)
for ax, title in zip(axes, ("(a) Class accuracy", "(b) Help (%)", "(c) Harm (%)")):
    intervention_axis(ax, title)
axes[0].set_ylim(0.60, 1.02); axes[0].set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
axes[1].set_ylim(-0.8, 32); axes[1].set_yticks([0, 10, 20, 30])
axes[2].set_ylim(-0.4, 15); axes[2].set_yticks([0, 5, 10, 15])
finish_figure(fig, axes[0], 2, 3)
fig.savefig(OUT / f"cub_intervention_{CUB_SPLIT}.pdf"); fig.savefig(OUT / f"cub_intervention_{CUB_SPLIT}.png", dpi=200); plt.close(fig)
print("saved to", OUT)

# ---------- CUB: concepts after partial intervention ----------
# These are saved aggregate means, not extra runs or reconstructed error bars.
fig, axes = plt.subplots(1, 2, figsize=(W, H3))
concept_source = {}
for key, label, color in (("A0", "ECBM (official)", AQUA), ("D", "GA-ECBM", ORANGE)):
    rows = CS[0][f"{key}_{CUB_SPLIT}"]["curve"]          # first seed only (figure not used in the paper)
    concept_source[key] = []
    unverified_x, unverified_y = [], []
    for row in rows:
        ratio = row["ratio"]
        if ratio == 0:
            value = row["concept_accuracy"]
        elif ratio < 1:
            value = row["unverified_concept_accuracy"]
        else:
            value = None  # no unverified concepts at full intervention
        concept_source[key].append({"ratio":ratio,"unverified_concept_accuracy":value,
                                    "overall_concept_accuracy":row["overall_concept_accuracy"]})
        if value is not None:
            unverified_x.append(100 * ratio); unverified_y.append(value)
    axes[0].plot(unverified_x, unverified_y, color=color, lw=1.6, **curve_style(key), ms=3.5,
                 mec="white", mew=.5, label=label)
    axes[1].plot([r["ratio"]*100 for r in rows], [r["overall_concept_accuracy"] for r in rows],
                 color=color, lw=1.6, **curve_style(key), ms=3.5, mec="white", mew=.5, label=label)
for ax, title in zip(axes, ("(a) Unverified concept accuracy", "(b) Overall concept accuracy")):
    intervention_axis(ax, title); ax.set_ylabel("Accuracy")
axes[0].set_ylim(.945,.98)
axes[0].set_yticks([.95,.96,.97,.98]);axes[1].set_ylim(0,1.02)
axes[1].set_yticks([0,.25,.5,.75,1.])
finish_figure(fig, axes[0], 2, 2)
fig.savefig(OUT/f"cub_concept_propagation_{CUB_SPLIT}.pdf")
fig.savefig(OUT/f"cub_concept_propagation_{CUB_SPLIT}.png",dpi=220)
plt.close(fig)
(OUT/f"cub_concept_propagation_{CUB_SPLIT}.json").write_text(json.dumps({
    "source":str(CUB_FILES[0]),"split":CUB_SPLIT,"training_seeds":1,
    "intermediate_subset_repeats":5,"full_verification_unverified_metric":None,
    "curves":concept_source},indent=2)+"\n")
audit["theta"] = {"labels": labels, "evidence": dtheta, "checklist_weights": weights,
                  "rho": float(rho.correlation), "p": float(rho.pvalue)}
(OUT/"figure_data_audit.json").write_text(json.dumps(audit, indent=2)+"\n")

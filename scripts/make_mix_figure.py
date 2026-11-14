"""Figure: help-harm trade-off when ECBM's two Derm7pt intervention procedures are interpolated (analysis 2a, revised).

Usage: python3 make_mix_figure.py <mix_valselected.json> <output dir>
Reads the output of ECBM_Derm7pt_kaggle/analysis_2a/mix_valselected.py (test, six seeds, per-k thresholds selected on
validation for every model; the two alphas selected on validation are marked).
Same fonts, colours and legend position as make_figures.py.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# FIG_WIDTH_IN=4.8: LNCS text width, fonts sized for it (see make_figures.py).
W = float(os.environ.get("FIG_WIDTH_IN", "7.16"))
NARROW = W < 6
FS = {"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
      "legend.fontsize": 6.5} if NARROW else {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                                              "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5}
plt.rcParams.update({"font.family": "DejaVu Sans", **FS, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": AXIS,
                     "axes.linewidth": 0.8, "pdf.fonttype": 42})

R = json.loads(SRC.read_text())
alphas = sorted((a for a in R["test"] if a != "ga"), key=float)
mix = {key: [R["test"][a][key][0] for a in alphas] for key in R["test"]["ga"]}
ga = {key: value[0] for key, value in R["test"]["ga"].items()}
x_mix = [h * 100 for h in mix["harm13"]]
x_ga = ga["harm13"] * 100
SELECTED = {str(R["alpha_match"]): "matched harm", str(R["alpha_net"]): "best net"}   # chosen on validation

PANELS = [("(a) Help, k = 1–3", "help13", 100, "Help (% of cases)"),
          ("(b) ROC-AUC, k = 1–3", "auc13", 1, "ROC-AUC"),
          ("(c) ROC-AUC, k = 7", "auc7", 1, "ROC-AUC")]
fig, axes = plt.subplots(1, 3, figsize=(W, 1.95 if NARROW else 2.15))
for ax, (title, key, scale, ylabel) in zip(axes, PANELS):
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0, pad=3)
    ax.set_title(title, loc="left", pad=7)
    ax.set_xlabel("Harm, k = 1–3 (%)" if NARROW else "Harm, k = 1–3 (% of cases)", labelpad=5)
    ax.set_ylabel(ylabel)
    y_mix = [v * scale for v in mix[key]]
    if key == "help13":
        lim = max(max(x_mix), max(y_mix)) * 1.05
        ax.plot([0, lim], [0, lim], color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
        xmax = max(x_mix) * 1.05
        ax.text(xmax * (0.45 if NARROW else 0.6), xmax * 0.33, "help = harm", color=INK2, fontsize=6 if NARROW else 6.5,
                ha="left", va="top")
    ax.plot(x_mix, y_mix, color=MUTED, lw=1.2, marker="o", ms=2.4, mec="white", mew=0.3, zorder=2,
            label="ECBM interpolated (α = 0 … 1)")
    ax.plot(x_mix[0], y_mix[0], color=BLUE, marker="^", ms=6, mec="white", mew=0.6, ls="none", zorder=3,
            label="ECBM 1:1:0.01 (α = 0)")
    ax.plot(x_mix[-1], y_mix[-1], color=AQUA, marker="s", ms=5.5, mec="white", mew=0.6, ls="none", zorder=3,
            label="ECBM concept-only (α = 1)")
    ax.plot(x_ga, ga[key] * scale, color=ORANGE, marker="o", ms=6.5, mec="white", mew=0.6, ls="none", zorder=4,
            label="GA-ECBM")
    sel = [alphas.index(a) for a in SELECTED]
    ax.plot([x_mix[i] for i in sel], [y_mix[i] for i in sel], ls="none", marker="o", ms=6, mfc="none", mec=INK,
            mew=0.9, zorder=3, label="α chosen on validation")
    for a in SELECTED:
        i = alphas.index(a)
        if key == "help13" and a == str(R["alpha_match"]):
            offset = (-27, 4) if NARROW else (-30, 3)      # upper left: clear of the curve and the help = harm line
        else:
            offset = (5, -9)
        ax.annotate(f"α={float(a):g}", (x_mix[i], y_mix[i]), xytext=offset, textcoords="offset points",
                    fontsize=6 if NARROW else 6.5, color=INK2)
    ax.set_xlim(-0.6, max(x_mix) * 1.05)
if NARROW:
    fig.subplots_adjust(left=0.09, right=0.975, bottom=0.2, top=0.7, wspace=0.45)
else:
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.21, top=0.74, wspace=0.35)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3 if NARROW else 5, handlelength=1.6,
           columnspacing=0.9, handletextpad=0.4, bbox_to_anchor=(0.52, 0.995), fontsize=6.2 if NARROW else 6.6)
fig.savefig(OUT / "derm7pt_mix_tradeoff.pdf")
fig.savefig(OUT / "derm7pt_mix_tradeoff.png", dpi=200)
print("saved", OUT / "derm7pt_mix_tradeoff.pdf")

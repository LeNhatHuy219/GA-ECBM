# GA-ECBM: Gated Additive Energy-Based Concept Bottleneck Models for Reliable Concept Interventions

This repo is the official implementation of our paper:

**GA-ECBM: Gated Additive Energy-Based Concept Bottleneck Models for Reliable Concept Interventions**

Huy Nhat-Le, Khanh Quoc Tran-Duong

Faculty of Information Technology, Ton Duc Thang University, Ho Chi Minh City, Vietnam

*Under submission.* [Paper: coming soon]

## Overview of GA-ECBM

<p align="center">
<img src="figures/overview.png" alt="GA-ECBM inference and intervention" width="90%"/>
</p>

Concept bottleneck models let an expert correct the predicted concepts at test time, but a correction can also turn a
correct prediction into a wrong one. In the official implementation of energy-based concept bottleneck models (ECBMs),
the second intervention round predicts the label from the concepts alone, so the image evidence is discarded once a few
concepts have been verified.

GA-ECBM keeps the image and concept energies of ECBM and changes two things:

- **Additive concept–label energy.** The neural concept–label energy is replaced by a table of per-state scores,
  `E_global(c, y) = -(b_y + Σ_j θ_j(y, c_j))`. The label and concept posteriors for any set of verified concepts are then
  exact and closed-form, so no iterative energy minimization is needed.
- **Gated image evidence.** The image–label energy is multiplied by a learned gate `w(k)` that decreases with the
  number `k` of verified concept groups, so the image evidence is down-weighted instead of discarded.

Training adds a marginal-likelihood term, an intervention term and a monotonicity term to the ECBM objective.

## Installation

### Prerequisites

We ran all experiments on one NVIDIA T4 GPU (16 GB) on Kaggle. The notebooks install their own dependencies (CUB uses
PyTorch Lightning 1.6.4, as the official ECBM code). The analysis and figure scripts run on a CPU:

```bash
git clone https://github.com/LeNhatHuy219/GA-ECBM.git && cd GA-ECBM
pip install -r requirements.txt
```

### Dataset Preparation

+ [**CUB-200-2011**](https://data.caltech.edu/records/65de6-vp158)
  We use the train/val/test split and the 112 concepts of vanilla
  [CBMs](https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/README.md). The notebooks download the images and
  the processed split automatically.

+ [**Derm7pt**](http://derm.cs.sfu.ca/)
  Download the dataset from its website and attach `release_v0` (with `images/` and `meta/`) as a Kaggle dataset. We use
  the official split (413/203/395 cases); melanoma is any diagnosis starting with "melanoma". Derm7pt is licensed
  CC BY-NC-ND 4.0, so this repo contains no Derm7pt images, metadata or labels.

## Run Experiments

### 1. CUB-200-2011

GA-ECBM runs inside the official ECBM code ([xmed-lab/ECBM](https://github.com/xmed-lab/ECBM), commit `d38c43a`). For
each seed (42, 123, 2026), run the three notebooks in [`cub/notebooks/seed<seed>/`](cub/notebooks) in order:

1. `ecbm_cub*_kaggle.ipynb`: train ECBM with the official code (about 4 h);
2. `ga_ecbm_cub*_kaggle.ipynb`: train GA-ECBM (about 4 h);
3. `*score_kaggle.ipynb`: attach the outputs of steps 1 and 2 and score both models on the same intervention protocol
   (about 3 h, most of it ECBM's iterative inference).

Each notebook clones the official repository, applies [`cub/official_repo.patch`](cub/official_repo.patch) and writes
the GA-ECBM model ([`cub/ga_ecbm.py`](cub/ga_ecbm.py)) and its LightningModule
([`cub/ga_litmodel.py`](cub/ga_litmodel.py)) into it. We trained for 100 instead of 300 epochs (Kaggle session limit)
and in FP16.

### 2. Derm7pt

Run [`derm7pt/ecbm_derm7pt_paper_kaggle.ipynb`](derm7pt/ecbm_derm7pt_paper_kaggle.ipynb) (about 75 min). It trains
ECBM and GA-ECBM with seeds 42, 123, 2026, 11, 222 and 3333 on a ResNet-101 backbone, then scores the unintervened
models and all 127 subsets of the seven checklist criteria on the validation and test sets.

### 3. Tables and Figures

The outputs of our runs are in [`results/`](results). To recompute the tables and figures of the paper from them:

```bash
# CUB (Table 2)
python scripts/cub_three_seeds.py results/cub/cub_three_seeds.json \
    results/cub/seed42/results.json results/cub/seed123/results.json results/cub/seed2026/results.json

# Derm7pt: first re-attach the labels from your own copy of the dataset
python derm7pt/analysis/attach_labels.py /path/to/release_v0/meta
python derm7pt/analysis/paper_numbers.py /path/to/release_v0/meta    # Tables 3-4 and the Derm7pt numbers in the text
python derm7pt/analysis/mix_valselected.py                            # comparison with interpolated ECBM procedures

# Figures
FIG_WIDTH_IN=4.8 python scripts/make_figures.py results/derm7pt/runs.json \
    results/cub/seed42/results.json,results/cub/seed123/results.json,results/cub/seed2026/results.json figures test
FIG_WIDTH_IN=4.8 python scripts/make_mix_figure.py results/derm7pt/analysis/mix_valselected.json figures
```

`paper_numbers.py` prints each value next to the one in the paper.

## Results

### CUB-200-2011

| Model | Class acc. | Concept acc. | Overall concept acc. | Intervention AUC | Harm (%) |
|---|---|---|---|---|---|
| *Published, test set (5 seeds, 300 epochs)* | | | | | |
| ECBM | 0.812 ± 0.006 | 0.973 ± 0.001 | 0.713 ± 0.009 | – | – |
| *Ours, validation set (3 seeds, 100 epochs)* | | | | | |
| ECBM | 0.723 ± 0.012 | 0.958 ± 0.003 | 0.559 ± 0.013 | 0.838 ± 0.014 | 2.68 ± 0.14 |
| GA-ECBM | 0.740 ± 0.006 | 0.963 ± 0.001 | 0.634 ± 0.008 | 0.876 ± 0.003 | 0.01 ± 0.01 |
| *Ours, test set (3 seeds, 100 epochs)* | | | | | |
| ECBM | 0.738 ± 0.025 | 0.960 ± 0.005 | 0.575 ± 0.029 | 0.844 ± 0.015 | 2.83 ± 0.21 |
| GA-ECBM | 0.760 ± 0.003 | 0.965 ± 0.001 | 0.651 ± 0.006 | **0.886 ± 0.003** | **0.03 ± 0.01** |
| *p*<sub>Holm</sub> (test) | 0.31 | 0.31 | 0.11 | 0.11 | 0.01 |

Overall concept accuracy: all 112 concepts correct. Intervention AUC: normalized area under the class-accuracy curve
from 0% to 100% of the 28 concept groups verified. Harm: share of all images turned from correct to wrong by an
intervention, averaged over 10–100%. Mean ± s.d. over seeds; ECBM uses the official iterative intervention;
*p*<sub>Holm</sub>: paired *t*-test over the three seeds, Holm-adjusted over the five columns. Without intervention,
GA-ECBM was higher in every seed, but no difference is significant after Holm correction.

<p align="center">
<img src="figures/cub_intervention_test.png" alt="CUB intervention" width="85%"/>
</p>

Verifying 10% of the concept groups lowered ECBM's accuracy (0.738 → 0.709) and turned 6.1% of the images from correct
to wrong. GA-ECBM's accuracy rose with every level in every seed, and its harm never exceeded 0.12%.

### Derm7pt

**Without intervention** (test set, mean ± s.d. over six seeds; *p*: paired *t*-test over seeds)

| Metric | ECBM | GA-ECBM | *p* |
|---|---|---|---|
| Accuracy | 0.787 ± 0.018 | 0.790 ± 0.023 | 0.86 |
| Balanced accuracy | 0.737 ± 0.017 | 0.752 ± 0.032 | 0.25 |
| Macro F1 | 0.729 ± 0.015 | 0.738 ± 0.027 | 0.56 |
| Sensitivity | 0.634 ± 0.055 | 0.673 ± 0.081 | 0.27 |
| Specificity | 0.840 ± 0.037 | 0.830 ± 0.038 | 0.74 |
| ROC-AUC | 0.837 ± 0.012 | 0.839 ± 0.019 | 0.87 |
| Concept accuracy | 0.679 ± 0.010 | 0.665 ± 0.018 | 0.19 |
| Concept macro F1 | 0.362 ± 0.036 | 0.362 ± 0.057 | 1.00 |
| All seven concepts correct | 0.072 ± 0.010 | 0.081 ± 0.011 | 0.0501 |

Melanoma is the positive class. Decision thresholds maximize validation macro F1 and are applied unchanged to the test
set. No difference is significant; the test results show no detectable difference, not equivalence.

**Under intervention** (test set, mean ± s.d. over six seeds)

| Metric | ECBM (1:1:0.01) | ECBM (concept-only) | GA-ECBM | *p* | *p*<sub>Holm</sub> |
|---|---|---|---|---|---|
| Macro F1<sub>1–3</sub> | 0.729 ± 0.014 | 0.756 ± 0.011 | 0.765 ± 0.024 | 0.39 | 0.50 |
| ROC-AUC<sub>1–3</sub> | 0.838 ± 0.013 | 0.854 ± 0.012 | 0.869 ± 0.019 | 0.25 | 0.50 |
| Help<sub>1–3</sub> (%) | 0.1 ± 0.2 | 9.0 ± 1.9 | 4.5 ± 1.3 | 0.005 | 0.025 |
| Harm<sub>1–3</sub> (%) | 0.2 ± 0.5 | 6.9 ± 1.6 | 1.9 ± 0.7 | 0.0003 | 0.003 |
| Macro F1<sub>7</sub> | 0.729 ± 0.016 | 0.817 ± 0.012 | **0.849 ± 0.011** | 0.008 | 0.033 |
| ROC-AUC<sub>7</sub> | 0.840 ± 0.014 | 0.917 ± 0.005 | **0.934 ± 0.007** | 0.002 | 0.013 |
| Help<sub>1–7</sub> (%) | 0.3 ± 0.3 | 11.8 ± 1.9 | 8.2 ± 1.4 | 0.022 | 0.067 |
| Harm<sub>1–7</sub> (%) | 0.4 ± 0.9 | 7.7 ± 1.4 | 2.8 ± 0.9 | 0.0006 | 0.005 |

Subscripts: number of verified concepts (1–3: mean over k = 1, 2, 3; 1–7: mean over all k), averaging all subsets of
each size. Help and harm: share of all test cases turned from wrong to correct and from correct to wrong. *p*: paired
*t*-test over seeds, GA-ECBM versus ECBM (concept-only), raw and Holm-adjusted over the eight metrics. The two ECBM
columns share one model and differ in how a correction is propagated: ECBM (1:1:0.01) keeps all three energies with the
weights used for prediction; ECBM (concept-only) predicts from the concepts alone after a correction, like the official
second intervention round.

<p align="center">
<img src="figures/derm7pt_intervention.png" alt="Derm7pt intervention" width="85%"/>
</p>

Without intervention the two models performed alike. Compared with the concept-only procedure, GA-ECBM harmed fewer
cases and reached a higher ROC-AUC with all criteria verified, but it also corrected fewer cases, so the net benefit for
one to three verified criteria was similar (2.66 versus 2.15 points).

<p align="center">
<img src="figures/derm7pt_mix_tradeoff.png" alt="Help-harm trade-off" width="85%"/>
</p>

Interpolating the two ECBM procedures traces a help–harm trade-off (gray). At the same harm, GA-ECBM helped more cases;
at the same net benefit, it harmed about half as many.

## Repository Structure

```
cub/
  ga_ecbm.py, ga_litmodel.py       GA-ECBM for the official ECBM code
  official_repo.patch              our edits to xmed-lab/ECBM
  notebooks/seed{42,123,2026}/     Kaggle notebooks (train ECBM, train GA-ECBM, score)
derm7pt/
  ecbm_derm7pt_paper_kaggle.ipynb  ECBM and GA-ECBM on Derm7pt (6 seeds)
  analysis/                        statistics, interpolation analysis, label re-attachment
scripts/                           CUB aggregation and figures
results/                           metrics, intervention curves, per-case predictions and logs of our runs
figures/                           figures of the paper
```

The notebooks are the exact versions we ran; their comments are in Vietnamese, and
[`cub/README.md`](cub/README.md) and [`derm7pt/README.md`](derm7pt/README.md) describe each section in English. The
Derm7pt per-case predictions in `results/derm7pt/per_case/` contain model outputs only;
`derm7pt/analysis/attach_labels.py` adds the labels from your copy of the dataset. Model checkpoints are not included.

## Reference

The BibTeX entry will be added once the author list and the publication are final.

## Acknowledgement

This code builds on the official [ECBM](https://github.com/xmed-lab/ECBM) implementation (MIT License) and the CUB
preprocessing of [Concept Bottleneck Models](https://github.com/yewsiang/ConceptBottleneck). Our code is released under
the MIT License; see [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

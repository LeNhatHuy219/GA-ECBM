# Derm7pt

`ecbm_derm7pt_paper_kaggle.ipynb` re-implements ECBM for the seven categorical Derm7pt criteria (28 states) and adds
GA-ECBM. Both models use a ResNet-101 backbone at 224 × 224, batch size 32, hidden size 32, SGD and early stopping on the
validation ECBM loss (patience 15, at most 100 epochs). Without intervention, ECBM is evaluated by exact minimization of
its energy over all 8,640 concept profiles.

Section headings in the notebook are in Vietnamese:

| § | Content |
|---|---|
| 1 | Configuration (seeds 42, 123, 2026, 11, 222, 3333; hyperparameters; `EVALUATE_TEST`) |
| 2 | Data: official split (413/203/395 cases), melanoma label (diagnosis starting with "melanoma"), state mapping |
| 3–4 | ECBM model and loss |
| 5 | Exact ECBM inference by enumeration |
| 6 | GA-ECBM: additive scores, gate (initial `a_i = -1.5`, so `w(7) ≈ 0.24`), exact posterior, losses |
| 7 | Shared training loop |
| 8 | Shared scoring: thresholds per number of verified concepts chosen on validation, all 127 intervention subsets, help/harm, paired DeLong test |
| 9 | Train and score every seed; writes `runs.json` and `per_case_seed*.npz` |
| 10 | Tables, paired t-tests, learned scores versus the seven-point checklist; writes `summary.json` |

Procedures: `A_joint` = ECBM with the inference weights 1 : 1 : 0.01; `A_author` = ECBM concept-only (the discrete
analogue of the official second intervention round, weights 0 : 0 : 3); `D` = GA-ECBM.

## Analyses (`analysis/`, CPU)

| Script | Paper | Needs |
|---|---|---|
| `attach_labels.py` | — | your Derm7pt `meta/` folder; run once |
| `paper_numbers.py` | Tables 3–4, Sect. 5.2, Sect. 5.4 (Derm7pt) | labels; `meta/` for the log-odds baseline |
| `mix_valselected.py` | Sect. 5.3, Fig. 4 | labels |
| `mix_tradeoff.py`, `paired.py` | Sect. 5.3, fixed k = 0 threshold | labels |
| `extract_val_scores.py` | validation scores used by `mix_valselected.py` | checkpoints and the Derm7pt images |

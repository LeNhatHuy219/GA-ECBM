# CUB-200-2011

GA-ECBM is added to the official ECBM code (`xmed-lab/ECBM`, commit `d38c43a`) with as few changes as possible:

- `ga_ecbm.py`: `GAECBM` subclasses the official `ECBM`, removes the MLP concept–label energy, and adds the additive
  score table `theta` (200 classes × 112 concepts), the label bias and the 28-step gate (initial `a_i = -3.1`, so
  `w(28) ≈ 0.29`). `posterior()` gives the exact label and concept posteriors for any set of verified concepts.
  `ga_losses()` keeps the official 3 : 1 : 1 loss and adds the marginal, intervention and monotonicity terms.
- `ga_litmodel.py`: `GALitModel` subclasses the official `LitModel` (same optimizer, scheduler and checkpointing).
- `official_repo.patch`: the edits to the official files (see the main README).

To use the modules outside the notebooks: clone the official repository at commit `d38c43a`, `git apply
official_repo.patch`, copy the two modules into the clone, set `CUB_DATA_DIR`, and run `main.py` as in the official
README.

## Notebooks (`notebooks/seed<seed>/`)

Section headings in the notebooks are in Vietnamese; the GA-ECBM training and scoring notebooks share this structure:

| § | Content |
|---|---|
| 1 | Configuration: epochs, seed, gate initialization, intervention levels, decision criteria fixed before the run, `EVALUATE_TEST` |
| 2–3 | Utilities, GPU check |
| 4–5 | Clone the official code at the fixed commit, install PyTorch Lightning 1.6.4 and torchmetrics 0.9.3 |
| 6–7 | Download CUB-200-2011 and the processed splits, check them, place them where the official code reads them |
| 8 | Minimal edits to the official code and the GA-ECBM modules (written into the clone, saved to `patches.diff`) |
| 9 | Numerical checks of the closed-form posterior against brute-force enumeration |
| 10–11 | GPU memory and speed benchmark (chooses FP16), short end-to-end smoke run |
| 12–13 | Train GA-ECBM (resumes from a checkpoint when one is attached; skipped in the scoring notebooks), learning curves |
| 14 | Shared scoring protocol: 0–100% of the 28 groups verified, five random subsets per level, help/harm, unverified-concept accuracy |
| 15–16 | Score GA-ECBM (closed form) and the official ECBM checkpoint (official Adam-based inference) with the same protocol |
| 17 | Comparison with the criteria fixed before the run; writes `results.json`, `summary.json`, `comparison.csv` |

The ECBM notebooks (`ecbm_cub*_kaggle.ipynb`) run the official training (§ 1–12) and, for seed 42, the official
test-set evaluation.

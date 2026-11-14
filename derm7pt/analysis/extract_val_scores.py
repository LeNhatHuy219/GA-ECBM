"""Recompute the validation scores of every intervention subset from the paper checkpoints (validation images only).

The paper notebook saved per-subset scores for the test set but only the k = 0 scores for validation. The
interpolation analysis (mix_valselected.py) selects its mixing weight on validation data, so it needs the validation
scores of all 128 subsets. This script runs the notebook's own cells (configuration, data, both models, exact
inference, shared scoring) unchanged except EVALUATE_TEST = False, so no test image is loaded.

Usage: python derm7pt/analysis/extract_val_scores.py <folder containing Derm7pt/release_v0> <checkpoint folder>
The checkpoint folder holds baseline_seed_{s}.pth (ECBM) and gated_seed_{s}.pth (GA-ECBM), i.e. the folder
ecbm_derm7pt_paper/ of the notebook output; checkpoints are not distributed with this repository.
Writes results/derm7pt/with_labels/val_scores_seed{s}.npz (same content as the released val_scores plus labels) and
checks that the k = 0 scores equal the per-case validation scores saved by the notebook.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELLED, NOTEBOOK, RESULTS, runs  # noqa: E402

data_parent, ckpt_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
os.environ.setdefault("ECBM_WORK", tempfile.mkdtemp(prefix="ga_ecbm_val_"))
os.chdir(data_parent)                         # the notebook searches ./Derm7pt, ./data and . for the dataset
cells = json.loads(NOTEBOOK.read_text())["cells"]
g = {"__name__": "notebook"}
for i in (3, 5, 7, 11, 13, 17):               # config, data, ECBM, exact inference, GA-ECBM, shared scoring
    src = "".join(cells[i]["source"])
    if i == 3:
        assert src.count("EVALUATE_TEST = True") == 1
        src = src.replace("EVALUATE_TEST = True", "EVALUATE_TEST = False")
    exec(compile(src, f"notebook cell {i}", "exec"), g)
assert g["EVALUATE_TEST"] is False
torch, DEVICE = g["torch"], g["DEVICE"]
RUNS = runs()
LABELLED.mkdir(parents=True, exist_ok=True)
for s in g["SEEDS"]:
    _, val_loader, _ = g["get_dataloaders"](s)
    saved = np.load(RESULTS / "per_case" / f"per_case_seed{s}.npz")
    scores = {}
    for tag, make, variants in (("baseline", g["create_model"], ("A_joint", "A_author")), ("gated", g["create_gated_model"], ("D",))):
        model = make()
        model.load_state_dict(torch.load(ckpt_dir / f"{tag}_seed_{s}.pth", map_location=DEVICE))
        model.eval()
        feat, y, c = g["collect"](model, val_loader)
        for v in variants:
            S, C, M = g["score_all"](v, model, feat, c)
            pred, _ = g["predictions"](y, S, S, True)
            res = g["summarize"](y, S, C, pred, c.cpu().numpy(), M)
            ref = RUNS[str(s)][v]["val"]
            print(f"seed {s} {v:8s}: max|k=0 score - saved| {np.abs(S[0] - saved[f'{v}_val_score']).max():.1e} | "
                  f"harm_k1_3 {res['harm_k1_3']:.4f} (runs.json {ref['harm_k1_3']:.4f}) | auc_k7 {res['auc_k7']:.4f} "
                  f"({ref['auc_k7']:.4f})", flush=True)
            scores[v] = S
        del model
    released = np.load(RESULTS / "val_scores" / f"val_scores_seed{s}.npz")
    out = {"ecbm_joint": scores["A_joint"], "ecbm_concept_only": scores["A_author"], "ga_ecbm": scores["D"]}
    print(f"seed {s}: max|recomputed - released val_scores| = "
          f"{max(np.abs(out[k] - released[k]).max() for k in out):.1e}", flush=True)
    np.savez_compressed(LABELLED / f"val_scores_seed{s}.npz", y=np.asarray(y), **out)

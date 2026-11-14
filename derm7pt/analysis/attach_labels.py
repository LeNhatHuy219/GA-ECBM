"""Re-attach the Derm7pt melanoma labels to the released per-case outputs.

The released files in results/derm7pt/per_case and results/derm7pt/val_scores contain model outputs only. Derm7pt is
licensed CC BY-NC-ND 4.0 and its images may not be redistributed, so the ground-truth labels are not included either;
this script rebuilds them from your own copy of the dataset (http://derm.cs.sfu.ca/).

Usage: python derm7pt/analysis/attach_labels.py <path to release_v0/meta>
Label rule (as in the paper notebook): melanoma = diagnosis starting with "melanoma" (90/61/101 in train/val/test).
Case order: rows of meta.csv in the official valid/test split, sorted by row index (= the notebook's data order).
Writes results/derm7pt/with_labels/{per_case_seed*.npz, val_scores_seed*.npz} with the original key names (*_y).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELLED, RESULTS, SEEDS  # noqa: E402

meta_dir = Path(sys.argv[1])
meta = pd.read_csv(meta_dir / "meta.csv")
y_all = meta["diagnosis"].astype(str).str.lower().str.startswith("melanoma").astype(int).values
index = {split: np.sort(pd.read_csv(meta_dir / f)["indexes"].values)
         for split, f in (("train", "train_indexes.csv"), ("val", "valid_indexes.csv"), ("test", "test_indexes.csv"))}
counts = {split: (len(i), int(y_all[i].sum())) for split, i in index.items()}
assert counts == {"train": (413, 90), "val": (203, 61), "test": (395, 101)}, counts
LABELLED.mkdir(parents=True, exist_ok=True)
for s in SEEDS:
    z = dict(np.load(RESULTS / "per_case" / f"per_case_seed{s}.npz"))
    for split in ("val", "test"):
        assert (z.pop(f"{split}_case_index") == index[split]).all(), "case order differs from the official split"
        for model in ("A_joint", "A_author", "D"):
            z[f"{model}_{split}_y"] = y_all[index[split]]
    np.savez_compressed(LABELLED / f"per_case_seed{s}.npz", **z)
    v = dict(np.load(RESULTS / "val_scores" / f"val_scores_seed{s}.npz"))
    assert (v.pop("val_case_index") == index["val"]).all()
    np.savez_compressed(LABELLED / f"val_scores_seed{s}.npz", y=y_all[index["val"]], **v)
print(f"labels attached for {len(SEEDS)} seeds -> {LABELLED}  (cases train/val/test, melanoma: {counts})")

"""Shared paths and helpers for the Derm7pt analyses (no model is trained or evaluated here)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "derm7pt"
LABELLED = RESULTS / "with_labels"        # written by attach_labels.py from your own Derm7pt copy; git-ignored
OUT = RESULTS / "analysis"
NOTEBOOK = ROOT / "derm7pt" / "ecbm_derm7pt_paper_kaggle.ipynb"
SEEDS = ["42", "123", "2026", "11", "222", "3333"]


def runs():
    return json.loads((RESULTS / "runs.json").read_text())


def labelled(name):
    path = LABELLED / name
    if not path.exists():
        raise SystemExit(f"{path} not found. Run first:\n  python derm7pt/analysis/attach_labels.py <Derm7pt release_v0/meta>")
    return path


def notebook_functions():
    """best_threshold and val_halves, executed verbatim from the paper notebook (section 8)."""
    src = "".join(json.loads(NOTEBOOK.read_text())["cells"][17]["source"])
    ns = {}
    exec("import numpy as np\n" + src[src.index("def best_threshold"):src.index("@torch.no_grad()")], ns)
    return ns["best_threshold"], ns["val_halves"]

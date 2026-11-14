"""Check that cub/ga_ecbm.py and cub/ga_litmodel.py are the code that ran on Kaggle.

Every GA-ECBM CUB notebook embeds the module sources as the strings GA_ECBM_SOURCE and GA_LITMODEL_SOURCE and writes
them into the cloned official repository. This script extracts them from all six notebooks, checks that they are
identical across notebooks, and that the English-commented files in cub/ have the same syntax tree once docstrings are
removed (comments are not part of the syntax tree).
Usage: python tests/check_cub_modules.py
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def embedded(notebook):
    out = {}
    for cell in json.loads(notebook.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("GA_ECBM_SOURCE", "GA_LITMODEL_SOURCE"):
                        out[target.id] = node.value.value
    return out


def without_docstrings(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


notebooks = sorted((ROOT / "cub/notebooks").rglob("ga_ecbm_cub*_kaggle.ipynb"))
assert len(notebooks) == 6, notebooks
sources = [embedded(nb) for nb in notebooks]
for name, local in (("GA_ECBM_SOURCE", "cub/ga_ecbm.py"), ("GA_LITMODEL_SOURCE", "cub/ga_litmodel.py")):
    texts = {s[name] for s in sources}
    assert len(texts) == 1, f"{name} differs between notebooks"
    assert without_docstrings(texts.pop()) == without_docstrings((ROOT / local).read_text()), f"{local} differs in code"
    print(f"OK  {local}: same code as the {len(notebooks)} notebooks (docstrings and comments aside)")

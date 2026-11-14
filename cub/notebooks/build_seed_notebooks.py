"""Tạo notebook seed mới cho CUB từ đúng code đã chạy ở seed 42 (Kaggle ehehee/ecbm-cub, ehehee/ga-ecbm-cub).

Chỉ đổi những chỗ bắt buộc:
- random_seed trong configs/cub.json (file này ghi đè tham số dòng lệnh của main.py);
- precision cố định 16 như seed 42 (trên T4 đo bộ nhớ cũng tự chọn 16, dòng này chỉ để chắc chắn);
- ECBM: tắt hai lần chấm test tự động (main.py tự chấm sau train, mục 13 suy luận gradient trên test)
  trừ khi EVALUATE_TEST = True; phần chạy thử dùng ảnh validation thay cho ảnh test;
- GA-ECBM: kiểm tra checkpoint ECBM ở Input đúng seed, tránh gắn nhầm bản seed khác.

Chạy: python3 build_seed_notebooks.py  → ecbm_cub_seed{123,2026}_kaggle.ipynb, ga_ecbm_cub_seed{123,2026}_kaggle.ipynb
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (123, 2026)


def src(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def replace_once(cell, old, new):
    text = src(cell)
    assert text.count(old) == 1, (old[:80], text.count(old))
    set_src(cell, text.replace(old, new))


def find(nb, marker):
    hits = [c for c in nb["cells"] if src(c).lstrip().startswith(marker)]
    assert len(hits) == 1, (marker, len(hits))
    return hits[0]


def wrap_if_test(cell, skipped_message):
    """Chạy nội dung cell chỉ khi EVALUATE_TEST = True."""
    lines = src(cell).splitlines()
    header, body = lines[0], lines[1:]
    assert header.startswith("# =====")
    text = "\n".join([header, "if EVALUATE_TEST:"] + ["    " + l if l.strip() else l for l in body]
                     + ["else:", f"    print({skipped_message!r})"])
    compile(text, "<cell>", "exec")
    set_src(cell, text)


def ecbm_notebook(seed):
    nb = copy.deepcopy(json.load(open(HERE / "source_ecbm-cub.ipynb")))
    title = nb["cells"][0]
    set_src(title, f"# ECBM gốc trên CUB, seed {seed}\n\n"
            f"> **Bản thêm seed (13/11/2026).** Giống hệt notebook `ecbm-cub` (seed 42): cùng code gốc, dữ liệu, "
            f"100 epoch, SGD 0.01, batch 64, FP16. **Chỉ khác:** `random_seed = {seed}`; tắt chấm test "
            f"(`main.py` tự chấm sau khi train, và mục 13–14) trừ khi `EVALUATE_TEST = True`; phần chạy thử dùng ảnh "
            f"validation thay cho ảnh test. Checkpoint `last.ckpt` của notebook này là Input cho "
            f"`ga-ecbm-cub-seed{seed}`.\n\n"
            "**Cài đặt Kaggle:** Accelerator **GPU T4 x2**, Internet **On**, rồi **Save Version → Save & Run All**. "
            "Khoảng 4.5 giờ.\n\n" + src(title).split("\n", 1)[1].lstrip("\n"))

    cfg = find(nb, "# ===== 1. Cấu hình")
    replace_once(cfg, 'EPOCHS = 100   # chạy thử; bài gốc train 300 epoch (configs/cub.json)\n',
                 'EPOCHS = 100   # như seed 42; bài gốc train 300 epoch (configs/cub.json)\n'
                 f'SEED = {seed}     # seed 42 đã chạy ở notebook ecbm-cub\n'
                 'EVALUATE_TEST = False   # chỉ bật khi đã đồng ý chấm tập test\n'
                 'os.environ["ECBM_RUN_TEST"] = "1" if EVALUATE_TEST else "0"   # main.py chỉ tự chấm test khi = 1\n')

    patch = find(nb, "# ===== 8. Sửa tối thiểu")
    replace_once(patch, 'edit_json(REPO / "configs/cub.json", epochs=EPOCHS, enable_progress_bar=False)',
                 'edit_json(REPO / "configs/cub.json", epochs=EPOCHS, enable_progress_bar=False, random_seed=SEED)\n'
                 'replace_once(REPO / "main.py", "    trainer.test(model=model,ckpt_path=None,dataloaders=test_loader)",\n'
                 '             "    if os.environ.get(\'ECBM_RUN_TEST\') == \'1\':\\n'
                 '        trainer.test(model=model,ckpt_path=None,dataloaders=test_loader)")')

    bench = find(nb, "# ===== 9. Đo bộ nhớ GPU và tốc độ")
    replace_once(bench, 'print("Khác cấu hình tác giả:"',
                 'TRAIN_OVERRIDES["precision"] = 16   # cố định như seed 42 (ecbm-cub và ga-ecbm-cub đều chạy FP16)\n'
                 'print("Khác cấu hình tác giả:"')

    smoke = find(nb, "# ===== 10. Chạy thử nhanh")
    replace_once(smoke, "        pickle.dump(SPLITS[split][:size], handle)",
                 "        source = split if (split != \"test\" or EVALUATE_TEST) else \"val\"   # không đụng ảnh test\n"
                 "        pickle.dump(SPLITS[source][:size], handle)")

    train = find(nb, "# ===== 11. Train ECBM")
    replace_once(train, 'print("main.py tự chấm test (đọc trực tiếp năng lượng, chưa phải số so với bài):", DIRECT_TEST)',
                 'print("Global seed trong log:", re.findall(r"Global seed set to \\d+",\n'
                 '      (OUT / "train_stdout.log").read_text(errors="replace"))[-1:])\n'
                 'assert f"random_seed={SEED}" in (TRAIN_DIR / "log.txt").read_text(errors="replace"), "Seed không đúng"\n'
                 'if EVALUATE_TEST:\n'
                 '    print("main.py tự chấm test (đọc trực tiếp năng lượng, chưa phải số so với bài):", DIRECT_TEST)')
    for marker, message in (("# ===== 13. Suy luận gradient", "Bỏ qua mục 13: EVALUATE_TEST = False (không chấm test)."),
                            ("# ===== 14. So sánh với số trong bài", "Bỏ qua mục 14: EVALUATE_TEST = False.")):
        wrap_if_test(find(nb, marker), message)
    last = find(nb, "# ===== 14. So sánh với số trong bài")
    set_src(last, src(last) + "\n    state = load_state()\n    (OUT / \"summary.json\").write_text(json.dumps({\n"
            "        \"seed\": SEED, \"epochs\": EPOCHS, \"train_dir\": state.get(\"train_dir\"),\n"
            "        \"train_hours\": state.get(\"train_hours\"), \"train_overrides\": state.get(\"train_overrides\"),\n"
            "        \"test_evaluated\": False, \"versions\": state.get(\"versions\")}, indent=2, ensure_ascii=False))\n"
            "    print(\"Đã lưu\", OUT / \"summary.json\")")
    compile(src(last), "<cell>", "exec")
    return nb


def ga_notebook(seed):
    nb = copy.deepcopy(json.load(open(HERE / "source_ga-ecbm-cub.ipynb")))
    title = nb["cells"][0]
    body = src(title).split("\n", 1)[1]
    body = body.replace("2. **Add Input → Your Work → `ECBM_CUB`** (bản tái lập đã chạy xong).",
                        f"2. **Add Input → Your Work → `ECBM_CUB seed {seed}`** (notebook `ecbm-cub-seed{seed}` đã chạy "
                        "xong). **Chỉ gắn đúng bản seed này**; mục 12 kiểm tra seed và báo lỗi nếu gắn nhầm. Có thể không gắn "
                        f"nếu sẽ chấm sau bằng notebook `ga-ecbm-cub-seed{seed}-score` (train GA song song với ECBM).")
    assert f"ecbm-cub-seed{seed}" in body
    set_src(title, f"# GA-ECBM trên CUB, seed {seed}\n\n"
            f"> **Bản thêm seed (13/11/2026).** Giống hệt notebook `ga-ecbm-cub` (seed 42). **Chỉ khác:** "
            f"`random_seed = {seed}`; precision cố định 16 như seed 42; kiểm tra checkpoint ECBM ở Input đúng seed {seed}. "
            "Chấm test sau: chạy lại chính notebook này với `EVALUATE_TEST = True` và Input là Output của lần chạy này "
            "(mục 12 thấy checkpoint đủ 100 epoch nên bỏ qua train), giống cách đã làm ở `ga-ecbm-cub-test`.\n"
            + body)

    cfg = find(nb, "# ===== 1. Cấu hình")
    replace_once(cfg, "EPOCHS = 100                    # bằng lần tái lập ECBM gốc\n",
                 "EPOCHS = 100                    # bằng lần tái lập ECBM gốc\n"
                 f"SEED = {seed}                      # seed 42 đã chạy ở notebook ga-ecbm-cub\n")
    replace_once(cfg, '"intervention_repeats": INTERVENTION_REPEATS,',
                 '"intervention_repeats": INTERVENTION_REPEATS, "seed": SEED,')

    patch = find(nb, "# ===== 8. Sửa tối thiểu")
    replace_once(patch, 'edit_json(REPO / "configs/cub.json", epochs=EPOCHS, enable_progress_bar=False)',
                 'edit_json(REPO / "configs/cub.json", epochs=EPOCHS, enable_progress_bar=False, random_seed=SEED)')

    bench = find(nb, "# ===== 10. Đo bộ nhớ GPU và tốc độ")
    replace_once(bench, 'print("Khác cấu hình tác giả:"',
                 'TRAIN_OVERRIDES["precision"] = 16   # cố định như seed 42 (ecbm-cub và ga-ecbm-cub đều chạy FP16)\n'
                 'print("Khác cấu hình tác giả:"')

    train = find(nb, "# ===== 12. Train GA-ECBM")
    replace_once(train, "RESUME = RESUME[0] if RESUME else None\n",
                 "RESUME = RESUME[0] if RESUME else None\n"
                 "# Kiểm tra Input trước khi train vài giờ: checkpoint ECBM gốc phải đúng seed này\n"
                 "_ga_inputs = set(ga_checkpoints(INPUT))\n"
                 "_ecbm_inputs = [p for p in (INPUT.rglob(\"exp/*/last.ckpt\") if INPUT.exists() else []) if p not in _ga_inputs]\n"
                 "_wrong = [str(p) for p in _ecbm_inputs\n"
                 "          if f\"random_seed={SEED}\" not in (p.parent / \"log.txt\").read_text(errors=\"replace\")]\n"
                 "assert not _wrong, f\"Input có checkpoint ECBM không phải seed {SEED}: {_wrong}. Chỉ gắn ecbm-cub-seed{SEED}.\"\n"
                 "print(f\"Checkpoint ECBM seed {SEED} ở Input: {[str(p) for p in _ecbm_inputs] or 'CHƯA GẮN (sẽ không so sánh được)'}\")\n"
                 "if RESUME is not None:\n"
                 "    assert f\"random_seed={SEED}\" in (RESUME.parent / \"log.txt\").read_text(errors=\"replace\"), \\\n"
                 "        f\"Checkpoint GA-ECBM ở Input không phải seed {SEED}: {RESUME}\"\n")
    replace_once(train, '    print(f"Train xong sau {train_seconds / 3600:.2f} giờ. Checkpoint: {GA_CKPT}")',
                 '    print(f"Train xong sau {train_seconds / 3600:.2f} giờ. Checkpoint: {GA_CKPT}")\n'
                 'assert f"random_seed={SEED}" in (GA_CKPT.parent / "log.txt").read_text(errors="replace"), "Seed không đúng"')

    score = find(nb, "# ===== 14. Cách chấm chung")
    replace_once(score, "A0_CKPT = find_author_checkpoint()\n",
                 "A0_CKPT = find_author_checkpoint()\n"
                 "if A0_CKPT is not None:\n"
                 "    assert f\"random_seed={SEED}\" in (A0_CKPT.parent / \"log.txt\").read_text(errors=\"replace\"), \\\n"
                 "        f\"Checkpoint ECBM ở Input không phải seed {SEED}: {A0_CKPT}. Chỉ gắn notebook ecbm-cub-seed{SEED}.\"\n")
    summ = find(nb, "# ===== 17. So sánh và kết luận")
    replace_once(summ, '"epochs": EPOCHS, "seeds": 1,', '"epochs": EPOCHS, "seeds": 1, "seed": SEED,')
    return nb


SCORE_CHECK = '''

# ---- Bản chấm: kiểm tra Input ngay từ đầu, tránh lỡ train lại từ đầu vài giờ ----
if SCORE_ONLY:
    import torch
    _ga = ga_checkpoints(INPUT)
    _ga_done = [p for p in _ga if int(torch.load(p, map_location="cpu", weights_only=False)["epoch"]) + 1 >= EPOCHS
                and f"random_seed={SEED}" in (p.parent / "log.txt").read_text(errors="replace")]
    _ecbm = [p for p in (INPUT.rglob("exp/*/last.ckpt") if INPUT.exists() else []) if p not in set(_ga)]
    _ecbm_ok = [p for p in _ecbm if f"random_seed={SEED}" in (p.parent / "log.txt").read_text(errors="replace")]
    assert _ga_done, f"Chưa gắn Output của ga-ecbm-cub-seed{SEED} (cần checkpoint GA-ECBM seed {SEED} đủ {EPOCHS} epoch)."
    assert _ecbm_ok and len(_ecbm_ok) == len(_ecbm), f"Cần gắn đúng Output của ecbm-cub-seed{SEED}; đang thấy: {_ecbm}"
    print(f"Input đúng: GA-ECBM {_ga_done[0]} | ECBM {_ecbm_ok[0]}")
'''


def score_notebook(seed):
    """Lượt chấm riêng (như ga-ecbm-cub-test của seed 42): không train, chấm GA-ECBM và ECBM cùng seed."""
    nb = ga_notebook(seed)
    title = nb["cells"][0]
    body = src(title).split("\n", 1)[1]
    set_src(title, f"# Chấm GA-ECBM và ECBM gốc trên CUB, seed {seed} (không train lại)\n\n"
            f"> **Lượt chấm (13/11/2026)**, giống cách `ga-ecbm-cub-test` chấm seed 42. Notebook này là "
            f"`ga-ecbm-cub-seed{seed}` với `SCORE_ONLY = True`: mục 12 thấy checkpoint GA-ECBM đủ 100 epoch ở Input nên "
            "bỏ qua train, rồi chấm cả hai mô hình bằng cùng một code.\n>\n"
            f"> **Input (bắt buộc):** Output của `ga-ecbm-cub-seed{seed}`, Output của `ecbm-cub-seed{seed}`, dataset "
            "ảnh CUB (`wenewone/cub2002011`). Thiếu hoặc gắn nhầm seed thì mục 2 báo lỗi ngay, trước khi tốn GPU.\n>\n"
            "> **Tập test:** `EVALUATE_TEST = True` — user đồng ý chấm test CUB seed 123 và 2026 ngày 13/11/2026. "
            "Chấm validation và test cho cả hai mô hình, khoảng 3.2 giờ.\n" + body)
    cfg = find(nb, "# ===== 1. Cấu hình")
    replace_once(cfg, f"SEED = {seed}                      # seed 42 đã chạy ở notebook ga-ecbm-cub\n",
                 f"SEED = {seed}                      # seed 42 đã chạy ở notebook ga-ecbm-cub\n"
                 "SCORE_ONLY = True               # lượt chấm: không train, bắt buộc có checkpoint ở Input\n")
    replace_once(cfg, "EVALUATE_TEST = False           # chỉ bật khi muốn chấm tập test\n",
                 "EVALUATE_TEST = True            # user đồng ý chấm test CUB seed 123 và 2026 ngày 13/11/2026\n")
    util = find(nb, "# ===== 2. Hàm tiện ích")
    set_src(util, src(util).rstrip("\n") + SCORE_CHECK)
    return nb


if __name__ == "__main__":
    for seed in SEEDS:
        for name, nb in ((f"ecbm_cub_seed{seed}_kaggle.ipynb", ecbm_notebook(seed)),
                         (f"ga_ecbm_cub_seed{seed}_kaggle.ipynb", ga_notebook(seed)),
                         (f"ga_ecbm_cub_seed{seed}_score_kaggle.ipynb", score_notebook(seed))):
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    compile(src(cell), name, "exec")
            json.dump(nb, open(HERE / name, "w"), ensure_ascii=False, indent=1)
            print("đã tạo", name)

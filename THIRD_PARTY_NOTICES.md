# Third-party notices

## ECBM (xmed-lab/ECBM)

The CUB experiments run the official ECBM implementation, cloned at commit
`d38c43aeb22a44465aa5344af5a3b7bad72211ae` from <https://github.com/xmed-lab/ECBM>. This repository does not copy that
code; it contains a patch with a few context lines from it (`cub/official_repo.patch`, `results/cub/seed*/patches.diff`)
and modules that subclass its classes (`cub/ga_ecbm.py`, `cub/ga_litmodel.py`). The Derm7pt notebook re-implements the
ECBM design for categorical concepts. ECBM is distributed under the following license:

```
MIT License

Copyright (c) 2024 xmed-lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Reference: X. Xu, Y. Qin, L. Mi, H. Wang, X. Li. Energy-Based Concept Bottleneck Models: Unifying Prediction, Concept
Intervention, and Probabilistic Interpretations. ICLR 2024.

## Datasets (not included)

- CUB-200-2011: C. Wah, S. Branson, P. Welinder, P. Perona, S. Belongie. The Caltech-UCSD Birds-200-2011 Dataset.
  Technical Report CNS-TR-2011-001, Caltech, 2011. Concept annotation and splits: P. W. Koh et al. Concept Bottleneck
  Models. ICML 2020. Downloaded by the notebooks from the original sources.
- Derm7pt: J. Kawahara, S. Daneshvar, G. Argenziano, G. Hamarneh. Seven-Point Checklist and Skin Lesion Classification
  Using Multitask Multimodal Neural Nets. IEEE JBHI 23(2), 2019. Licensed CC BY-NC-ND 4.0; images may not be
  redistributed. No Derm7pt images, metadata or labels are included in this repository.

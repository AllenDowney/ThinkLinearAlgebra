---
jupyter:
  jupytext:
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

*Think Linear Algebra* is not for sale yet, but if you would like to support this project, you can [buy me a coffee](https://buymeacoffee.com/allendowney).

# Trilateration

TODO: intro

<!-- #region tags=["remove-print"] -->
[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ThinkLinearAlgebra/blob/main/chapters/trilateration.ipynb).
<!-- #endregion -->

```python tags=["remove-cell"]
# temporary hack to make autoreload work on Colab
import importlib, sys
sys.modules["imp"] = importlib

%load_ext autoreload
%autoreload 2
```

```python tags=["remove-print", "hide-cell"]
from os.path import basename, exists


def download(url, filename=None):
    if filename is None:
        filename = basename(url)
        
    if not exists(filename):
        from urllib.request import urlretrieve

        local, _ = urlretrieve(url, filename)
        print("Downloaded " + local)


download("https://github.com/AllenDowney/ThinkLinearAlgebra/raw/main/utils.py")
```

```python tags=["hide-cell", "remove-print"]
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

from utils import decorate, underride
```

```python tags=["remove-cell"]
from utils import set_precision

set_precision(4)
```

## Trilateration

```python
import numpy as np

def trilateration_algebraic(beacons, ranges, ref_idx=0):
    """
    Algebraic LS trilateration in 2D.
    beacons: (m,2) array of [xi, yi]
    ranges:  (m,)  array of ri
    ref_idx: index k to use as reference equation
    Returns: x_hat, y_hat
    """
    C = np.asarray(beacons, float)
    r = np.asarray(ranges, float)
    k = ref_idx

    idx = [i for i in range(len(C)) if i != k]
    xi, yi = C[idx, 0], C[idx, 1]
    xk, yk = C[k, 0], C[k, 1]
    ri2, rk2 = r[idx]**2, r[k]**2

    A = np.column_stack([2*(xk - xi), 2*(yk - yi)])
    b = ri2 - rk2 + xk**2 - xi**2 + yk**2 - yi**2

    # LS solve via QR/SVD under the hood
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol[0], sol[1]

def trilateration_refine_geometric(beacons, ranges, p0, steps=5, lam=0.0):
    """
    Few Gauss–Newton / LM refinement steps for geometric LS.
    lam=0 gives Gauss–Newton; lam>0 gives Levenberg–Marquardt damping.
    """
    p = np.array(p0, float)
    C = np.asarray(beacons, float)
    r = np.asarray(ranges, float)

    for _ in range(steps):
        d = np.linalg.norm(p - C, axis=1)
        e = d - r
        # Jacobian m x 2
        J = (p - C) / d[:, None]
        # Solve (J^T J + lam I) Δ = -J^T e
        JTJ = J.T @ J
        g = J.T @ e
        H = JTJ + lam * np.eye(2)
        delta = -np.linalg.solve(H, g)
        p = p + delta
        if np.linalg.norm(delta) < 1e-9:
            break
    return p

```

```python
# Example beacons & noisy ranges
beacons = np.array([[0.,0.], [10.,0.], [0.,10.], [10.,10.]])
true_p = np.array([4.2, 6.5])
ranges = np.linalg.norm(beacons - true_p, axis=1) + 0.2*np.random.randn(4)
```

```python
# Algebraic LS (initializer)
p0 = trilateration_algebraic(beacons, ranges, ref_idx=0)
p0
```

```python
# Geometric refinement (few tiny 2x2 solves)
p_hat = trilateration_refine_geometric(beacons, ranges, p0, steps=6, lam=1e-3)
p_hat
```

<!-- #region tags=["remove-print"] -->
[Think Linear Algebra](https://allendowney.github.io/ThinkLinearAlgebra/index.html)

Copyright 2025 [Allen B. Downey](https://allendowney.com)

Code license: [MIT License](https://mit-license.org/)

Text license: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
<!-- #endregion -->

```python

```

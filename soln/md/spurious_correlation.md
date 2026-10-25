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

*Think Linear Algebra* is not for sale yet, but if you would like to support this project, you can [contribute through Green Tea Press](https://greenteapress.com/support/?book=tla).

# Correlation



<!-- #region tags=["remove-print"] -->
[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ThinkLinearAlgebra/blob/main/nb/track.ipynb).
<!-- #endregion -->

```python tags=["remove-print", "hide-cell"]
%load_ext nb_black
```

```python tags=["remove-print", "hide-cell"]
from os.path import basename, exists


def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve

        local, _ = urlretrieve(url, filename)
        print("Downloaded " + local)


download("https://github.com/AllenDowney/ThinkLinearAlgebra/raw/main/utils.py")
```

```python tags=["hide-cell", "remove-print"]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import decorate
```

## Section

Spurious correlation from Tyler Vigen's Spurious Correlations

https://www.tylervigen.com/spurious/correlation/21184_motor-vehicle-thefts-in-new-mexico_correlates-with_car-crashes-in-the-us

Motor vehicle theft rate per 100,000 residents in New Mexico
Source: FBI Criminal Justice Information Services

Total annual vehicle crash count in the US
Source: Statista

```python
array1 = np.array([346.3,377.9,405.1,504.8,513.2,582.2,717.2,619.9,467.1,403.6,389.8,401.6,384.5,415.2,411.7,472.2,451.9,404,323.5,254.1,259.8,262.2,286.7,301.6,])
array2 = np.array([6117,6000,6106,6496,6699,6770,6624,6335,6279,6394,6323,6316,6328,6181,6159,5973,6024,5811,5505,5419,5338,5615,5687,6064,])
array1_name = "Motor vehicle thefts in New Mexico"
array2_name = "Car crashes in the US"
years = np.arange(1991, 2015)
years


```

```python
def standardize(seq):
    array = np.asarray(seq)
    return (array - array.mean()) / array.std()
```

```python
array1_deviations = standardize(array1)
array2_deviations = standardize(array2)
product = array1_deviations * array2_deviations
np.mean(product)
```

```python

fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True, figsize=(8, 8))

ax1.plot(years, array1_deviations)
ax1.set_ylabel("Array 1 Deviations")

ax2.plot(years, array2_deviations)
ax2.set_ylabel("Array 2 Deviations")

ax3.plot(years, product)
ax3.set_ylabel("Product")
ax3.set_xlabel("Years")

decorate()

```

```python
np.mean(product)
```

```python
(array1_deviations @ array2_deviations) / len(years)
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

<!-- #region tags=["remove-print"] -->
[Think Linear Algebra](https://allendowney.github.io/ThinkLinearAlgebra/index.html)

Copyright 2025 [Allen B. Downey](https://allendowney.com)

Code license: [MIT License](https://mit-license.org/)

Text license: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
<!-- #endregion -->

```python

```

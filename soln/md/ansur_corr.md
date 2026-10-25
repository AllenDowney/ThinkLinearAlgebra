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

# Correlated Body Parts


[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/examples/ansur_corr.ipynb).

```python tags=["hide-cell"]
# Install empirical dist if we don't already have it

try:
    import empiricaldist
except ImportError:
    !pip install empiricaldist
```

```python tags=["hide-cell"]
# download utils.py

from os.path import basename, exists

def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve
        local, _ = urlretrieve(url, filename)
        print('Downloaded ' + local)
        
download("https://github.com/AllenDowney/ProbablyOverthinkingIt/raw/book/notebooks/utils.py")
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## ANSUR

Here's the ANSUR data, originally downloaded from [The OPEN Design Lab](https://www.openlab.psu.edu/ansur2/).

```python
download("https://github.com/AllenDowney/ProbablyOverthinkingIt/raw/book/data/ANSURIIFEMALEPublic.csv")
download("https://github.com/AllenDowney/ProbablyOverthinkingIt/raw/book/data/ANSURIIMALEPublic.csv")
```

```python tags=["remove-cell"]
ansur_female = pd.read_csv("ANSURIIFEMALEPublic.csv", encoding="ISO-8859-1")
ansur_female.shape
```

```python tags=["remove-cell"]
ansur_male = pd.read_csv("ANSURIIMALEPublic.csv", encoding="ISO-8859-1")
ansur_male.shape
```

<!-- #region tags=["remove-cell"] -->
[*The Measurer’s Handbook*](http://tools.openlab.psu.edu/publicData/ANSURII-TR11-017.pdf) used in the ANSUR survey is 253 pages, and includes details on "sampling strategy and measuring techniques" as well as descriptions and diagrams of the landmarks and measurements between them.


<!-- #endregion -->

Since heights are in mm, I converted to cm.

```python tags=["remove-cell"]
height_male = ansur_male["stature"] / 10
height_female = ansur_female["stature"] / 10
```

```python
len(ansur_male)
```

```python
len(ansur_female)
```

## Correlations

Here are the names of the measurements.

```python
df = ansur_male
h = df['earlength']
y = df['earprotrusion']
ratio = 0.8 * y/h
```

```python tags=["remove-cell"]
measurements = [column for column in df.columns if column.lower() == column]
measurements.remove('subjectid')
measurements
```

```python tags=["remove-cell"]
n = len(measurements)
n
```

```python
n * (n-1)
```

```python
corr = df[measurements].corr().replace(1, np.nan)
```

```python

```

```python
corr.loc['functionalleglength', 'sleevelengthspinewrist']
```

```python
corr.loc['sleevelengthspinewrist', 'chestcircumference']
```

```python
corr.stack().sort_values()[-60:-30]
```

```python
sort_by_mean = corr.mean().sort_values()
sort_by_mean
```

```python
sort_by_mag = corr.abs().mean().sort_values()
sort_by_mag
```

```python
from empiricaldist import Cdf

cdf = Cdf.from_seq(corr.abs().mean())
cdf.plot()
```

```python
sort_by_std = corr.std().sort_values()
sort_by_std[:10]
```

```python
color_map = dict(earprotrusion='C0',
                earbreadth='C1',
                tragiontopofhead='C2',
                mentonsellionlength='C3',
                interpupillarybreadth='C4',
                earlength='C5',
                headlength='C6',
                earangle='C9',
                elbowrestheight='C8')
```

```python
from empiricaldist import Cdf
from utils import decorate

for column in sort_by_std.index:
    # print(column)
    if column in color_map:
        options = dict(color=color_map[column], lw=1, label=column)
    else:
        options = dict(color='gray', lw=0.5, alpha=0.3)
    
    cdf = Cdf.from_seq(corr[column].dropna())
    cdf.plot(**options)
    
decorate(xlabel='Correlation', ylabel='CDF')
```

```python
corr['elbowrestheight'].sort_values()
```

```python

# Convert the DataFrame to a NumPy array
array = corr.to_numpy()

# Set elements below the main diagonal to NaN
array[np.tril_indices(len(array))] = np.nan

# Convert the modified NumPy array back to a DataFrame
stacked = pd.DataFrame(array, columns=corr.columns, index=corr.index).stack()

stacked.sort_values()[:30]
```

```python
n = len(df)
n
```

```python
corrs = []
for i in range(1000):
    x = np.random.normal(size=n)
    y = np.random.normal(size=n)
    c = np.corrcoef(x, y)[0][1]
    corrs.append(c)
    

```

```python
corr['earprotrusion'].sort_values().dropna()
```

```python
cdf = Cdf.from_seq(corrs)
cdf.plot(color='gray')

cdf = Cdf.from_seq(corr['earprotrusion'].dropna())
options = dict(color='C0', lw=1, label=column)
cdf.plot(**options)


```

<!-- #region tags=["remove-print"] -->
Probably Overthinking It

Copyright 2022 Allen Downey 

The code in this notebook and `utils.py` is under the [MIT license](https://mit-license.org/).
<!-- #endregion -->

```python tags=["remove-cell"]

```

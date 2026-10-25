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

# Linear Algebra with Penguins


Copyright 2024 Allen B. Downey

License: [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
def decorate(**options):
    """Decorate the current axes.
    
    Call decorate with keyword arguments like
    decorate(title='Title',
             xlabel='x',
             ylabel='y')
             
    The keyword arguments can be any of the axis properties
    https://matplotlib.org/api/axes_api.html
    """
    ax = plt.gca()
    ax.set(**options)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)

    plt.tight_layout()
```

```python
def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    d: dictionary
    options: keyword args to add to d
    """
    for key, val in options.items():
        d.setdefault(key, val)

    return d
```

```python
def plot_vector2d(vector2d, origin=[0, 0], **options):
    """
    
    Modified from: https://github.com/ageron/handson-ml3/blob/main/math_linear_algebra.ipynb
    """
    underride(options, head_width=0.1, head_length=0.15, length_includes_head=True)
    return plt.arrow(origin[0], origin[1], vector2d[0], vector2d[1], **options)
```

```python
# Load the data files from https://github.com/allisonhorst/palmerpenguins
# With gratitude to Allison Horst (@allison_horst)

import os

if not os.path.exists('penguins_raw.csv'):
    !wget https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins_raw.csv
```

The dataset is stored in a CSV file, which contains one row for each penguin and one column for each variable.

I'll use Pandas to read the CSV file and put the results in a `DataFrame`.

```python
df = pd.read_csv('penguins_raw.csv')
df.shape
```

```python
df.head()
```

Three species of penguins are represented in the dataset: Adelie, Chinstrap and Gentoo, as shown in this illustration (by Allison Horst, available under the [CC-BY](https://creativecommons.org/licenses/by/2.0/) license):

<img width="400" src="https://github.com/allisonhorst/palmerpenguins/blob/main/man/figures/lter_penguins.png?raw=true">

In this dataset we are told that there are three species, and we are told which species each penguin belongs to.
But for purposes of clustering, we'll pretend we don't have this information and we'll see whether the algorithm "discovers" the different species.


The measurements we'll use are:

* Body Mass in grams (g).

* Flipper Length in millimeters (mm).

* Culmen Length in millimeters.  

* Culmen Depth in millimeters.

If you are not familiar with the word "culmen", it refers to the [top margin of the beak](https://en.wikipedia.org/wiki/Bird_measurement#Culmen), as shown in the following illustration (also by Allison Horst):

<img width="400" src="https://github.com/allisonhorst/palmerpenguins/blob/main/man/figures/culmen_depth.png?raw=true">

```python
species = 'Chinstrap penguin (Pygoscelis antarctica)'

chinstrap = df.query(f'Species == "{species}"')
chinstrap.head()
```

```python
var1 = 'Flipper Length (mm)'
var2 = 'Culmen Length (mm)'
var3 = 'Culmen Depth (mm)'
var4 = 'Body Mass (g)'
```

```python
A = chinstrap[[var1, var4]].to_numpy()
n, m = A.shape
n, m
```

```python
for (x, y) in A:
    plt.plot(x, y, 'o', color='C0', alpha=0.4)
    
decorate(xlabel=var1, ylabel=var4)
```

```python
lengths, weights = A.transpose()
plt.plot(lengths, weights, 'o', alpha=0.4)
```

```python
def scatter(A):
    xs, ys = A.transpose()
    plt.plot(xs, ys, 'o', alpha=0.4)
```

```python
decorate(xlabel=var1, ylabel=var4)
```

```python
np.corrcoef(lengths, weights)
```

```python
plt.plot(lengths)
decorate(xlabel=var1)
```

```python
plt.plot(weights, color='C1')
decorate(xlabel=var4)
```

```python
means = np.sum(A, axis=0) / n
means
```

```python
deviations = A - means
```

```python
assert np.allclose(np.mean(deviations, axis=0), [0, 0])
```

```python
msd = np.sum(deviations**2, axis=0) / n
```

```python
stds = np.sqrt(msd)
stds
```

```python
assert np.allclose(stds, np.std(A, axis=0))
```

```python
standardized = deviations / stds
standardized.shape
```

```python
standard_lengths, standard_weights = standardized.transpose()
```

```python
plt.plot(standard_lengths, alpha=0.5)
plt.plot(standard_weights, alpha=0.5)
```

```python
product = standard_lengths * standard_weights

plt.plot(product, color='C2')
```

```python
r = np.sum(product) / n
r
```

## Dot Product

```python
np.dot(standard_lengths, standard_weights) / n
```

```python
np.dot(standardized.T, standardized) / n
```

```python
corr = standardized.T @ standardized / n
corr
```

```python
corr = np.corrcoef(A.T)
corr
```

## Covariance matrix

```python
np.dot(deviations.T, deviations) / n
```

```python
np.var(lengths)
```

```python
np.var(weights)
```

```python
np.sum(deviations.prod(axis=1)) / n
```

```python
np.cov(lengths, weights)
```

```python
for i, row in enumerate(deviations.T):
    for j, col in enumerate(deviations.T):
        print(i, j, np.dot(row, col) / n)
```

## Eigenvectors



```python
eigenvalues, eigenvectors = np.linalg.eig(corr)
```

```python
eigenvalues
```

```python
eigenvectors
```

```python
np.cos(np.pi/4)
```

```python
v = [1, 0.2]
res = []

for i in range(5):
    res.append(v)
    v = corr @ v
    
vectors = np.array(res)
vectors.shape
```

```python
for row in vectors:
    plot_vector2d(row, color='C0')
    
lim = [0, 4]
decorate(xlim=lim, ylim=lim)
```

## Random

```python
random = np.random.normal(size=(50, 2))

scatter(random)
```

```python
corr
```

```python
chol = np.linalg.cholesky(corr)
chol.shape
```

```python
correlated = chol @ random.T
correlated.shape
```

```python
np.corrcoef(correlated)
```

```python
scatter(correlated.T)
```

```python
res = []
for i in range(1000):
    random = np.random.normal(size=(50, 2))
    correlated = chol @ random.T
    r = np.corrcoef(correlated)[0][1]
    res.append(r)
    
np.mean(res)
```

## Ovals

```python

```

```python
xs = np.linspace(0, 2*np.pi, 50)
```

```python
P = np.array([np.sin(xs), np.cos(xs)])
P.shape
```

```python
def connect(A):
    xs, ys = A
    plt.plot(xs, ys, '-', alpha=0.4)
```

```python
connect(P)
```

```python
Q = chol @ P
Q.shape
```

```python
connect(P)
connect(Q)
scatter(standardized)
```

```python
cov
```

```python

```

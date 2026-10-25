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

# The Principal Components of Anthropometry

 the ANSUR-II dataset, which includes 93 measurements from a sample of U.S. military personnel


[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ThinkLinearAlgebra/blob/chapters/correlation.ipynb).

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

```python tags=["hide-cell"]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

```python tags=["remove-cell"]
from utils import set_precision

set_precision(4)
```

## The data

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

Since heights are in mm, I converted to cm.

```python tags=["remove-cell"]
height_male = ansur_male["stature"] / 10
height_female = ansur_female["stature"] / 10
```

```python
def add_earratio(df):
    df['earratio'] = df['earprotrusion'] / df['earbreadth']
```

```python tags=["remove-cell"]
measurements = [column for column in ansur_female.columns if column.lower() == column]
measurements
```

```python tags=["remove-cell"]
len(measurements)
```

```python
ansur_male[measurements]
```

```python
df = ansur_male[measurements]
```

## Principle Component Analysis


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In this example, we will first standardize the data and then perform PCA.

# Step 3.1: Create a StandardScaler step
scaler = StandardScaler()

# Step 3.2: Create a PCA step
pca = PCA(n_components=10) 

# Step 4: Create the pipeline
steps = [
    ('scaler', scaler),  # Standardize the data
    ('pca', pca)        # Perform PCA
]

pipeline = Pipeline(steps)

# Fit the pipeline to your data
pipeline.fit(df)

# Now, you can access the transformed data and principal components if needed.
transformed_data = pipeline.transform(df)  # Transformed data after PCA
principal_components = pca.components_    # Principal components (eigenvectors)

# You can also access the explained variance ratio to see how much variance is explained by each component.
explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_ratio
```

Here's a visualization of explained variance versus number of components.
I lifted this from [The Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html).

```python
ys = np.cumsum(pca.explained_variance_ratio_)
xs = np.arange(len(ys)) + 1
ys
```

```python
plt.bar(xs, ys)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

plt.savefig('ansur_pca1.png', dpi=300)
```

With one component, we can capture 44% of the variation in the measurements.
With two components, we're up to 62%.
After that, the gains are smaller (as we expect), but with 10 measurements, we get up to 78%.

## Loadings

Looking at the loadings, we can see which measurements contribute the most to each of the components, so we can get a sense of which characteristics each component captures. 

```python
loadings = pca.components_  # Get the loadings (eigenvectors)

# Create a DataFrame to store the loadings with feature names as columns
loadings_df = pd.DataFrame(loadings, columns=measurements)
loadings_df
```

```python
loadings_transposed = loadings_df.transpose()
loadings_transposed
```

```python
num_top_features = 5

# Iterate through each principal component
for i, col in loadings_transposed.items():
    print(f"Principal Component {i+1}:")
    
    # Sort the features by absolute loading values for the current principal component
    sorted_features = col.sort_values(key=abs, ascending=False)
    
    # Select the top N features with the highest absolute loadings
    top_features = sorted_features[:num_top_features].round(3)
    
    # Print the top correlated factors for the current principal component
    for name, val in top_features.items():
        print(-val, '\t', name)
    print()
```

I won't explain all of the measurements, but if there are any you are curious about, you can look them up in [*The Measurer’s Handbook*](http://tools.openlab.psu.edu/publicData/ANSURII-TR11-017.pdf) which includes details on "sampling strategy and measuring techniques" as well as descriptions and diagrams of the landmarks and measurements between them.

* Not surprisingly, the first component is loaded with measurements of height. If you want to predict someone's measurements, and can only use one number, choose height.

* The second component is loaded with measurements of girth. Again, no surprises so far.

* The third component seems to capture torso length. And that makes sense, too. One you know how tall someone is, it helps to know how that height is split between torso and legs.

* The fourth component seems to capture hand and foot size (with sitting height thrown in just to remind us that PCA is not obligated to find components that align perfectly with the axes we expect).

* Component 5 is all about the shoulders.

* Component 6 is all about the head.

After that, things are not so neat. But two things are worth noting:

* Component 7 is mostly related to the dimensions of the pelvis, but...

* Components 7, 8, and 10 are surprisingly loaded up with ear measurements.

As we saw in the previous article, there seems to be something special about ears.
Once you have exhausted the information carried by the most obvious measurements, the dimensions of the ear seem to be strangely salient.

## Singular Value Decomposition

```python
# --- PCA via SVD (correlation PCA: center & scale first) ---

# 1) Standardize each column (match StandardScaler: ddof=0)
mu = df.mean(axis=0)
sigma = df.std(axis=0, ddof=0)
X_std = (df - mu) / sigma
X_std = X_std.to_numpy()  # (n, p)

n, p = X_std.shape
n_components = 10  # match your sklearn PCA(n_components=10)

# 2) SVD of standardized matrix
#    X_std = U @ diag(S) @ Vt
U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
```

```python

```

```python
# 3) PCA quantities
# Components / loadings (each row is a PC direction in feature space)
components = Vt[:n_components, :]                 # shape (k, p)

# Scores (transformed data): either U*S or X_std @ V
scores = U[:, :n_components] * S[:n_components]   # shape (n, k)

# Explained variance (eigenvalues of correlation matrix)
explained_variance = (S**2) / (n - 1)             # length p (since full_matrices=False, len(S)=min(n,p))
explained_variance = explained_variance[:n_components]

# Explained variance ratio
total_var = (S**2).sum() / (n - 1)
explained_variance_ratio = explained_variance / total_var

# For parity with sklearn names:
transformed_data = scores
principal_components = components

# Quick check of shapes
print("scores:", scores.shape, "components:", components.shape)
print("explained_variance_ratio[:5]:", explained_variance_ratio[:5])
```

```python
# --- Cumulative explained variance plot (like your existing code) ---
ys = np.cumsum(explained_variance_ratio)
xs = np.arange(1, len(ys) + 1)

plt.bar(xs, ys)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.tight_layout()
plt.savefig('ansur_pca_svd_cumvar.png', dpi=300)
```

```python
# --- Loadings DataFrame (same structure you had) ---
loadings_df = pd.DataFrame(principal_components, columns=measurements)
loadings_transposed = loadings_df.transpose()
loadings_df  # display if in a notebook
```

```python
# --- Print top loadings per component (by absolute value) ---
num_top_features = 5
for i, col in loadings_transposed.items():
    print(f"\nPrincipal Component {i+1}:")
    top_features = col.reindex(col.abs().sort_values(ascending=False).index)[:num_top_features].round(3)
    for name, val in top_features.items():
        print(f"{val:+.3f}\t{name}")
```

<!-- #region tags=["remove-print"] -->
[Think Linear Algebra](https://allendowney.github.io/ThinkLinearAlgebra/index.html)

Copyright 2025 [Allen B. Downey](https://allendowney.com)

Code license: [MIT License](https://mit-license.org/)

Text license: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
<!-- #endregion -->

```python

```

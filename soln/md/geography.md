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

# Geography

This file contains leftover bits I might get back to.



<!-- #region tags=["remove-print"] -->
[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ThinkLinearAlgebra/blob/main/nb/track.ipynb).
<!-- #endregion -->

```python
# temporary hack to make autoreload work on Colab
import importlib, sys
sys.modules["imp"] = importlib

%load_ext autoreload
%autoreload 2
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

from utils import decorate, plot_vector, plot_vectors, scatter
```

```python
from utils import set_precision

set_precision(4)
```

## 3D Coordinate Transformations

The simple 2D conversion works well for most applications, but it's also interesting to see how GPS coordinates relate to 3D geometry.
In this section, we'll convert from latitude-longitude-elevation to a local Cartesian coordinate system using vectors from Earth's center.

This approach is more complex than necessary for practical use, but it demonstrates important concepts in linear algebra: coordinate systems, basis vectors, and projection matrices.

First, we need to account for the fact that Earth is not a perfect sphere.
The following function computes the radius of Earth at a given latitude, taking into account that Earth is slightly flattened at the poles.

```python
def earth_radius(lat_deg):
    a = 6378137.0          # equatorial radius (m)
    b = 6356752.3          # polar radius (m)
    lat = np.deg2rad(lat_deg)
    return np.sqrt(
        ((a**2 * np.cos(lat))**2 + (b**2 * np.sin(lat))**2) /
        ((a * np.cos(lat))**2 + (b * np.sin(lat))**2)
    )
```

We'll compute the average radius for our track location.

```python
lat = df['latitude'].mean()
R = earth_radius(lat)
lat, R
```

The result shows the average latitude of our track and the corresponding radius in meters.

Now we'll convert latitude, longitude, and elevation to 3D Cartesian coordinates in an Earth-centered reference frame.
This is a standard transformation used in geodesy.

```python
def latlon_to_cartesian(lat_deg, lon_deg, radius=1.0):
    """
    Convert latitude and longitude (in degrees) to Cartesian coordinates.

    Parameters
    ----------
    lat_deg : array_like
        Geodetic latitude in degrees (north positive).
    lon_deg : array_like
        Longitude in degrees (east positive).
    radius : float or array_like, optional
        Radius of the sphere. Default is 1.0 for unit vectors.

    Returns
    -------
    xyz : ndarray of shape (N, 3)
        Cartesian coordinates [x, y, z] in an Earth-centered, Earth-fixed frame.

    Notes
    -----
    - The coordinate frame is right-handed:
        +x → (lat=0°, lon=0°)
        +y → (lat=0°, lon=90°E)
        +z → (North Pole)
    - This is the standard spherical-to-Cartesian conversion used for
      geocentric coordinates (no ellipsoid flattening applied).
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return sph2cart(lon, lat, radius)
```

We extract the latitude, longitude, and elevation from the DataFrame, and add the elevation to Earth's radius.

```python
lat, lon, elev = df[["latitude", "longitude", "elevation"]].values.T
R + elev
```

Now we can convert all the GPS points to 3D Cartesian coordinates.
Each point is represented as a vector from the center of Earth.

```python
r = latlon_to_cartesian(lat, lon, R + elev)
```

We'll compute the mean position, which gives us a reference point near the center of our track.

```python
r0 = r.mean(axis=0)
r0
```

And we can normalize this vector to get a unit vector in the radial direction.

```python
from utils import norm, normalize

r_hat = normalize(r0)
r_hat
```

## Creating a Local Coordinate System

For our analysis, we need a local coordinate system with axes that are East, North, and Up relative to our location on Earth.
We'll use the cross product to create these basis vectors.

The cross product of two vectors produces a third vector that is perpendicular to both.
We'll start by crossing the global "up" direction (north pole) with our radial vector to get an "east" direction.

```python
# local basis via vector operations
z_hat = np.array([0, 0, 1])
e = np.cross(z_hat, r0)
e_hat = normalize(e)
norm(e_hat)
```

Now we can use another cross product to get the "north" direction, which is perpendicular to both the radial and east directions.

```python
n_hat = np.cross(r_hat, e_hat)
norm(n_hat)
```

The result confirms that we have three orthonormal basis vectors: east (`e_hat`), north (`n_hat`), and radial (`r_hat`).

To visualize these coordinate systems, we'll draw a 3D globe with our local basis vectors.

```python
import numpy as np

def draw_globe(ax, R=3.0, n_lat=9, n_lon=12, alpha=0.3):
    """Draw a globe of radius R with lines of latitude and longitude.
    """
    # ---- base surface ----
    lat_grid = np.linspace(-90, 90, 60)
    lon_grid = np.linspace(-180, 180, 120)
    lat, lon = np.meshgrid(lat_grid, lon_grid)
    xyz = latlon_to_cartesian(lat.ravel(), lon.ravel(), radius=np.full(lat.size, R))
    X, Y, Z = [a.reshape(lat.shape) for a in xyz.T]
    ax.plot_surface(X, Y, Z, color='C0', alpha=0.04, linewidth=0)

    # ---- latitude circles ----
    lats = np.linspace(-60, 60, n_lat)
    lons = np.linspace(-180, 180, 361)
    for lat in lats:
        xyz = latlon_to_cartesian(np.full_like(lons, lat), lons, radius=np.full_like(lons, R))
        ax.plot(*xyz.T, color='k', linewidth=0.5, alpha=alpha)

    # ---- longitude lines ----
    lons = np.linspace(-180, 180, n_lon, endpoint=False)
    lats = np.linspace(-90, 90, 181)
    for lon in lons:
        xyz = latlon_to_cartesian(lats, np.full_like(lats, lon), radius=np.full_like(lats, R))
        ax.plot(*xyz.T, color='k', linewidth=0.5, alpha=alpha)

```

```python
from utils import setup_3D


def globe_diagram(R):
    fig, [ax] = setup_3D(show_grid=False)
    ax.view_init(elev=20, azim=-45, roll=0)

    draw_globe(ax, R)

    lim = [-6, 6]
    decorate(xlim=lim, ylim=lim, zlim=lim, 
             xlabel='i', ylabel='j', zlabel='k', 
             aspect='equal')
```

For visualization, we'll scale down the radius by a factor of a million so the vectors fit on the same scale as the globe.

```python
r0_scaled = r0 / 1e6
R_scaled = norm(r0_scaled)
z_scaled = z_hat * R_scaled
```

Here's a 3D diagram showing the reference position vector and the plane used to compute the east direction.

```python
from utils import plot_vector_3D, plot_vectors_3D, label_vectors_3D, plot_plane

globe_diagram(R_scaled)

plot_vector_3D(r0_scaled, label='$r0$', color='C1', alpha=1)

plot_vector_3D(z_scaled, label='$\hat{z}$', color='C1', alpha=1)

plot_plane(r0_scaled, z_scaled, alpha=0.2)

plot_vector_3D(e_hat, label='$\hat{e}$', scale=5, alpha=1)
```

And here's a diagram showing all three basis vectors of our local coordinate system.

```python
globe_diagram(R_scaled)

plot_vectors_3D([r0_scaled, z_scaled], color='C1', alpha=1)

origins = [r0_scaled] * 3
plot_vectors_3D([r_hat, e_hat, n_hat], origins, scale=5, alpha=1)
label_vectors_3D(['$\hat{r}$', '$\hat{e}$', '$\hat{n}$'], 
                 [r_hat, e_hat, n_hat], origins, scale=5)
```

## Projecting to Local Coordinates

Now that we have a local coordinate system, we can transform all of our GPS points into this system.
We'll use a projection matrix whose columns are our basis vectors.

When we multiply a vector by this matrix, it computes the projections onto each basis vector, giving us coordinates in meters relative to our reference point.

```python
# projection matrix
T = np.column_stack([e_hat, n_hat, r0])

# local coords (in meters)
d = r - r0
enr = d @ T
df["east_m"], df["north_m"], df["up_m"] = enr.T
```

Now we can plot the track in our local coordinate system, with axes in meters.

```python
plt.plot(df["east_m"], df["north_m"], marker=".", lw=0)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("GPS Track")
plt.axis("equal");
```

This plot looks similar to the lat-lon plot, but now the coordinates are in meters and represent actual distances.

We can verify this by computing the distance from start to finish in the local coordinate system.

```python
cols = ["east_m", "north_m"]
start = df.loc[0, cols]
end = df.loc[df.index[-1], cols]
v = end - start
np.linalg.norm(v)
```

This should match the haversine distance we computed earlier.





## OpenSky

```python

```

```python
import os
from traffic.data import opensky

opensky.username = os.environ["OPENSKY_USERNAME"]
opensky.password = os.environ["OPENSKY_PASSWORD"]

```

```python
from traffic.data import airports
from traffic.core import Flight
import pandas as pd

bos = airports["KBOS"]
dub = airports["EIDW"]

# Query flights on a given date
flights = opensky.history(
    start="2025-06-01 00:00",
    stop="2025-06-02 00:00",
    departure_airport=bos,
    arrival_airport=dub,
)

flights

```

```python

```

<!-- #region tags=["remove-print"] -->
[Think Linear Algebra](https://allendowney.github.io/ThinkLinearAlgebra/index.html)

Copyright 2025 [Allen B. Downey](https://allendowney.com)

Code license: [MIT License](https://mit-license.org/)

Text license: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
<!-- #endregion -->

"""This file contains code for use with "DataWorld"
by Allen B. Downey, available from greenteapress.com

Copyright 2024 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html

"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from numpy.linalg import norm

# Make the figures smaller to save some screen real estate.
# The figures generated for the book have DPI 400, so scaling
# them by a factor of 4 restores them to the size in the notebooks.
plt.rcParams['figure.dpi'] = 75
plt.rcParams['figure.figsize'] = [6, 3.5]

def remove_spines():
    """Remove the spines of a plot but keep the ticks visible."""
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Ensure ticks stay visible
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def value_counts(series, **options):
    """Counts the values in a series and returns sorted.

    series: pd.Series

    returns: pd.Series
    """
    options = underride(options, dropna=False)
    return series.value_counts(**options).sort_index()


def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    d: dictionary
    options: keyword args to add to d
    """
    for key, val in options.items():
        d.setdefault(key, val)

    return d


def decorate(**options):
    """Decorate the current axes.
    Call decorate with keyword arguments like
    decorate(title='Title',
             xlabel='x',
             ylabel='y')
    The keyword arguments can be any of the axis properties
    https://matplotlib.org/api/axes_api.html
    In addition, you can use `legend=False` to suppress the legend.
    And you can use `loc` to indicate the location of the legend
    (the default value is 'best')
    """
    loc = options.pop("loc", "best")
    if options.pop("legend", True):
        legend(loc=loc)

    plt.gca().set(**options)
    plt.tight_layout()


def legend(**options):
    """Draws a legend only if there is at least one labeled item.
    options are passed to plt.legend()
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
    """
    underride(options, loc="best")

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, **options)


def polar_to_cartesian(r, theta):
    """Convert polar coordinates (r, theta) to Cartesian (x, y)."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def cartesian_to_polar(x, y):
    """Convert Cartesian coordinates (x, y) to polar (r, theta)."""
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return r, theta

def cartesian_to_complex(x, y):
    """Convert Cartesian coordinates (x, y) to complex plane representation."""
    return x + 1j * y

def polar_to_complex(r, theta):
    """Convert polar coordinates (r, theta) to complex plane representation."""
    return r * np.exp(1j * theta)


def plot_vectors(vectors, origin=None, start=0, end=None, labels=None, label_pos=None, **options):
    """Plot a set of vectors.

    Args:
        vectors: list of vectors or array with one row per vector
        origin: list of vectors or array with one row per vector
        start: integer slice index
        end: integer slice index
        labels: list of string labels
        label_pos: list of locations as integer clock positions
        options: passed to plt.quiver

    """
    scale = options.pop("scale", 1)
    vectors = np.asarray(vectors) * scale

    underride(
        options,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="C0",
        alpha=0.6,
    )
    if origin is None:
        origin = np.zeros_like(vectors)
    else:
        origin = np.asarray(origin)

    if labels is not None:
        if label_pos is None:
            label_pos = [12] * len(vectors)
        label_vectors(vectors, origin, labels, label_pos)

    us, vs = vectors[start:end].transpose()
    xs, ys = origin[start:end].transpose()

    # draw invisible points at the heads and tails of the vectors
    plt.scatter(xs, ys, s=0)
    plt.scatter(xs + us, ys + vs, s=0)
    return plt.quiver(xs, ys, us, vs, **options)


def plot_vector(v, origin=None, *args, **options):
    """Draw a single vector.

    Args:
        v: array-like
        origin: array-like
        options: passed to plt.quiver
    """
    if origin is None:
        origin = np.zeros_like(v)
    else:
        origin = np.asarray(origin)
    return plot_vectors([v], [origin], *args, **options)


def label_vectors(vectors, origins, labels, label_pos):
    """Label the vectors with a string at a given position.

    Args:
        vectors: list of vectors or array with one row per vector
        origins: list of vectors or array with one row per vector
        labels: list of string labels
        label_pos: list of locations as integer clock positions
    """
    for vector, origin, label, pos in zip(vectors, origins, labels, label_pos):
        label_vector(vector, origin, label, pos)


def label_vector(vector, origin, label, pos, offset=4):
    """Label a vector with a string at a given position.

    Args:
        vector: array-like
        origin: array-like
        label: string
        pos: integer clock position
        offset: offset of the label from the vector
    """
    u = normalize(vector)
    v = vector_perp(u)

    pos %= 12
    mag_u = np.array([4, 4, 3, 2, 1, 0, 0, 0, 1, 2, 3, 4])[pos] * norm(vector) / 4
    offset_u = np.array([1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0])[pos] * offset
    offset_v = np.array([0, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1])[pos] * offset

    x, y = origin + u * (mag_u + offset_u) + v * offset_v
    plt.text(x, y, label, fontsize=12, ha="center", va="center")


def scatter(vectors, start=0, end=None, **options):
    """Plot a set of vectors.

    Args:
        vectors: list of vectors or array with one row per vector
        start: integer slice index
        end: integer slice index
        options: passed to plt.scatter
    """
    underride(options, s=6)
    xs, ys = vectors[start:end].transpose()
    plt.scatter(xs, ys, **options)


def cartesian_product(arrays):
    """Compute the cartesian product of a list of arrays.

    From: https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points

    Args:
        arrays: sequence of sequences

    returns: array of shape (n, len(arrays))
    """
    la = len(arrays)
    arrays = [np.asarray(a) for a in arrays]
    dtype = np.result_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def cartesian_product(arrays):
    """Compute the Cartesian product of a list of arrays.

    Args:
        arrays (list of array-like): A sequence of 1D sequences.

    Returns:
        numpy.ndarray: A 2D array of shape (n, len(arrays)), where `n` is the 
        total number of combinations, and `len(arrays)` is the number of input arrays.
    """
    # Ensure all inputs are NumPy arrays
    arrays = [np.asarray(a) for a in arrays]

    # Determine the output dtype
    dtype = np.result_type(*[a.dtype for a in arrays])

    # Initialize an empty array to hold the result
    num_arrays = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [num_arrays], dtype=dtype)

    # Fill the array using np.ix_
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a

    # Reshape the array to have shape (n, len(arrays))
    return arr.reshape(-1, num_arrays)


def normalize(v):
    """Normalize a vector.
    
    Args:
        v: NumPy array
        
    Returns: NumPy array  
    """
    return v / norm(v)


def vector_perp(v):
    """Compute the vector perpendicular to a given vector.

    Args:
        v: NumPy array

    Returns: NumPy array
    """
    x, y = v
    perp = np.array([-y, x])
    return normalize(perp)
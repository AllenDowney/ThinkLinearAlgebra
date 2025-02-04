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
plt.rcParams["figure.dpi"] = 75
plt.rcParams["figure.figsize"] = [6, 3.5]


def remove_spines():
    """Remove the spines of a plot but keep the ticks visible."""
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Ensure ticks stay visible
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")


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


def plot_vectors(
    vectors, origin=None, start=0, end=None, labels=None, label_pos=None, **options
):
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


def plot_vector(v, origin=None, label=None, *args, **options):
    """Draw a single vector.

    Args:
        v: array-like
        origin: array-like
        label: string
        options: passed to plt.quiver
    """
    if origin is None:
        origin = np.zeros_like(v)
    else:
        origin = np.asarray(origin)
    return plot_vectors([v], [origin], labels=[label], *args, **options)


def label_vectors(vectors, origins, labels, label_positions=None, **options):
    """Label the vectors with a string at a given position.

    Args:
        vectors: list of vectors or array with one row per vector
        origins: list of vectors or array with one row per vector
        labels: list of string labels
        label_positions: list of locations as integer clock positions
    """
    if label_positions is None:
        label_positions = np.full_like(labels, 12)

    for vector, origin, label, label_pos in zip(
        vectors, origins, labels, label_positions
    ):
        label_vector(vector, origin, label, label_pos, **options)


def label_vector(vector, origin, label, label_pos=12, offset=0.1, **options):
    """Label a vector with a string at a given position.

    Args:
        vector: array-like
        origin: array-like
        label: string
        label_pos: integer clock position
        offset: offset of the label from the vector
    """
    v_mag = norm(vector)
    v_hat = normalize(vector)
    w_hat = vector_perp(v_hat)

    i = label_pos % 12
    v = np.array([4, 4, 3, 2, 1, 0, 0, 0, 1, 2, 3, 4])[i] / 4 * vector

    offset_v = np.array([1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0])[i] * offset * v_mag
    offset_w = np.array([0, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1])[i] * offset * v_mag

    x, y = origin + v + v_hat * offset_v + w_hat * offset_w

    underride(options, ha="center", va="center")
    plt.text(x, y, label, **options)


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
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def vector_perp(v):
    """Compute a vector perpendicular to a given nD vector.

    Args:
        v: NumPy array (shape: (n,))

    Returns:
        NumPy array (a unit vector perpendicular to v)
    """
    v = np.asarray(v)
    n = len(v)
    if np.allclose(v, np.zeros(n)):
        raise ValueError("Zero vector has no perpendicular vector.")

    if n == 2:
        # 2D case: Rotate by 90 degrees
        return normalize(np.array([-v[1], v[0]]))

    elif n == 3:
        # 3D case: Find any perpendicular vector using cross product

        # Pick an arbitrary non-parallel vector
        if np.allclose(v[0:2], [0, 0]):
            perp = np.array([1, 0, 0])  # Use X-axis if v is along Z
        else:
            perp = np.array([0, 0, 1])  # Otherwise, use Z-axis

        return normalize(np.cross(v, perp))

    else:
        # nD case: Use vector projection to remove parallel component

        # Generate a random vector and ensure it's not collinear
        random_vec = np.random.randn(n)
        while np.allclose(np.dot(v, random_vec), 0):
            random_vec = np.random.randn(n)

        # Compute perpendicular component using vector projection
        projection = (np.dot(random_vec, v) / np.dot(v, v)) * v
        perp = random_vec - projection

        return normalize(perp)

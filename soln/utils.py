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



## Linear Algebra helper functions

def normalize(v):
    """Normalize a vector.

    Args:
        v: NumPy array

    Returns: NumPy array
    """
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def scalar_projection(a, b):
    """Compute the scalar projection of vector a onto vector b.
    
    Args:
        a: NumPy array
        b: NumPy array

    Returns: float
    """
    return np.dot(a, b) / norm(b)

def vector_projection(a, b):
    """Compute the vector projection of a onto b.
    
    Args:
        a: NumPy array
        b: NumPy array

    Returns: NumPy array
    """
    return (np.dot(a, b) / np.dot(b, b)) * b

def vector_rejection(a, b):
    """Compute the component of a that is perpendicular to b.
    
    Args:
        a: NumPy array
        b: NumPy array

    Returns: NumPy array
    """
    return a - vector_projection(a, b)

def angle_between(a, b, degrees=True):
    """Compute the angle between two vectors.
    
    Args:
        a: NumPy array
        b: NumPy array
        degrees: bool, whether to return the angle in degrees
    """
    cos_theta = np.dot(a, b) / (norm(a) * norm(b))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to handle numerical errors
    return np.degrees(angle) if degrees else angle

def rotate_2d(v, angle, degrees=True):
    """Rotate a 2D vector by a given angle.
    
    Args:
        v: NumPy array
        angle: float
        degrees: bool, whether the angle is in degrees

    Returns: NumPy array
    """
    if degrees:
        angle = np.radians(angle)
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
                           [np.sin(angle),  np.cos(angle)]])
    return rot_matrix @ v

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
    

def gram_schmidt(vectors):
    """Perform Gram-Schmidt orthogonalization on a set of vectors.
    
    Args:
        vectors: list of vectors or array with one row per vector

    Returns: NumPy array
    """
    # Convert input vectors to float arrays
    vectors = [np.asarray(v, dtype=float) for v in vectors]

    basis = []
    for v in vectors:
        for b in basis:
            v -= vector_projection(v, b)
        basis.append(normalize(v))
    return np.array(basis)


def polar_to_cartesian(r, theta):
    """Convert polar coordinates (r, theta) to Cartesian (x, y).
    
    Args:
        r: float
        theta: float angle in radians

    Returns: tuple of floats
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def cartesian_to_polar(x, y):
    """Convert Cartesian coordinates (x, y) to polar (r, theta).
    
    Args:
        x: float
        y: float

    Returns: tuple of floats
    """
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return r, theta


def cartesian_to_complex(x, y):
    """Convert Cartesian coordinates (x, y) to complex plane representation.
    
    Args:
        x: float
        y: float

    Returns: complex number
    """
    return x + 1j * y


def polar_to_complex(r, theta):
    """Convert polar coordinates (r, theta) to complex plane representation.
    
    Args:
        r: float
        theta: float angle in radians

    Returns: complex number
    """
    return r * np.exp(1j * theta)

def cartesian_product(arrays):
    """Compute the Cartesian product of a list of arrays.

    Modified from: https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points

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






## Visualization helper functions

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


## Vector plotting functions

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


def plot_vector(v, origin=None, label=None, label_pos=12, **options):
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
    return plot_vectors([v], [origin], labels=[label], label_pos=[label_pos], **options)


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


def plot_rejection(a, b, **kwargs):
    """Draw the rejection of vector a from b (the component of a perpendicular to b).

    Args:
        a: vector being projected (e.g., displacement vector)
        b: direction vector (e.g., truss member)
        kwargs: passed to plt.plot (e.g., color, linestyle)
    """
    underride(kwargs, color='gray', linestyle=':')
    rej = vector_rejection(a, b)
    start = a - rej
    end = a
    plt.plot([start[0], end[0]], [start[1], end[1]], **kwargs)


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


import matplotlib.lines as mlines
import matplotlib.patches as mpatches

def diagram_truss(nodes, u=None, add_pin=True, add_labels=True, add_vectors=True):
    """Plot a truss diagram.

    Args:
        nodes: list of nodes
        u: vector as np.array
        add_pin: bool, whether to add a pin at C and triangles at the anchors
        add_labels: bool, whether to add labels to the nodes
        add_vectors: bool, whether to add vectors to the truss members
    """
    # Get current axis
    ax = plt.gca()
    ax.set_aspect('equal')
    remove_spines()
    
    # Calculate the vectors from C to A and C to B
    A, B, C = nodes
    r_CA = A - C
    r_CB = B - C

    # Calculate the lengths of the truss members
    L_CA = norm(r_CA)
    L_CB = norm(r_CB)

    # Options for the truss members
    line_options = dict(
        linewidth=20,
        solid_capstyle='round',
        color='gray',
        alpha=0.1,
        transform=ax.transData
    )
    
    anchor_width = 0.1 * L_CA
    anchor_height = 0.1 * L_CA

    for start in [A, B]:
        # Draw truss member
        line = mlines.Line2D([start[0], C[0]], [start[1], C[1]], **line_options)
        ax.add_artist(line)

        # Draw upward-pointing triangle with tip at base point
        if add_pin:
            x, y = start
            triangle = mpatches.Polygon([
                [x - anchor_width / 2, y - anchor_height],
                [x + anchor_width / 2, y - anchor_height],
                [x, y]
            ], closed=True, color='black')
            ax.add_artist(triangle)

    if add_pin:
        # Draw a circle at C to represent a pin
        pin_radius = 0.05 * L_CA
        pin_options = dict(facecolor='black', edgecolor='none')
        pin = mpatches.Circle(C, pin_radius, **pin_options)
        ax.add_artist(pin)

    if add_labels:
        text_options = dict(fontsize=12, ha='center')
        offset = 0.1 * L_CA
        ax.text(A[0] - offset, A[1] + offset, 'A', **text_options)
        ax.text(B[0] + offset, B[1] + offset, 'B', **text_options)
        ax.text(C[0], C[1] + offset, 'C', **text_options)

    if add_vectors:
        if add_labels:
            vector_options = dict(labels=['$r_{CA}$', '$r_{CB}$'], 
                     label_pos=[1, -1])
        else:
            vector_options = dict()
        plot_vectors([r_CA, r_CB], [C, C], **vector_options)

    if u is not None:
        plot_vector(u, nodes[2], label='u', color='C1')
        u_CA = vector_projection(u, r_CA)
        u_CB = vector_projection(u, r_CB)
        plot_vector(u_CA, nodes[2], color='C4', label='$u_{CA}$', label_pos=1)
        plot_vector(u_CB, nodes[2], color='C4', label='$u_{CB}$', label_pos=-1)

        plot_rejection(u, r_CA)
        plot_rejection(u, r_CB)


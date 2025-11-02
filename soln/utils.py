"""This file contains code for use with "DataWorld"
by Allen B. Downey, available from greenteapress.com

Copyright 2024 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
from mpl_toolkits.mplot3d import Axes3D

from numpy.linalg import norm
import networkx as nx
import sympy as sp

# Make the figures smaller to save some screen real estate.
# The figures generated for the book have DPI 400, so scaling
# them by a factor of 4 restores them to the size in the notebooks.
plt.rcParams["figure.dpi"] = 75
plt.rcParams["figure.figsize"] = [6, 3.5]

def set_precision(precision=3, legacy='1.25'):
    from IPython import get_ipython

    # Register a pretty printer for floats
    def float_printer(x, p, cycle):
        p.text(f"{x:.{precision}f}")

    formatters = get_ipython().display_formatter.formatters['text/plain']
    formatters.for_type(float, float_printer)
    formatters.for_type(np.float64, float_printer)

    # Set display precision for NumPy and Pandas
    np.set_printoptions(precision=precision, legacy=legacy)
    pd.options.display.float_format = (f"{{:.{precision}f}}").format


def value_counts(seq, **options):
    """Make a series of values and the number of times they appear.

    Returns a DataFrame because they get rendered better in Jupyter.

    Args:
        seq: sequence
        options: passed to pd.Series.value_counts

    returns: pd.Series
    """
    options = underride(options, dropna=False)
    series = pd.Series(seq).value_counts(**options).sort_index()
    series.index.name = "values"
    series.name = "counts"
    return pd.DataFrame(series)


## Linear Algebra helper functions

def normalize(v):
    """Normalize a vector.

    Args:
        v: NumPy array

    Returns: NumPy array
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def scalar_projection(a, b):
    """Compute the scalar projection of vector a onto vector b.
    
    Args:
        a: NumPy array
        b: NumPy array

    Returns: float
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.dot(a, b) / norm(b)

def vector_projection(a, b):
    """Compute the vector projection of a onto b.
    
    Args:
        a: NumPy array
        b: NumPy array

    Returns: NumPy array
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return (np.dot(a, b) / np.dot(b, b)) * b

def vector_rejection(a, b):
    """Compute the component of a that is perpendicular to b.
    
    Args:
        a: NumPy array
        b: NumPy array

    Returns: NumPy array
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a - vector_projection(a, b)

def cross2d(x, y):
    """Compute the 2D cross product (determinant) of two vectors.
    
    This is the version recommended to replace the deprecated np.cross for 2D arrays.
    https://numpy.org/doc/stable/reference/generated/numpy.cross.html

    Args:
        x: NumPy array
        y: NumPy array

    Returns: 
        array or float
    """
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

def angle_between(a, b, degrees=True):
    """Compute the angle between two vectors.
    
    Args:
        a: NumPy array
        b: NumPy array
        degrees: bool, whether to return the angle in degrees
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
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
    v = np.asarray(v, dtype=float)
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
    v = np.asarray(v, dtype=float)
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


def pol2cart(r, phi):
    """Convert polar coordinates to Cartesian coordinates.
    
    Args:
        r: float or array of floats
        phi: float or array of floats

    Returns: NumPy array (2,) or (N, 2)
    """
    return r * np.array([np.cos(phi), np.sin(phi)])


def cart2pol(v):
    """Convert Cartesian coordinates to polar coordinates.
    
    Args:
        v: NumPy array (2,) or (N, 2)

    Returns: tuple of floats or tuple of arrays
    """
    x, y = np.transpose(v)
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    return r, phi


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
        alpha=0.9,
    )
    if origin is None:
        origin = np.zeros_like(vectors)
    else:
        origin = np.asarray(origin)

    if labels is not None:
        if label_pos is None:
            label_pos = [12] * len(vectors)
        label_vectors(labels, vectors, origin, label_pos)

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


def label_vectors(labels, vectors, origins=None, label_positions=None, **options):
    """Label the vectors with a string at a given position.

    Args:
        labels: list of string labels
        vectors: list of vectors or array with one row per vector
        origins: list of vectors or array with one row per vector
        label_positions: list of locations as integer clock positions
    """
    if origins is None:
        origins = [None] * len(labels)

    if label_positions is None:
        label_positions = [12] * len(labels)

    for label, vector, origin, label_pos in zip(
        labels, vectors, origins, label_positions
    ):
        label_vector(label, vector, origin, label_pos, **options)


def label_vector(label, vector, origin=None, label_pos=12, offset=10, **options):
    """Label a vector with a string at a given clock-face position.

    Args:
        label: string
        vector: array-like
        origin: array-like
        label_pos: integer clock position
        offset: distance of the label from the vector, in points
    """
    if origin is None:
        origin = np.zeros_like(vector)

    v_hat = normalize(vector)
    w_hat = vector_perp(v_hat)

    # fraction of vector length to move along v
    i = label_pos % 12
    v = np.array([4, 4, 3, 2, 1, 0, 0, 0, 1, 2, 3, 4])[i] / 4 * vector

    # unit step in v_hat or w_hat
    offset_v = np.array([1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0])[i]
    offset_w = np.array([0, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1])[i]

    # base anchor in data coordinates
    x0, y0 = origin + v

    # direction for the offset in screen points
    dx, dy = offset_v * v_hat + offset_w * w_hat

    ax = plt.gca()
    base = ax.transData
    offset_transform = mtransforms.offset_copy(
        base, fig=ax.figure, x=dx*offset, y=dy*offset, units="points"
    )

    underride(options, ha="center", va="center", fontsize=12)
    plt.text(x0, y0, label, transform=offset_transform, **options)


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


def plot_angle_between(a, b, radius=1, **kwargs):
    """Plot the angle between two vectors.
    
    Args:
        a: vector
        b: vector
        radius: float, the radius of the arc
        kwargs: passed to plt.patches.Arc
    """
    underride(kwargs, color='gray')
    
    # Calculate the angles of a and b
    _, angles = cart2pol([a, b])
    
    # Create the arc 
    import matplotlib.patches as patches
    arc = patches.Arc((0, 0), 
                      width=2*radius, height=2*radius,
                      theta1=np.degrees(angles[0]), 
                      theta2=np.degrees(angles[1]),
                      **kwargs)
    
    # Add the arc to the current axes
    ax = plt.gca()
    ax.add_patch(arc)

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


def setup_3D(nrows=1, ncols=1, show_grid=True, **subplot_kw):
    """Setup 3D subplots with configurable options.

    Args:
        nrows: number of rows of subplots
        ncols: number of columns of subplots
        show_grid: whether to show grid lines on the axes
        subplot_kw: keyword arguments passed to plt.subplots (e.g., figsize, dpi)
    
    Returns:
        fig, axes: matplotlib figure and axes objects
    """
    # Set default subplot arguments and ensure 3D projection
    underride(subplot_kw, projection='3d')
    
    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, subplot_kw=subplot_kw)
    fig.subplots_adjust(wspace=0.4)

    # Handle single axis case
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Configure all axes panes
    for ax in axes:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Set pane colors to white for better visibility
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        ax.grid(show_grid)
    
    return fig, axes


def plot_vectors_3D(
    vectors,
    origin=None,
    start=0,
    end=None,
    scale=1,
    **options
):
    """Plot a set of vectors in 3D.

    Args:
        vectors: list of vectors or array with one row per vector (shape: (N, 3))
        origin: list of vectors or array with one row per vector (default: all at (0,0,0))
        start: integer slice index
        end: integer slice index
        scale: factor to multiply vectors
        labels: list of string labels
        label_pos: list of locations as integer clock positions (only for 2D)
        options: passed to plt.quiver
    """
    vectors = np.asarray(vectors) * scale
    dim = vectors.shape[1]
    if dim != 3:
        raise ValueError("plot_vectors_3D requires 3D vectors.")

    if origin is None:
        origin = np.zeros_like(vectors)
    else:
        origin = np.asarray(origin)

    underride(
        options,
        color="C0",
        alpha=0.9,
        arrow_length_ratio=0.1,
    )
    
    us, vs, ws = vectors[start:end].T
    xs, ys, zs = origin[start:end].T

    ax = plt.gca()
    ax.scatter(xs, ys, zs, s=0)
    ax.scatter(xs + us, ys + vs, zs + zs, s=0)
    ax.quiver(xs, ys, zs, us, vs, ws, **options)


def label_vectors_3D(labels, vectors, origins=None, scale=1, offset=0.1, **options):
    """Label 3D vectors with strings positioned near the head of each arrow.

    Args:
        labels: list of string labels
        vectors: list of vectors or array with one row per vector (shape: (N, 3))
        origins: list of vectors or array with one row per vector (default: all at (0,0,0))
        scale: factor to multiply vectors
        offset: fraction of vector length to offset the label from the head (default: 0.1)
        options: passed to plt.text (e.g., fontsize, color)
    """
    vectors = np.asarray(vectors) * scale
    if vectors.shape[1] != 3:
        raise ValueError("label_vectors_3D requires 3D vectors.")
    
    if origins is None:
        origins = np.zeros_like(vectors)
    else:
        origins = np.asarray(origins)
    
    if len(labels) != len(vectors):
        raise ValueError("Number of labels must match number of vectors.")
    
    ax = plt.gca()
    
    for label, vector, origin in zip(labels, vectors, origins):
        # Calculate the head position of the vector
        head_pos = origin + vector
        
        # Calculate label position slightly offset from the head
        # Use a small offset in the direction of the vector
        label_offset = vector * offset
        label_pos = head_pos + label_offset
        
        # Set default text options
        underride(options,
            fontsize=12,
            ha='center',
            va='center',
            color='black'
        )
        
        ax.text(label_pos[0], label_pos[1], label_pos[2], label, **options)


def plot_vector_3D(v, origin=None, label=None, scale=1, **options):
    """Draw a single 3D vector.

    Args:
        v: array-like (shape: (3,))
        origin: array-like (shape: (3,))
        label: string
        scale: factor to multiply vector
        options: passed to plt.quiver
    """
    if origin is None:
        origin = np.zeros_like(v)
    else:
        origin = np.asarray(origin)

    if label is not None:
        label_vectors_3D([label], [v], [origin], scale=scale)
    return plot_vectors_3D([v], [origin], scale=scale, **options)


def plot_plane(v1, v2, origin=None, **options):
    """Plot a shaded plane spanned by two vectors in 3D.

    Args:
        v1: First vector defining the plane (array-like, shape (3,))
        v2: Second vector defining the plane (array-like, shape (3,))
        origin: Origin point of the plane (default: [0, 0, 0])
        options: Passed to plot_surface (e.g., color, alpha)
    """
    v1, v2 = np.asarray(v1), np.asarray(v2)

    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("plot_plane requires 3D vectors.")

    if origin is None:
        origin = np.zeros(3)
    else:
        origin = np.asarray(origin)

    # Generate a mesh grid for the plane
    u = [0, 1]
    v = [0, 1]
    U, V = np.meshgrid(u, v)

    # Plane equation: P = origin + U * v1 + V * v2
    X, Y, Z = [origin[i] + U * v1[i] + V * v2[i] for i in range(3)]

    underride(options, color="gray", alpha=0.3)

    # Plot the plane
    ax = plt.gca()
    ax.plot_surface(X, Y, Z, **options)


## Track function

import xml.etree.ElementTree as ET

def clean_gpx(input_path, output_path):
    """
    Create a privacy-safe GPX file containing only lat/lon/elevation/time.
    Removes metadata and extensions.
    """
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Remove top-level metadata, routes, and waypoints
    for tag in ["metadata", "rte", "wpt"]:
        for elem in root.findall(f".//{tag}"):
            root.remove(elem)

    # Remove all extensions anywhere in the tree
    for elem in root.findall(".//extensions"):
        parent = elem.getparent() if hasattr(elem, "getparent") else None
        if parent is not None:
            parent.remove(elem)
        else:
            # fallback for stdlib ElementTree (no getparent)
            for p in root.iter():
                for child in list(p):
                    if child.tag == elem.tag:
                        p.remove(child)

    # Keep only lat, lon, ele, and time for trackpoints
    for trkpt in root.findall(".//trkpt"):
        for child in list(trkpt):
            if child.tag not in ["ele", "time"]:
                trkpt.remove(child)

    # Strip any namespaces from tags for simplicity
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    # Write out clean GPX with XML declaration
    ET.indent(tree, space="  ", level=0)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    print(f"✅ Clean GPX written to: {output_path}")


## Pool table plotting functions


def decorate_plane(lim=None, **options):
    """Decorate the current axes with the given options.

    Args:
        lim: tuple of floats, the limits of the plot
        options: passed to decorate
    """
    underride(options, aspect='equal')
    remove_spines()
    if lim is not None:
        options['xlim'] = lim
        options['ylim'] = lim
    decorate(**options)


def draw_circles(vs, radius=1.125, **options):
    underride(options, color="C1", alpha=0.8, lw=0)
    ax = options.pop("ax", plt.gca())
    patches = [plt.Circle((x, y), radius, **options) for x, y in vs]
    for patch in patches:
        ax.add_patch(patch)
    return patches


def draw_table(table_width=100, table_height=50, ticks=False):
    fig, ax = plt.subplots(figsize=(5, 2.5))

    ax.add_patch(plt.Rectangle((0, 0), table_width, table_height,
                                fill='forestgreen', alpha=0.05))

    xs = [0, table_height, table_width]
    ys = [0, table_height]
    pockets = cartesian_product([xs, ys])
    draw_circles(pockets, radius=1.75, color="forestgreen", alpha=0.4)

    # Remove spines, ticks and labels
    remove_spines()
    if not ticks:
        remove_ticks()

    # Set the aspect ratio and limits
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2, table_width+2)  
    ax.set_ylim(-2, table_height+2)

    return fig, ax


def draw_collision(t, cue, target, v1, v2, label=None):
    """Draw a collision between a cue and a target.

    Args:
        t: float, the time of the collision
        cue: vector, the position of the cue
        target: vector, the position of the target
        v1: vector, the velocity of the cue
        v2: vector, the velocity of the target
        label: string, the label of the collision
    """
    draw_table(ticks=True)
    pos1 = cue + v1 * t
    pos2 = target + v2 * t
    draw_circles([pos1, pos2])


def remove_ticks():
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


## Regression helper functions

def add_constant(X):
    """Add an intercept column of ones to a 1D or 2D array-like X.
    
    Args:
        X: array-like, shape (n,) or (n, m)
    
    Returns:
        array of shape (n, m+1) with a column of ones prepended
    """
    X = np.asarray(X)

    # If X is 1D, make it a column vector
    if X.ndim == 1:
        X = X[:, np.newaxis]

    n = X.shape[0]
    ones = np.ones((n, 1))
    return np.column_stack([ones, X])


## Circuit plotting functions

def draw_circuit_graph(G, pos=None, **options):
    """Draw a circuit graph.

    Args:
        G: A NetworkX graph.
        pos: A dictionary of positions.
        options: passed to nx.draw_networkx_edges.
    """
    underride(options, node_color='lightblue', node_size=1000)
    G.graph['graph'] = {'rankdir': 'LR'}
    if pos is None:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

    nx.draw_networkx_nodes(G, pos, **options)
    nx.draw_networkx_labels(G, pos)

    edge_labels = {(u, v): str(d['resistance']) for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.axis('off')
    plt.tight_layout()


## Truss plotting functions

import matplotlib.lines as mlines
import matplotlib.patches as mpatches



def diagram_truss(nodes, subs=None, u=None, lim=1,
                  add_pin=True, add_labels=True, add_vectors=False,
                  **kwargs):
    """Plot a simple truss with three nodes.

    Args:
        nodes: list of nodes
        u: vector as np.array
        lim: float, the limit of the plot
        add_pin: bool, whether to add a pin at C and triangles at the anchors
        add_labels: bool, whether to add labels to the nodes
        add_vectors: bool, whether to add vectors to the truss members
    """
    # Get current axis
    ax = plt.gca()
    ax.axis(kwargs.pop("axis", "off"))
    ax.set(**kwargs)
    ax.set_aspect('equal')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim/4)
    remove_spines()
    
    # Calculate the vectors from C to A and C to B
    nodes = [sympy_to_numpy(node.transpose(), subs) for node in nodes]
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
            ], closed=True, color='C0', alpha=0.8)
            ax.add_artist(triangle)

    if add_pin:
        # Draw a circle at C to represent a pin
        pin_radius = 0.05 * L_CA
        pin_options = dict(facecolor='C0', edgecolor='none', alpha=0.8)
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
        u = sympy_to_numpy(u, subs)
        plot_vector(u, nodes[2], label='u', color='C1')
        u_CA = vector_projection(u, r_CA)
        u_CB = vector_projection(u, r_CB)
        plot_vector(u_CA, nodes[2], color='C4', label='$u_{CA}$', label_pos=1)
        plot_vector(u_CB, nodes[2], color='C4', label='$u_{CB}$', label_pos=-1)

        plot_rejection(u, r_CA)
        plot_rejection(u, r_CB)


def draw_truss_graph(G, subs, label_nodes=True, **options):
    """Draw a diagram of a truss represented by a NetworkX graph.

    Args:
        G: A NetworkX graph.
        subs: A dictionary of substitutions.
        label_nodes: bool, whether to label the nodes.
        options: passed to nx.draw_networkx_edges.
    """
    pos = {}
    for node in G.nodes:
        p = gget(G, node, 'pos')
        p = p.subs(subs).evalf()
        pos[node] = [float(x) for x in p]

    underride(options, edge_color='C0')
    nx.draw_networkx_edges(G, pos, width=2, **options)
    if label_nodes:
        node_options = dict(node_size=600, node_color='white', edgecolors='C0')
        nx.draw_networkx_nodes(G, pos, **node_options)
        nx.draw_networkx_labels(G, pos, font_size=11)

    xs, ys = zip(*pos.values())
    pad = 0.05 * max(max(xs) - min(xs), max(ys) - min(ys))
    plt.xlim(min(xs) - pad, max(xs) + pad)
    plt.ylim(min(ys) - pad, max(ys) + pad)

    plt.gca().set_aspect('equal')
    plt.axis('off')

def plot_force_field(nodes, subs, F_field, U, lim, displacements, ticks):
    diagram_truss(nodes, subs, add_pin=False, add_labels=False, axis='on')

    options = dict(angles='xy', scale_units='xy', scale=3e-7)
    plot_vectors(F_field.T, U.T, color='C0', alpha=0.9, **options)

    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    plt.xticks(displacements, ticks)
    plt.yticks(displacements, ticks)

    plt.xlabel(r'x displacement in $\mu$m')
    plt.ylabel(r'y displacement in $\mu$m')
    plt.grid(True)


## NetworkX helper functions
def gset(G, item=None, **kwargs):
    """Set attributes on a node, edge, or the graph itself.

    Args:
        G: A NetworkX graph.
        item: Node ID, edge tuple (u, v), or None for the graph itself.
        **kwargs: Arbitrary key-value pairs to set as attributes.
    """
    if item is None:
        container = G.graph
    elif item in G.edges:
        container = G.edges[item]
    elif item in G.nodes:
        container = G.nodes[item]
    else:
        raise ValueError(f"Item {item} not found in graph")

    container.update(**kwargs)

def gget(G, item=None, *keys):
    """Get one or more attributes from a node, edge, or the graph itself.

    Args:
        G: A NetworkX graph.
        item: Node ID, edge tuple (u, v), or None for the graph itself.
        *keys: One or more attribute names.

    Returns:
        A single value if one key is provided, or a tuple of values if multiple.
    """
    if item is None:
        container = G.graph
    elif item in G.edges:
        container = G.edges[item]
    elif item in G.nodes:
        container = G.nodes[item]
    else:
        raise ValueError(f"Item {item} not found in graph")

    if len(keys) == 1:
        return container[keys[0]]
    return tuple(container[key] for key in keys)


## SymPy helper functions

def sympy_to_numpy(expr, subs=None, evalf=True, dtype=float, squeeze=True):
    """Convert a SymPy Matrix to a NumPy array.

    Args:
        expr: SymPy Matrix
        subs: dictionary of substitutions
        evalf: bool, whether to evaluate the expression
        dtype: dtype of the NumPy array
        squeeze: bool, whether to convert 1-row or 1-col matrices to 1D

    Returns: NumPy array
    """
    if subs:
        expr = expr.subs(subs)
    if evalf:
        expr = expr.evalf()

    arr = np.array(expr, dtype=dtype)

    # If the matrix represents a vector, return a 1D array
    if squeeze:
        if 1 in arr.shape:
            return arr.ravel()

    return arr



def numpy_to_sympy(array):
    """Convert a NumPy array to a SymPy expression.

    Args:
        array: NumPy array

    Returns: SymPy expression
    """ 
    return sp.Matrix(array)


def divide_block_matrix(K, factor):
    """Divide a block matrix by a scalar factor.

    Args:
        K: block matrix
        factor: scalar SymPy expression to factor out (e.g., k/4)

    Returns:
        A SymPy BlockMatrix object showing the divided matrix
    """
    rows, cols = K.blockshape
    blocks = [[K.blocks[i, j] / factor for j in range(cols)] for i in range(rows)]
    return sp.BlockMatrix(blocks)


def factor_matrix(K, factor):
    """Factor out a scalar factor from a matrix.

    Args:
        K: matrix
        factor: scalar SymPy expression to factor out (e.g., k/4)

    Returns:
        A SymPy Expr object showing the factorized matrix
    """
    if isinstance(K, sp.BlockMatrix):
        divided = divide_block_matrix(K, factor)
    else:
        divided = K / factor
    return sp.MatMul(factor, divided, evaluate=False).simplify()

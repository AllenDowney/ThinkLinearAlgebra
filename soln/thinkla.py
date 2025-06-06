import numpy as np

def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def scalar_projection(a, b):
    """Compute the scalar projection of vector a onto vector b."""
    return np.dot(a, b) / np.linalg.norm(b)

def vector_projection(a, b):
    """Compute the vector projection of a onto b."""
    return (np.dot(a, b) / np.dot(b, b)) * b

def vector_rejection(a, b):
    """Compute the component of a that is perpendicular to b."""
    return a - vector_projection(a, b)

def angle_between(a, b, degrees=True):
    """Compute the angle between two vectors."""
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to handle numerical errors
    return np.degrees(angle) if degrees else angle

def rotate_2d(v, angle, degrees=True):
    """Rotate a 2D vector by a given angle."""
    if degrees:
        angle = np.radians(angle)
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
                           [np.sin(angle),  np.cos(angle)]])
    return rot_matrix @ v

def find_perpendicular(v):
    """Find a perpendicular vector to v."""
    v = np.asarray(v)
    if len(v) == 2:
        return np.array([-v[1], v[0]])  # Rotate 90 degrees
    elif len(v) == 3:
        return np.cross(v, np.array([1, 0, 0]) if v[0] == 0 else np.array([0, 1, 0]))
    else:
        raise ValueError("Finding a perpendicular is only implemented for 2D and 3D.")


def gram_schmidt(vectors):
    """Perform Gram-Schmidt orthogonalization on a set of vectors."""
    basis = []
    for v in vectors:
        for b in basis:
            v -= np.dot(v, b) / np.dot(b, b) * b
        basis.append(v / np.linalg.norm(v))
    return np.array(basis)

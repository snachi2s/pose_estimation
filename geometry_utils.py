"""
Geometry utility functions for 3D processing
Handles plane fitting, vector normalization, and bounding box calculations
"""

import numpy as np
import traceback

def fit_plane_svd(points):
    """
    Fits a plane to a set of 3D points using SVD
    
    Args:
        points: Nx3 array of 3D points
        
    Returns:
        Tuple of (normal, centroid)
    """
    if points.shape[0] < 3:
        print("Warning: Less than 3 points provided to fit_plane_svd.")
        return None, None
    
    try:
        # Calculate centroid
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Calculate covariance matrix
        if points_centered.shape[0] > points_centered.shape[1]:
            covariance_matrix = np.cov(points_centered, rowvar=False)
        else:
            covariance_matrix = np.cov(points_centered)
        
        # SVD decomposition
        u, s, vh = np.linalg.svd(covariance_matrix)
        normal = vh[-1, :]
        
        # Ensure normal points somewhat towards camera Z-
        if normal[2] > 0:
            normal = -normal
        
        # Normalize normal vector
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            print("Warning: Normal vector near zero length in fit_plane_svd.")
            return None, None
            
        normal = normal / norm_len
        return normal, centroid
        
    except np.linalg.LinAlgError as e_svd:
        print(f"SVD computation failed in fit_plane_svd: {e_svd}")
        return None, None
    except Exception as e_svd:
        print(f"Error in fit_plane_svd: {e_svd}")
        traceback.print_exc()
        return None, None

def normalize_vector(v):
    """
    Normalizes a vector, handling potential zero vectors or None input
    
    Args:
        v: Vector to normalize
        
    Returns:
        Normalized vector or None if normalization fails
    """
    if v is None:
        return None
        
    try:
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            print("Warning: Attempting to normalize near-zero vector.")
            return None
            
        return v / norm
        
    except Exception as e:
        print(f"Error normalizing vector {v}: {e}")
        return None

def get_bounding_box_corners(center, x_axis, y_axis, z_axis, L, W, H):
    """
    Calculates the 8 corners of a bounding box given center, axes, and dimensions
    
    Args:
        center: 3D center point of the box
        x_axis, y_axis, z_axis: Orientation axes of the box
        L, W, H: Length, width, and height of the box
        
    Returns:
        8x3 array of corner coordinates
    """
    if any(v is None for v in [center, x_axis, y_axis, z_axis, L, W, H]):
        print("Warning: Invalid input to get_bounding_box_corners (None value).")
        return None
        
    if any(dim <= 1e-9 for dim in [L, W, H]):
        print(f"Warning: Non-positive or near-zero dimension in get_bounding_box_corners "
              f"(L={L:.4f}, W={W:.4f}, H={H:.4f}). Cannot create box.")
        return None
    
    corners = np.zeros((8, 3))
    hl = L / 2.0
    hw = W / 2.0
    hh = H / 2.0
    
    x_ax = normalize_vector(x_axis)
    y_ax = normalize_vector(y_axis)
    z_ax = normalize_vector(z_axis)
    
    if any(v is None or np.linalg.norm(v) < 1e-6 for v in [x_ax, y_ax, z_ax]):
        print("Warning: Near-zero axis vector in get_bounding_box_corners after normalization.")
        return None
    
    # Define the 8 corners of the box
    corner_vectors = [
        -hl*x_ax - hw*y_ax + hh*z_ax,  # Top face, clockwise from back-left
        +hl*x_ax - hw*y_ax + hh*z_ax,
        +hl*x_ax + hw*y_ax + hh*z_ax,
        -hl*x_ax + hw*y_ax + hh*z_ax,
        -hl*x_ax - hw*y_ax - hh*z_ax,  # Bottom face, clockwise from back-left
        +hl*x_ax - hw*y_ax - hh*z_ax,
        +hl*x_ax + hw*y_ax - hh*z_ax,
        -hl*x_ax + hw*y_ax - hh*z_ax
    ]
    
    for i, vec in enumerate(corner_vectors):
        corners[i] = center + vec
        
    return corners
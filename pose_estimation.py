"""
Pose estimation module
Estimates the pose (orientation and dimensions) of a detected box
"""

import numpy as np
import traceback
from sklearn.decomposition import PCA
from geometry_utils import fit_plane_svd, normalize_vector, get_bounding_box_corners

# Try importing Shapely for OBB if available
try:
    from shapely import MultiPoint, minimum_rotated_rectangle
    SHAPELY_AVAILABLE = True
except ImportError:
    try:
        from shapely.geometry import MultiPoint
        # minimum_rotated_rectangle was an attribute in older versions
        SHAPELY_AVAILABLE = True
        # Define a wrapper for compatibility
        def minimum_rotated_rectangle(geom):
            return geom.minimum_rotated_rectangle
    except ImportError:
        SHAPELY_AVAILABLE = False

def calculate_box_pose(points_box, points_pallet, box_plane_found, pallet_plane_found,
                       pose_method, assume_box_height, dim_percentile, box_pallet_min_points):
    """
    Calculate box pose (orientation, dimensions, and transformation matrix)
    
    Args:
        points_box: Points belonging to box surface
        points_pallet: Points belonging to pallet surface
        box_plane_found: Boolean indicating if box plane was found
        pallet_plane_found: Boolean indicating if pallet plane was found
        pose_method: Method for calculating orientation ('OBB' or 'PCA')
        assume_box_height: Default box height if pallet not found
        dim_percentile: Percentile for dimension calculation
        box_pallet_min_points: Minimum points to consider a plane valid
        
    Returns:
        Dictionary containing box pose information
    """
    result = {
        'T_box_camera': None,         # 4x4 transformation matrix (box to camera)
        'T_camera_box': None,         # 4x4 transformation matrix (camera to box)
        'box_corners': None,          # 8x3 array of corner coordinates
        'box_pose_calculated': False, # Flag indicating successful calculation
        'box_dimensions': {           # Box dimensions in meters
            'L': 0.0,  # Length (along x_box_prime)
            'W': 0.0,  # Width (along y_box_prime)
            'H': 0.0   # Height (along z_box_prime/normal)
        },
        'C_box_top': None,           # Center of box top surface
        'R_box': None,               # 3x3 rotation matrix
        't_box_center': None,        # 3x1 translation vector to box center
        'axes': {                    # Box coordinate axes
            'x': None,  # x_box_prime (length direction)
            'y': None,  # y_box_prime (width direction)
            'z': None   # z_box_prime/normal (height direction)
        }
    }
    
    # Check if we have enough box points to calculate pose
    if not box_plane_found or points_box.shape[0] < 3:
        print("Box plane not found or too few points. Cannot calculate pose.")
        return result
    
    try:
        # 1. Refine Box Top Plane using SVD
        n_box_svd, C_box_top_svd = fit_plane_svd(points_box)
        if n_box_svd is None or C_box_top_svd is None:
            raise ValueError("SVD box plane fit failed.")
            
        n_box = n_box_svd
        C_box_top = C_box_top_svd
        result['C_box_top'] = C_box_top
        result['axes']['z'] = n_box
        
        print(f"Refined Box Top Plane: Normal={np.round(n_box,3)}, Centroid={np.round(C_box_top,3)}")
        
        # 2. Calculate Height (H)
        box_H = assume_box_height
        
        if pallet_plane_found and points_pallet.shape[0] >= 3:
            n_pallet, C_pallet = fit_plane_svd(points_pallet)
            
            if (n_pallet is not None and C_pallet is not None and 
                np.abs(np.dot(n_box, n_pallet)) > 0.8):
                # Calculate distance between planes
                dist_to_pallet = np.abs(np.dot(points_box - C_pallet, n_pallet))
                calc_H = np.mean(dist_to_pallet)
                
                if calc_H > 0.005 and calc_H < 2.0:
                    box_H = calc_H
                    print(f"Calculated H from planes: {box_H:.4f}m")
                else:
                    print(f"Warn: Unreasonable calc H {calc_H:.4f}m. Using assumed H.")
            else:
                print("Warn: Pallet plane unreliable. Using assumed H.")
        else:
            print("Warn: Pallet not found/reliable. Using assumed H.")
            
        result['box_dimensions']['H'] = box_H
        
        # 3. Determine In-Plane Orientation & Dimensions (L, W)
        # Project points onto plane
        points_on_plane = points_box - np.outer(
            np.dot(points_box - C_box_top, n_box), n_box
        )
        points_on_plane_centered = points_on_plane - C_box_top
        
        if points_on_plane_centered.shape[0] >= 2:
            # --- OBB Method (Oriented Bounding Box) ---
            if pose_method == 'OBB' and SHAPELY_AVAILABLE:
                print("Attempting OBB for orientation + Percentiles for dimensions...")
                try:
                    # A) Get initial 2D basis (PCA)
                    pca_init = PCA(n_components=2)
                    pca_init.fit(points_on_plane_centered)
                    basis_x_3d = normalize_vector(pca_init.components_[0])
                    basis_y_3d = normalize_vector(pca_init.components_[1])
                    
                    if basis_x_3d is None or basis_y_3d is None:
                        raise ValueError("PCA basis invalid.")
                        
                    # Ensure right-handed coordinate system
                    if np.dot(np.cross(basis_x_3d, basis_y_3d), n_box) < 0:
                        basis_y_3d = -basis_y_3d
                    
                    # B) Convert points to 2D
                    coords_2d_x = np.dot(points_on_plane_centered, basis_x_3d)
                    coords_2d_y = np.dot(points_on_plane_centered, basis_y_3d)
                    points_2d = np.column_stack((coords_2d_x, coords_2d_y))
                    
                    if points_2d.shape[0] < 3:
                        raise ValueError("Not enough points for MultiPoint.")
                    
                    # C) Find minimum rotated rectangle for ORIENTATION
                    multi_point = MultiPoint(points_2d)
                    min_rect_polygon = minimum_rotated_rectangle(multi_point)
                    rect_coords_2d = np.array(min_rect_polygon.exterior.coords)[:-1]
                    
                    # D) Get OBB AXES (Normalized 2D)
                    side1_vec_2d = rect_coords_2d[1] - rect_coords_2d[0]
                    side2_vec_2d = rect_coords_2d[3] - rect_coords_2d[0]
                    edge1_len = np.linalg.norm(side1_vec_2d)
                    edge2_len = np.linalg.norm(side2_vec_2d)
                    
                    # Determine axes (long=L candidate, short=W candidate)
                    if edge1_len >= edge2_len:
                        long_axis_2d = normalize_vector(side1_vec_2d)
                        short_axis_2d = normalize_vector(side2_vec_2d)
                    else:
                        long_axis_2d = normalize_vector(side2_vec_2d)
                        short_axis_2d = normalize_vector(side1_vec_2d)
                        
                    if long_axis_2d is None or short_axis_2d is None:
                        raise ValueError("OBB axis normalization failed.")
                    
                    # E) Project points onto OBB axes and Calculate L/W using PERCENTILES
                    print(f"Using percentile {dim_percentile} for dimension calculation on OBB axes.")
                    projected_obb_L = np.dot(points_2d, long_axis_2d)
                    projected_obb_W = np.dot(points_2d, short_axis_2d)
                    lower_p = (100.0 - dim_percentile) / 2.0
                    upper_p = 100.0 - lower_p
                    box_L = np.percentile(projected_obb_L, upper_p) - np.percentile(projected_obb_L, lower_p)
                    box_W = np.percentile(projected_obb_W, upper_p) - np.percentile(projected_obb_W, lower_p)
                    
                    # F) Convert OBB AXES back to 3D for final pose
                    x_box_prime = long_axis_2d[0] * basis_x_3d + long_axis_2d[1] * basis_y_3d
                    y_box_prime = short_axis_2d[0] * basis_x_3d + short_axis_2d[1] * basis_y_3d
                    x_box_prime = normalize_vector(x_box_prime)  # Renormalize
                    y_box_prime = normalize_vector(y_box_prime)
                    
                    if x_box_prime is None or y_box_prime is None:
                        raise ValueError("OBB final 3D axis invalid.")
                    
                    # G) Ensure right-handed system
                    z_check_obb = np.cross(x_box_prime, y_box_prime)
                    if np.dot(z_check_obb, n_box) < 0:
                        y_box_prime = -y_box_prime
                        print("Flipped OBB Y' axis for right-handed system.")
                    
                    result['box_dimensions']['L'] = box_L
                    result['box_dimensions']['W'] = box_W
                    result['axes']['x'] = x_box_prime
                    result['axes']['y'] = y_box_prime
                    
                    print(f"Estimated Dimensions (OBB Orient + Percentile {dim_percentile}): "
                          f"L={box_L:.3f}, W={box_W:.3f}, H={box_H:.3f}m")
                    
                except Exception as e_obb:
                    print(f"Error during OBB+Percentile calc: {e_obb}. Falling back to PCA.")
                    traceback.print_exc()
                    pose_method = 'PCA'
                    result['axes']['x'], result['axes']['y'] = None, None  # Trigger fallback
            
            # --- PCA Method (Fallback or if Shapely unavailable) ---
            if pose_method == 'PCA' or result['axes']['x'] is None:
                print("Using PCA method for orientation and dimensions...")
                try:
                    pca = PCA(n_components=2)
                    pca.fit(points_on_plane_centered)
                    x_box_prime = normalize_vector(pca.components_[0])
                    y_box_prime = normalize_vector(pca.components_[1])
                    
                    if x_box_prime is None or y_box_prime is None:
                        raise ValueError("PCA axis normalization failed.")
                    
                    # Ensure right-handed system
                    if np.dot(np.cross(x_box_prime, y_box_prime), n_box) < 0:
                        y_box_prime = -y_box_prime
                        print("Flipped PCA Y' axis.")
                    
                    # Project points onto PCA axes
                    proj_x = np.dot(points_on_plane_centered, x_box_prime)
                    proj_y = np.dot(points_on_plane_centered, y_box_prime)
                    lower_p = (100.0 - dim_percentile) / 2.0
                    upper_p = 100.0 - lower_p
                    box_L = np.percentile(proj_x, upper_p) - np.percentile(proj_x, lower_p)
                    box_W = np.percentile(proj_y, upper_p) - np.percentile(proj_y, lower_p)
                    
                    result['box_dimensions']['L'] = box_L
                    result['box_dimensions']['W'] = box_W
                    result['axes']['x'] = x_box_prime
                    result['axes']['y'] = y_box_prime
                    
                    print(f"Estimated Dimensions (PCA Percentile {dim_percentile}): "
                          f"L={box_L:.3f}, W={box_W:.3f}, H={box_H:.3f}m")
                    
                except Exception as e:
                    print(f"Error PCA fallback: {e}")
                    result['axes']['x'], result['axes']['y'] = None, None
            
            # --- Post Orientation/Dimension Processing ---
            if result['axes']['x'] is not None and result['axes']['y'] is not None:
                # 4. Calculate Pose (T_box_camera)
                R_box = np.column_stack((
                    result['axes']['x'], 
                    result['axes']['y'], 
                    result['axes']['z']
                ))
                t_box_center = C_box_top - n_box * (result['box_dimensions']['H'] / 2.0)
                
                # Create T_box_camera (box to camera transformation)
                T_box_camera = np.identity(4)
                T_box_camera[:3, :3] = R_box
                T_box_camera[:3, 3] = t_box_center
                
                # Calculate T_camera_box (camera to box transformation)
                # Inverse of the transformation matrix
                T_camera_box = np.identity(4)
                # Transpose of rotation matrix = inverse for orthogonal matrices
                R_camera_box = R_box.T
                # Translation vector in the new coordinate system
                t_camera_box = -np.dot(R_camera_box, t_box_center)
                
                T_camera_box[:3, :3] = R_camera_box
                T_camera_box[:3, 3] = t_camera_box
                
                # Alternative direct inversion method
                # T_camera_box = np.linalg.inv(T_box_camera)
                
                result['R_box'] = R_box
                result['t_box_center'] = t_box_center
                result['T_box_camera'] = T_box_camera
                result['T_camera_box'] = T_camera_box
                result['box_pose_calculated'] = True
                
                print("Calculated Box Pose (T_box_camera):\n", np.round(T_box_camera, 3))
                print("\nCalculated Camera Pose (T_camera_box):\n", np.round(T_camera_box, 3))
                
                # 5. Calculate Corners
                box_corners = get_bounding_box_corners(
                    t_box_center,
                    result['axes']['x'],
                    result['axes']['y'],
                    result['axes']['z'],
                    result['box_dimensions']['L'],
                    result['box_dimensions']['W'],
                    result['box_dimensions']['H']
                )
                
                if box_corners is not None:
                    result['box_corners'] = box_corners
                    print(f"Calculated {box_corners.shape[0]} Bounding Box corners.")
                else:
                    print("Warning: Could not calculate bounding box corners.")
            else:
                print("Warning: Could not determine valid in-plane axes. Cannot calculate pose/corners.")
        else:
            print("Warning: Not enough points on plane for OBB/PCA.")
            
    except Exception as e:
        print(f"Error calculating box pose: {e}")
        traceback.print_exc()
    
    return result
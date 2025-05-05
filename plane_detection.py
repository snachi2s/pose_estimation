"""
Plane detection module
Handles detection and classification of box and pallet planes
"""

import numpy as np
import traceback
from sklearn.linear_model import RANSACRegressor

def detect_box_pallet_planes(current_pts, box_pallet_plane_enabled, 
                            box_pallet_min_points, box_pallet_max_trials,
                            box_pallet_ransac_threshold, box_pallet_sample_size):
    """
    Detect box and pallet planes using RANSAC
    
    Args:
        current_pts: Input point cloud
        box_pallet_plane_enabled: Boolean flag to enable/disable plane detection
        box_pallet_min_points: Minimum points to consider a plane valid
        box_pallet_max_trials: Maximum RANSAC iterations
        box_pallet_ransac_threshold: Threshold for RANSAC inliers
        box_pallet_sample_size: Number of points to sample for RANSAC
        
    Returns:
        Dictionary containing plane1_data and plane2_data
    """
    plane1_data = None
    plane2_data = None
    planes_data = {'plane1': None, 'plane2': None}
    
    if not box_pallet_plane_enabled or current_pts.shape[0] < box_pallet_min_points:
        if not box_pallet_plane_enabled:
            print("Skipping Box/Pallet fit (disabled).")
        else:
            print(f"Skipping Box/Pallet fit (few points: {current_pts.shape[0]} < {box_pallet_min_points}).")
        return planes_data
    
    # Initialize RANSAC regressor
    ransac_bp = RANSACRegressor(
        max_trials=box_pallet_max_trials,
        residual_threshold=box_pallet_ransac_threshold,
        min_samples=3,
        random_state=43
    )
    
    # --- Plane 1 ---
    print("Searching for Plane 1...")
    n_cur = current_pts.shape[0]
    
    try:
        s_size1 = min(n_cur, box_pallet_sample_size)
        if s_size1 >= 3:
            # Sample points for RANSAC
            s_idx1 = np.random.choice(n_cur, s_size1, replace=False) if n_cur > s_size1 else np.arange(n_cur)
            
            # Fit plane using RANSAC
            ransac_bp.fit(current_pts[s_idx1, :2], current_pts[s_idx1, 2])
            a1, b1 = ransac_bp.estimator_.coef_
            c1 = ransac_bp.estimator_.intercept_
            
            # Calculate residuals and find inliers
            res1 = np.abs(a1*current_pts[:, 0] + b1*current_pts[:, 1] + c1 - current_pts[:, 2])
            inl_m1 = res1 < box_pallet_ransac_threshold
            n_in1 = np.sum(inl_m1)
            
            if n_in1 >= box_pallet_min_points:
                pl1_pts = current_pts[inl_m1]
                pl1_z = np.mean(pl1_pts[:, 2])
                plane1_data = {'mask': inl_m1, 'avg_z': pl1_z, 'points': pl1_pts}
                planes_data['plane1'] = plane1_data
                print(f"Plane 1 found: {n_in1} pts, Z={pl1_z:.3f}m")
            else:
                print(f"Plane 1 too small ({n_in1}<{box_pallet_min_points}).")
        else:
            print("Not enough points for P1 sample.")
    except Exception as e:
        print(f"Error P1: {e}")
        traceback.print_exc()
    
    # --- Plane 2 ---
    pts_rem_p2 = np.array([])
    idx_rem_p2 = np.array([])
    
    if plane1_data:
        pts_rem_p2 = current_pts[~plane1_data['mask']]
        idx_rem_p2 = np.where(~plane1_data['mask'])[0]
        print(f"Searching for Plane 2 in {pts_rem_p2.shape[0]} pts...")
    else:
        print("Skipping P2 search (P1 not found).")
    
    if pts_rem_p2.shape[0] >= box_pallet_min_points:
        try:
            n_rem = pts_rem_p2.shape[0]
            s_size2 = min(n_rem, box_pallet_sample_size)
            
            if s_size2 >= 3:
                # Sample points for RANSAC
                s_idx2 = np.random.choice(n_rem, s_size2, replace=False) if n_rem > s_size2 else np.arange(n_rem)
                
                # Fit plane using RANSAC
                ransac_bp.fit(pts_rem_p2[s_idx2, :2], pts_rem_p2[s_idx2, 2])
                a2, b2 = ransac_bp.estimator_.coef_
                c2 = ransac_bp.estimator_.intercept_
                
                # Calculate residuals and find inliers
                res2 = np.abs(a2*pts_rem_p2[:, 0] + b2*pts_rem_p2[:, 1] + c2 - pts_rem_p2[:, 2])
                inl_m2_loc = res2 < box_pallet_ransac_threshold
                n_in2 = np.sum(inl_m2_loc)
                
                if n_in2 >= box_pallet_min_points:
                    pl2_pts = pts_rem_p2[inl_m2_loc]
                    pl2_z = np.mean(pl2_pts[:, 2])
                    
                    # Convert local mask to global mask
                    pl2_mask_glob = np.zeros(n_cur, dtype=bool)
                    pl2_mask_glob[idx_rem_p2[inl_m2_loc]] = True
                    
                    plane2_data = {'mask': pl2_mask_glob, 'avg_z': pl2_z, 'points': pl2_pts}
                    planes_data['plane2'] = plane2_data
                    print(f"Plane 2 found: {n_in2} pts, Z={pl2_z:.3f}m")
                else:
                    print(f"Plane 2 too small ({n_in2}<{box_pallet_min_points}).")
            else:
                print("Not enough points for P2 sample.")
        except Exception as e:
            print(f"Error P2: {e}")
            traceback.print_exc()
    elif plane1_data:
        print(f"Not enough remaining pts ({pts_rem_p2.shape[0]}) for P2.")
    
    return planes_data

def assign_planes_to_points(current_pts, current_clrs, planes_data):
    """
    Assign points to detected planes (box, pallet, other)
    
    Args:
        current_pts: Input point cloud
        current_clrs: Input colors
        planes_data: Dictionary containing plane information
        
    Returns:
        Tuple of (points_box, colors_box, points_pallet, colors_pallet, 
                  points_other, colors_other, box_plane_found, pallet_plane_found)
    """
    n_cur = current_pts.shape[0]
    box_mask = np.zeros(n_cur, dtype=bool)
    pallet_mask = np.zeros(n_cur, dtype=bool)
    
    plane1_data = planes_data.get('plane1')
    plane2_data = planes_data.get('plane2')
    
    box_plane_found = False
    pallet_plane_found = False
    
    # Assign Box/Pallet based on detected planes
    if plane1_data and plane2_data:
        if plane1_data['avg_z'] < plane2_data['avg_z']:
            box_mask = plane1_data['mask']
            pallet_mask = plane2_data['mask']
            print("Assign: P1->Box, P2->Pallet")
        else:
            box_mask = plane2_data['mask']
            pallet_mask = plane1_data['mask']
            print("Assign: P2->Box, P1->Pallet")
        
        box_plane_found = True
        pallet_plane_found = True
        
    elif plane1_data:
        print("Only P1 found. Assign->Box.")
        box_mask = plane1_data['mask']
        box_plane_found = True
    
    # Split points into categories
    other_mask = ~(box_mask | pallet_mask)
    points_box = current_pts[box_mask]
    points_pallet = current_pts[pallet_mask]
    points_other = current_pts[other_mask]
    
    # Split colors into categories
    colors_box = None
    colors_pallet = None
    colors_other = None
    
    if current_clrs is not None:
        try:
            if np.any(box_mask):
                colors_box = current_clrs[box_mask]
            
            if np.any(pallet_mask):
                colors_pallet = current_clrs[pallet_mask]
            
            if np.any(other_mask):
                colors_other = current_clrs[other_mask]
            
            # Validate shapes
            if colors_box is not None and colors_box.shape[0] != points_box.shape[0]:
                colors_box = None
            
            if colors_pallet is not None and colors_pallet.shape[0] != points_pallet.shape[0]:
                colors_pallet = None
            
            if colors_other is not None and colors_other.shape[0] != points_other.shape[0]:
                colors_other = None
                
        except Exception as e:
            print(f"Warn: Error assigning final colors: {e}")
    
    print(f"Final Counts - Box:{points_box.shape[0]}, Pallet:{points_pallet.shape[0]}, "
          f"Other:{points_other.shape[0]}")
    
    return (points_box, colors_box, points_pallet, colors_pallet, 
            points_other, colors_other, box_plane_found, pallet_plane_found)
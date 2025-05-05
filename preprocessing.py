"""
Preprocessing module for point cloud data
Handles backprojection, filtering, and outlier removal
"""

import numpy as np
import traceback
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors

def backproject_to_camera(depth_image, color_image, fx, fy, cx, cy, 
                          min_depth_th_native, max_depth_th_native, depth_scale, H, W):
    """
    Backproject 2D depth/color image to 3D camera coordinates
    
    Args:
        depth_image: Depth image data
        color_image: Color image data (or None)
        fx, fy, cx, cy: Camera intrinsic parameters
        min_depth_th_native, max_depth_th_native: Depth thresholds in native units
        depth_scale: Scale factor to convert depth to meters
        H, W: Image height and width
        
    Returns:
        Tuple of (points_cam, colors_raw_normalized)
    """
    points_cam = np.array([])
    colors_raw_normalized = None
    
    try:
        # Create coordinate grids and flatten
        v, u = np.indices((H, W))
        u_f = u.flatten()
        v_f = v.flatten()
        d_f = depth_image.flatten()
        
        # Apply depth thresholds
        valid_mask = ((d_f > min_depth_th_native) & 
                      (d_f < max_depth_th_native) & 
                      np.isfinite(d_f) & 
                      (d_f != 0))
        
        u_v = u_f[valid_mask]
        v_v = v_f[valid_mask]
        d_m = d_f[valid_mask] * depth_scale
        
        # Backproject to 3D
        Z = d_m
        X = (u_v - cx) * Z / fx
        Y = (v_v - cy) * Z / fy
        points_cam = np.vstack((X, Y, Z)).T
        
        print(f"Generated {points_cam.shape[0]} valid points.")
        
        # Process colors if available
        if color_image is not None:
            try:
                colors_flat = color_image.reshape(-1, 3)
                colors_raw_filtered = colors_flat[valid_mask]
                temp_colors_normalized = None
                
                # Normalize colors based on data type
                if (np.issubdtype(colors_raw_filtered.dtype, np.integer) and 
                    np.min(colors_raw_filtered) >= 0 and 
                    np.max(colors_raw_filtered) <= 255):
                    temp_colors_normalized = colors_raw_filtered.astype(np.float64) / 255.0
                elif (np.issubdtype(colors_raw_filtered.dtype, np.floating) and 
                      np.min(colors_raw_filtered) >= 0.0 and 
                      np.max(colors_raw_filtered) <= 1.0):
                    temp_colors_normalized = colors_raw_filtered
                elif np.issubdtype(colors_raw_filtered.dtype, np.floating):
                    print("Warn: Raw color float values out of [0, 1] range. Clamping.")
                    temp_colors_normalized = np.clip(colors_raw_filtered, 0.0, 1.0)
                
                # Validate color shape
                if (temp_colors_normalized is not None and 
                    temp_colors_normalized.shape[0] == points_cam.shape[0]):
                    colors_raw_normalized = temp_colors_normalized
                    print(f"Filtered {colors_raw_normalized.shape[0]} colors (normalized).")
                elif temp_colors_normalized is not None:
                    print(f"Error: Color shape mismatch ({temp_colors_normalized.shape[0]} vs "
                          f"{points_cam.shape[0]}). Disabling color.")
                else:
                    print("Disabling color due to invalid range/type.")
                    
            except Exception as e:
                print(f"Error processing colors: {e}. Disabling color.")
                
    except Exception as e:
        print(f"Error during back-projection: {e}")
        traceback.print_exc()
        raise
        
    if points_cam.shape[0] == 0:
        print("Error: No points after back-projection. Exiting.")
        raise ValueError("No valid points after back-projection")
        
    return points_cam, colors_raw_normalized

def filter_background_plane(points_cam, colors_raw_normalized,
                           ransac_initial_sample_size, ransac_initial_max_trials,
                           ransac_initial_residual_threshold, plane_distance_threshold,
                           min_plane_distance_from_camera, box_pallet_min_points):
    """
    Detect and remove the background plane
    
    Args:
        points_cam: 3D points in camera coordinates
        colors_raw_normalized: Normalized RGB colors 
        ransac_initial_sample_size: Number of points to sample for RANSAC
        ransac_initial_max_trials: Maximum number of RANSAC iterations
        ransac_initial_residual_threshold: Threshold for RANSAC inliers
        plane_distance_threshold: Distance threshold for plane points
        min_plane_distance_from_camera: Minimum Z distance for plane removal
        box_pallet_min_points: Minimum points to consider a plane valid
        
    Returns:
        Tuple of (points_wo_plane, colors_wo_plane)
    """
    points_wo_plane = points_cam.copy()
    colors_wo_plane = colors_raw_normalized.copy() if colors_raw_normalized is not None else None
    
    if points_cam.shape[0] >= ransac_initial_sample_size:
        try:
            # Sample points for RANSAC
            sample_indices = np.random.choice(points_cam.shape[0], ransac_initial_sample_size, replace=False)
            pts_s = points_cam[sample_indices]
            
            # Fit plane using RANSAC (modeling Z as a function of X,Y)
            X_r = pts_s[:, :2]
            y_r = pts_s[:, 2]
            
            ransac_initial = RANSACRegressor(
                max_trials=ransac_initial_max_trials,
                residual_threshold=ransac_initial_residual_threshold,
                min_samples=3,
                random_state=42
            )
            
            ransac_initial.fit(X_r, y_r)
            a, b = ransac_initial.estimator_.coef_
            c = ransac_initial.estimator_.intercept_
            
            # Apply to all points
            X_all = points_cam[:, :2]
            y_all = points_cam[:, 2]
            residuals_all = np.abs(a*X_all[:,0] + b*X_all[:,1] + c - y_all)  # Z residual
            inlier_mask_all = residuals_all < plane_distance_threshold
            n_inliers = np.sum(inlier_mask_all)
            
            if n_inliers > box_pallet_min_points:
                avg_z_inliers = np.mean(points_cam[inlier_mask_all, 2])
                
                if avg_z_inliers >= min_plane_distance_from_camera:
                    print(f"Removing far plane ({n_inliers} pts, avg Z={avg_z_inliers:.3f}m)...")
                    non_plane_mask = ~inlier_mask_all
                    points_wo_plane = points_cam[non_plane_mask]
                    
                    if colors_raw_normalized is not None:
                        if colors_raw_normalized.shape[0] == points_cam.shape[0]:
                            colors_wo_plane = colors_raw_normalized[non_plane_mask]
                            if colors_wo_plane.shape[0] != points_wo_plane.shape[0]:
                                colors_wo_plane = None
                                print("Error: Color mismatch after plane removal.")
                        else:
                            colors_wo_plane = None
                            print("Error: Color mismatch before plane removal.")
                            
                    print(f"{points_wo_plane.shape[0]} points remaining.")
                else:
                    print(f"Initial plane too close (avg Z={avg_z_inliers:.3f}m). Keeping.")
            else:
                print("No significant initial plane found.")
                
        except Exception as e:
            print(f"Error initial RANSAC: {e}")
            traceback.print_exc()
    else:
        print("Skipping initial plane removal (few points).")
        
    return points_wo_plane, colors_wo_plane

def filter_by_x_range(points_wo_plane, colors_wo_plane, x_filter_min, x_filter_max):
    """
    Filter points by X-axis range
    
    Args:
        points_wo_plane: Points after background plane removal
        colors_wo_plane: Colors after background plane removal
        x_filter_min: Minimum X value
        x_filter_max: Maximum X value
        
    Returns:
        Tuple of (pts_post_xfilt, clr_post_xfilt)
    """
    pts_post_xfilt = np.array([])
    clr_post_xfilt = None
    
    if points_wo_plane.shape[0] > 0:
        try:
            x_mask = (points_wo_plane[:, 0] >= x_filter_min) & (points_wo_plane[:, 0] <= x_filter_max)
            pts_post_xfilt = points_wo_plane[x_mask]
            
            print(f"Removed {points_wo_plane.shape[0] - pts_post_xfilt.shape[0]} points, "
                  f"{pts_post_xfilt.shape[0]} remaining.")
            
            if colors_wo_plane is not None:
                if colors_wo_plane.shape[0] == points_wo_plane.shape[0]:
                    clr_post_xfilt = colors_wo_plane[x_mask]
                    
                    if clr_post_xfilt.shape[0] != pts_post_xfilt.shape[0]:
                        clr_post_xfilt = None
                        print("Error: Color mismatch post X-filt.")
                else:
                    clr_post_xfilt = None
                    print("Error: Color mismatch pre X-filt.")
                    
        except Exception as e:
            print(f"Error X-filt: {e}")
            pts_post_xfilt = points_wo_plane
            clr_post_xfilt = colors_wo_plane
    else:
        print("No points before X-filt.")
        
    if pts_post_xfilt.shape[0] == 0:
        print("Warning: No points after X-filt.")
        
    return pts_post_xfilt, clr_post_xfilt

def apply_sor(pts_post_xfilt, clr_post_xfilt, sor_enabled, sor_k_neighbors, sor_std_ratio):
    """
    Apply Statistical Outlier Removal (SOR)
    
    Args:
        pts_post_xfilt: Points after X-range filtering
        clr_post_xfilt: Colors after X-range filtering
        sor_enabled: Whether SOR is enabled
        sor_k_neighbors: Number of neighbors for SOR
        sor_std_ratio: Standard deviation ratio for SOR threshold
        
    Returns:
        Tuple of (pts_denoised, clr_denoised)
    """
    pts_denoised = np.array([])
    clr_denoised = None
    
    if sor_enabled and pts_post_xfilt.shape[0] > sor_k_neighbors:
        print(f"Applying SOR (k={sor_k_neighbors}, std={sor_std_ratio})...")
        
        try:
            # Find nearest neighbors
            nn = NearestNeighbors(n_neighbors=sor_k_neighbors + 1, algorithm='auto', n_jobs=-1)
            nn.fit(pts_post_xfilt)
            distances, _ = nn.kneighbors(pts_post_xfilt)
            
            # Calculate mean distances to neighbors
            mean_distances = np.mean(distances[:, 1:], axis=1)
            
            # Determine threshold based on standard deviation
            dist_thresh = np.mean(mean_distances) + sor_std_ratio * np.std(mean_distances)
            sor_inlier_mask = mean_distances < dist_thresh
            
            n_outliers = pts_post_xfilt.shape[0] - np.sum(sor_inlier_mask)
            print(f"SOR Threshold: {dist_thresh:.4f}. Removed {n_outliers}, "
                  f"retaining {np.sum(sor_inlier_mask)}.")
            
            pts_denoised = pts_post_xfilt[sor_inlier_mask]
            
            if clr_post_xfilt is not None:
                if clr_post_xfilt.shape[0] == pts_post_xfilt.shape[0]:
                    clr_denoised = clr_post_xfilt[sor_inlier_mask]
                    
                    if clr_denoised.shape[0] != pts_denoised.shape[0]:
                        clr_denoised = None
                        print("Error: Color mismatch post SOR.")
                else:
                    clr_denoised = None
                    print("Error: Color mismatch pre SOR.")
                    
        except Exception as e:
            print(f"Error SOR: {e}")
            pts_denoised = pts_post_xfilt
            clr_denoised = clr_post_xfilt
    else:
        if not sor_enabled:
            print("Skipping SOR (disabled).")
        else:
            print(f"Skipping SOR (few points: {pts_post_xfilt.shape[0]} <= {sor_k_neighbors}).")
            
        pts_denoised = pts_post_xfilt
        clr_denoised = clr_post_xfilt
        
    if pts_denoised.shape[0] == 0:
        print("Warning: No points after SOR.")
        
    return pts_denoised, clr_denoised
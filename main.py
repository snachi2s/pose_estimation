#!/usr/bin/env python3
"""
3D Box Detection and Pose Estimation from Point Cloud Data
Main script that orchestrates the processing pipeline
"""

import numpy as np
import traceback
from config import Config
from data_loader import load_data, validate_data, infer_depth_units
from preprocessing import backproject_to_camera, filter_background_plane, filter_by_x_range, apply_sor
from plane_detection import detect_box_pallet_planes, assign_planes_to_points
from pose_estimation import calculate_box_pose
from visualization import visualize_results

def main():
    """Main function to run the box detection pipeline"""
    print("--- Script Start ---")
    
    try:
        # Load data
        print("\nLoading data...")
        intrinsics, color_image, depth_image = load_data(
            Config.INTRINSICS_FILE, 
            Config.COLOR_FILE, 
            Config.DEPTH_FILE
        )
        
        # Validate data
        print("\nValidating data...")
        fx, fy, cx, cy = validate_data(intrinsics, color_image, depth_image)
        H, W = depth_image.shape[:2]
        
        # Infer depth units
        print("\nInferring depth units...")
        depth_scale, min_depth_th_native, max_depth_th_native = infer_depth_units(
            depth_image, 
            Config.INITIAL_MIN_DEPTH_THRESHOLD, 
            Config.INITIAL_MAX_DEPTH_THRESHOLD
        )
        
        # Backproject to camera coordinates
        print("\nPerforming back-projection...")
        points_cam, colors_raw_normalized = backproject_to_camera(
            depth_image, color_image, fx, fy, cx, cy, 
            min_depth_th_native, max_depth_th_native, depth_scale, H, W
        )
        
        # Remove background plane
        print("\nAttempting background plane removal...")
        points_wo_plane, colors_wo_plane = filter_background_plane(
            points_cam, colors_raw_normalized,
            Config.RANSAC_INITIAL_SAMPLE_SIZE,
            Config.RANSAC_INITIAL_MAX_TRIALS,
            Config.RANSAC_INITIAL_RESIDUAL_THRESHOLD,
            Config.PLANE_DISTANCE_THRESHOLD,
            Config.MIN_PLANE_DISTANCE_FROM_CAMERA,
            Config.BOX_PALLET_MIN_POINTS
        )
        
        # Filter by X range
        print(f"\nFiltering by X range [{Config.X_FILTER_MIN:.2f}m, {Config.X_FILTER_MAX:.2f}m]...")
        pts_post_xfilt, clr_post_xfilt = filter_by_x_range(
            points_wo_plane, colors_wo_plane,
            Config.X_FILTER_MIN, Config.X_FILTER_MAX
        )
        
        # Apply Statistical Outlier Removal (SOR)
        pts_denoised, clr_denoised = apply_sor(
            pts_post_xfilt, clr_post_xfilt,
            Config.SOR_ENABLED, Config.SOR_K_NEIGHBORS, Config.SOR_STD_RATIO
        )
        
        # Detect box and pallet planes
        print(f"\nFitting Box/Pallet planes (RANSAC)...")
        planes_data = detect_box_pallet_planes(
            pts_denoised, Config.BOX_PALLET_PLANE_ENABLED,
            Config.BOX_PALLET_MIN_POINTS, Config.BOX_PALLET_MAX_TRIALS,
            Config.BOX_PALLET_RANSAC_THRESHOLD, Config.BOX_PALLET_SAMPLE_SIZE
        )
        
        # Assign points to detected planes
        points_box, colors_box, points_pallet, colors_pallet, points_other, colors_other, box_plane_found, pallet_plane_found = assign_planes_to_points(
            pts_denoised, clr_denoised, planes_data
        )
        
        # Calculate box pose
        print(f"\nCalculating Box Pose (Method: {Config.POSE_METHOD})...")
        box_pose_result = calculate_box_pose(
            points_box, points_pallet, box_plane_found, pallet_plane_found,
            Config.POSE_METHOD, Config.ASSUME_BOX_HEIGHT, Config.DIM_PERCENTILE,
            Config.BOX_PALLET_MIN_POINTS
        )
        
        # Visualize results
        print("\nPreparing 3D visualization...")
        visualize_results(
            points_box, colors_box, 
            points_pallet, colors_pallet, 
            points_other, colors_other,
            box_pose_result
        )
        
        print("\n--- Script Finished ---")
        
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    main()
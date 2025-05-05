"""
Data loading and validation module
Handles loading and checking input data files
"""

import numpy as np
import traceback
import sys

def load_data(intrinsics_file, color_file, depth_file):
    """
    Load intrinsics, color, and depth data from specified files
    
    Args:
        intrinsics_file: Path to intrinsics matrix file
        color_file: Path to color image file
        depth_file: Path to depth image file
        
    Returns:
        Tuple of (intrinsics, color_image, depth_image)
    """
    try:
        intrinsics = np.load(intrinsics_file)
        color_image = np.load(color_file)
        depth_image = np.load(depth_file)
        
        print("--- Debug: Loaded Array Shapes ---")
        print(f"Intrinsics: {getattr(intrinsics, 'shape', 'N/A')}, "
              f"Color: {getattr(color_image, 'shape', 'N/A')}, "
              f"Depth: {getattr(depth_image, 'shape', 'N/A')}")
              
        return intrinsics, color_image, depth_image
        
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading: {e}")
        traceback.print_exc()
        sys.exit(1)

def validate_data(intrinsics, color_image, depth_image):
    """
    Validate loaded data and extract intrinsic parameters
    
    Args:
        intrinsics: Intrinsics matrix
        color_image: Color image data
        depth_image: Depth image data
        
    Returns:
        Tuple of intrinsic parameters (fx, fy, cx, cy)
    """
    # Validate intrinsics
    print("Validating intrinsics...")
    if not isinstance(intrinsics, np.ndarray) or intrinsics.shape != (3, 3):
        print(f"Error: Intrinsics matrix shape {getattr(intrinsics, 'shape', 'N/A')} != (3, 3).")
        sys.exit(1)
    
    try:
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        
        if not all(isinstance(val, (int, float)) and val > 0 for val in [fx, fy, cx, cy]):
            raise ValueError("Non-positive/numeric intrinsics.")
            
        print(f"Intrinsics (fx,fy,cx,cy): ({fx:.2f}, {fy:.2f}, {cx:.2f}, {cy:.2f})")
        print("Intrinsics validated.")
        
    except Exception as e:
        print(f"Error: Invalid intrinsics content: {e}")
        sys.exit(1)
    
    # Validate image data
    print("Validating image data...")
    color_data_valid = False
    
    if not isinstance(depth_image, np.ndarray) or depth_image.ndim < 2:
        print(f"Error: Invalid depth data shape: {getattr(depth_image, 'shape', 'N/A')}")
        sys.exit(1)
        
    H, W = depth_image.shape[:2]
    print(f"Depth dimensions: H={H}, W={W}")
    
    if isinstance(color_image, np.ndarray) and color_image.ndim >= 2:
        if (H == color_image.shape[0] and 
            W == color_image.shape[1] and 
            color_image.ndim == 3 and 
            color_image.shape[2] == 3):
            print("Color image shape seems valid.")
            color_data_valid = True
        else:
            print(f"Warn: Color shape {color_image.shape} mismatch/invalid. Disabling color.")
    else:
        print("Warn: Color data invalid. Disabling color.")
        
    if not color_data_valid:
        color_image = None
        print("Proceeding without color.")
        
    print("Image data validated.")
    
    return fx, fy, cx, cy

def infer_depth_units(depth_image, min_depth_th_scaled, max_depth_th_scaled):
    """
    Infer depth units (meters, millimeters) from depth image data
    
    Args:
        depth_image: Depth image data
        min_depth_th_scaled: Minimum depth threshold in meters
        max_depth_th_scaled: Maximum depth threshold in meters
        
    Returns:
        Tuple of (depth_scale, min_depth_th_native, max_depth_th_native)
    """
    d_scale = 1.0
    
    if np.issubdtype(depth_image.dtype, np.integer):
        max_val_int = np.max(depth_image) if np.any(depth_image > 0) else 0
        if max_val_int > 10000:
            print(f"Depth integer (max={max_val_int}), assuming mm.")
            d_scale = 0.001
        else:
            print(f"Warn: Depth integer (max={max_val_int}) low. "
                  f"Assuming meters/unknown. scale=1.0.")
    
    elif np.issubdtype(depth_image.dtype, np.floating):
        finite_depth = depth_image[np.isfinite(depth_image)]
        max_val_float = np.max(finite_depth) if finite_depth.size > 0 else 0
        
        if max_val_float > 50:
            print(f"Warn: Depth float (max={max_val_float:.2f}) > 50 suggests mm? scale=0.001.")
            d_scale = 0.001
        else:
            print(f"Depth float (max={max_val_float:.2f}), assuming meters (m).")
    
    else:
        print(f"Warn: Unknown depth dtype: {depth_image.dtype}. Assuming meters, scale=1.0.")
    
    min_depth_th_native = min_depth_th_scaled / d_scale
    max_depth_th_native = max_depth_th_scaled / d_scale
    
    print(f"Depth Thresholds (Native): [{min_depth_th_native:.2f}, {max_depth_th_native:.2f}], "
          f"Scale to meters: {d_scale}")
          
    return d_scale, min_depth_th_native, max_depth_th_native
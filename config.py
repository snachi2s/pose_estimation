"""
Configuration module for box detection pipeline
Centralizes all parameters and settings
"""

import numpy as np

# Try importing Shapely for OBB, fallback to PCA if unavailable
try:
    # Try importing Shapely >= 2.0
    from shapely import MultiPoint, minimum_rotated_rectangle
    SHAPELY_AVAILABLE = True
    print("Shapely >= 2.0 imported successfully. Will prioritize OBB for orientation.")
except ImportError:
    try:
        # Try importing older Shapely < 2.0
        from shapely.geometry import MultiPoint
        # minimum_rotated_rectangle was an attribute in older versions
        SHAPELY_AVAILABLE = True
        print("Shapely < 2.0 imported successfully. Will prioritize OBB for orientation.")
        # Define a wrapper for compatibility
        def minimum_rotated_rectangle(geom):
            return geom.minimum_rotated_rectangle
    except ImportError:
        print("Warning: shapely library not found or version incompatible.")
        print("Install using: pip install shapely")
        print("Falling back to PCA for box dimensions and orientation.")
        SHAPELY_AVAILABLE = False

class Config:
    """Configuration class for all parameters used in the application"""
    
    # File paths
    INTRINSICS_FILE = "data/intrinsics.npy"
    COLOR_FILE = "data/one-box.color.npdata.npy"
    DEPTH_FILE = "data/one-box.depth.npdata.npy"
    
    # Depth filtering thresholds
    INITIAL_MIN_DEPTH_THRESHOLD = 0.1
    INITIAL_MAX_DEPTH_THRESHOLD = 5.0
    
    # RANSAC Initial Plane detection parameters (Floor/Background)
    PLANE_DISTANCE_THRESHOLD = 0.02
    MIN_PLANE_DISTANCE_FROM_CAMERA = 1.0  # Min distance for initial plane to be removed
    RANSAC_INITIAL_SAMPLE_SIZE = 1000
    RANSAC_INITIAL_MAX_TRIALS = 100
    RANSAC_INITIAL_RESIDUAL_THRESHOLD = PLANE_DISTANCE_THRESHOLD  # Alias for clarity
    
    # X-axis filtering range (meters) after initial plane removal
    X_FILTER_MIN = -0.75
    X_FILTER_MAX = 0.75
    
    # Statistical Outlier Removal (SOR) parameters (using sklearn)
    SOR_ENABLED = True
    SOR_K_NEIGHBORS = 20
    SOR_STD_RATIO = 1.0  # Might need tuning based on noise level
    
    # Secondary Plane Fitting (Box/Pallet) parameters (using sklearn RANSAC)
    BOX_PALLET_PLANE_ENABLED = True
    BOX_PALLET_RANSAC_THRESHOLD = 0.015  # Tighter threshold for box/pallet surfaces
    BOX_PALLET_MIN_POINTS = 50           # Minimum points to consider a plane valid
    BOX_PALLET_SAMPLE_SIZE = 500         # Sample size for RANSAC fitting
    BOX_PALLET_MAX_TRIALS = 50           # Max trials for RANSAC fitting
    
    # Pose Estimation & Visualization
    POSE_METHOD = 'OBB' if SHAPELY_AVAILABLE else 'PCA'  # Prioritize OBB if available
    ASSUME_BOX_HEIGHT = 0.1             # Used if pallet plane isn't found/reliable
    BOUNDING_BOX_COLOR = 'lime'         # Use lime green for better visibility
    BOUNDING_BOX_LINEWIDTH = 1.5
    # Percentile used for dimension calculation in BOTH OBB (modified) and PCA methods
    DIM_PERCENTILE = 98  # Use this percentile range for L/W calculation
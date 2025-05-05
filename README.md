# 3D Box Detection and Pose Estimation

A modular system for detecting and estimating the pose of boxes in point cloud data. This application processes depth and color data to segment the scene, detect box and pallet planes, and calculate the box's orientation and dimensions.

## Algorithm Overview

The pipeline uses a multi-stage approach to detect boxes and estimate their pose:

1. **Data Loading & Validation**: Loads depth, color, and camera intrinsic data, validating formats and dimensions.

2. **Preprocessing**:
   - Back-projection of depth/color data to 3D camera coordinates
   - Background plane removal using RANSAC
   - X-axis range filtering to focus on the region of interest
   - Statistical Outlier Removal (SOR) for noise reduction

3. **Plane Detection**:
   - Detection of box and pallet planes using RANSAC
   - Point classification into box, pallet, and other categories

4. **Pose Estimation**:
   - Refinement of box plane orientation using SVD
   - Calculation of box height using distance to pallet plane
   - Orientation determination using either:
     - Oriented Bounding Box (OBB) method via Shapely (preferred)
     - Principal Component Analysis (PCA) method (fallback)
   - Box dimension calculation using percentile range of projected points
   - Construction of box pose transformation matrix and bounding box

5. **Visualization**:
   - 3D rendering of segmented point cloud
   - Display of box coordinate axes and bounding box

## Module Descriptions

### `main.py`
Entry point that orchestrates the entire processing pipeline. Calls functions from each module in the correct order and passes data between stages.

### `config.py`
Centralizes all configuration parameters including thresholds, file paths, and processing options. Attempts to import Shapely for OBB calculation, with fallback to PCA.

### `data_loader.py`
Handles loading and validation of input data:
- `load_data()`: Loads data files
- `validate_data()`: Validates data formats and extracts intrinsic parameters
- `infer_depth_units()`: Determines depth scale (meters or millimeters)

### `preprocessing.py`
Implements point cloud preprocessing operations:
- `backproject_to_camera()`: Projects depth data to 3D points
- `filter_background_plane()`: Removes background/floor plane
- `filter_by_x_range()`: Filters points by X coordinate
- `apply_sor()`: Applies Statistical Outlier Removal

### `geometry_utils.py`
Provides geometric utility functions:
- `fit_plane_svd()`: Fits a plane to points using SVD
- `normalize_vector()`: Normalizes a vector with error handling
- `get_bounding_box_corners()`: Calculates box corners from pose

### `plane_detection.py`
Detects and classifies box and pallet planes:
- `detect_box_pallet_planes()`: Finds planes using RANSAC
- `assign_planes_to_points()`: Classifies points by plane association

### `pose_estimation.py`
Calculates box pose and dimensions:
- `calculate_box_pose()`: Determines orientation, dimensions, and transformation

### `visualization.py`
Handles 3D visualization of results:
- `visualize_results()`: Renders point cloud, axes, and bounding box

## Usage

1. Place depth, color, and intrinsics files in the working directory
2. Adjust parameters in `config.py` if needed
3. Run the application with:
   ```
   python main.py
   ```

## Dependencies

- NumPy: For numerical operations
- Matplotlib: For 3D visualization
- scikit-learn: For RANSAC plane fitting, PCA, and nearest neighbors
- Shapely (optional): For Oriented Bounding Box calculation

## Notes

- The code prioritizes Shapely's OBB method but falls back to PCA if Shapely is unavailable
- Box height is calculated from the distance to the pallet plane, or uses an assumed value if unavailable
- Parameters in `config.py` can be adjusted to tune the pipeline for different scenarios
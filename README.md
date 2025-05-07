# 3D Box Detection and Pose Estimation

A modular system for detecting and estimating the pose of boxes in point cloud data. This application processes depth data to segment the scene, detect box and pallet planes, and calculate the box's orientation and dimensions.

## Usage

## Using Docker 

1. Assuming Docker is available on your system.
2. Clone this repository.
   ```
   git clone https://github.com/snachi2s/pose_estimation.git
   ```
3. Build the docker container with the necessary requirements using the dockerfile
   ```
   docker build -t box-detection:latest .
   ```
5. Once Docker is successfully built, we need to connect our local display to the Docker container.
   ```
   xhost +local:docker
   ```
6. Now, run the container in interactive mode using
   ```
   docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --rm box-detection:latest
   ```
7. That's it!! Now execute the main script inside the container terminal
   ```
   python main.py
   ```

## Without Docker

1. Clone the repository
2. Executing main script is enough to see the visualization along with the estimated pose matrix:
   ```
   python main.py
   ```
   
   
## Algorithm Overview

Before delving into the algorithm, my initial assumption about the scene are as follows,
- As provided in the `one-box.depth.npdata.npy`, the scene always have a good visibility og ground and ground will be the largest plane formed in the scene.
- For the tasks such as palletizing/depalletizing/bin-picking, the camera always looks at the objects in the scene from a top-view, thus I considered the assumption that box plane will be the first plane normal to the camera depth-axis. 

## Short summary of the algorithm 

The algorithm takes in the depth data and converts it into point clouds using camera parameters. Then it uses RANSAC to fit planes that are normal to the camera depth-axis. Among the fitted planes by RANSAC, the largest plane which should also be the farthest from the camera, is eliminated, which results in the ground plane removal. Then, statistical based outlier rejection is done. Then, another iteration of RANSAC is done by which the box and pallet plane are found based on their distance with respect to the camera. The distance between the planes is considered to be the box height, as it is trivial. Then, the Oriented Bounding Box method is used to find the orientation of the box.  

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

## Dependencies

- NumPy: For numerical operations
- Matplotlib: For 3D visualization
- scikit-learn: For RANSAC plane fitting, PCA, and nearest neighbors
- Shapely (optional): For Oriented Bounding Box calculation

## Notes

- The code prioritizes Shapely's OBB method but falls back to PCA if Shapely is unavailable
- Box height is calculated from the distance to the pallet plane, or uses an assumed value if unavailable(mostly not the case)
- Parameters in `config.py` can be adjusted to tune the pipeline for different scenarios

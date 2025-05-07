"""
Visualization module
Handles 3D visualization of point cloud and box detection results
"""

import numpy as np
import matplotlib.pyplot as plt
from config import Config

def visualize_results(points_box, colors_box, points_pallet, colors_pallet, 
                     points_other, colors_other, box_pose_result):
    """
    Visualize point cloud and detected box in 3D
    
    Args:
        points_box: Points belonging to box surface
        colors_box: Colors of box points
        points_pallet: Points belonging to pallet surface
        colors_pallet: Colors of pallet points
        points_other: Other points
        colors_other: Colors of other points
        box_pose_result: Box pose information from pose_estimation
    """
    # Create 3D figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    plotted_something = False
    
    try:
        # Determine the title based on pose method used
        plot_title = f"Point Cloud Segmentation & Box Pose ({Config.POSE_METHOD} used)"
        alpha_pts = 0.15
        
        # Plot points
        if points_other.shape[0] > 0:
            ax.scatter(
                points_other[:, 0], 
                points_other[:, 1], 
                points_other[:, 2],
                c=colors_other if colors_other is not None else 'lightgray',
                s=1,
                label=f'Other ({points_other.shape[0]})',
                alpha=alpha_pts*0.8
            )
            plotted_something = True
            
        if points_pallet.shape[0] > 0:
            ax.scatter(
                points_pallet[:, 0], 
                points_pallet[:, 1], 
                points_pallet[:, 2],
                c=colors_pallet if colors_pallet is not None else 'blue',
                s=2,
                label=f'Pallet (Est. {points_pallet.shape[0]})',
                alpha=alpha_pts
            )
            plotted_something = True
            
        if points_box.shape[0] > 0:
            ax.scatter(
                points_box[:, 0], 
                points_box[:, 1], 
                points_box[:, 2],
                c=colors_box if colors_box is not None else 'red',
                s=3,
                label=f'Box (Est. {points_box.shape[0]})',
                alpha=alpha_pts*1.2
            )
            plotted_something = True
            
        # Plot Box Axes
        if (box_pose_result['box_pose_calculated'] and 
            box_pose_result['T_box_camera'] is not None):
            
            t_c = box_pose_result['T_box_camera'][:3, 3]
            R = box_pose_result['T_box_camera'][:3, :3]
            ax_len = 0.1
            
            print("Plotting calculated box axes...")
            ax.quiver(
                t_c[0], t_c[1], t_c[2],
                R[0, 0], R[1, 0], R[2, 0],
                length=ax_len, normalize=True, color='r',
                label="Box X'(L)", linewidth=1.5
            )
            ax.quiver(
                t_c[0], t_c[1], t_c[2],
                R[0, 1], R[1, 1], R[2, 1],
                length=ax_len, normalize=True, color='g',
                label="Box Y'(W)", linewidth=1.5
            )
            ax.quiver(
                t_c[0], t_c[1], t_c[2],
                R[0, 2], R[1, 2], R[2, 2],
                length=ax_len, normalize=True, color='b',
                label="Box Z'(H)", linewidth=1.5
            )
            plotted_something = True
            
        # Plot Bounding Box Edges
        if (box_pose_result['box_pose_calculated'] and 
            box_pose_result['box_corners'] is not None and 
            box_pose_result['box_corners'].shape == (8, 3)):
            
            print(f"Plotting bounding box (color {Config.BOUNDING_BOX_COLOR})")
            box_corners = box_pose_result['box_corners']
            
            # Define box edges (pairs of vertex indices)
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Top face
                (4, 5), (5, 6), (6, 7), (7, 4),  # Bottom face
                (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
            ]
            
            plotted_edge = False
            for i, j in edges:
                label = 'Detected Box Pose' if not plotted_edge else ""
                ax.plot(
                    *zip(box_corners[i], box_corners[j]),
                    color=Config.BOUNDING_BOX_COLOR,
                    linewidth=Config.BOUNDING_BOX_LINEWIDTH,
                    label=label
                )
                plotted_edge = True
                
            plotted_something = True
            
        # Final Plot Configuration
        if plotted_something:
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title(plot_title, fontsize=10)
            ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.95), fontsize='small')
            
            try:
                # Set aspect ratio based on point cloud extent
                all_pts_list = [p for p in [points_other, points_pallet, points_box] if p.shape[0] > 0]
                
                if box_pose_result['box_corners'] is not None:
                    all_pts_list.append(box_pose_result['box_corners'])
                    
                if all_pts_list:
                    all_pts = np.concatenate(all_pts_list, axis=0)
                    if all_pts.shape[0] > 1:
                        ranges = np.maximum(np.ptp(all_pts, axis=0), 1e-6)
                        ax.set_box_aspect(ranges / np.max(ranges))
                        print(f"Set aspect ratio.")
                    else:
                        ax.set_aspect('equal')
                else:
                    ax.set_aspect('equal')
                    
            except Exception as e:
                print(f"Warn: Could not set aspect ratio: {e}")
                ax.set_aspect('equal')
                
            # Set initial view angle
            ax.view_init(elev=-75, azim=-90)
            
        else:
            ax.set_title("No Valid Points/Box to Visualize")
            
        plt.tight_layout()
        plt.show()
        print("Visualization closed.")
        
    except Exception as e:
        print(f"Error during visualization: {e}")

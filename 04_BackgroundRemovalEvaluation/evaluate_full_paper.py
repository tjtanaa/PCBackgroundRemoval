import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
)
import math
import time
from numba import cuda
from scipy.spatial.transform import Rotation
from matplotlib.path import Path

# Set up CUDA environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@cuda.jit
def wna_number_cuda_jit(points, polygon_vertices, wn):
    """
    CUDA kernel to compute the winding number for each point with respect to a polygon.

    Args:
        points: 2D array of point coordinates (x, y)
        polygon_vertices: 2D array of polygon vertex coordinates (x, y)
        wn: Output array to store winding numbers
    """
    row, col = cuda.grid(2)

    if row < points.shape[0] and col < polygon_vertices.shape[0] - 1:
        if polygon_vertices[col, 1] <= points[row, 1]:
            if polygon_vertices[col + 1, 1] > points[row, 1]:
                if (
                    (polygon_vertices[col + 1, 0] - polygon_vertices[col, 0])
                    * (points[row, 1] - polygon_vertices[col, 1])
                    - (points[row, 0] - polygon_vertices[col, 0])
                    * (polygon_vertices[col + 1, 1] - polygon_vertices[col, 1])
                ) > 0:
                    cuda.atomic.add(wn, row, 1)

        else:
            if polygon_vertices[col + 1, 1] <= points[row, 1]:
                if (
                    (polygon_vertices[col + 1, 0] - polygon_vertices[col, 0])
                    * (points[row, 1] - polygon_vertices[col, 1])
                    - (points[row, 0] - polygon_vertices[col, 0])
                    * (polygon_vertices[col + 1, 1] - polygon_vertices[col, 1])
                ) < 0:
                    cuda.atomic.add(wn, row, -1)


def filter_points_by_polygon(points_xy, polygon_vertices):
    """
    Filter points based on whether they are inside a polygon using winding number algorithm.

    Args:
        points_xy: numpy array of shape (N, 2) containing x,y coordinates of points
        polygon_vertices: numpy array of shape (M, 2) containing polygon vertices

    Returns:
        mask: boolean array indicating which points are inside the polygon
    """
    # # Copy the arrays to the device
    # points_global_mem = cuda.to_device(points_xy)
    # vertices_global_mem = cuda.to_device(polygon_vertices)

    # # Allocate memory on the device for the winding number results
    # wn_global_mem = cuda.device_array((points_xy.shape[0]), dtype=np.int32)

    # # Configure the CUDA blocks and grid
    # threadsperblock = (128, 8)
    # blockspergrid_x = int(math.ceil(points_xy.shape[0] / threadsperblock[0]))
    # blockspergrid_y = int(math.ceil(polygon_vertices.shape[0] / threadsperblock[1]))
    # blockspergrid = (blockspergrid_x, blockspergrid_y)

    # # Start the kernel to compute winding numbers
    # wna_number_cuda_jit[blockspergrid, threadsperblock](
    #     points_global_mem, vertices_global_mem, wn_global_mem
    # )
    # cuda.synchronize()

    # # Copy the result back to the host
    # wn = wn_global_mem.copy_to_host()

    # # Clear CUDA memory
    # cuda.current_context().deallocations.clear()

    # # Return mask for points inside the polygon (wn > 0)
    # return wn < 0
    valid_mask = polygon_vertices.contains_points(points_xy)
    return valid_mask


def filter_point_cloud_by_polygon(point_cloud, valid_polygon, invalid_polygons):
    """
    Filter a point cloud to keep only points inside the valid polygon and outside all invalid polygons.

    Args:
        point_cloud: numpy array of shape (N, C) where N is the number of points and C is the number of channels
        valid_polygon: numpy array of shape (M, 2) containing valid polygon vertices
        invalid_polygons: list of numpy arrays, each of shape (P, 2) containing invalid polygon vertices

    Returns:
        filtered_point_cloud: numpy array containing only valid points
        mask: boolean array indicating which points were kept
    """
    # Extract x and y coordinates from the point cloud
    points_xy = point_cloud[:, :2].astype(np.float32)

    # Make Paths
    invalid_polygons = [
        Path(invalid_poly, closed=True) for invalid_poly in invalid_polygons
    ]
    valid_polygons = [Path(valid_polygon, closed=True)]

    # Get mask for points inside the valid polygon
    valid_mask = filter_points_by_polygon(points_xy, valid_polygons[0])

    # Initialize mask for points outside all invalid polygons
    invalid_mask = np.zeros(points_xy.shape[0], dtype=bool)

    # print(f"valid_mask: {np.sum(valid_mask)}")
    # Process each invalid polygon
    for invalid_polygon in invalid_polygons:
        # Get mask for points inside this invalid polygon
        inside_invalid = filter_points_by_polygon(points_xy, invalid_polygon)
        # Update the invalid mask
        invalid_mask = np.logical_or(invalid_mask, inside_invalid)
        # print(f"invalid_mask: {np.sum(~invalid_mask)}")

    # Final mask: points inside valid polygon AND outside all invalid polygons
    final_mask = np.logical_and(valid_mask, ~invalid_mask)

    # Apply the mask to filter the point cloud
    filtered_point_cloud = point_cloud[final_mask]

    return filtered_point_cloud, final_mask


def load_point_cloud(file_path, channels=7):
    """
    Load a point cloud from a binary file.

    Args:
        file_path: Path to the binary file
        channels: Number of channels in the point cloud

    Returns:
        point_cloud: Numpy array of shape (N, channels)
    """
    try:
        point_cloud = np.fromfile(file_path, "<f4")
        point_cloud = point_cloud.reshape(-1, channels)
        return point_cloud
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def load_and_transform_point_cloud(file_path, rxyz_offset, xyz_offset, channels=7):
    """
    Load a point cloud from a binary file and apply transformations.

    Args:
        file_path: Path to the binary file
        rxyz_offset: Rotation offset in radians [rx, ry, rz]
        xyz_offset: Translation offset [x, y, z]
        channels: Number of channels in the point cloud

    Returns:
        transformed_point_cloud: Transformed point cloud
    """
    # Load point cloud
    try:
        point_cloud = np.fromfile(file_path, "<f4")
        point_cloud = point_cloud.reshape(-1, channels)

        # Extract XYZ coordinates
        xyz = point_cloud[:, :3]

        # Apply rotation
        r = Rotation.from_euler("xyz", rxyz_offset, degrees=False)
        rotated_xyz = r.apply(xyz)

        # Apply translation
        transformed_xyz = rotated_xyz + xyz_offset

        # Update point cloud with transformed coordinates
        transformed_point_cloud = point_cloud.copy()
        transformed_point_cloud[:, :3] = transformed_xyz

        return transformed_point_cloud

    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def load_ground_truth(file_path):
    """
    Load ground truth data from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        bounding_boxes: List of bounding box dictionaries
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Extract bounding box information
        bounding_boxes = data.get("bounding_boxes", [])
        return bounding_boxes
    except Exception as e:
        print(f"Error loading ground truth file {file_path}: {e}")
        return []


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def visualize_point_cloud(point_cloud, output_prefix, point_size=0.1, alpha=0.5):
    """
    Generate multi-view visualizations of a point cloud and save them as PNG files.

    Args:
        point_cloud: numpy array of shape (N, C) where N is the number of points
        output_prefix: prefix for output filenames
        point_size: size of points in the visualization
        alpha: transparency of points
    """
    # Extract x, y, z coordinates
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # Get intensity if available (for coloring)
    if point_cloud.shape[1] > 3:
        intensity = point_cloud[:, 3]
    else:
        intensity = np.ones_like(x)

    # Calculate bounds for consistent scaling
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    # Define different viewing angles
    views = [
        {"elev": 30, "azim": 45, "name": "perspective"},
        {"elev": 90, "azim": 0, "name": "top"},
        {"elev": 0, "azim": 0, "name": "front"},
        {"elev": 0, "azim": 90, "name": "side"},
    ]

    # Create visualizations for each view
    for view in views:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Plot the point cloud
        scatter = ax.scatter(
            x, y, z, c=intensity, s=point_size, alpha=alpha, cmap="viridis"
        )

        # Set axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set the viewing angle
        ax.view_init(elev=view["elev"], azim=view["azim"])

        # Set equal scaling for all axes
        max_range = max([x_max - x_min, y_max - y_min, z_max - z_min])
        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        mid_z = (z_max + z_min) / 2
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

        # Add a colorbar
        fig.colorbar(scatter, ax=ax, label="Intensity")

        # Set title
        ax.set_title(f'Polygon Filtered Point Cloud - {view["name"]} view')

        # Save the figure
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_{view["name"]}.png', dpi=300)
        plt.close(fig)

    # Create a combined visualization with all views
    fig = plt.figure(figsize=(16, 12))

    for i, view in enumerate(views):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")

        # Plot the point cloud
        scatter = ax.scatter(
            x, y, z, c=intensity, s=point_size, alpha=alpha, cmap="viridis"
        )

        # Set axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set the viewing angle
        ax.view_init(elev=view["elev"], azim=view["azim"])

        # Set equal scaling for all axes
        max_range = max([x_max - x_min, y_max - y_min, z_max - z_min])
        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        mid_z = (z_max + z_min) / 2
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

        # Set title
        ax.set_title(f'{view["name"]} view')

    # Add a colorbar
    cbar = fig.colorbar(scatter, ax=fig.axes, label="Intensity", shrink=0.6)

    # Set overall title
    plt.suptitle("Polygon Filtered Point Cloud - Multiple Views", fontsize=16)

    # Save the combined figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig(f"{output_prefix}_combined.png", dpi=300)
    plt.close(fig)


def filter_bounding_boxes_by_polygon(bounding_boxes, point_cloud, polygon_mask):
    """
    Filter bounding boxes based on whether they contain any points after polygon filtering.

    Args:
        bounding_boxes: List of bounding box dictionaries
        point_cloud: Numpy array of shape (N, channels) containing the original point cloud
        polygon_mask: Boolean mask indicating which points are inside the polygon

    Returns:
        filtered_boxes: List of bounding box dictionaries that contain points after polygon filtering
    """
    filtered_boxes = []

    # Apply polygon mask to get filtered point cloud
    filtered_point_cloud = point_cloud[polygon_mask]
    # print(f"len(filtered_point_cloud): {len(filtered_point_cloud)}")

    for box in bounding_boxes:
        # Extract box parameters
        center_x = box["center"]["x"]
        center_y = box["center"]["y"]
        center_z = box["center"]["z"]
        width = box["width"]  # X-axis
        length = box["length"]  # Y-axis
        height = box["height"]  # Z-axis
        angle = box["angle"]  # Counter-clockwise rotation around Z-axis

        # Extract points from filtered point cloud
        points = filtered_point_cloud[:, :3].copy()

        # Translate points to have the box center at origin
        centered_points = points - np.array([center_x, center_y, center_z])

        # Rotate points around Z-axis to align with box orientation
        rotation_matrix = np.array(
            [
                [np.cos(-angle), -np.sin(-angle), 0],
                [np.sin(-angle), np.cos(-angle), 0],
                [0, 0, 1],
            ]
        )

        rotated_points = np.dot(centered_points, rotation_matrix.T)

        # Check which points are inside the box
        half_width = width / 2
        half_length = length / 2
        half_height = height / 2

        box_mask = (
            (rotated_points[:, 0] >= -half_width)
            & (rotated_points[:, 0] <= half_width)
            & (rotated_points[:, 1] >= -half_length)
            & (rotated_points[:, 1] <= half_length)
            & (rotated_points[:, 2] >= -half_height)
            & (rotated_points[:, 2] <= half_height)
        )

        # If any points are inside the box, keep this box
        if np.any(box_mask):
            filtered_boxes.append(box)

    return filtered_boxes


def create_ground_truth_mask(point_cloud, bounding_boxes):
    """
    Create a ground truth mask based on bounding boxes.

    Args:
        point_cloud: Numpy array of shape (N, channels)
        bounding_boxes: List of bounding box dictionaries

    Returns:
        ground_truth_mask: Boolean mask indicating foreground points (True = foreground)
    """
    # Initialize mask (all points are background by default)
    ground_truth_mask = np.zeros(len(point_cloud), dtype=bool)

    # For each bounding box, mark points inside as foreground
    for box in bounding_boxes:
        # Extract box parameters
        center_x = box["center"]["x"]
        center_y = box["center"]["y"]
        center_z = box["center"]["z"]
        width = box["width"]  # X-axis
        length = box["length"]  # Y-axis
        height = box["height"]  # Z-axis
        angle = box["angle"]  # Counter-clockwise rotation around Z-axis

        # Extract points
        points = point_cloud[:, :3]

        # Translate points to have the box center at origin
        centered_points = points - np.array([center_x, center_y, center_z])

        # Rotate points around Z-axis to align with box orientation
        rotation_matrix = np.array(
            [
                [np.cos(-angle), -np.sin(-angle), 0],
                [np.sin(-angle), np.cos(-angle), 0],
                [0, 0, 1],
            ]
        )

        rotated_points = np.dot(centered_points, rotation_matrix.T)

        # Check which points are inside the box
        half_width = width / 2
        half_length = length / 2
        half_height = height / 2

        box_mask = (
            (rotated_points[:, 0] >= -half_width)
            & (rotated_points[:, 0] <= half_width)
            & (rotated_points[:, 1] >= -half_length)
            & (rotated_points[:, 1] <= half_length)
            & (rotated_points[:, 2] >= -half_height)
            & (rotated_points[:, 2] <= half_height)
        )

        # Update the ground truth mask (union of all box masks)
        ground_truth_mask = ground_truth_mask | box_mask

    return ground_truth_mask


def evaluate_filtering_with_polygon(
    original_pc_dir,
    filtered_pc_dir,
    ground_truth_dir,
    output_dir,
    strategy_name,
    rxyz_offset=None,
    xyz_offset=None,
):
    """
    Evaluate the background filtering algorithm with additional point-in-polygon filtering.
    Ground truth bounding boxes are filtered based on polygon filtering results.

    Args:
        original_pc_dir: Directory containing original point cloud files
        filtered_pc_dir: Directory containing filtered point cloud files
        ground_truth_dir: Directory containing ground truth files
        output_dir: Directory to save evaluation results
        strategy_name: Name of the filtering strategy being evaluated
        rxyz_offset: Optional rotation offset for point cloud transformation
        xyz_offset: Optional translation offset for point cloud transformation
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define channels
    channels = ["x", "y", "z", "intensity", "range", "ambient", "reflectivity"]

    # Define the polygon regions
    polygon_data = {
        "valid_polygon": [
            [
                [-40.15195465, 21.1319828],
                [-10.74100018, 40.74986267],
                [13.46200657, 16.93098259],
                [17.09595299, -17.68921471],
                [5.38323689, -30.59368324],
                [-34.55211258, -27.11372948],
                [-31.93655586, -12.17062569],
                [-38.57237244, 2.28900456],
                [-36.83390808, 4.86979866],
                [-36.27791977, 15.94988251],
                [-40.15195465, 21.1319828],
            ]
        ],
        "invalid_polygon": [
            [
                [14.06089973, 4.8097167],
                [7.93772078, 7.14294147],
                [1.01086497, 14.94453907],
                [-3.50242281, 20.53687096],
                [-9.6521225, 28.86942101],
                [-19.97583771, 43.58998489],
                [-27.50993919, 63.10228729],
                [15.24509335, 45.59996796],
                [34.25726318, 41.24291611],
                [68.79250336, 5.66090012],
                [72.19792175, 3.14087272],
                [71.34347534, 2.3449049],
                [14.06089973, 4.8097167],
            ],
            [
                [12.28229237, -63.71089172],
                [2.52607179, -44.00482941],
                [2.72164583, -28.84677696],
                [4.55272865, -24.5709095],
                [12.55642128, -13.50995064],
                [23.44750404, -5.89971209],
                [25.3551178, -5.90315723],
                [57.7483902, -15.74698639],
                [23.77217865, -61.75832748],
                [12.28229237, -63.71089172],
            ],
            [
                [4.442984104, 16.430439],
                [6.122646809, 15.61476994],
                [8.548356056, 20.53249359],
                [8.693145752, 21.18700981],
                [0.881210983, 29.20153427],
                [-8.533398628, 38.50027084],
                [-10.85100555, 40.51189423],
                [-11.37808037, 40.2118454],
                [-5.307154655, 29.45556831],
                [2.023125172, 19.02893066],
                [4.442984104, 16.430439],
            ],
            [
                [-30.444232940674, 26.416440963745],
                [-28.086193084717, 23.437868118286],
                [-27.477630615234, 21.470422744751],
                [-25.11106300354, 21.431261062622],
                [-20.096929550171, 23.804498672485],
                [-17.585401535034, 25.758008956909],
                [-15.988693237305, 27.667018890381],
                [-15.392958641052, 28.876237869263],
                [-14.995494842529, 31.180606842041],
                [-17.254079818726, 36.121360778809],
                [-24.440565109253, 31.578065872192],
                [-29.277206420898, 28.077560424805],
                [-30.444232940674, 26.416440963745],
            ],
            [
                [-38.180648803711, 1.788941502571],
                [-35.749618530273, 4.014994621277],
                [-34.713665008545, 5.220662593842],
                [-33.918006896973, 7.263592243195],
                [-33.498958587646, 9.368733406067],
                [-33.63676071167, 11.10545539856],
                [-33.961994171143, 12.720592498779],
                [-35.797843933105, 15.43830871582],
                [-40.076484680176, 21.147472381592],
                [-38.498512268066, 2.323858499527],
                [-38.180648803711, 1.788941502571],
            ],
            [
                [-16.639026641846, 0.935104370117],
                [-15.261664390564, 0.210890129209],
                [-13.805727005005, -0.093149825931],
                [-11.88335609436, -0.201968207955],
                [-9.198356628418, 0.4722699821],
                [-8.04657459259, 1.241767048836],
                [-6.759798049927, 2.348032712936],
                [-5.439073562622, 4.268807411194],
                [-4.757977485657, 6.454925060272],
                [-4.6365442276, 7.849351406097],
                [-4.797882080078, 9.365277290344],
                [-5.117852210999, 10.638592720032],
                [-5.817307472229, 11.974423408508],
                [-6.419658660889, 12.807320594788],
                [-7.032185077667, 13.458583831787],
                [-7.678003787994, 14.011194229126],
                [-9.005589485168, 14.841164588928],
                [-11.711576461792, 15.167936325073],
                [-13.597177505493, 15.10563659668],
                [-16.376735687256, 14.676927566528],
                [-16.926254272461, 14.410374641418],
                [-18.869678497314, 11.830725669861],
                [-19.348215103149, 10.90552520752],
                [-19.259969711304, 9.121381759644],
                [-19.661476135254, 6.571251869202],
                [-19.579166412354, 6.209203243256],
                [-18.690006256104, 4.354350090027],
                [-18.486806869507, 3.472965240479],
                [-16.639026641846, 0.935104370117],
            ],
            [
                [-33.163913726807, -5.605916976929],
                [-33.621814727783, -7.393526077271],
                [-33.417724609375, -8.386545181274],
                [-32.9040184021, -19.143211364746],
                [-33.302612304688, -20.666603088379],
                [-31.806688308716, -20.026891708374],
                [-23.317153930664, -15.723935127258],
                [-19.98571395874, -14.077204704285],
                [-12.80609703064, -14.642774581909],
                [-6.687733650208, -15.232246398926],
                [-2.711044549942, -15.564800262451],
                [-0.162731736898, -14.410130500793],
                [2.42479634285, -11.657358169556],
                [2.924674510956, -10.765809059143],
                [3.044909477234, -10.285830497742],
                [2.684117794037, -9.226300239563],
                [2.496987819672, -8.919030189514],
                [2.115324020386, -8.605619430542],
                [-4.727807998657, -8.357259750366],
                [-7.544390201569, -9.461579322815],
                [-12.056007385254, -10.05403137207],
                [-15.986765861511, -9.561311721802],
                [-21.258243560791, -7.627488613129],
                [-24.667051315308, -6.048822402954],
                [-28.584348678589, -5.258367061615],
                [-33.163913726807, -5.605916976929],
            ],
            [
                [15.90147495, -12.09426117],
                [8.405247688, -23.13452339],
                [6.570725918, -26.60411644],
                [6.005459785, -29.71888542],
                [8.603281975, -26.88464355],
                [10.13683128, -25.31586266],
                [13.22369099, -21.87047577],
                [16.1186142, -18.54201508],
                [16.9290638, -16.82001495],
                [16.67486191, -14.42174625],
                [16.28601456, -12.72824574],
                [15.52775574, -12.03287506],
                [14.56326389, -12.92064285],
                [15.90147495, -12.09426117],
            ],
        ],
    }

    # Convert polygon data to numpy arrays
    valid_polygon = np.array(polygon_data["valid_polygon"][0], dtype=np.float32)
    invalid_polygons = [
        np.array(poly, dtype=np.float32) for poly in polygon_data["invalid_polygon"]
    ]

    # Initialize results dictionary
    results = {
        "file": [],
        "accuracy_before_polygon": [],
        "precision_before_polygon": [],
        "recall_before_polygon": [],
        "accuracy_after_polygon": [],
        "precision_after_polygon": [],
        "recall_after_polygon": [],
        "tp_before": [],
        "tn_before": [],
        "fp_before": [],
        "fn_before": [],
        "tp_after": [],
        "tn_after": [],
        "fp_after": [],
        "fn_after": [],
        "total_points": [],
        "points_after_polygon": [],
        "original_boxes": [],
        "filtered_boxes": [],
    }

    # Get list of filtered point cloud files
    filtered_files = [f for f in os.listdir(filtered_pc_dir) if f.endswith(".bin")]

    frame=0
    for filtered_file in tqdm(filtered_files, desc=f"Evaluating {strategy_name}"):
        frame+=1
        # Extract original file name from filtered file name
        original_file = filtered_file.replace("frame_", "")

        # Get corresponding ground truth file
        ground_truth_file = original_file.replace(".bin", ".bin.json")

        # Construct full paths
        original_path = os.path.join(original_pc_dir, original_file)
        filtered_path = os.path.join(filtered_pc_dir, filtered_file)

        # Find the correct subdirectory in ground_truth_dir that contains the ground truth file
        ground_truth_path = None
        for root, dirs, files in os.walk(ground_truth_dir):
            if ground_truth_file in files:
                ground_truth_path = os.path.join(root, ground_truth_file)
                break

        if not ground_truth_path:
            print(
                f"Warning: Ground truth file for {original_file} not found. Skipping."
            )
            continue

        # Load point clouds
        original_pc = load_and_transform_point_cloud(
            original_path, rxyz_offset, xyz_offset, len(channels)
        )

        filtered_pc = load_and_transform_point_cloud(
            filtered_path, rxyz_offset, xyz_offset, len(channels)
        )

        if original_pc is None or filtered_pc is None:
            print(
                f"Warning: Could not load point clouds for {original_file}. Skipping."
            )
            continue

        # Load ground truth bounding boxes
        original_boxes = load_ground_truth(ground_truth_path)
        print(f"len(original_boxes): {len(original_boxes)}")

        if not original_boxes:
            print(f"Warning: No bounding boxes found for {original_file}. Skipping.")
            continue

        # Apply polygon filtering to the original point cloud
        polygon_filtered_original_pc, polygon_mask = filter_point_cloud_by_polygon(
            original_pc, valid_polygon, invalid_polygons
        )

        # # # To use this function in the script, add the following code where you want to generate the visualizations:
        # # # After the line where polygon_filtered_original_pc is created:
        # visualize_point_cloud(
        #     polygon_filtered_original_pc, "polygon_filtered_point_cloud"
        # )

        # exit()
        # Filter bounding boxes based on polygon filtering results
        filtered_boxes = filter_bounding_boxes_by_polygon(
            original_boxes, original_pc, polygon_mask
        )
        # print(f"len(filtered_boxes): {len(filtered_boxes)}")
        # print(filtered_boxes)

        visualize = False
        # for box in filtered_boxes:
        #     if box['object_id'] == "publicminibus":
        #         visualize = True
        #         break
            
        # for box in filtered_boxes:
        #     if box['center']['x'] > 0 and box['object_id'] == "publicminibus":
        #         visualize = True
        #         break

        # Create ground truth mask for original point cloud using original boxes
        original_gt_mask = create_ground_truth_mask(original_pc, original_boxes)

        # Create ground truth mask for filtered point cloud using original boxes
        filtered_gt_mask_original_boxes = create_ground_truth_mask(
            filtered_pc, original_boxes
        )

        # Calculate metrics before polygon filtering (using original boxes)
        tp_before = np.sum(filtered_gt_mask_original_boxes)
        fp_before = len(filtered_pc) - tp_before
        fn_before = np.sum(original_gt_mask) - tp_before
        tn_before = len(original_pc) - np.sum(original_gt_mask) - fp_before

        # Calculate metrics before polygon filtering
        accuracy_before = (
            (tp_before + tn_before) / (tp_before + tn_before + fp_before + fn_before)
            if (tp_before + tn_before + fp_before + fn_before) > 0
            else 0
        )
        precision_before = (
            tp_before / (tp_before + fp_before) if (tp_before + fp_before) > 0 else 0
        )
        recall_before = (
            tp_before / (tp_before + fn_before) if (tp_before + fn_before) > 0 else 0
        )

        # Apply polygon filtering to the filtered point cloud
        polygon_filtered_pc, polygon_mask_filtered = filter_point_cloud_by_polygon(
            filtered_pc, valid_polygon, invalid_polygons
        )
        
        if visualize:
            visualize_point_cloud(
                polygon_filtered_original_pc, f"polygon_filtered_point_cloud_{frame}"
            )

            visualize_point_cloud(
                filtered_pc, f"filtered_pc_{frame}"
            )

            visualize_point_cloud(
                polygon_filtered_pc, f"polygon_filtered_pc_{frame}"
            )

            # exit()
        # Create ground truth mask for polygon-filtered point cloud using filtered boxes
        polygon_filtered_gt_mask = create_ground_truth_mask(
            polygon_filtered_pc, filtered_boxes
        )

        # Create ground truth mask for polygon_filtered_original_pc using filtered boxes
        original_gt_mask_filtered_boxes = create_ground_truth_mask(
            polygon_filtered_original_pc, filtered_boxes
        )

        # Calculate metrics after polygon filtering (using filtered boxes)
        tp_after = np.sum(polygon_filtered_gt_mask)
        fp_after = len(polygon_filtered_pc) - tp_after
        fn_after = np.sum(original_gt_mask_filtered_boxes) - tp_after
        tn_after = (
            len(polygon_filtered_original_pc)
            - np.sum(original_gt_mask_filtered_boxes)
            - fp_after
        )

        # Calculate metrics after polygon filtering
        accuracy_after = (
            (tp_after + tn_after) / (tp_after + tn_after + fp_after + fn_after)
            if (tp_after + tn_after + fp_after + fn_after) > 0
            else 0
        )
        precision_after = (
            tp_after / (tp_after + fp_after) if (tp_after + fp_after) > 0 else 0
        )
        recall_after = (
            tp_after / (tp_after + fn_after) if (tp_after + fn_after) > 0 else 0
        )

        # Store results
        results["file"].append(original_file)
        results["accuracy_before_polygon"].append(accuracy_before)
        results["precision_before_polygon"].append(precision_before)
        results["recall_before_polygon"].append(recall_before)
        results["accuracy_after_polygon"].append(accuracy_after)
        results["precision_after_polygon"].append(precision_after)
        results["recall_after_polygon"].append(recall_after)
        results["tp_before"].append(tp_before)
        results["tn_before"].append(tn_before)
        results["fp_before"].append(fp_before)
        results["fn_before"].append(fn_before)
        results["tp_after"].append(tp_after)
        results["tn_after"].append(tn_after)
        results["fp_after"].append(fp_after)
        results["fn_after"].append(fn_after)
        results["total_points"].append(len(original_pc))
        results["points_after_polygon"].append(len(polygon_filtered_pc))
        results["original_boxes"].append(len(original_boxes))
        results["filtered_boxes"].append(len(filtered_boxes))

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Calculate average metrics
    avg_accuracy_before = results_df["accuracy_before_polygon"].mean()
    avg_precision_before = results_df["precision_before_polygon"].mean()
    avg_recall_before = results_df["recall_before_polygon"].mean()
    avg_accuracy_after = results_df["accuracy_after_polygon"].mean()
    avg_precision_after = results_df["precision_after_polygon"].mean()
    avg_recall_after = results_df["recall_after_polygon"].mean()

    # Add summary row
    summary = pd.DataFrame(
        {
            "file": ["AVERAGE"],
            "accuracy_before_polygon": [avg_accuracy_before],
            "precision_before_polygon": [avg_precision_before],
            "recall_before_polygon": [avg_recall_before],
            "accuracy_after_polygon": [avg_accuracy_after],
            "precision_after_polygon": [avg_precision_after],
            "recall_after_polygon": [avg_recall_after],
            "tp_before": [results_df["tp_before"].sum()],
            "tn_before": [results_df["tn_before"].sum()],
            "fp_before": [results_df["fp_before"].sum()],
            "fn_before": [results_df["fn_before"].sum()],
            "tp_after": [results_df["tp_after"].sum()],
            "tn_after": [results_df["tn_after"].sum()],
            "fp_after": [results_df["fp_after"].sum()],
            "fn_after": [results_df["fn_after"].sum()],
            "total_points": [results_df["total_points"].sum()],
            "points_after_polygon": [results_df["points_after_polygon"].sum()],
            "original_boxes": [results_df["original_boxes"].sum()],
            "filtered_boxes": [results_df["filtered_boxes"].sum()],
        }
    )

    results_df = pd.concat([results_df, summary], ignore_index=True)

    # Save results to CSV
    results_file = os.path.join(
        output_dir, f"{strategy_name}_evaluation_with_filtered_boxes.csv"
    )
    results_df.to_csv(results_file, index=False)

    # Generate plots
    generate_evaluation_plots_with_filtered_boxes(results_df, output_dir, strategy_name)

    print(f"Evaluation results for {strategy_name}:")
    print(f"  Before Polygon Filtering (Original Boxes):")
    print(f"    Average Accuracy: {avg_accuracy_before:.4f}")
    print(f"    Average Precision: {avg_precision_before:.4f}")
    print(f"    Average Recall: {avg_recall_before:.4f}")
    print(f"  After Polygon Filtering (Filtered Boxes):")
    print(f"    Average Accuracy: {avg_accuracy_after:.4f}")
    print(f"    Average Precision: {avg_precision_after:.4f}")
    print(f"    Average Recall: {avg_recall_after:.4f}")
    print(f"  Total original boxes: {results_df['original_boxes'].sum()}")
    print(f"  Total filtered boxes: {results_df['filtered_boxes'].sum()}")
    print(f"Results saved to {results_file}")

    return results_df


def generate_evaluation_plots_with_filtered_boxes(
    results_df, output_dir, strategy_name
):
    """
    Generate plots from evaluation results with filtered bounding boxes.

    Args:
        results_df: DataFrame with evaluation results
        output_dir: Directory to save plots
        strategy_name: Name of the filtering strategy
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Remove the summary row for plotting
    plot_df = results_df[results_df["file"] != "AVERAGE"].copy()

    # Plot accuracy, precision, and recall before and after polygon filtering
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 1, 1)
    plt.plot(
        plot_df["file"],
        plot_df["accuracy_before_polygon"],
        marker="o",
        label="Accuracy",
    )
    plt.plot(
        plot_df["file"],
        plot_df["precision_before_polygon"],
        marker="s",
        label="Precision",
    )
    plt.plot(
        plot_df["file"], plot_df["recall_before_polygon"], marker="^", label="Recall"
    )
    plt.xlabel("File")
    plt.ylabel("Score")
    plt.title(f"{strategy_name} - Metrics Before Polygon Filtering (Original Boxes)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=90)

    plt.subplot(2, 1, 2)
    plt.plot(
        plot_df["file"], plot_df["accuracy_after_polygon"], marker="o", label="Accuracy"
    )
    plt.plot(
        plot_df["file"],
        plot_df["precision_after_polygon"],
        marker="s",
        label="Precision",
    )
    plt.plot(
        plot_df["file"], plot_df["recall_after_polygon"], marker="^", label="Recall"
    )
    plt.xlabel("File")
    plt.ylabel("Score")
    plt.title(f"{strategy_name} - Metrics After Polygon Filtering (Filtered Boxes)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            plots_dir, f"{strategy_name}_metrics_comparison_filtered_boxes.png"
        )
    )

    # Plot confusion matrix components before polygon filtering
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 1, 1)
    plt.bar(plot_df["file"], plot_df["tp_before"], label="True Positives")
    plt.bar(
        plot_df["file"],
        plot_df["tn_before"],
        bottom=plot_df["tp_before"],
        label="True Negatives",
    )
    plt.bar(
        plot_df["file"],
        plot_df["fp_before"],
        bottom=plot_df["tp_before"] + plot_df["tn_before"],
        label="False Positives",
    )
    plt.bar(
        plot_df["file"],
        plot_df["fn_before"],
        bottom=plot_df["tp_before"] + plot_df["tn_before"] + plot_df["fp_before"],
        label="False Negatives",
    )
    plt.xlabel("File")
    plt.ylabel("Count")
    plt.title(
        f"{strategy_name} - Confusion Matrix Before Polygon Filtering (Original Boxes)"
    )
    plt.legend()
    plt.grid(True, axis="y")
    plt.xticks(rotation=90)

    plt.subplot(2, 1, 2)
    plt.bar(plot_df["file"], plot_df["tp_after"], label="True Positives")
    plt.bar(
        plot_df["file"],
        plot_df["tn_after"],
        bottom=plot_df["tp_after"],
        label="True Negatives",
    )
    plt.bar(
        plot_df["file"],
        plot_df["fp_after"],
        bottom=plot_df["tp_after"] + plot_df["tn_after"],
        label="False Positives",
    )
    plt.bar(
        plot_df["file"],
        plot_df["fn_after"],
        bottom=plot_df["tp_after"] + plot_df["tn_after"] + plot_df["fp_after"],
        label="False Negatives",
    )
    plt.xlabel("File")
    plt.ylabel("Count")
    plt.title(
        f"{strategy_name} - Confusion Matrix After Polygon Filtering (Filtered Boxes)"
    )
    plt.legend()
    plt.grid(True, axis="y")
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            plots_dir, f"{strategy_name}_confusion_comparison_filtered_boxes.png"
        )
    )

    # Plot percentage of points in each category before and after polygon filtering
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 1, 1)
    total_points = plot_df["total_points"]
    plt.bar(
        plot_df["file"],
        100 * plot_df["tp_before"] / total_points,
        label="True Positives",
    )
    plt.bar(
        plot_df["file"],
        100 * plot_df["tn_before"] / total_points,
        bottom=100 * plot_df["tp_before"] / total_points,
        label="True Negatives",
    )
    plt.bar(
        plot_df["file"],
        100 * plot_df["fp_before"] / total_points,
        bottom=100 * (plot_df["tp_before"] + plot_df["tn_before"]) / total_points,
        label="False Positives",
    )
    plt.bar(
        plot_df["file"],
        100 * plot_df["fn_before"] / total_points,
        bottom=100
        * (plot_df["tp_before"] + plot_df["tn_before"] + plot_df["fp_before"])
        / total_points,
        label="False Negatives",
    )
    plt.xlabel("File")
    plt.ylabel("Percentage")
    plt.title(
        f"{strategy_name} - Point Classification Percentages Before Polygon Filtering (Original Boxes)"
    )
    plt.legend()
    plt.grid(True, axis="y")
    plt.xticks(rotation=90)

    plt.subplot(2, 1, 2)
    plt.bar(
        plot_df["file"],
        100 * plot_df["tp_after"] / total_points,
        label="True Positives",
    )
    plt.bar(
        plot_df["file"],
        100 * plot_df["tn_after"] / total_points,
        bottom=100 * plot_df["tp_after"] / total_points,
        label="True Negatives",
    )
    plt.bar(
        plot_df["file"],
        100 * plot_df["fp_after"] / total_points,
        bottom=100 * (plot_df["tp_after"] + plot_df["tn_after"]) / total_points,
        label="False Positives",
    )
    plt.bar(
        plot_df["file"],
        100 * plot_df["fn_after"] / total_points,
        bottom=100
        * (plot_df["tp_after"] + plot_df["tn_after"] + plot_df["fp_after"])
        / total_points,
        label="False Negatives",
    )
    plt.xlabel("File")
    plt.ylabel("Percentage")
    plt.title(
        f"{strategy_name} - Point Classification Percentages After Polygon Filtering (Filtered Boxes)"
    )
    plt.legend()
    plt.grid(True, axis="y")
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            plots_dir, f"{strategy_name}_percentages_comparison_filtered_boxes.png"
        )
    )

    # Plot direct comparison of metrics before and after polygon filtering
    plt.figure(figsize=(12, 8))

    # Prepare data for bar chart
    metrics = ["Accuracy", "Precision", "Recall"]
    before_values = [
        results_df.iloc[-1]["accuracy_before_polygon"],
        results_df.iloc[-1]["precision_before_polygon"],
        results_df.iloc[-1]["recall_before_polygon"],
    ]
    after_values = [
        results_df.iloc[-1]["accuracy_after_polygon"],
        results_df.iloc[-1]["precision_after_polygon"],
        results_df.iloc[-1]["recall_after_polygon"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(
        x - width / 2, before_values, width, label="Before Polygon (Original Boxes)"
    )
    plt.bar(x + width / 2, after_values, width, label="After Polygon (Filtered Boxes)")

    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.title(f"{strategy_name} - Performance Metrics Comparison")
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, axis="y")

    # Add value labels on top of bars
    for i, v in enumerate(before_values):
        plt.text(i - width / 2, v + 0.01, f"{v:.4f}", ha="center")

    for i, v in enumerate(after_values):
        plt.text(i + width / 2, v + 0.01, f"{v:.4f}", ha="center")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            plots_dir, f"{strategy_name}_metrics_bar_comparison_filtered_boxes.png"
        )
    )

    # Plot bounding box filtering statistics
    plt.figure(figsize=(12, 6))

    # Plot number of bounding boxes before and after filtering
    plt.bar(plot_df["file"], plot_df["original_boxes"], label="Original Boxes")
    plt.bar(plot_df["file"], plot_df["filtered_boxes"], label="Filtered Boxes")
    plt.xlabel("File")
    plt.ylabel("Number of Bounding Boxes")
    plt.title(f"{strategy_name} - Bounding Box Filtering Statistics")
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{strategy_name}_bounding_box_stats.png"))

    # Plot percentage of bounding boxes kept after filtering
    plt.figure(figsize=(12, 6))

    percentage_boxes_kept = 100 * plot_df["filtered_boxes"] / plot_df["original_boxes"]
    plt.bar(plot_df["file"], percentage_boxes_kept, color="orange", alpha=0.7)
    plt.axhline(
        y=percentage_boxes_kept.mean(),
        color="r",
        linestyle="-",
        label=f"Avg: {percentage_boxes_kept.mean():.2f}%",
    )
    plt.xlabel("File")
    plt.ylabel("Bounding Boxes Kept (%)")
    plt.title(
        f"{strategy_name} - Percentage of Bounding Boxes Kept After Polygon Filtering"
    )
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{strategy_name}_boxes_kept_percentage.png"))


def compare_strategies_with_filtered_boxes(evaluation_dir, output_dir):
    """
    Compare different filtering strategies based on evaluation results with filtered bounding boxes.

    Args:
        evaluation_dir: Directory containing evaluation results
        output_dir: Directory to save comparison results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all evaluation CSV files
    csv_files = [
        f
        for f in os.listdir(evaluation_dir)
        if f.endswith("_evaluation_with_filtered_boxes.csv")
    ]

    if not csv_files:
        print("No evaluation results found.")
        return

    # Initialize comparison dictionary
    comparison = {
        "strategy": [],
        "avg_accuracy_before": [],
        "avg_precision_before": [],
        "avg_recall_before": [],
        "avg_accuracy_after": [],
        "avg_precision_after": [],
        "avg_recall_after": [],
        "accuracy_improvement": [],
        "precision_improvement": [],
        "recall_improvement": [],
        "total_points": [],
        "points_after_polygon": [],
        "percentage_points_kept": [],
        "original_boxes": [],
        "filtered_boxes": [],
        "percentage_boxes_kept": [],
    }

    # Load each evaluation file and extract summary metrics
    for csv_file in csv_files:
        # Extract strategy name from file name
        strategy_name = csv_file.replace("_evaluation_with_filtered_boxes.csv", "")

        # Load evaluation results
        results_df = pd.read_csv(os.path.join(evaluation_dir, csv_file))

        # Get summary row (last row)
        summary = results_df.iloc[-1]

        # Calculate improvements
        accuracy_improvement = (
            summary["accuracy_after_polygon"] - summary["accuracy_before_polygon"]
        )
        precision_improvement = (
            summary["precision_after_polygon"] - summary["precision_before_polygon"]
        )
        recall_improvement = (
            summary["recall_after_polygon"] - summary["recall_before_polygon"]
        )
        percentage_points_kept = (
            100 * summary["points_after_polygon"] / summary["total_points"]
        )
        percentage_boxes_kept = (
            100 * summary["filtered_boxes"] / summary["original_boxes"]
        )

        # Store comparison metrics
        comparison["strategy"].append(strategy_name)
        comparison["avg_accuracy_before"].append(summary["accuracy_before_polygon"])
        comparison["avg_precision_before"].append(summary["precision_before_polygon"])
        comparison["avg_recall_before"].append(summary["recall_before_polygon"])
        comparison["avg_accuracy_after"].append(summary["accuracy_after_polygon"])
        comparison["avg_precision_after"].append(summary["precision_after_polygon"])
        comparison["avg_recall_after"].append(summary["recall_after_polygon"])
        comparison["accuracy_improvement"].append(accuracy_improvement)
        comparison["precision_improvement"].append(precision_improvement)
        comparison["recall_improvement"].append(recall_improvement)
        comparison["total_points"].append(summary["total_points"])
        comparison["points_after_polygon"].append(summary["points_after_polygon"])
        comparison["percentage_points_kept"].append(percentage_points_kept)
        comparison["original_boxes"].append(summary["original_boxes"])
        comparison["filtered_boxes"].append(summary["filtered_boxes"])
        comparison["percentage_boxes_kept"].append(percentage_boxes_kept)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison)

    # Sort by accuracy after polygon filtering (descending)
    comparison_df = comparison_df.sort_values("avg_accuracy_after", ascending=False)

    # Save comparison to CSV
    comparison_file = os.path.join(
        output_dir, "strategy_comparison_with_filtered_boxes.csv"
    )
    comparison_df.to_csv(comparison_file, index=False)

    # Generate comparison plots
    generate_comparison_plots_with_filtered_boxes(comparison_df, output_dir)

    print("Strategy comparison with filtered bounding boxes:")
    print(
        comparison_df[
            [
                "strategy",
                "avg_accuracy_before",
                "avg_accuracy_after",
                "accuracy_improvement",
            ]
        ]
    )
    print(f"Comparison saved to {comparison_file}")

    return comparison_df


def generate_comparison_plots_with_filtered_boxes(comparison_df, output_dir):
    """
    Generate plots comparing different filtering strategies with filtered bounding boxes.

    Args:
        comparison_df: DataFrame with comparison results
        output_dir: Directory to save plots
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot average metrics before and after polygon filtering
    plt.figure(figsize=(15, 10))

    # Set up the plot
    x = np.arange(len(comparison_df))
    width = 0.15

    # Plot accuracy
    plt.subplot(3, 1, 1)
    plt.bar(
        x - width, comparison_df["avg_accuracy_before"], width, label="Before Polygon"
    )
    plt.bar(x, comparison_df["avg_accuracy_after"], width, label="After Polygon")
    plt.bar(
        x + width, comparison_df["accuracy_improvement"], width, label="Improvement"
    )
    plt.xlabel("Strategy")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.xticks(x, comparison_df["strategy"])
    plt.legend()
    plt.grid(True, axis="y")

    # Plot precision
    plt.subplot(3, 1, 2)
    plt.bar(
        x - width, comparison_df["avg_precision_before"], width, label="Before Polygon"
    )
    plt.bar(x, comparison_df["avg_precision_after"], width, label="After Polygon")
    plt.bar(
        x + width, comparison_df["precision_improvement"], width, label="Improvement"
    )
    plt.xlabel("Strategy")
    plt.ylabel("Precision")
    plt.title("Precision Comparison")
    plt.xticks(x, comparison_df["strategy"])
    plt.legend()
    plt.grid(True, axis="y")

    # Plot recall
    plt.subplot(3, 1, 3)
    plt.bar(
        x - width, comparison_df["avg_recall_before"], width, label="Before Polygon"
    )
    plt.bar(x, comparison_df["avg_recall_after"], width, label="After Polygon")
    plt.bar(x + width, comparison_df["recall_improvement"], width, label="Improvement")
    plt.xlabel("Strategy")
    plt.ylabel("Recall")
    plt.title("Recall Comparison")
    plt.xticks(x, comparison_df["strategy"])
    plt.legend()
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_dir, "strategy_metrics_comparison_with_filtered_boxes.png")
    )

    # Plot percentage of points and boxes kept after polygon filtering
    plt.figure(figsize=(12, 6))

    # Set up the plot
    x = np.arange(len(comparison_df))
    width = 0.35

    plt.bar(
        x - width / 2,
        comparison_df["percentage_points_kept"],
        width,
        label="Points Kept (%)",
    )
    plt.bar(
        x + width / 2,
        comparison_df["percentage_boxes_kept"],
        width,
        label="Boxes Kept (%)",
    )

    plt.xlabel("Strategy")
    plt.ylabel("Percentage")
    plt.title("Percentage of Points and Boxes Kept After Polygon Filtering")
    plt.xticks(x, comparison_df["strategy"])
    plt.legend()
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "strategy_percentage_kept_comparison.png"))

    # Plot absolute numbers of points and boxes
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 1, 1)
    plt.bar(
        comparison_df["strategy"], comparison_df["total_points"], label="Total Points"
    )
    plt.bar(
        comparison_df["strategy"],
        comparison_df["points_after_polygon"],
        label="Points After Polygon",
    )
    plt.xlabel("Strategy")
    plt.ylabel("Number of Points")
    plt.title("Point Count Before and After Polygon Filtering")
    plt.legend()
    plt.grid(True, axis="y")
    plt.xticks(rotation=45)

    plt.subplot(2, 1, 2)
    plt.bar(
        comparison_df["strategy"],
        comparison_df["original_boxes"],
        label="Original Boxes",
    )
    plt.bar(
        comparison_df["strategy"],
        comparison_df["filtered_boxes"],
        label="Filtered Boxes",
    )
    plt.xlabel("Strategy")
    plt.ylabel("Number of Boxes")
    plt.title("Bounding Box Count Before and After Filtering")
    plt.legend()
    plt.grid(True, axis="y")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "strategy_count_comparison.png"))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate background filtering algorithm with point-in-polygon filtering and filtered bounding boxes"
    )
    parser.add_argument(
        "--original_pc_dir",
        required=True,
        help="Directory containing original point cloud files",
    )
    parser.add_argument(
        "--filtered_pc_dir",
        required=True,
        help="Directory containing filtered point cloud files",
    )
    parser.add_argument(
        "--ground_truth_dir",
        required=True,
        help="Directory containing ground truth files",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--strategy_name",
        default="default",
        help="Name of the filtering strategy being evaluated",
    )

    args = parser.parse_args()

    # Define transformation parameters (if needed)
    rxyz_offset = [0.0705718, -0.2612746, -0.017035]  # Rotation in radians
    xyz_offset = [0, 0, 5.7]  # Translation

    # Evaluate filtering with polygon filtering and filtered bounding boxes
    evaluate_filtering_with_polygon(
        args.original_pc_dir,
        args.filtered_pc_dir,
        args.ground_truth_dir,
        args.output_dir,
        args.strategy_name,
        rxyz_offset,
        xyz_offset,
    )

    # If multiple strategies have been evaluated, compare them
    compare_strategies_with_filtered_boxes(args.output_dir, args.output_dir)


if __name__ == "__main__":
    main()

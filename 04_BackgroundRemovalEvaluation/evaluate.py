import os
import json
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_and_transform_point_cloud(file_path, rxyz_offset, xyz_offset):
    """
    Load a point cloud from a binary file and apply transformations.

    Args:
        file_path: Path to the binary file
        rxyz_offset: Rotation offset in radians [rx, ry, rz]
        xyz_offset: Translation offset [x, y, z]

    Returns:
        transformed_point_cloud: Transformed point cloud
    """
    # Define channels
    channels = ["x", "y", "z", "intensity", "range", "ambient", "reflectivity"]

    # Load point cloud
    try:
        point_cloud = np.fromfile(file_path, "<f4")
        point_cloud = point_cloud.reshape(-1, len(channels))

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


def extract_points_in_box(point_cloud, box):
    """
    Extract points that fall within a 3D bounding box.

    Args:
        point_cloud: Numpy array of shape (N, 7) containing point cloud data
        box: Dictionary containing bounding box information

    Returns:
        points_in_box: Numpy array containing points inside the bounding box
    """
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
    # Note: We use negative angle because we're rotating the points in the opposite direction
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

    mask = (
        (rotated_points[:, 0] >= -half_width)
        & (rotated_points[:, 0] <= half_width)
        & (rotated_points[:, 1] >= -half_length)
        & (rotated_points[:, 1] <= half_length)
        & (rotated_points[:, 2] >= -half_height)
        & (rotated_points[:, 2] <= half_height)
    )

    return point_cloud[mask]


def process_data(data_dir, label_dir, output_dir):
    """
    Process all point cloud files and extract points within bounding boxes.

    Args:
        data_dir: Directory containing point cloud data
        label_dir: Directory containing label data
        output_dir: Directory to save extracted point clouds
    """
    # Define transformation parameters
    rxyz_offset = [0.0705718, -0.2612746, -0.017035]  # Rotation in radians
    xyz_offset = [0, 0, 5.7]  # Translation

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all directories in label_dir
    label_dirs = [
        d for d in os.listdir(label_dir) if os.path.isdir(os.path.join(label_dir, d))
    ]

    for dir_name in label_dirs:
        # Skip evaluators.json files
        if dir_name.endswith(".json"):
            continue

        label_dir_path = os.path.join(label_dir, dir_name)
        data_dir_path = os.path.join(data_dir, dir_name)

        # Create output subdirectory
        output_subdir = os.path.join(output_dir, dir_name)
        os.makedirs(output_subdir, exist_ok=True)

        # Get all JSON files in the label directory
        json_files = [
            f
            for f in os.listdir(label_dir_path)
            if f.endswith(".json") and f != "evaluators.json"
        ]

        for json_file in json_files:
            # Get corresponding bin file
            bin_file = json_file[:-5]  # Remove .json extension
            bin_path = os.path.join(data_dir_path, bin_file)

            # Check if bin file exists
            if not os.path.exists(bin_path):
                print(f"Warning: Bin file {bin_path} does not exist")
                continue

            # Load point cloud
            point_cloud = load_and_transform_point_cloud(
                bin_path, rxyz_offset, xyz_offset
            )
            if point_cloud is None:
                continue

            # Load label
            json_path = os.path.join(label_dir_path, json_file)
            with open(json_path, "r") as f:
                label_data = json.load(f)

            # Process each bounding box
            for i, box in enumerate(label_data.get("bounding_boxes", [])):
                # Extract points in box
                points_in_box = extract_points_in_box(point_cloud, box)

                # Save extracted points
                object_id = box.get("object_id", "unknown")
                track_id = box.get("track_id", i)
                output_file = f"{bin_file[:-4]}_{object_id}_{track_id}.npy"
                output_path = os.path.join(output_subdir, output_file)

                np.save(output_path, points_in_box)

                print(
                    f"Saved {len(points_in_box)} points for {object_id} (track {track_id}) in {output_path}"
                )

                # Optional: Visualize the extracted points
                # visualize_points(points_in_box, box, output_path.replace('.npy', '.png'))


def visualize_points(points, box, output_path=None):
    """
    Visualize the extracted points in 3D.

    Args:
        points: Numpy array of shape (N, 7) containing point cloud data
        box: Dictionary containing bounding box information
        output_path: Path to save the visualization (optional)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot points
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 3], cmap="viridis"
    )

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set title
    object_id = box.get("object_id", "unknown")
    track_id = box.get("track_id", "unknown")
    ax.set_title(f"Object: {object_id}, Track: {track_id}")

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Save or show
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Define directories
    data_dir = "/home/akk/southgate/dettrain_20220711/south_gate_1680_8Feb2022/Data"
    label_dir = "/home/akk/southgate/dettrain_20220711/south_gate_1680_8Feb2022/Label"
    output_dir = "/home/akk/repos/PCBackgroundRemoval/04_BackgroundRemovalEvaluation/ExtractedBoxes"

    # Process data
    process_data(data_dir, label_dir, output_dir)

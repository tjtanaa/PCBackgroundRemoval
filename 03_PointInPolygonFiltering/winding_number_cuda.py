from __future__ import division
import os
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from numba import cuda
import json
import glob
from scipy.spatial.transform import Rotation

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

    if row < points.shape[0] and col < polygon_vertices.shape[0]-1:
        if(polygon_vertices[col, 1] <= points[row, 1]):
            if(polygon_vertices[col+1, 1] > points[row, 1]):
                if(
                    (
                        (polygon_vertices[col+1, 0] - polygon_vertices[col, 0]) 
                        * (points[row, 1] - polygon_vertices[col, 1])
                    - (points[row, 0] - polygon_vertices[col, 0]) 
                        * (polygon_vertices[col+1, 1] - polygon_vertices[col, 1])
                    ) > 0
                ):
                    cuda.atomic.add(wn, row, 1)

        else:
            if(polygon_vertices[col+1, 1] <= points[row, 1]):
                if(
                    (
                        (polygon_vertices[col+1, 0] - polygon_vertices[col, 0]) 
                        * (points[row, 1] - polygon_vertices[col, 1])
                    - (points[row, 0] - polygon_vertices[col, 0]) 
                        * (polygon_vertices[col+1, 1] - polygon_vertices[col, 1])
                    ) < 0
                ):
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
    # Copy the arrays to the device
    points_global_mem = cuda.to_device(points_xy)
    vertices_global_mem = cuda.to_device(polygon_vertices)

    # Allocate memory on the device for the winding number results
    wn_global_mem = cuda.device_array((points_xy.shape[0]), dtype=np.int32)

    # Configure the CUDA blocks and grid
    threadsperblock = (128, 8)
    blockspergrid_x = int(math.ceil(points_xy.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(polygon_vertices.shape[0] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Start the kernel to compute winding numbers
    wna_number_cuda_jit[blockspergrid, threadsperblock](
        points_global_mem, vertices_global_mem, wn_global_mem
    )
    cuda.synchronize()

    # Copy the result back to the host
    wn = wn_global_mem.copy_to_host()

    # Clear CUDA memory
    cuda.current_context().deallocations.clear()

    # Return mask for points inside the polygon (wn > 0)
    return wn >= 0

def filter_point_cloud(point_cloud, valid_polygon, invalid_polygons):
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

    # Get mask for points inside the valid polygon
    valid_mask = filter_points_by_polygon(points_xy, valid_polygon)

    # Initialize mask for points outside all invalid polygons
    invalid_mask = np.zeros(points_xy.shape[0], dtype=bool)

    # # Process each invalid polygon
    # for invalid_polygon in invalid_polygons:
    #     # Get mask for points inside this invalid polygon
    #     inside_invalid = filter_points_by_polygon(points_xy, invalid_polygon)
    #     # Update the invalid mask
    #     invalid_mask = np.logical_or(invalid_mask, inside_invalid)

    # Final mask: points inside valid polygon AND outside all invalid polygons
    final_mask = np.logical_and(valid_mask, ~invalid_mask)

    # Apply the mask to filter the point cloud
    filtered_point_cloud = point_cloud[final_mask]

    return filtered_point_cloud, final_mask

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
        point_cloud = point_cloud[:112144,:]

        # Extract XYZ coordinates
        xyz = point_cloud[:, :3]

        # Apply rotation in the correct order: X, then Y, then Z
        # Create rotation matrices for each axis
        rx, ry, rz = rxyz_offset

        # Use scipy's Rotation class with 'xyz' sequence (extrinsic rotations)
        # This applies rotations in the order: first around x, then around y, then around z
        r = Rotation.from_euler('xyz', [rx, ry, rz], degrees=False)
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

def run_benchmark(point_cloud_dir, rxyz_offset, xyz_offset):
    """
    Run benchmark on point cloud files from a directory.

    Args:
        point_cloud_dir: Directory containing point cloud binary files
        rxyz_offset: Rotation offset in radians [rx, ry, rz]
        xyz_offset: Translation offset [x, y, z]

    Returns:
        results: Dictionary containing benchmark results
    """
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
                [-40.15195465, 21.1319828]
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
                [14.06089973, 4.8097167]
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
                [12.28229237, -63.71089172]
            ]
        ]
    }

    # Convert polygon data to numpy arrays
    valid_polygon = np.array(polygon_data["valid_polygon"][0], dtype=np.float32)
    invalid_polygons = [np.array(poly, dtype=np.float32) for poly in polygon_data["invalid_polygon"]]

    # Find all binary files in the directory
    file_pattern = os.path.join(point_cloud_dir, "*.bin")
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        print(f"No binary files found in {point_cloud_dir}")
        return None

    # Store results
    results = {
        "file_names": [],
        "filtering_times": [],
        "percentage_kept": [],
        "num_points_kept": []
    }

    print(f"Found {len(file_paths)} point cloud files")
    print(f"Applying rotation [X,Y,Z]: [{rxyz_offset[0]:.6f}, {rxyz_offset[1]:.6f}, {rxyz_offset[2]:.6f}] rad")
    print(f"Applying translation: [{xyz_offset[0]:.1f}, {xyz_offset[1]:.1f}, {xyz_offset[2]:.1f}]")
    print(f"{'File':<30} {'Time (ms)':<15} {'Points Kept':<15} {'Percentage (%)':<15}")
    print("-" * 75)

    for file_path in file_paths:
        file_name = os.path.basename(file_path)

        # Load and transform point cloud
        point_cloud = load_and_transform_point_cloud(file_path, rxyz_offset, xyz_offset)

        if point_cloud is None:
            continue

        # Measure filtering time (only the filtering part, not loading/transforming)
        start_time = time.time()
        filtered_point_cloud, mask = filter_point_cloud(point_cloud, valid_polygon, invalid_polygons)
        end_time = time.time()

        # Calculate time in milliseconds
        elapsed_time = (end_time - start_time) * 1000

        # Calculate percentage and number of points kept
        percentage_kept = (filtered_point_cloud.shape[0] / point_cloud.shape[0]) * 100
        num_points_kept = filtered_point_cloud.shape[0]

        # Store results
        results["file_names"].append(file_name)
        results["filtering_times"].append(elapsed_time)
        results["percentage_kept"].append(percentage_kept)
        results["num_points_kept"].append(num_points_kept)

        print(f"{file_name:<30} {elapsed_time:<15.2f} {num_points_kept:<15} {percentage_kept:<15.2f}")

    # Calculate averages
    if results["filtering_times"]:
        results["avg_filtering_time"] = np.mean(results["filtering_times"][1:])
        results["avg_percentage_kept"] = np.mean(results["percentage_kept"][1:])
        results["avg_num_points_kept"] = np.mean(results["num_points_kept"][1:])

        print("\nSummary:")
        print(f"Average filtering time: {results['avg_filtering_time']:.2f} ms")
        print(f"Average percentage of points kept: {results['avg_percentage_kept']:.2f}%")
        print(f"Average number of points kept: {int(results['avg_num_points_kept'])}")

    return results

def plot_results(results):
    """
    Plot benchmark results.
    """
    if not results or not results["filtering_times"]:
        print("No results to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Sort results by filtering time
    sorted_indices = np.argsort(results["filtering_times"])
    sorted_files = [results["file_names"][i] for i in sorted_indices]
    sorted_times = [results["filtering_times"][i] for i in sorted_indices]
    sorted_percentages = [results["percentage_kept"][i] for i in sorted_indices]

    # Plot filtering times
    ax1.bar(range(len(sorted_files)), sorted_times, color='blue', alpha=0.7)
    ax1.set_xlabel('Point Cloud File')
    ax1.set_ylabel('Filtering Time (ms)')
    ax1.set_title('Polygon Filtering Performance')
    ax1.set_xticks(range(len(sorted_files)))
    ax1.set_xticklabels(sorted_files, rotation=90)
    ax1.axhline(y=results["avg_filtering_time"], color='r', linestyle='-', label=f'Avg: {results["avg_filtering_time"]:.2f} ms')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot percentage of points kept
    ax2.bar(range(len(sorted_files)), sorted_percentages, color='green', alpha=0.7)
    ax2.set_xlabel('Point Cloud File')
    ax2.set_ylabel('Points Kept (%)')
    ax2.set_title('Filtering Efficiency')
    ax2.set_xticks(range(len(sorted_files)))
    ax2.set_xticklabels(sorted_files, rotation=90)
    ax2.axhline(y=results["avg_percentage_kept"], color='r', linestyle='-', label=f'Avg: {results["avg_percentage_kept"]:.2f}%')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('point_cloud_filtering_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Define the directory containing point cloud binary files
    point_cloud_dir = "/home/akk/southgate/dettrain_20220711/south_gate_1680_8Feb2022/Data/2020_12_02=08_08_18_845_dir"  # Change this to your directory path

    # Define transformation parameters
    rxyz_offset = [0.0705718, -0.2612746, -0.017035]  # Rotation in radians [rx, ry, rz]
    xyz_offset = [0, 0, 5.7]  # Translation [x, y, z]

    # Run the benchmark
    results = run_benchmark(point_cloud_dir, rxyz_offset, xyz_offset)

    if results:
        # Plot the results
        plot_results(results)

        # Save results to file
        with open('benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("\nBenchmark completed. Results saved to 'benchmark_results.json'")
        print("Plot saved to 'point_cloud_filtering_benchmark.png'")
    else:
        print("Benchmark failed. No results to save.")

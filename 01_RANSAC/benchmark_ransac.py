import numpy as np
import time
import argparse
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_point_cloud(file_path):
    """
    Load point cloud data from a file.
    For this example, we'll create synthetic data if no file is provided.
    """
    if file_path:
        # Implement your file loading logic here
        # Example: return np.load(file_path)
        pass
    else:
        # Create synthetic point cloud data with the specified channels
        num_points = 2048 * 128

        # Generate random points with a ground plane and some noise
        x = np.random.uniform(-10, 10, num_points)
        y = np.random.uniform(-10, 10, num_points)

        # Create a ground plane with z = 0.1*x + 0.05*y - 2 + noise
        z_ground = 0.1 * x + 0.05 * y - 2 + np.random.normal(0, 0.1, num_points)

        # Add some non-ground points
        non_ground_indices = np.random.choice(num_points, num_points // 3)
        z_ground[non_ground_indices] += np.random.uniform(0.5, 3, len(non_ground_indices))

        # Create other channels
        intensity = np.random.uniform(0, 1, num_points)
        range_vals = np.sqrt(x**2 + y**2 + z_ground**2)
        ambient = np.random.uniform(0, 0.5, num_points)
        reflectivity = np.random.uniform(0, 1, num_points)
        placeholder1 = np.zeros(num_points)
        placeholder2 = np.zeros(num_points)

        # Create structured array
        dtype = np.dtype([
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("intensity", np.float32),
            ("range", np.float32),
            ("ambient", np.float32),
            ("reflectivity", np.float32),
            ("placeholder1", np.float32),
            ("placeholder2", np.float32)
        ])

        point_cloud = np.zeros(num_points, dtype=dtype)
        point_cloud["x"] = x
        point_cloud["y"] = y
        point_cloud["z"] = z_ground
        point_cloud["intensity"] = intensity
        point_cloud["range"] = range_vals
        point_cloud["ambient"] = ambient
        point_cloud["reflectivity"] = reflectivity
        point_cloud["placeholder1"] = placeholder1
        point_cloud["placeholder2"] = placeholder2

        return point_cloud

def ransac_ground_plane(points, max_iterations=100, distance_threshold=0.1):
    """
    Detect ground plane using RANSAC algorithm.

    Args:
        points: Nx3 array of points (x, y, z)
        max_iterations: Maximum number of RANSAC iterations
        distance_threshold: Maximum distance for a point to be considered an inlier

    Returns:
        plane_model: The coefficients of the plane (a, b, c, d) where ax + by + cz + d = 0
        inliers: Boolean mask of inlier points
    """
    # Extract XYZ coordinates
    X = np.column_stack((points["x"], points["y"], points["z"]))

    # Create RANSAC model
    ransac = linear_model.RANSACRegressor(
        linear_model.LinearRegression(),
        max_trials=max_iterations,
        residual_threshold=distance_threshold,
        random_state=42
    )

    # Fit model
    # For a plane, we're predicting z from x and y
    X_train = X[:, :2]  # x and y coordinates
    y_train = X[:, 2]   # z coordinates

    ransac.fit(X_train, y_train)
    inlier_mask = ransac.inlier_mask_

    # Get plane coefficients (ax + by + c = z)
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_

    # Convert to standard form ax + by + cz + d = 0
    # z = ax + by + c => ax + by - z + c = 0
    plane_model = np.array([a, b, -1, c])

    return plane_model, inlier_mask

def visualize_results(points, inliers, plane_model):
    """Visualize the point cloud and detected ground plane"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot inliers (ground) and outliers (non-ground)
    ax.scatter(points["x"][inliers], points["y"][inliers], points["z"][inliers], 
               c='green', s=1, label='Ground')
    ax.scatter(points["x"][~inliers], points["y"][~inliers], points["z"][~inliers], 
               c='red', s=1, label='Non-ground')

    # Plot the ground plane
    x_range = np.linspace(min(points["x"]), max(points["x"]), 10)
    y_range = np.linspace(min(points["y"]), max(points["y"]), 10)
    X, Y = np.meshgrid(x_range, y_range)

    # Plane equation: ax + by + cz + d = 0 => z = (-ax - by - d) / c
    a, b, c, d = plane_model
    Z = (-a * X - b * Y - d) / c

    ax.plot_surface(X, Y, Z, alpha=0.2, color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.savefig('ground_plane_detection.png')
    plt.close()

def benchmark_ransac(point_cloud, num_runs=10, visualize=False):
    """Benchmark the RANSAC algorithm"""
    total_time = 0
    best_plane = None
    best_inliers = None

    for i in range(num_runs):
        start_time = time.time()
        plane_model, inliers = ransac_ground_plane(point_cloud)
        end_time = time.time()

        run_time = end_time - start_time
        total_time += run_time

        if i == 0:
            best_plane = plane_model
            best_inliers = inliers

        print(f"Run {i+1}/{num_runs}: {run_time:.4f} seconds")
        print(f"  Plane equation: {plane_model[0]:.4f}x + {plane_model[1]:.4f}y + {plane_model[2]:.4f}z + {plane_model[3]:.4f} = 0")
        print(f"  Ground points: {np.sum(inliers)} / {len(inliers)} ({100 * np.sum(inliers) / len(inliers):.2f}%)")

    avg_time = total_time / num_runs
    print(f"\nAverage execution time: {avg_time:.4f} seconds")

    if visualize:
        visualize_results(point_cloud, best_inliers, best_plane)

    return avg_time, best_plane, best_inliers

def main():
    parser = argparse.ArgumentParser(description='Benchmark RANSAC ground plane detection on Jetson Orin')
    parser.add_argument('--input', type=str, help='Path to point cloud file (optional)')
    parser.add_argument('--runs', type=int, default=10, help='Number of benchmark runs')
    parser.add_argument('--visualize', action='store_true', help='Visualize the results')
    args = parser.parse_args()

    print("Loading point cloud data...")
    point_cloud = load_point_cloud(args.input)
    print(f"Loaded {len(point_cloud)} points")

    print("\nStarting RANSAC benchmark...")
    avg_time, best_plane, best_inliers = benchmark_ransac(point_cloud, args.runs, args.visualize)

    # Print system info
    try:
        import subprocess
        jetson_info = subprocess.check_output(['cat', '/proc/device-tree/model']).decode('utf-8').strip()
        print(f"\nSystem: {jetson_info}")
    except:
        print("\nCouldn't determine Jetson model")

    print(f"Final results:")
    print(f"  Average execution time: {avg_time:.4f} seconds")
    print(f"  Plane equation: {best_plane[0]:.4f}x + {best_plane[1]:.4f}y + {best_plane[2]:.4f}z + {best_plane[3]:.4f} = 0")
    print(f"  Ground points: {np.sum(best_inliers)} / {len(best_inliers)} ({100 * np.sum(best_inliers) / len(best_inliers):.2f}%)")

    if args.visualize:
        print("Visualization saved as 'ground_plane_detection.png'")

if __name__ == "__main__":
    main()
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
        num_points = 128*2048

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

def filter_ground_points_naive(points, plane_model, distance_threshold=0.1):
    """
    Filter ground points using the plane equation (naive implementation).

    Args:
        points: Point cloud with x, y, z fields
        plane_model: Plane coefficients [a, b, c, d] where ax + by + cz + d = 0
        distance_threshold: Maximum distance to be considered as ground

    Returns:
        ground_mask: Boolean mask of ground points
    """
    a, b, c, d = plane_model

    # Calculate distance from each point to the plane
    # Distance = |ax + by + cz + d| / sqrt(a² + b² + c²)
    numerator = np.abs(a * points["x"] + b * points["y"] + c * points["z"] + d)
    denominator = np.sqrt(a**2 + b**2 + c**2)
    distances = numerator / denominator

    # Points with distance less than threshold are ground points
    ground_mask = distances < distance_threshold

    return ground_mask

def filter_ground_points_vectorized(points, plane_model, distance_threshold=0.1):
    """
    Filter ground points using the plane equation (vectorized implementation).

    Args:
        points: Point cloud with x, y, z fields
        plane_model: Plane coefficients [a, b, c, d] where ax + by + cz + d = 0
        distance_threshold: Maximum distance to be considered as ground

    Returns:
        ground_mask: Boolean mask of ground points
    """
    a, b, c, d = plane_model

    # Extract coordinates as a single operation
    coords = np.column_stack((points["x"], points["y"], points["z"]))

    # Calculate distance from each point to the plane
    # Distance = |ax + by + cz + d| / sqrt(a² + b² + c²)
    plane_normal = np.array([a, b, c])
    norm_factor = np.linalg.norm(plane_normal)

    # Vectorized distance calculation
    distances = np.abs(np.dot(coords, plane_normal) + d) / norm_factor

    # Points with distance less than threshold are ground points
    ground_mask = distances < distance_threshold

    return ground_mask

def visualize_filtered_points(points, ground_mask, plane_model):
    """Visualize the filtered point cloud"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground and non-ground points
    ax.scatter(points["x"][ground_mask], points["y"][ground_mask], points["z"][ground_mask], 
               c='green', s=1, label='Ground')
    ax.scatter(points["x"][~ground_mask], points["y"][~ground_mask], points["z"][~ground_mask], 
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

    plt.savefig('ground_filtering_result.png')
    plt.close()

def benchmark_ground_filtering(point_cloud, plane_model, num_runs=10, visualize=False):
    """Benchmark the ground filtering algorithms"""
    methods = {
        "Naive Implementation": filter_ground_points_naive,
        "Vectorized Implementation": filter_ground_points_vectorized
    }

    results = {}

    for method_name, method_func in methods.items():
        print(f"\nBenchmarking {method_name}:")
        total_time = 0
        ground_mask = None

        for i in range(num_runs):
            start_time = time.time()
            ground_mask = method_func(point_cloud, plane_model)
            end_time = time.time()

            run_time = end_time - start_time
            total_time += run_time

            print(f"  Run {i+1}/{num_runs}: {run_time:.6f} seconds")
            print(f"  Ground points: {np.sum(ground_mask)} / {len(ground_mask)} ({100 * np.sum(ground_mask) / len(ground_mask):.2f}%)")

        avg_time = total_time / num_runs
        print(f"  Average execution time: {avg_time:.6f} seconds")

        results[method_name] = {
            "avg_time": avg_time,
            "ground_mask": ground_mask
        }

    # Find the fastest method
    fastest_method = min(results.items(), key=lambda x: x[1]["avg_time"])
    print(f"\nFastest method: {fastest_method[0]} ({fastest_method[1]['avg_time']:.6f} seconds)")

    if visualize:
        # Visualize using the fastest method's results
        visualize_filtered_points(point_cloud, fastest_method[1]["ground_mask"], plane_model)

    return results

def main():
    parser = argparse.ArgumentParser(description='Benchmark ground plane filtering on Jetson Orin')
    parser.add_argument('--input', type=str, help='Path to point cloud file (optional)')
    parser.add_argument('--runs', type=int, default=10, help='Number of benchmark runs')
    parser.add_argument('--visualize', action='store_true', help='Visualize the results')
    parser.add_argument('--points', type=int, default=(128*2048), help='Number of points in synthetic data')
    args = parser.parse_args()

    print("Loading point cloud data...")
    point_cloud = load_point_cloud(args.input)
    print(f"Loaded {len(point_cloud)} points")

    print("\nDetecting ground plane using RANSAC...")
    start_time = time.time()
    plane_model, _ = ransac_ground_plane(point_cloud)
    ransac_time = time.time() - start_time

    print(f"RANSAC completed in {ransac_time:.4f} seconds")
    print(f"Plane equation: {plane_model[0]:.4f}x + {plane_model[1]:.4f}y + {plane_model[2]:.4f}z + {plane_model[3]:.4f} = 0")

    print("\nStarting ground filtering benchmark...")
    results = benchmark_ground_filtering(point_cloud, plane_model, args.runs, args.visualize)

    # Print system info
    try:
        import subprocess
        jetson_info = subprocess.check_output(['cat', '/proc/device-tree/model']).decode('utf-8').strip()
        print(f"\nSystem: {jetson_info}")
    except:
        print("\nCouldn't determine Jetson model")

    # Print summary
    print("\nBenchmark Summary:")
    print(f"  Point cloud size: {len(point_cloud)} points")
    print(f"  RANSAC time: {ransac_time:.6f} seconds")

    for method_name, result in results.items():
        print(f"  {method_name}: {result['avg_time']:.6f} seconds")

    if args.visualize:
        print("Visualization saved as 'ground_filtering_result.png'")

    # Optional: Test with different point cloud sizes
    if args.input is None:  # Only for synthetic data
        print("\nTesting scalability with different point cloud sizes:")
        sizes = [16384, 64*2048, 128*2048, 700000, 1700000]
        times_naive = []
        times_vectorized = []

        for size in sizes:
            print(f"\nTesting with {size} points:")
            # Generate synthetic data with specified size
            num_points = size
            x = np.random.uniform(-10, 10, num_points)
            y = np.random.uniform(-10, 10, num_points)
            z = 0.1 * x + 0.05 * y - 2 + np.random.normal(0, 0.1, num_points)

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

            test_cloud = np.zeros(num_points, dtype=dtype)
            test_cloud["x"] = x
            test_cloud["y"] = y
            test_cloud["z"] = z

            # Use the same plane model for all tests

            # Benchmark naive implementation
            start_time = time.time()
            filter_ground_points_naive(test_cloud, plane_model)
            naive_time = time.time() - start_time
            times_naive.append(naive_time)

            # Benchmark vectorized implementation
            start_time = time.time()
            filter_ground_points_vectorized(test_cloud, plane_model)
            vectorized_time = time.time() - start_time
            times_vectorized.append(vectorized_time)

            print(f"  Naive: {naive_time:.6f} seconds")
            print(f"  Vectorized: {vectorized_time:.6f} seconds")
            print(f"  Speedup: {naive_time/vectorized_time:.2f}x")

        # Plot scalability results
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, times_naive, 'o-', label='Naive Implementation')
        plt.plot(sizes, times_vectorized, 'o-', label='Vectorized Implementation')
        plt.xlabel('Number of Points')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Ground Filtering Performance Scalability')
        plt.legend()
        plt.grid(True)
        plt.savefig('scalability_results.png')
        print("\nScalability results saved as 'scalability_results.png'")

if __name__ == "__main__":
    main()
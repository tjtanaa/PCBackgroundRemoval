import numpy as np
import cupy as cp
import pandas as pd
import os
import time
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import gc


def load_statistics(stats_file):
    """
    Load the pre-computed point cloud statistics from CSV file.

    Args:
        stats_file: Path to the CSV file with point statistics

    Returns:
        DataFrame containing the statistics
    """
    print(f"Loading statistics from {stats_file}...")
    return pd.read_csv(stats_file)


def filter_point_cloud(point_cloud, stats_df, strategy_index):
    """
    Filter point cloud based on the selected range strategy.

    Args:
        point_cloud: NumPy array of point cloud data
        stats_df: DataFrame with point statistics
        strategy_index: Index of the strategy to use (1-4)

    Returns:
        Filtered point cloud and number of points removed
    """
    # Define range columns based on strategy
    if strategy_index == 1:
        lower_col, upper_col = "min_range", "max_range"
    elif strategy_index == 2:
        lower_col, upper_col = "quantile_0.05", "quantile_0.95"
    elif strategy_index == 3:
        lower_col, upper_col = "quantile_0.25", "quantile_0.75"
    elif strategy_index == 4:
        lower_col, upper_col = "quantile_0.01", "quantile_0.99"
    else:
        raise ValueError(f"Invalid strategy index: {strategy_index}")

    # Get range channel index (assuming it's at index 4 based on the channel list)
    range_idx = 4

    # Create a mask for points to keep
    keep_mask = np.ones(len(point_cloud), dtype=bool)

    # Process each point
    for i in range(len(point_cloud)):
        # Get point range value
        point_range = point_cloud[i, range_idx]

        # Get statistics for this point
        point_stats = stats_df.iloc[i]

        # Check if statistics are valid (not NaN)
        if pd.isna(point_stats[lower_col]) or pd.isna(point_stats[upper_col]):
            # Keep the point if stats are missing
            continue

        # Check if point range is within the specified range
        if point_stats[lower_col] <= point_range <= point_stats[upper_col]:
            # Mark for removal if within range
            keep_mask[i] = False

    # Count removed points
    removed_count = np.sum(~keep_mask)

    # Filter the point cloud
    filtered_point_cloud = point_cloud[keep_mask]

    return filtered_point_cloud, removed_count


def filter_point_cloud_gpu(point_cloud, stats_df, strategy_index):
    """
    Filter point cloud based on the selected range strategy using GPU acceleration.

    Args:
        point_cloud: NumPy array of point cloud data
        stats_df: DataFrame with point statistics
        strategy_index: Index of the strategy to use (1-4)

    Returns:
        Filtered point cloud and number of points removed
    """

    # Define range columns based on strategy
    if strategy_index == 1:
        lower_col, upper_col = "min_range", "max_range"
    elif strategy_index == 2:
        lower_col, upper_col = "quantile_0.05", "quantile_0.95"
    elif strategy_index == 3:
        lower_col, upper_col = "quantile_0.25", "quantile_0.75"
    elif strategy_index == 4:
        lower_col, upper_col = "quantile_0.01", "quantile_0.99"
    else:
        raise ValueError(f"Invalid strategy index: {strategy_index}")

    # Get range channel index (assuming it's at index 4 based on the channel list)
    range_idx = 4

    # Transfer point cloud to GPU
    point_cloud_gpu = cp.array(point_cloud)

    # Start timing
    start_time = time.time()

    # Extract range values
    range_values = point_cloud_gpu[:, range_idx]

    # Create arrays for lower and upper bounds
    lower_bounds = np.array(stats_df[lower_col])
    upper_bounds = np.array(stats_df[upper_col])

    # Create mask for valid statistics (not NaN)
    valid_stats = ~(np.isnan(lower_bounds) | np.isnan(upper_bounds))

    # Transfer bounds to GPU
    lower_bounds_gpu = cp.array(lower_bounds)
    upper_bounds_gpu = cp.array(upper_bounds)
    valid_stats_gpu = cp.array(valid_stats)

    # Create initial keep mask (keep all points)
    keep_mask_gpu = cp.ones(len(point_cloud), dtype=bool)

    # For points with valid stats, check if range is within bounds
    within_range = (range_values >= lower_bounds_gpu) & (
        range_values <= upper_bounds_gpu
    )

    # Only apply the within_range condition for points with valid stats
    points_to_remove = within_range & valid_stats_gpu

    # Invert to get points to keep
    keep_mask_gpu = ~points_to_remove

    # Count removed points
    removed_count = int(cp.sum(points_to_remove).get())

    # Filter the point cloud
    filtered_point_cloud = cp.asnumpy(point_cloud_gpu[keep_mask_gpu])

    # End timing
    end_time = time.time()
    processing_time = end_time - start_time

    # Free GPU memory
    del (
        point_cloud_gpu,
        range_values,
        lower_bounds_gpu,
        upper_bounds_gpu,
        valid_stats_gpu,
        keep_mask_gpu,
    )
    cp.get_default_memory_pool().free_all_blocks()

    return filtered_point_cloud, removed_count, processing_time


def warm_up_gpu(point_cloud, stats_df, num_warmup=5):
    """
    Perform GPU warm-up iterations before actual benchmarking.

    Args:
        point_cloud: NumPy array of point cloud data
        stats_df: DataFrame with point statistics
        num_warmup: Number of warm-up iterations
    """
    print(f"Performing {num_warmup} GPU warm-up iterations...")

    for i in range(num_warmup):
        # Run each strategy once
        for strategy_idx in range(1, 5):
            _, _, _ = filter_point_cloud_gpu(point_cloud, stats_df, strategy_idx)

        # Clear GPU memory after each iteration
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    # Final synchronization to ensure all operations are complete
    cp.cuda.Stream.null.synchronize()
    print("GPU warm-up complete")


def benchmark_filtering(
    input_dir, stats_file, output_dir, use_gpu=True, num_frames=10, num_warmup=5
):
    """
    Benchmark the point cloud filtering using different strategies.

    Args:
        input_dir: Directory containing point cloud data files
        stats_file: Path to the CSV file with point statistics
        output_dir: Directory to save filtered point clouds and benchmark results
        use_gpu: Whether to use GPU acceleration
        num_frames: Number of frames to process for benchmarking
        num_warmup: Number of warm-up iterations for GPU
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load statistics
    stats_df = load_statistics(stats_file)

    # Define strategies
    strategies = {
        # 1: "min_max_range",
        2: "quantile_05_95",
        # 3: "quantile_25_75",
        # 4: "quantile_01_99"
    }

    # Initialize benchmark results
    benchmark_results = {
        "strategy": [],
        "frame": [],
        "processing_time": [],
        "points_removed": [],
        "points_remaining": [],
    }

    # Define channel attributes
    channels = ["x", "y", "z", "intensity", "range", "ambient", "reflectivity"]

    pc_filename_list = [
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if ".bin" in filename
    ]
    # Perform GPU warm-up if using GPU
    if use_gpu:
        warmup_point_cloud = np.fromfile(pc_filename_list[0], "<f4")
        warmup_point_cloud = warmup_point_cloud.reshape(-1, len(channels))
        warm_up_gpu(warmup_point_cloud, stats_df, num_warmup)

    # Process each frame
    for file_path in tqdm(pc_filename_list[:num_frames]):
        # Load point cloud data

        point_cloud = np.fromfile(file_path, "<f4")
        point_cloud = point_cloud.reshape(-1, len(channels))

        # Process with each strategy
        for strategy_idx, strategy_name in strategies.items():
            # # Start timing
            # start_time = time.time()

            # Filter point cloud
            if use_gpu:
                filtered_point_cloud, removed_count, processing_time = (
                    filter_point_cloud_gpu(point_cloud, stats_df, strategy_idx)
                )
            # else:
            #     filtered_point_cloud, removed_count = filter_point_cloud(
            #         point_cloud, stats_df, strategy_idx
            #     )

            # # End timing
            # end_time = time.time()
            # processing_time = end_time - start_time

            # Save benchmark results
            benchmark_results["strategy"].append(strategy_name)
            benchmark_results["frame"].append(file_path.split(os.sep)[-1])
            benchmark_results["processing_time"].append(processing_time)
            benchmark_results["points_removed"].append(removed_count)
            benchmark_results["points_remaining"].append(len(filtered_point_cloud))

            # Save filtered point cloud
            # output_file = os.path.join(output_dir, f"frame_{file_path.split(os.sep)[-1]}.npy")
            # np.save(output_file, filtered_point_cloud)
            filtered_point_cloud.tofile(
                os.path.join(output_dir, f"frame_{file_path.split(os.sep)[-1]}")
            )
            # Clear memory between runs
            if use_gpu:
                cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
    # Create benchmark DataFrame
    benchmark_df = pd.DataFrame(benchmark_results)

    # Save benchmark results
    benchmark_file = os.path.join(output_dir, "benchmark_results.csv")
    benchmark_df.to_csv(benchmark_file, index=False)

    # Generate benchmark plots
    generate_benchmark_plots(benchmark_df, output_dir)

    print(f"Benchmark results saved to {benchmark_file}")
    print(f"Filtered point clouds saved to {output_dir}")


def generate_benchmark_plots(benchmark_df, output_dir):
    """
    Generate plots from benchmark results.

    Args:
        benchmark_df: DataFrame with benchmark results
        output_dir: Directory to save plots
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Group by strategy
    grouped = benchmark_df.groupby("strategy")

    # Plot processing time by strategy
    plt.figure(figsize=(12, 6))
    for strategy, group in grouped:
        plt.plot(group["frame"], group["processing_time"], marker="o", label=strategy)
    plt.xlabel("Frame Index")
    plt.ylabel("Processing Time (seconds)")
    plt.title("Processing Time by Strategy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "processing_time.png"))

    # Plot points removed by strategy
    plt.figure(figsize=(12, 6))
    for strategy, group in grouped:
        plt.plot(group["frame"], group["points_removed"], marker="o", label=strategy)
    plt.xlabel("Frame Index")
    plt.ylabel("Points Removed")
    plt.title("Points Removed by Strategy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "points_removed.png"))

    # Plot points remaining by strategy
    plt.figure(figsize=(12, 6))
    for strategy, group in grouped:
        plt.plot(group["frame"], group["points_remaining"], marker="o", label=strategy)
    plt.xlabel("Frame Index")
    plt.ylabel("Points Remaining")
    plt.title("Points Remaining by Strategy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "points_remaining.png"))

    # Calculate average metrics by strategy
    avg_metrics = (
        benchmark_df.groupby("strategy")
        .agg(
            {
                "processing_time": "mean",
                "points_removed": "mean",
                "points_remaining": "mean",
            }
        )
        .reset_index()
    )

    # Plot average processing time by strategy
    plt.figure(figsize=(10, 6))
    plt.bar(avg_metrics["strategy"], avg_metrics["processing_time"])
    plt.xlabel("Strategy")
    plt.ylabel("Average Processing Time (seconds)")
    plt.title("Average Processing Time by Strategy")
    plt.grid(True, axis="y")
    plt.savefig(os.path.join(plots_dir, "avg_processing_time.png"))

    # Plot average points removed by strategy
    plt.figure(figsize=(10, 6))
    plt.bar(avg_metrics["strategy"], avg_metrics["points_removed"])
    plt.xlabel("Strategy")
    plt.ylabel("Average Points Removed")
    plt.title("Average Points Removed by Strategy")
    plt.grid(True, axis="y")
    plt.savefig(os.path.join(plots_dir, "avg_points_removed.png"))

    # Save average metrics
    avg_metrics_file = os.path.join(plots_dir, "average_metrics.csv")
    avg_metrics.to_csv(avg_metrics_file, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark point cloud filtering based on range statistics"
    )
    parser.add_argument(
        "--input_dir", required=True, help="Directory containing point cloud data files"
    )
    parser.add_argument(
        "--stats_file", required=True, help="Path to the CSV file with point statistics"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save filtered point clouds and benchmark results",
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=10,
        help="Number of frames to process for benchmarking",
    )

    args = parser.parse_args()

    benchmark_filtering(
        args.input_dir,
        args.stats_file,
        args.output_dir,
        use_gpu=not args.cpu,
        num_frames=args.num_frames,
    )


if __name__ == "__main__":
    main()

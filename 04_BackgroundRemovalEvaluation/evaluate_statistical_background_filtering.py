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


def load_ground_truth(file_path):
    """
    Load ground truth data from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        ground_truth_mask: Boolean mask indicating foreground points
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Extract bounding box information
        bounding_boxes = data.get("bounding_boxes", [])

        # Create a mask for foreground points (points in any bounding box)
        # This will be filled in by the create_ground_truth_mask function
        return bounding_boxes
    except Exception as e:
        print(f"Error loading ground truth file {file_path}: {e}")
        return []


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


def evaluate_filtering(
    original_pc_dir,
    filtered_pc_dir,
    ground_truth_dir,
    output_dir,
    strategy_name,
    rxyz_offset=None,
    xyz_offset=None,
):
    """
    Evaluate the background filtering algorithm by comparing with ground truth.

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

    # Initialize results dictionary
    results = {
        "file": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "tp": [],
        "tn": [],
        "fp": [],
        "fn": [],
        "total_points": [],
    }

    # Get list of filtered point cloud files
    filtered_files = [f for f in os.listdir(filtered_pc_dir) if f.endswith(".bin")]

    for filtered_file in tqdm(filtered_files, desc=f"Evaluating {strategy_name}"):
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
        original_pc = load_point_cloud(original_path, len(channels))
        filtered_pc = load_point_cloud(filtered_path, len(channels))

        if original_pc is None or filtered_pc is None:
            print(
                f"Warning: Could not load point clouds for {original_file}. Skipping."
            )
            continue

        # Apply transformation if provided
        if rxyz_offset is not None and xyz_offset is not None:
            from scipy.spatial.transform import Rotation

            # Extract XYZ coordinates
            xyz = original_pc[:, :3]

            # Apply rotation
            r = Rotation.from_euler("xyz", rxyz_offset, degrees=False)
            rotated_xyz = r.apply(xyz)

            # Apply translation
            transformed_xyz = rotated_xyz + xyz_offset

            # Update point cloud with transformed coordinates
            transformed_pc = original_pc.copy()
            transformed_pc[:, :3] = transformed_xyz
            original_pc = transformed_pc

        # Apply transformation to filtered point cloud if provided
        if rxyz_offset is not None and xyz_offset is not None:
            from scipy.spatial.transform import Rotation

            # Extract XYZ coordinates
            xyz = filtered_pc[:, :3]

            # Apply rotation
            r = Rotation.from_euler("xyz", rxyz_offset, degrees=False)
            rotated_xyz = r.apply(xyz)

            # Apply translation
            transformed_xyz = rotated_xyz + xyz_offset

            # Update point cloud with transformed coordinates
            transformed_pc = filtered_pc.copy()
            transformed_pc[:, :3] = transformed_xyz
            filtered_pc = transformed_pc

        # Load ground truth bounding boxes
        bounding_boxes = load_ground_truth(ground_truth_path)

        if not bounding_boxes:
            print(f"Warning: No bounding boxes found for {original_file}. Skipping.")
            continue

        # Create ground truth mask for original point cloud (True = foreground, False = background)
        original_gt_mask = create_ground_truth_mask(original_pc, bounding_boxes)

        # Create ground truth mask for filtered point cloud (True = foreground, False = background)
        filtered_gt_mask = create_ground_truth_mask(filtered_pc, bounding_boxes)

        # Count points in each category
        # For original point cloud:
        # - Foreground points: points inside any bounding box
        # - Background points: points outside all bounding boxes
        original_foreground_count = np.sum(original_gt_mask)
        original_background_count = len(original_pc) - original_foreground_count

        # For filtered point cloud:
        # - Foreground points: points inside any bounding box
        # - Background points: points outside all bounding boxes
        filtered_foreground_count = np.sum(filtered_gt_mask)
        filtered_background_count = len(filtered_pc) - filtered_foreground_count

        # Calculate metrics
        # - True Positive (TP): Foreground points that were kept in the filtered point cloud
        # - True Negative (TN): Background points that were removed in the filtered point cloud
        # - False Positive (FP): Background points that were kept in the filtered point cloud
        # - False Negative (FN): Foreground points that were removed in the filtered point cloud

        tp = filtered_foreground_count
        fp = filtered_background_count
        fn = original_foreground_count - filtered_foreground_count
        tn = original_background_count - filtered_background_count

        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Store results
        results["file"].append(original_file)
        results["accuracy"].append(accuracy)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["tp"].append(tp)
        results["tn"].append(tn)
        results["fp"].append(fp)
        results["fn"].append(fn)
        results["total_points"].append(len(original_pc))

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Calculate average metrics
    avg_accuracy = results_df["accuracy"].mean()
    avg_precision = results_df["precision"].mean()
    avg_recall = results_df["recall"].mean()

    # Add summary row
    summary = pd.DataFrame(
        {
            "file": ["AVERAGE"],
            "accuracy": [avg_accuracy],
            "precision": [avg_precision],
            "recall": [avg_recall],
            "tp": [results_df["tp"].sum()],
            "tn": [results_df["tn"].sum()],
            "fp": [results_df["fp"].sum()],
            "fn": [results_df["fn"].sum()],
            "total_points": [results_df["total_points"].sum()],
        }
    )

    results_df = pd.concat([results_df, summary], ignore_index=True)

    # Save results to CSV
    results_file = os.path.join(output_dir, f"{strategy_name}_evaluation.csv")
    results_df.to_csv(results_file, index=False)

    # Generate plots
    generate_evaluation_plots(results_df, output_dir, strategy_name)

    print(f"Evaluation results for {strategy_name}:")
    print(f"  Average Accuracy: {avg_accuracy:.4f}")
    print(f"  Average Precision: {avg_precision:.4f}")
    print(f"  Average Recall: {avg_recall:.4f}")
    print(f"Results saved to {results_file}")

    return results_df


def generate_evaluation_plots(results_df, output_dir, strategy_name):
    """
    Generate plots from evaluation results.

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

    # Plot accuracy, precision, and recall
    plt.figure(figsize=(12, 6))
    plt.plot(plot_df["file"], plot_df["accuracy"], marker="o", label="Accuracy")
    plt.plot(plot_df["file"], plot_df["precision"], marker="s", label="Precision")
    plt.plot(plot_df["file"], plot_df["recall"], marker="^", label="Recall")
    plt.xlabel("File")
    plt.ylabel("Score")
    plt.title(f"{strategy_name} - Evaluation Metrics")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{strategy_name}_metrics.png"))

    # Plot confusion matrix components
    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["file"], plot_df["tp"], label="True Positives")
    plt.bar(
        plot_df["file"], plot_df["tn"], bottom=plot_df["tp"], label="True Negatives"
    )
    plt.bar(
        plot_df["file"],
        plot_df["fp"],
        bottom=plot_df["tp"] + plot_df["tn"],
        label="False Positives",
    )
    plt.bar(
        plot_df["file"],
        plot_df["fn"],
        bottom=plot_df["tp"] + plot_df["tn"] + plot_df["fp"],
        label="False Negatives",
    )
    plt.xlabel("File")
    plt.ylabel("Count")
    plt.title(f"{strategy_name} - Confusion Matrix Components")
    plt.legend()
    plt.grid(True, axis="y")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{strategy_name}_confusion.png"))

    # Plot percentage of points in each category
    plt.figure(figsize=(12, 6))
    total_points = plot_df["total_points"]
    plt.bar(plot_df["file"], 100 * plot_df["tp"] / total_points, label="True Positives")
    plt.bar(
        plot_df["file"],
        100 * plot_df["tn"] / total_points,
        bottom=100 * plot_df["tp"] / total_points,
        label="True Negatives",
    )
    plt.bar(
        plot_df["file"],
        100 * plot_df["fp"] / total_points,
        bottom=100 * (plot_df["tp"] + plot_df["tn"]) / total_points,
        label="False Positives",
    )
    plt.bar(
        plot_df["file"],
        100 * plot_df["fn"] / total_points,
        bottom=100 * (plot_df["tp"] + plot_df["tn"] + plot_df["fp"]) / total_points,
        label="False Negatives",
    )
    plt.xlabel("File")
    plt.ylabel("Percentage")
    plt.title(f"{strategy_name} - Point Classification Percentages")
    plt.legend()
    plt.grid(True, axis="y")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{strategy_name}_percentages.png"))


def compare_strategies(evaluation_dir, output_dir):
    """
    Compare different filtering strategies based on evaluation results.

    Args:
        evaluation_dir: Directory containing evaluation results
        output_dir: Directory to save comparison results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all evaluation CSV files
    csv_files = [f for f in os.listdir(evaluation_dir) if f.endswith("_evaluation.csv")]

    if not csv_files:
        print("No evaluation results found.")
        return

    # Initialize comparison dictionary
    comparison = {
        "strategy": [],
        "avg_accuracy": [],
        "avg_precision": [],
        "avg_recall": [],
        "total_tp": [],
        "total_tn": [],
        "total_fp": [],
        "total_fn": [],
        "total_points": [],
    }

    # Load each evaluation file and extract summary metrics
    for csv_file in csv_files:
        # Extract strategy name from file name
        strategy_name = csv_file.replace("_evaluation.csv", "")

        # Load evaluation results
        results_df = pd.read_csv(os.path.join(evaluation_dir, csv_file))

        # Get summary row (last row)
        summary = results_df.iloc[-1]

        # Store comparison metrics
        comparison["strategy"].append(strategy_name)
        comparison["avg_accuracy"].append(summary["accuracy"])
        comparison["avg_precision"].append(summary["precision"])
        comparison["avg_recall"].append(summary["recall"])
        comparison["total_tp"].append(summary["tp"])
        comparison["total_tn"].append(summary["tn"])
        comparison["total_fp"].append(summary["fp"])
        comparison["total_fn"].append(summary["fn"])
        comparison["total_points"].append(summary["total_points"])

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison)

    # Sort by accuracy (descending)
    comparison_df = comparison_df.sort_values("avg_accuracy", ascending=False)

    # Save comparison to CSV
    comparison_file = os.path.join(output_dir, "strategy_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)

    # Generate comparison plots
    generate_comparison_plots(comparison_df, output_dir)

    print("Strategy comparison:")
    print(comparison_df[["strategy", "avg_accuracy", "avg_precision", "avg_recall"]])
    print(f"Comparison saved to {comparison_file}")

    return comparison_df


def generate_comparison_plots(comparison_df, output_dir):
    """
    Generate plots comparing different filtering strategies.

    Args:
        comparison_df: DataFrame with comparison results
        output_dir: Directory to save plots
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot average metrics
    plt.figure(figsize=(10, 6))
    x = np.arange(len(comparison_df))
    width = 0.25

    plt.bar(x - width, comparison_df["avg_accuracy"], width, label="Accuracy")
    plt.bar(x, comparison_df["avg_precision"], width, label="Precision")
    plt.bar(x + width, comparison_df["avg_recall"], width, label="Recall")

    plt.xlabel("Strategy")
    plt.ylabel("Score")
    plt.title("Comparison of Filtering Strategies")
    plt.xticks(x, comparison_df["strategy"])
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "strategy_comparison_metrics.png"))

    # Plot confusion matrix components
    plt.figure(figsize=(12, 6))

    # Calculate percentages
    total_points = comparison_df["total_points"]
    tp_pct = 100 * comparison_df["total_tp"] / total_points
    tn_pct = 100 * comparison_df["total_tn"] / total_points
    fp_pct = 100 * comparison_df["total_fp"] / total_points
    fn_pct = 100 * comparison_df["total_fn"] / total_points

    plt.bar(comparison_df["strategy"], tp_pct, label="True Positives")
    plt.bar(comparison_df["strategy"], tn_pct, bottom=tp_pct, label="True Negatives")
    plt.bar(
        comparison_df["strategy"],
        fp_pct,
        bottom=tp_pct + tn_pct,
        label="False Positives",
    )
    plt.bar(
        comparison_df["strategy"],
        fn_pct,
        bottom=tp_pct + tn_pct + fp_pct,
        label="False Negatives",
    )

    plt.xlabel("Strategy")
    plt.ylabel("Percentage")
    plt.title("Point Classification Percentages by Strategy")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "strategy_comparison_percentages.png"))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate background filtering algorithm"
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

    # Evaluate filtering
    evaluate_filtering(
        args.original_pc_dir,
        args.filtered_pc_dir,
        args.ground_truth_dir,
        args.output_dir,
        args.strategy_name,
        rxyz_offset,
        xyz_offset,
    )

    # If multiple strategies have been evaluated, compare them
    compare_strategies(args.output_dir, args.output_dir)


if __name__ == "__main__":
    main()

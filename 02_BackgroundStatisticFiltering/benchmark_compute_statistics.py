import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def process_point_cloud_statistics(input_directory, output_file):
    """
    Process point cloud data across 250 frames to calculate range statistics.

    Args:
        input_directory: Directory containing point cloud data files
        output_file: Path to output CSV file
    """
    # Define channel attributes
    channels = ["x", "y", "z", "intensity", "range", "ambient", "reflectivity"]

    # Number of points in each point cloud
    num_points = 262144

    # Initialize array to store valid range values for each point across all frames
    # Using a list of lists for efficient appending
    point_range_values = [[] for _ in range(num_points)]

    print("Processing point cloud frames...")

    pc_filename_list = [
        os.path.join(input_directory, filename)
        for filename in os.listdir(input_directory)
        if ".bin" in filename
    ]
    # print(pc_filename_list)

    # Process each frame
    for file_path in tqdm(pc_filename_list[:]):
        # Load point cloud data (assuming files are named consistently)
        # Adjust the file loading according to your actual data format
        point_cloud = np.fromfile(file_path, "<f4")
        point_cloud = point_cloud.reshape(-1, len(channels))
        # print(point_cloud)

        # Check each point
        for point_idx in range(num_points):
            point = point_cloud[point_idx]

            # Check if the point has non-zero values in any channel
            # Excluding the range channel itself from this check
            non_range_channels = [point[i] for i in range(3) if channels[i] != "range"]

            # Get the range value (assuming it's at index 4 based on the channel list)
            range_value = point[4]

            # If any non-range channel has a non-zero value, consider this range value
            if any(val != 0.0 for val in non_range_channels) and range_value != 0.0:
                point_range_values[point_idx].append(range_value)

    print("Calculating statistics...")

    # Calculate statistics for each point
    statistics = []
    quantiles = [0.01, 0.05, 0.25, 0.75, 0.95, 0.99]

    for point_idx in tqdm(range(num_points)):
        ranges = np.array(point_range_values[point_idx])

        # Skip points with no valid range values
        if len(ranges) == 0:
            # Add a row with NaN values
            statistics.append([point_idx] + [np.nan] * (2 + len(quantiles)))
            continue

        # Calculate min and max
        min_range = np.min(ranges)
        max_range = np.max(ranges)

        # Calculate quantiles
        quantile_values = (
            np.quantile(ranges, quantiles)
            if len(ranges) > 1
            else [np.nan] * len(quantiles)
        )

        # Add statistics for this point
        statistics.append([point_idx, min_range, max_range] + list(quantile_values))

    # Create DataFrame and save to CSV
    columns = [
        "point_idx",
        "min_range",
        "max_range",
        "quantile_0.01",
        "quantile_0.05",
        "quantile_0.25",
        "quantile_0.75",
        "quantile_0.95",
        "quantile_0.99",
    ]

    df = pd.DataFrame(statistics, columns=columns)
    df.to_csv(output_file, index=False)

    print(f"Statistics saved to {output_file}")


# Example usage
if __name__ == "__main__":
    input_dir = "/home/akk/southgate/background_dataset/background_frames1"
    output_file = "./range_statistics.csv"
    process_point_cloud_statistics(input_dir, output_file)

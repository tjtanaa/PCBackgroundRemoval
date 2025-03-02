#!/bin/bash
python benchmark_filter.py --input_dir '/home/akk/southgate/background_dataset/background_frames1' \
 --stats_file 'cpu_range_statistics.csv' \
 --output_dir 'filtered_point_cloud_gpu' \
 --num_frames 120
#!/bin/bash
python benchmark_filter.py --input_dir '/home/akk/southgate/dettrain_20220711/south_gate_1680_8Feb2022/Data/2020_12_02=08_08_18_845_dir' \
 --stats_file 'cpu_range_statistics.csv' \
 --output_dir 'filtered_point_cloud_gpu' \
 --num_frames 120
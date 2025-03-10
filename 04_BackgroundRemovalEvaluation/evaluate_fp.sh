#!/bin/bash

# # Define paths
# ORIGINAL_PC_PATH="/home/akk/southgate/statfilter_eval_debug/original_pc"
# FILTERED_PC_PATH="/home/akk/southgate/statfilter_eval_debug/filtered_pc"
# # STATS_FILE="/home/akk/repos/PCBackgroundRemoval/02_BackgroundStatisticFiltering/range_statistics.csv"
# GROUND_TRUTH_DIR="/home/akk/southgate/dettrain_20220711/south_gate_1680_8Feb2022/Label"
# EVAL_OUTPUT="/home/akk/southgate/fullpaperfilter_eval_debug/eval_result"


# Define paths
ORIGINAL_PC_PATH="/home/akk/southgate/south_gate_1680_8Feb2022/Data/2020_12_03=08_03_38_492_dir"
FILTERED_PC_PATH="/home/akk/southgate/bgrm_full_training_dataset/south_gate_1680_8Feb2022/Data/2020_12_03=08_03_38_492_dir"
# STATS_FILE="/home/akk/repos/PCBackgroundRemoval/02_BackgroundStatisticFiltering/range_statistics.csv"
GROUND_TRUTH_DIR="/home/akk/southgate/south_gate_1680_8Feb2022/Label/2020_12_03=08_03_38_492_dir"
EVAL_OUTPUT="/home/akk/southgate/fullpaperfilter_eval_debug/eval_result_2020_12_03=08_03_38_492_dir"

python evaluate_full_paper.py \
  --original_pc_dir $ORIGINAL_PC_PATH \
  --ground_truth_dir $GROUND_TRUTH_DIR \
  --filtered_pc_dir $FILTERED_PC_PATH \
  --output_dir $EVAL_OUTPUT \
  --strategy 2
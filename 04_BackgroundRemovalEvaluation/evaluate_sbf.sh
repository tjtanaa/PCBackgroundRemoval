#!/bin/bash

# Define paths
ORIGINAL_PC_PATH="/home/akk/southgate/statfilter_eval_debug/original_pc"
FILTERED_PC_PATH="/home/akk/southgate/statfilter_eval_debug/filtered_pc"
STATS_FILE="/home/akk/repos/PCBackgroundRemoval/02_BackgroundStatisticFiltering/range_statistics.csv"

# Generate filtered path
python ../02_BackgroundStatisticFiltering/benchmark_filter.py --input_dir $ORIGINAL_PC_PATH \
 --stats_file $STATS_FILE \
 --output_dir $FILTERED_PC_PATH \
 --num_frames 12000

# 3. For ground_truth_dir, we'll use the original Label directory
GROUND_TRUTH_DIR="/home/akk/southgate/dettrain_20220711/south_gate_1680_8Feb2022/Label"
EVAL_OUTPUT="/home/akk/southgate/statfilter_eval_debug/sbf_eval_result"

echo "Setup complete. You can now run the evaluation script with:"
echo "python evaluate_statistical_background_filtering.py \\"
echo "  --original_pc_dir $ORIGINAL_PC_PATH \\"
echo "  --filtered_pc_dir $FILTERED_PC_PATH \\"
echo "  --ground_truth_dir $GROUND_TRUTH_DIR \\"
echo "  --output_dir $EVAL_OUTPUT \\"
echo "  --strategy_name quantile_01_99"
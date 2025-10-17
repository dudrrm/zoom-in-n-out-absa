#!/bin/bash

# Example script for evaluating a model on all element orders
# This demonstrates the flexibility of the evaluation framework

# Configuration
TASK="acos"
DATASET="rest16"
MODEL_NAME="gpt-4o-mini"
TEST_DATA_DIR="../datasets/data"
DEMO_DIR="../datasets/few_shot"
OUTPUT_DIR="../outputs"

echo "Evaluating $MODEL_NAME on all element orders for $TASK-$DATASET"
echo "=================================================="

cd ../model_evaluation

# The main.py script already iterates through all orders by default
# This is just a demonstration of how to run it

python main.py \
  --task $TASK \
  --dataset $DATASET \
  --model_name_or_path $MODEL_NAME \
  --test_data_dir $TEST_DATA_DIR \
  --demo_dir $DEMO_DIR \
  --output_dir $OUTPUT_DIR \
  --shot 20 \
  --temp 0.0 \
  --exp_name "all_orders"

echo "=================================================="
echo "Evaluation on all orders completed!"
echo "Results saved to: $OUTPUT_DIR/$TASK-$DATASET/"


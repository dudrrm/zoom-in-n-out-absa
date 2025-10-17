#!/bin/bash

# Example script for model evaluation on ABSA tasks
# This script evaluates an LLM model on ABSA dataset

# Configuration
TASK="acos"
DATASET="rest16"
MODEL_NAME="gpt-4o-mini"
TEST_DATA_DIR="../datasets/data"
DEMO_DIR="../datasets/few_shot"
OUTPUT_DIR="../outputs"

# Model parameters
TEMP=0.0
SHOT=20
TARGET_ELEMENT="ACOS"

echo "Running Model Evaluation for $TASK-$DATASET with $MODEL_NAME"
echo "=================================================="

cd ../model_evaluation

python main.py \
  --task $TASK \
  --dataset $DATASET \
  --model_name_or_path $MODEL_NAME \
  --test_data_dir $TEST_DATA_DIR \
  --demo_dir $DEMO_DIR \
  --output_dir $OUTPUT_DIR \
  --target_element $TARGET_ELEMENT \
  --shot $SHOT \
  --temp $TEMP \
  --exp_name "evaluation"

echo "=================================================="
echo "Model evaluation completed!"
echo "Check results in: $OUTPUT_DIR/$TASK-$DATASET/"


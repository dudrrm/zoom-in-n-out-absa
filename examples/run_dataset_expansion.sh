#!/bin/bash

# Example script for dataset expansion (Zoom-In-N-Out)
# This script generates alternative expressions for aspect and opinion terms

# Configuration
TASK="asqp"
DATASET="rest16"
MODEL_NAME="gpt-4o-mini"
DATA_DIR="../datasets/data"
OUTPUT_DIR="../outputs"
LOG_DIR="../logs"

# Generation parameters
TEMP=0.7
SAMPLE=3
THRESHOLD=2

echo "Running Dataset Expansion for $TASK-$DATASET with $MODEL_NAME"
echo "=================================================="

cd ../dataset_expansion

python main.py \
  --task $TASK \
  --dataset $DATASET \
  --model-name $MODEL_NAME \
  --data-dir $DATA_DIR \
  --out-dir $OUTPUT_DIR \
  --log-dir $LOG_DIR \
  --temp $TEMP \
  --sample $SAMPLE \
  --threshold $THRESHOLD

echo "=================================================="
echo "Dataset expansion completed!"
echo "Check outputs in: $OUTPUT_DIR/$TASK-$DATASET/"


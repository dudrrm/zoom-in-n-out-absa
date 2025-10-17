# Example Scripts

This directory contains example scripts for running dataset expansion and model evaluation.

## Scripts

### 1. Dataset Expansion (`run_dataset_expansion.sh`)

Generates alternative expressions for aspect and opinion terms using the Zoom-In-N-Out approach.

```bash
bash run_dataset_expansion.sh
```

**Key parameters:**
- `TASK`: ABSA task type (acos, asqp, aste, tasd)
- `DATASET`: Dataset name (rest16, laptop16, etc.)
- `MODEL_NAME`: LLM for generation (gpt-4o, gpt-3.5-turbo, etc.)
- `TEMP`: Temperature for generation (higher = more diverse)
- `SAMPLE`: Number of samples per generation
- `THRESHOLD`: Minimum occurrences to accept alternative

### 2. Model Evaluation (`run_model_evaluation.sh`)

Evaluates an LLM model on ABSA tasks.

```bash
bash run_model_evaluation.sh
```

**Key parameters:**
- `MODEL_NAME`: Model to evaluate
- `TARGET_ELEMENT`: Element order (e.g., ACOS, ACSO)
- `SHOT`: Number of few-shot examples
- `TEMP`: Temperature for generation

### 3. All Orders Evaluation (`run_all_orders.sh`)

Evaluates a model on all 24 possible element orders.

```bash
bash run_all_orders.sh
```

This demonstrates the flexibility of the evaluation framework in handling different element orderings.

## Customization

Edit the configuration variables at the top of each script to customize:
- Task and dataset
- Model selection
- Data directories
- Generation parameters

## Prerequisites

1. Set up Python environment:
```bash
pip install -r ../requirements.txt
```

2. Set API keys (if using OpenAI):
```bash
export OPENAI_API_KEY="your-api-key"
```

3. Prepare your data in the correct format (see main README)

## Notes

- Make scripts executable: `chmod +x *.sh`
- Check logs in `./logs/` directory
- Check outputs in `./outputs/` directory
- Adjust paths in scripts based on your data location


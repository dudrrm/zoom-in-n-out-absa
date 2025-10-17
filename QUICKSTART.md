# Quick Start Guide

This guide will help you get started with the Zoom-In-N-Out ABSA framework quickly.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key (if using OpenAI)

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Task 1: Dataset Expansion

Generate alternative expressions for aspect and opinion terms:

### Basic Usage

```bash
cd dataset_expansion

python main.py \
  --task acos \
  --dataset rest16 \
  --model-name gpt-4o \
  --data-dir /path/to/your/data \
  --temp 0.7 \
  --sample 3
```

### Key Parameters

- `--task`: Task type (`acos`, `asqp`, `aste`, `tasd`)
- `--dataset`: Dataset name (e.g., `rest16`, `laptop16`)
- `--model-name`: LLM model for generation
- `--temp`: Temperature (higher = more diverse alternatives)
- `--sample`: Number of samples per generation
- `--threshold`: Minimum occurrences to accept (default: 1)

### Pipeline Steps

The expansion runs three steps automatically:
1. **Narrow**: Generate narrower expressions
2. **Widen**: Generate wider expressions
3. **Judge**: Validate with LLM

### Output

Results are saved in:
- `outputs/{task}-{dataset}/`: Intermediate and final results
- `logs/{task}-{dataset}/`: Detailed logs

## Task 2: Model Evaluation

Evaluate LLM models on ABSA tasks:

### Basic Usage

```bash
cd model_evaluation

python main.py \
  --task acos \
  --dataset rest16 \
  --model_name_or_path gpt-4o \
  --test_data_dir /path/to/data \
  --demo_dir /path/to/few_shot \
  --shot 20
```

### Key Parameters

- `--model_name_or_path`: Model to evaluate
- `--target_element`: Element order (e.g., `ACOS`, `ACSO`)
- `--shot`: Number of few-shot examples
- `--temp`: Temperature for generation (default: 0.0)

### Evaluate All Orders

The script automatically evaluates all 24 possible element orders by default.

### Output

Results are saved in:
- `outputs/{task}-{dataset}/{model}/`: Prediction results (JSON)
- `logs/{task}-{dataset}/{model}/`: Evaluation logs

## Using Example Scripts

We provide ready-to-use scripts in the `examples/` directory:

```bash
cd examples

# Dataset expansion
bash run_dataset_expansion.sh

# Model evaluation
bash run_model_evaluation.sh

# Evaluate all orders
bash run_all_orders.sh
```

Edit the configuration variables at the top of each script to customize.

## Common Issues

### 1. API Key Not Set

```
Error: OpenAI API key not found
```

**Solution:** Export your API key:
```bash
export OPENAI_API_KEY="your-key"
```

### 2. Demo File Not Found

```
Warning: Demo file not found
```

**Solution:** Check the `--demo_dir` path or use empty demos (the code will warn but continue).

### 3. Data Format Error

Ensure your data follows the format:
```
sentence####[('aspect', 'category', 'sentiment', 'opinion'), ...]
```

## Next Steps

1. **Customize Prompts**: Edit prompts in `dataset_expansion/prompts/` or `model_evaluation/prompts.py`
2. **Add Demos**: Add task-specific demo examples in `dataset_expansion/prompts/__init__.py`
3. **Evaluate Custom Models**: Use `--base-url` for custom API endpoints
4. **Analyze Results**: Use the JSON outputs for further analysis

## Tips

- Start with small `--sample` and `--threshold` values for testing
- Use `--now-debug` flag for debugging with a subset of data
- Check logs for detailed information about each step
- For open-source models, use `--base-url` to point to your vLLM or similar server

## Need Help?

- Check the full [README.md](README.md) for detailed documentation
- See example scripts in `examples/`
- Open an issue on GitHub (if repository is public)

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{yang-etal-2025-single,
    title = "Single Ground Truth Is Not Enough: Adding Flexibility to Aspect-Based Sentiment Analysis Evaluation",
    author = "Yang, Soyoung and Cho, Hojun and Lee, Jiyoung and Yoon, Sohee and Choi, Edward and Choo, Jaegul and Cho, Won Ik",
    booktitle = "Proceedings of NAACL 2025",
    year = "2025",
    url = "https://aclanthology.org/2025.naacl-long.603/",
}
```

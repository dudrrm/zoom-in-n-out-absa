# Zoom-in-Zoom-out ABSA: LLM Evaluation Framework

This repository contains a refactored implementation for **Aspect-Based Sentiment Analysis (ABSA)** evaluation, based on the NAACL 2025 paper. The codebase is designed for two primary tasks:

1. **Building diverse evaluation datasets** with multiple valid expressions
2. **Evaluating different LLM models** on ABSA tasks

## Paper

**Single Ground Truth Is Not Enough: Adding Flexibility to Aspect-Based Sentiment Analysis Evaluation**  
NAACL 2025  
Soyoung Yang, Hojun Cho, Jiyoung Lee, Sohee Yoon, Edward Choi, Jaegul Choo, Won Ik Cho  
[Paper Link](https://aclanthology.org/2025.naacl-long.603/)

## Key Features

- 🔄 **LangChain Integration**: Uses LangChain for flexible LLM integration
- 🎯 **Multiple LLM Support**: GPT models, HuggingFace models via OpenAI API, Google Gemini
- 📊 **Diverse Evaluation**: Build datasets with multiple valid ground truths
- 🚀 **Async Evaluation**: Efficient async processing for faster evaluation
- 🎲 **Element Ordering**: Support for all 24 possible element orderings (ACOS, AOSC, etc.)
- 🔒 **Constrained Decoding**: Optional regex-based constrained decoding for open-source models

## Installation

```bash
# Create conda environment
conda create -n absa-eval python=3.10
conda activate absa-eval

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
zoom-in-n-out-absa/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── config.py                   # Constants and configurations
├── models.py                   # LLM wrappers using LangChain
├── data_utils.py              # Data loading and processing
├── eval_utils.py              # Evaluation metrics
├── prompt.py                  # Prompt templates
├── methods.py                 # Prediction methods
├── utils.py                   # Utility functions
├── evaluate_llms.py           # Script for LLM evaluation (Task 2)
└── build_diverse_dataset.py   # Script for building diverse datasets (Task 1)
```

## Usage

### Step 1: Building Diverse Evaluation Datasets

Build evaluation datasets with multiple valid expressions to address the single ground truth limitation:

#### Using LLM to Generate Alternatives

```bash
python build_diverse_dataset.py \
    --mode llm \
    --task acos \
    --dataset rest16 \
    --model gpt-3.5-turbo \
    --n_alternatives 3 \
    --output_dir ./diverse_datasets
```

#### Using Manual Annotations

```bash
python build_diverse_dataset.py \
    --mode manual \
    --task acos \
    --dataset rest16 \
    --input manual_alternatives.json \
    --output_dir ./diverse_datasets
```

#### Convert to Evaluation Format

```bash
python build_diverse_dataset.py \
    --mode convert \
    --input diverse_datasets/acos_rest16_diverse_n3.json \
    --output_dir ./diverse_datasets
```

### Step 2: Evaluating Different LLM Models

Evaluate various LLM models on ABSA tasks:

#### Evaluate OpenAI GPT Models

```bash
python evaluate_llms.py \
    --model_name_or_path gpt-3.5-turbo \
    --task acos \
    --dataset rest16 \
    --target_element ACOS \
    --shot 20 \
    --output_dir ./outputs
```

#### Evaluate HuggingFace Models (via OpenAI-compatible API)

```bash
# First, start vLLM server or similar OpenAI-compatible API
# vllm serve meta-llama/Llama-3-8B --port 8000

python evaluate_llms.py \
    --model_name_or_path meta-llama/Llama-3-8B \
    --base-url http://localhost:8000/v1 \
    --task acos \
    --dataset rest16 \
    --constrained-decoding True \
    --output_dir ./outputs
```

#### Evaluate All Element Orders

```bash
python evaluate_llms.py \
    --model_name_or_path gpt-4 \
    --task acos \
    --dataset rest16 \
    --evaluate-all-orders \
    --output_dir ./outputs
```

## Datasets

The code expects datasets in the following structure:

```
datasets/
├── data/
│   ├── acos/
│   │   ├── rest16/
│   │   │   └── test.txt
│   │   └── laptop16-supcate/
│   │       └── test.txt
│   └── asqp/
│       ├── rest15/
│       │   └── test.txt
│       └── rest16/
│           └── test.txt
└── few_shot/
    ├── acos-rest16-20shot.json
    ├── acos-laptop16-supcate-20shot.json
    └── ...
```

**Note**: Dataset files should be linked or copied from the original ABSA project.

## Key Arguments

### Common Arguments

- `--task`: Task name (acos, asqp, aste, tasd)
- `--dataset`: Dataset name (rest16, laptop16-supcate, rest15, etc.)
- `--target_element`: Element order (ACOS, AOSC, OSAC, etc.)
- `--shot`: Number of demonstration examples (default: 20)
- `--seed`: Random seed (default: 42)

### LLM Evaluation Arguments

- `--model_name_or_path`: Model name or path
- `--base-url`: Base URL for OpenAI-compatible API (for HF models)
- `--temp`: Temperature for generation (default: 0.0)
- `--max_new_tokens`: Maximum tokens to generate (default: 256)
- `--constrained-decoding`: Use constrained decoding (default: False)
- `--max-concurrent`: Maximum concurrent requests (default: 5)
- `--evaluate-all-orders`: Evaluate all 24 element orders

### Diverse Dataset Arguments

- `--mode`: Mode (llm, manual, convert)
- `--model`: LLM model for generating alternatives
- `--n_alternatives`: Number of alternatives per quadruple (default: 3)
- `--input`: Input file for manual/convert mode

## Output Format

### Evaluation Results

Results are saved as JSON files with the following structure:

```json
{
  "preds": ["[A] food [C] food quality [S] positive [O] delicious", ...],
  "gt": ["[A] food [C] food quality [S] positive [O] delicious", ...],
  "idxs": [0, 1, 2, ...],
  "model": "gpt-3.5-turbo",
  "task": "acos",
  "dataset": "rest16",
  "target_element": "ACOS",
  "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "cost": 0.0}
}
```

### Diverse Dataset Format

```json
[
  {
    "sentence": "The food was delicious.",
    "original": [["food", "food quality", "positive", "delicious"]],
    "alternatives": [
      {
        "original": ["food", "food quality", "positive", "delicious"],
        "generated": [
          ["meal", "food quality", "positive", "tasty"],
          ["dish", "food quality", "positive", "excellent"]
        ]
      }
    ]
  }
]
```

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{yang-etal-2025-single,
    title = "Single Ground Truth Is Not Enough: Adding Flexibility to Aspect-Based Sentiment Analysis Evaluation",
    author = "Yang, Soyoung  and
      Cho, Hojun  and
      Lee, Jiyoung  and
      Yoon, Sohee  and
      Choi, Edward  and
      Choo, Jaegul  and
      Cho, Won Ik",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.603/",
    doi = "10.18653/v1/2025.naacl-long.603",
    pages = "12071--12096",
    ISBN = "979-8-89176-189-6",
}
```

## License

This code is based on the original ABSA project and maintains the same license.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

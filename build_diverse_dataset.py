#!/usr/bin/env python3
"""
Script for building diverse evaluation datasets with multiple valid expressions
This addresses the limitation of single ground truth in ABSA evaluation

Based on NAACL 2025 paper findings that show:
- Multiple valid ways to express the same sentiment
- Need for flexible evaluation metrics
- Importance of considering paraphrases and synonyms

Usage:
    # Interactive mode: Generate alternatives using LLM
    python build_diverse_dataset.py --task acos --dataset rest16 --mode llm --model gpt-3.5-turbo
    
    # Manual mode: Load manually annotated alternatives
    python build_diverse_dataset.py --task acos --dataset rest16 --mode manual --input alternatives.json
"""

import os
import argparse
import json
import asyncio
from datetime import datetime
import logging
from typing import List, Dict, Any

from data_utils import load_data
from utils import str2lower, str2bool, init_logging, save_json, load_json
from models import LLMs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback


DIVERSE_PROMPT = """You are an expert in aspect-based sentiment analysis. Given a quadruple (aspect, category, sentiment, opinion) extracted from a review, generate {n_alternatives} alternative valid expressions while maintaining the same meaning.

Original sentence: {sentence}
Original quadruple: [A] {aspect} [C] {category} [S] {sentiment} [O] {opinion}

Guidelines:
1. For aspect terms: Consider synonyms, paraphrases, or different spans that refer to the same entity
2. For opinion terms: Consider synonyms or alternative expressions of the same sentiment
3. For sentiment: Use equivalent sentiment expressions (positive/negative/neutral or great/bad/ok)
4. For category: Keep the same category (no alternatives needed)

Generate {n_alternatives} alternative quadruples in the same format. Each on a new line.
Only output the alternative quadruples, nothing else.
"""


async def generate_alternatives_llm(sentence: str, quadruple: tuple, 
                                   llm: LLMs, n_alternatives: int = 3) -> List[tuple]:
    """
    Generate alternative expressions for a quadruple using LLM
    
    Args:
        sentence: Original sentence
        quadruple: (aspect, category, sentiment, opinion)
        llm: LLM model
        n_alternatives: Number of alternatives to generate
    
    Returns:
        List of alternative quadruples
    """
    aspect, category, sentiment, opinion = quadruple
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in aspect-based sentiment analysis."),
        ("user", DIVERSE_PROMPT)
    ])
    
    chain = prompt | llm.model | StrOutputParser()
    
    try:
        with get_openai_callback() as cb:
            output = await chain.ainvoke({
                "sentence": sentence,
                "aspect": aspect,
                "category": category,
                "sentiment": sentiment,
                "opinion": opinion,
                "n_alternatives": n_alternatives
            })
            
            llm.prompt_tokens += cb.prompt_tokens
            llm.completion_tokens += cb.completion_tokens
        
        # Parse output to extract alternatives
        alternatives = []
        for line in output.strip().split('\n'):
            if line.strip() and '[A]' in line:
                # Parse the alternative quadruple
                # Format: [A] aspect [C] category [S] sentiment [O] opinion
                try:
                    parts = {}
                    for tag in ['[A]', '[C]', '[S]', '[O]']:
                        if tag in line:
                            idx = line.index(tag)
                            # Find next tag or end of line
                            next_idx = len(line)
                            for next_tag in ['[A]', '[C]', '[S]', '[O]']:
                                if next_tag != tag and next_tag in line[idx+3:]:
                                    next_idx = min(next_idx, line.index(next_tag, idx+3))
                            parts[tag] = line[idx+3:next_idx].strip()
                    
                    if len(parts) == 4:
                        alternatives.append((
                            parts['[A]'],
                            parts['[C]'],
                            parts['[S]'],
                            parts['[O]']
                        ))
                except Exception as e:
                    print(f"Failed to parse alternative: {line} - {e}")
                    continue
        
        return alternatives
    
    except Exception as e:
        print(f"Error generating alternatives: {e}")
        return []


async def build_diverse_dataset_llm(args):
    """
    Build diverse dataset using LLM to generate alternatives
    """
    logger = logging.getLogger()
    logger.info("Building diverse dataset using LLM")
    
    # Load LLM
    llm = LLMs(
        model_name=args.model,
        temp=args.temp,
        seed=args.seed,
        args=args
    )
    
    # Load original dataset
    data_path = os.path.join(args.data_dir, args.task, args.dataset, 'test.txt')
    logger.info(f"Loading data from: {data_path}")
    data = load_data(data_path, lowercase=False)
    
    # Generate alternatives for each sample
    diverse_data = []
    total = len(data)
    
    for idx, (sentence, quadruples) in enumerate(data):
        logger.info(f"Processing [{idx+1}/{total}]: {sentence}")
        
        # Store original and alternatives
        sample_data = {
            "sentence": sentence,
            "original": quadruples,
            "alternatives": []
        }
        
        # Generate alternatives for each quadruple
        for quad in quadruples:
            alternatives = await generate_alternatives_llm(
                sentence, quad, llm, n_alternatives=args.n_alternatives
            )
            sample_data["alternatives"].append({
                "original": quad,
                "generated": alternatives
            })
        
        diverse_data.append(sample_data)
        
        if (idx + 1) % 10 == 0:
            logger.info(f"Progress: {idx+1}/{total}, Usage: {llm.get_usage()}")
    
    # Save results
    output_path = os.path.join(
        args.output_dir, 
        f"{args.task}_{args.dataset}_diverse_n{args.n_alternatives}.json"
    )
    save_json(diverse_data, output_path)
    logger.info(f"Saved diverse dataset to: {output_path}")
    logger.info(f"Final usage: {llm.get_usage()}")


def build_diverse_dataset_manual(args):
    """
    Build diverse dataset from manually annotated alternatives
    """
    logger = logging.getLogger()
    logger.info("Building diverse dataset from manual annotations")
    
    # Load original dataset
    data_path = os.path.join(args.data_dir, args.task, args.dataset, 'test.txt')
    logger.info(f"Loading data from: {data_path}")
    original_data = load_data(data_path, lowercase=False)
    
    # Load manual alternatives
    logger.info(f"Loading manual alternatives from: {args.input}")
    manual_alternatives = load_json(args.input)
    
    # Combine original and manual alternatives
    diverse_data = []
    for idx, (sentence, quadruples) in enumerate(original_data):
        sample_data = {
            "sentence": sentence,
            "original": quadruples,
            "alternatives": manual_alternatives.get(str(idx), [])
        }
        diverse_data.append(sample_data)
    
    # Save results
    output_path = os.path.join(
        args.output_dir,
        f"{args.task}_{args.dataset}_diverse_manual.json"
    )
    save_json(diverse_data, output_path)
    logger.info(f"Saved diverse dataset to: {output_path}")


def convert_to_evaluation_format(args):
    """
    Convert diverse dataset to evaluation format
    This format groups all valid expressions for each sample
    """
    logger = logging.getLogger()
    logger.info("Converting to evaluation format")
    
    # Load diverse dataset
    diverse_data = load_json(args.input)
    
    # Convert to evaluation format
    eval_data = []
    for sample in diverse_data:
        sentence = sample["sentence"]
        all_valid = []
        
        # Include original
        all_valid.append(sample["original"])
        
        # Include alternatives
        for alt_group in sample.get("alternatives", []):
            if isinstance(alt_group, dict) and "generated" in alt_group:
                all_valid.extend(alt_group["generated"])
            elif isinstance(alt_group, list):
                all_valid.extend(alt_group)
        
        eval_data.append([sentence, all_valid])
    
    # Save
    output_path = os.path.join(
        args.output_dir,
        f"{os.path.basename(args.input).replace('.json', '_eval.json')}"
    )
    save_json(eval_data, output_path)
    logger.info(f"Saved evaluation format to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build diverse evaluation datasets with multiple valid expressions"
    )
    
    # Mode
    parser.add_argument("--mode", type=str, choices=["llm", "manual", "convert"],
                       required=True,
                       help="Mode: 'llm' to generate with LLM, 'manual' to load manual annotations, 'convert' to convert to eval format")
    
    # Data settings
    parser.add_argument("--task", type=str2lower, default="acos",
                       help="Task name (acos, asqp)")
    parser.add_argument("--dataset", type=str2lower, default="rest16",
                       help="Dataset name")
    parser.add_argument("--data_dir", type=str,
                       default="./datasets/data",
                       help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./diverse_datasets",
                       help="Output directory")
    
    # LLM mode settings
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                       help="LLM model for generating alternatives")
    parser.add_argument("--n_alternatives", type=int, default=3,
                       help="Number of alternatives to generate per quadruple")
    parser.add_argument("--temp", type=float, default=0.7,
                       help="Temperature for generation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Manual/Convert mode settings
    parser.add_argument("--input", type=str, default="",
                       help="Input file for manual or convert mode")
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="./logs",
                       help="Log directory")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logging
    date_str = datetime.today().strftime('%m%d_%H%M')
    log_path = os.path.join(
        args.log_dir,
        f"diverse_dataset_{args.mode}_{date_str}.log"
    )
    logger = init_logging(log_path, stdout=True)
    logger.info(f"Arguments: {args}")
    
    # Run appropriate mode
    if args.mode == "llm":
        asyncio.run(build_diverse_dataset_llm(args))
    elif args.mode == "manual":
        if not args.input:
            raise ValueError("--input is required for manual mode")
        build_diverse_dataset_manual(args)
    elif args.mode == "convert":
        if not args.input:
            raise ValueError("--input is required for convert mode")
        convert_to_evaluation_format(args)


if __name__ == "__main__":
    main()


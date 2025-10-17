#!/usr/bin/env python3
"""
Script for evaluating different LLM models on ABSA tasks
Based on NAACL 2025 paper: "Zoom-in-Zoom-out ABSA"

Usage:
    python evaluate_llms.py --model_name gpt-3.5-turbo --task acos --dataset rest16
    python evaluate_llms.py --model_name meta-llama/Llama-3-8B --base-url http://localhost:8000/v1 --task acos --dataset rest16
"""

import os
import argparse
import json
import asyncio
from datetime import datetime
import logging
import numpy as np

from eval_utils import compute_scores
from data_utils import prepare_dataset_wo_args, prepare_demo
from utils import str2lower, str2upper, str2bool, init_logging
from models import LLMs
from methods import apred
from config import ORDERS


async def evaluate_model(args, result_path):
    """
    Main evaluation function
    
    Args:
        args: Command line arguments
        result_path: Path to save results
    
    Returns:
        0 on success
    """
    logger = logging.getLogger()

    # Load LLM
    logger.info(f"Loading model: {args.model_name_or_path}")
    llms = LLMs(
        model_name=args.model_name_or_path,
        temp=args.temp, 
        top_p=args.top_p,
        n_out=args.n_out,
        max_new_tokens=args.max_new_tokens, 
        seed=args.seed,
        base_url=args.base_url,
        args=args
    )

    # Load test dataset
    test_fpath = os.path.join(args.test_data_dir, args.task, args.dataset, 'test.txt')
    logger.info(f"Loading test data from: {test_fpath}")
    test_xy = prepare_dataset_wo_args(
        fpath=test_fpath,
        target_element=args.target_element,
        task=args.task
    )

    # Load demonstration examples
    demo_path = os.path.join(args.demo_dir, f"{args.task}-{args.dataset}-{args.shot}shot")
    if args.ideal_samples:
        demo_path += "-ideal.json"
    else:
        demo_path += ".json"
    
    logger.info(f"Loading demo examples from: {demo_path}")
    demo = prepare_demo(
        demo_path, 
        trg_elements=args.target_element,
        delimeter=args.delimeter
    )
    
    if args.shuffle_demo:
        np.random.seed(args.seed)
        np.random.shuffle(demo)

    # Run evaluation
    gts = []
    preds = []
    idxs = []
    n_test = len(test_xy)

    async def wrapper(i, x, y, semaphore: asyncio.Semaphore):
        """Wrapper for async prediction with semaphore"""
        async with semaphore:
            return i, x, y, await apred(llms, x=x, demo=demo, args=args)

    # Evaluate with concurrency control
    index = 0
    semaphore = asyncio.Semaphore(args.max_concurrent)
    logger.info(f"Starting evaluation with max_concurrent={args.max_concurrent}")
    
    for coro in asyncio.as_completed([wrapper(i, x, y, semaphore) for i, (x, y) in enumerate(test_xy)]):
        i, x, y, pred = await coro
        index += 1

        gts.append(y)
        preds.append(pred)
        idxs.append(i)

        logger.info(f"\n[{index}/{n_test}] {i} <<INPUT>> {x}\n<<GT>> {y}\n<<Output>> {pred}")
        if "gpt" in args.model_name_or_path:
            logger.info(f"Usage so far: {llms.get_usage()}")

    # Save results
    logger.info(f"Saving results to: {result_path}")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            "preds": preds, 
            "gt": gts, 
            "idxs": idxs,
            "model": args.model_name_or_path,
            "task": args.task,
            "dataset": args.dataset,
            "target_element": args.target_element,
            "usage": llms.get_usage()
        }, f, indent=2, ensure_ascii=False)

    return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM models on ABSA tasks")
    
    # Model settings
    parser.add_argument("--model_name_or_path", type=str, required=True,
                       help="Model name or path (e.g., gpt-3.5-turbo, meta-llama/Llama-3-8B)")
    parser.add_argument("--base-url", type=str, default="",
                       help="Base URL for OpenAI-compatible API (for HuggingFace models)")
    
    # Task settings
    parser.add_argument("--task", type=str2lower, default="acos",
                       help="Task name (acos, asqp)")
    parser.add_argument("--dataset", type=str2lower, default="rest16",
                       help="Dataset name (rest16, laptop16-supcate, rest15)")
    parser.add_argument("--target_element", type=str2upper, default="ACOS",
                       help="Target element order (e.g., ACOS, AOSC)")
    
    # Data paths
    parser.add_argument("--test_data_dir", type=str, 
                       default="./datasets/data",
                       help="Test data directory")
    parser.add_argument("--demo_dir", type=str, 
                       default="./datasets/few_shot",
                       help="Demo examples directory")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for results")
    
    # Few-shot settings
    parser.add_argument("--shot", type=int, default=20,
                       help="Number of demonstration examples")
    parser.add_argument("--shuffle-demo", type=str2bool, default="True",
                       help="Whether to shuffle demo examples")
    parser.add_argument("--ideal-samples", action="store_true",
                       help="Use ideal samples for demonstration")
    
    # Prompt settings
    parser.add_argument("--delimeter", type=str, default="####",
                       help="Delimiter between quadruples")
    parser.add_argument("--sent_verb", type=str2bool, default="False",
                       help="Verbalize sentiment (positive -> great)")
    parser.add_argument("--cspace", type=str2bool, default="True",
                       help="Use spaces in category names")
    
    # Generation settings
    parser.add_argument("--temp", type=float, default=0.,
                       help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.,
                       help="Top-p for generation")
    parser.add_argument("--n_out", type=int, default=1,
                       help="Number of outputs to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum new tokens to generate")
    parser.add_argument("--constrained-decoding", type=str2bool, default="False",
                       help="Use constrained decoding")
    
    # Execution settings
    parser.add_argument("--max-concurrent", type=int, default=5,
                       help="Maximum concurrent requests")
    parser.add_argument("--evaluate-all-orders", action="store_true",
                       help="Evaluate all 24 element orders")
    parser.add_argument("--order-reverse", action="store_true",
                       help="Reverse the order list")
    
    # Other
    parser.add_argument("--exp_name", type=str, default="eval",
                       help="Experiment name")
    parser.add_argument("--date", type=str, default="",
                       help="Date string (default: today)")
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode")
    
    args = parser.parse_args()
    
    # Set date
    if not args.date:
        args.date = datetime.today().strftime('%m%d')
    
    # Validation
    if args.constrained_decoding:
        assert args.base_url != "", \
            f"Constrained decoding requires base_url for open-source models"
    
    # Setup output directory
    if args.output_dir.endswith("outputs"):
        args.output_dir = os.path.join(
            args.output_dir, 
            f"{args.task}-{args.dataset}", 
            args.model_name_or_path.replace("/", "--")
        )
    
    if 'llama' in args.model_name_or_path.lower():
        args.output_dir += f"_CD{int(args.constrained_decoding)}"
    
    # Setup log directory
    log_dir = os.path.join(
        "./logs", 
        f"{args.task}-{args.dataset}", 
        args.model_name_or_path.replace("/", "--")
    )
    if 'llama' in args.model_name_or_path.lower():
        log_dir += f"_CD{int(args.constrained_decoding)}"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which orders to evaluate
    if args.evaluate_all_orders:
        orders = ORDERS.copy()
        if args.order_reverse:
            orders = orders[::-1]
    else:
        orders = [args.target_element]
    
    # Run evaluation for each order
    for order in orders:
        args.target_element = order
        
        exp_name = (
            f"{args.date}_TRG={args.target_element}_"
            f"{os.path.basename(args.model_name_or_path)}_"
            f"temp{args.temp}_{args.exp_name}_"
            f"CD{int(args.constrained_decoding)}_{args.shot}-shot_"
            f"shuffled{int(args.shuffle_demo)}"
        )
        
        if args.ideal_samples:
            exp_name += "_ideal"
        
        log_path = os.path.join(log_dir, f"log_{exp_name}.log")
        result_path = os.path.join(args.output_dir, f"{exp_name}.json")
        
        # Skip if already done
        if os.path.exists(result_path):
            print(f"Experiment already done: {result_path}")
            continue
        
        # Initialize logger
        logger = init_logging(log_path, stdout=True)
        logger = logging.getLogger()
        logger.info(args)
        logger.info(f"Current experiment: {exp_name}")
        
        # Run evaluation
        asyncio.run(evaluate_model(args, result_path))


if __name__ == "__main__":
    main()


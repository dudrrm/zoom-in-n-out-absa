"""Main script for model evaluation on ABSA tasks

This script evaluates LLM models on ABSA datasets with support for:
- Multiple element orders
- Few-shot learning
- Constrained decoding (for open-source models)
- Async processing for efficiency
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


async def amain(args, result_path):
    """Main async function for model evaluation
    
    Args:
        args: Command line arguments
        result_path: Path to save results
        
    Returns:
        0 on success
    """
    logger = logging.getLogger()

    # Load predefined LLM
    llms = LLMs(model_name=args.model_name_or_path,
                temp=args.temp, top_p=args.top_p,
                n_out=args.n_out,
                max_new_tokens=args.max_new_tokens, seed=args.seed,
                args=args)

    # Load dataset
    test_fpath = os.path.join(args.test_data_dir, args.task, args.dataset, 'test.txt')
    if not os.path.exists(test_fpath):
        logger.error(f"Test file not found: {test_fpath}")
        return 1
    
    test_xy = prepare_dataset_wo_args(fpath=test_fpath,
                                      target_element=args.target_element,
                                      task=args.task)

    # Load demo examples
    demo_path = os.path.join(args.demo_dir, f"{args.task}-{args.dataset}-{args.shot}shot")
    if args.ideal_samples:
        demo_path += "-ideal.json"
    else:
        demo_path += ".json"

    if not os.path.exists(demo_path):
        logger.warning(f"Demo file not found: {demo_path}. Using empty demos.")
        demo = []
    else:
        demo = prepare_demo(demo_path, trg_elements=args.target_element,
                           delimeter=args.delimeter)
    
    if args.shuffle_demo and demo:
        np.random.seed(args.seed)
        np.random.shuffle(demo)

    # Generate predictions
    gts = []
    preds = []
    idxs = []
    n_test = len(test_xy)

    async def wrapper(i, x, y, semaphore: asyncio.Semaphore):
        """Wrapper for async prediction with semaphore"""
        async with semaphore:
            return i, x, y, await apred(llms, x=x, demo=demo, args=args)

    # Evaluate with async
    index = 0
    semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
    
    for coro in asyncio.as_completed([wrapper(i, x, y, semaphore) 
                                     for i, (x, y) in enumerate(test_xy)]):
        i, x, y, pred = await coro
        index = index + 1

        gts.append(y)
        preds.append(pred)
        idxs.append(i)

        logger.info(f"\n[{index}/{n_test}] {i} <<INPUT>> {x}\n<<GT>> {y}\n<<Output>> {pred}")
        if "gpt" in args.model_name_or_path:
            logger.info(f"Usage so far: {llms.get_usage()}")

    # Save results
    logger.info(f"Results will be saved at... {result_path}")
    with open(result_path, 'w') as f:
        json.dump({"preds": preds, "gt": gts, "idxs": idxs}, f, indent=2, ensure_ascii=False)

    # Compute and log scores (optional)
    try:
        scores, _, _ = compute_scores(preds, gts, verbose=False, delimeter=args.delimeter)
        exp_results = "[exact match] all preds | p, r, f1 | {:.2f}, {:.2f}, {:.2f}".format(
            scores["precision"], scores["recall"], scores["f1"])
        logger.info(exp_results)
    except Exception as e:
        logger.warning(f"Failed to compute scores: {e}")

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Model Evaluation for ABSA tasks"
    )
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode")

    # Model configuration
    parser.add_argument("--date", type=str, default="",
                        help="Date string (auto-generated if empty)")
    parser.add_argument("--exp_name", type=str, default="naive",
                        help="Experiment name")
    parser.add_argument("--model_name_or_path", type=str, default="gpt-4o",
                        help="Model name or path")
    parser.add_argument("--base-url", type=str, default="",
                        help="Base URL for custom API endpoints")
    
    # Prompt configuration
    parser.add_argument("--delimeter", type=str, default="####",
                        help="Delimiter between quadruples")
    parser.add_argument("--sent_verb", type=str2bool, default="False",
                        help="Verbalize sentiment or not")
    parser.add_argument("--cspace", type=str2bool, default="True",
                        help="Use space in category names")
    
    # Directory configuration
    parser.add_argument("--task", type=str2lower, default="acos",
                        choices=["acos", "asqp", "aste", "tasd"],
                        help="ABSA task type")
    parser.add_argument("--dataset", type=str2lower, default="rest16",
                        help="Dataset name")
    parser.add_argument("--test_data_dir", type=str, 
                        default="./datasets/data",
                        help="Directory containing test data")
    parser.add_argument("--demo_dir", type=str, 
                        default="./datasets/few_shot",
                        help="Directory containing few-shot demos")
    parser.add_argument("--output_dir", type=str, 
                        default="./outputs",
                        help="Output directory")
    
    # Dataset setup
    parser.add_argument("--target_element", type=str2upper, default="ACOS",
                        help="Element order (e.g., ACOS, ACSO)")
    parser.add_argument("--shot", type=int, default=20,
                        help="Number of few-shot examples")
    parser.add_argument("--shuffle-demo", type=str2bool, default="True",
                        help="Shuffle demo examples")
    parser.add_argument("--ideal-samples", action="store_true",
                        help="Use ideal samples for demos")
    parser.add_argument("--order-reverse", action="store_true",
                        help="Reverse order of element permutations")
    
    # Generation parameters
    parser.add_argument("--temp", type=float, default=0.,
                        help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.,
                        help="Top-p sampling parameter")
    parser.add_argument("--n_out", type=int, default=1,
                        help="Number of outputs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum new tokens")
    parser.add_argument("--constrained-decoding", type=str2bool, default="False",
                        help="Enable constrained decoding")
    
    args = parser.parse_args()
    
    # Process arguments
    if not args.date:
        args.date = datetime.today().strftime('%m%d')

    if args.constrained_decoding:    
        assert args.base_url != "", \
            f"Constrained decoding requires base-url for open-source models. " \
            f"Current model: {args.model_name_or_path}, Current base-url: {args.base_url}"

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

    # All possible element orders
    orders = [
        'AOSC', 'OCSA', 'OSAC', 'OSCA', 'OACS', 'AOCS', 
        'COAS', 'SAOC', 'OASC', 'SOAC', 'SOCA', 'ASOC', 
        'CAOS', 'SCAO', 'OCAS', 'COSA', 'CASO', 'CSAO', 
        'ACOS', 'ACSO', 'SCOA', 'CSOA', 'SACO', 'ASCO'
    ]

    if args.order_reverse:
        orders = orders[::-1]

    # Run evaluation for each order
    for order in orders:

        args.target_element = order
        
        exp_name = (
            f"{args.date}_TRG={args.target_element}_"
            f"{os.path.basename(args.model_name_or_path)}_temp{args.temp}_"
            f"{args.exp_name}_CD{int(args.constrained_decoding)}_"
            f"{args.shot}-shot_shuffled{int(args.shuffle_demo)}"
        )
        
        if args.ideal_samples:
            exp_name += "_ideal"

        log_log_path = os.path.join(log_dir, f"log_{exp_name}.log")
        result_path = os.path.join(args.output_dir, f"{exp_name}.json")

        # Skip if already done
        if os.path.exists(result_path):
            print(f"This experiment is already done at ... {result_path}")
            continue

        # Initialize logging
        logger = init_logging(log_log_path, stdout=True)
        logger = logging.getLogger()
        logger.info(args)
        logger.info(f"Current Exp name: {exp_name}")
        
        # Run evaluation
        asyncio.run(amain(args, result_path))


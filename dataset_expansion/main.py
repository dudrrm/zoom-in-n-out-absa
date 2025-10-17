"""Main script for dataset expansion (Zoom-In-N-Out)

This script generates alternative expressions for aspect and opinion terms
using the Zoom-In-N-Out approach:
1. Narrow: Generate narrower expressions
2. Widen: Generate wider expressions  
3. Judge: Validate generated alternatives using LLM
4. Merge: Combine aspect and opinion alternatives
"""

import os
import sys
import argparse
import json
import copy
import logging
from datetime import datetime
import random

# Add current directory to path first to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import from current directory (dataset_expansion)
from utils import str2bool, str2lower, str2upper, init_logging, load_data, count_tag
from modules import generate_alternative, generate_judge, merge_AO_wo_overlap

# Add model_evaluation directory to path to import LLM models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model_evaluation'))
from models import LLMs


def main(args):
    """Main function for dataset expansion
    
    Args:
        args: Command line arguments
        
    Returns:
        Merged results with alternative expressions
    """
    
    steps = ["narrow", "widen", "judge"]
    if args.judge_only:
        steps = ["judge"]
    elif args.gen_only:
        steps = ["narrow", "widen"]
        
    # Initialize experiment directories
    result_dir = os.path.join(args.out_dir, f"{args.task}-{args.dataset}")
    log_dir = os.path.join(args.log_dir, f"{args.task}-{args.dataset}")
    if args.now_debug and (not args.judge_only):
        result_dir = os.path.join(result_dir, "debug")
        log_dir = os.path.join(log_dir, "debug")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    exp_name = f"{args.date}_{args.model_name.replace('/', '-')}_tmp{args.temp}_sample{args.sample}"
    if args.judge_only:
        exp_name += f"_from_{args.load_pattern}"
    if args.now_debug:
        exp_name += "_debug"
    log_path = os.path.join(log_dir, f"{exp_name}.log")

    # Initialize logging
    init_logging(log_path, stdout=True)
    logger = logging.getLogger()
    
    # Load test data
    test_fpath = os.path.join(args.data_dir, 'test.txt')
    test_xy = load_data(test_fpath)

    if args.now_debug:
        # Sample subset for debugging
        test_idxs = random.sample(range(len(test_xy)), min(10, len(test_xy)))
        org_test_xy = copy.deepcopy(test_xy)
        test_xy = [org_test_xy[i] for i in test_idxs]

    # Target elements and conditions
    trgs = ["A", "O"]  # Aspect and Opinion
    conds = ["SOC", "CAS"]  # Conditions for each target
    results = {"A": [], "O": []}
    
    # Total time and cost tracking
    time_cost_all = 0
    total_cost = 0
    
    # Generate data for each target element
    for trg_e, cond_e in zip(trgs, conds):
        data_new = []
        if args.judge_only:
            # Load previously generated data
            load_path = os.path.join(result_dir, f"{args.load_pattern}_{trg_e}.json")
            with open(load_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded data from {load_path}")
            if args.now_debug:
                data_org = copy.deepcopy(data)
                data = [data_org[i] for i in test_idxs]
        else:
            data = test_xy

        for idx, step in enumerate(steps):

            # Start time
            start_time = datetime.now()

            # Load model
            temp = 0. if step == "judge" else args.temp
            llms = LLMs(model_name=args.model_name, temp=temp, 
                        max_new_tokens=args.max_new_tokens, seed=args.seed,
                        args=args)
 
            if step != "judge":  # Generate alternative expressions
                data_new, cost = generate_alternative(args, llms, trg_e, cond_e, data, step)
                
            elif step == "judge":  # Judge generated alternatives
                data_new, cost = generate_judge(args, llms, data, trg_e)
            
            del llms
            
            # Statistics
            tag_count = count_tag(data_new)
            logger.info(f"[Statistics] tags on TRG={trg_e} after {step} ... {tag_count}")
            logger.info(f"Cost of {step.capitalize()} step: ${cost:.4f}")

            # Save intermediate results
            fpath = os.path.join(result_dir, f"{exp_name}_{idx}-{step}_{trg_e}.json")

            with open(fpath, 'w') as f:
                json.dump(data_new, f, indent=2, ensure_ascii=True)
            logger.info(f"The results of Step {step} TRG={trg_e} are saved at ... {fpath}")

            # End time
            end_time = datetime.now()
            time_cost = end_time - start_time
            time_cost_all += time_cost.total_seconds()

            logger.info(f"Consuming time of {step.capitalize()} step: {end_time - start_time}")

            data = data_new
            total_cost += cost

        # Save results
        results[trg_e] = data_new

    # Total consuming time
    total_seconds = int(time_cost_all)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_cost_all_str = "{:02}:{:02}:{:02}".format(hours, minutes, seconds)
    logger.info(f"Total consuming time: {time_cost_all_str}")
    logger.info(f"Total cost: ${total_cost:.2f}")

    # Merge aspect and opinion predictions
    merged_result, merged_result_wo_tag = merge_AO_wo_overlap(results)

    if args.now_debug:
        import pdb
        pdb.set_trace()
    else:
        # Save final results with tag
        save_path = os.path.join(result_dir, f"{exp_name}_fin.json")
        with open(save_path, 'w') as f:
            json.dump(merged_result, f, indent=2, ensure_ascii=True)
        logger.info(f"The results are saved at ... {save_path}")

        # Save final results without tag (for evaluation)
        save_path = os.path.join(args.data_dir, f"test_{args.date}.txt")
        with open(save_path, 'w') as f:
            for x, ys in merged_result_wo_tag:
                f.write(f"{x}####{ys}\n")
        
        logger.info(f"The results are saved at ... {save_path}")

    return merged_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataset Expansion for ABSA using Zoom-In-N-Out approach"
    )

    # Experiment conditions
    parser.add_argument("--date", type=str, default="",
                        help="Date string for experiment name (auto-generated if empty)")
    parser.add_argument("--now-debug", action='store_true',
                        help="Debug mode (use subset of data)")
    parser.add_argument("--sample", type=int, default=1,
                        help="Number of samples per generation")
    parser.add_argument("--threshold", type=int, default=1,
                        help="Minimum occurrences to accept alternative")

    # Pipeline control
    parser.add_argument("--gen-only", action='store_true',
                        help="Only generate alternatives without judging")
    parser.add_argument("--judge-only", action='store_true',
                        help="Only judge previously generated alternatives")
    parser.add_argument("--load-pattern", type=str, default="",
                        help="Pattern to load for judge-only mode")
     
    # Task and dataset
    parser.add_argument("--task", type=str2lower, default="acos",
                        choices=["acos", "asqp", "aste", "tasd"],
                        help="ABSA task type")
    parser.add_argument("--dataset", type=str2lower, default="rest16",
                        help="Dataset name (e.g., rest16, laptop16)")

    # Model parameters
    parser.add_argument("--model-name", type=str, default="gpt-4o",
                        help="LLM model name")
    parser.add_argument("--seed", type=float, default=None,
                        help="Random seed")
    parser.add_argument("--temp", type=float, default=0.,
                        help="Temperature for generation")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum new tokens to generate")
    parser.add_argument("--base-url", type=str, default="",
                        help="Base URL for custom API endpoints")
    
    # Directories
    parser.add_argument("--data-dir", type=str, 
                        default="./datasets/data",
                        help="Directory containing test data")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="Directory for logs")
    parser.add_argument("--out-dir", type=str, default="./outputs",
                        help="Directory for outputs")

    args = parser.parse_args()

    # Process arguments
    if args.data_dir.endswith('data'):
        args.data_dir = os.path.join(args.data_dir, args.task, args.dataset)

    if not args.date:
        args.date = datetime.today().strftime('%m%d')

    if args.judge_only:
        assert args.load_pattern != "", "Must provide --load-pattern for judge-only mode"

    # Run main
    main(args)


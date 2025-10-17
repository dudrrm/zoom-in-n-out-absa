"""Utility functions for dataset expansion"""

import os
import json
import sys
import argparse
import logging
from collections import Counter
from typing import List, Tuple, Union, Dict


def load_data(path: str, lowercase: bool = True):
    """Load dataset from file
    
    Args:
        path: Path to data file (.txt or .json)
        lowercase: Whether to lowercase the text
        
    Returns:
        List of [sentence, quadruples] pairs
    """
    data = list()

    if path.endswith("txt"):
        with open(path, 'r') as f:
            d_str = [l.strip() for l in f.readlines() if len(l) > 0]
        for line in d_str:
            if lowercase:
                line = line.lower()
            words, tuples = line.split("####")
            data.append([words, [[tuple(q + ["original"])] for q in eval(tuples)]])

    elif path.endswith("json"):
        with open(path, 'r') as f:
            data = json.load(f)

    else:
        raise NotImplementedError(f"Unsupported file format: {path}")

    return data


def str2bool(v):
    """Convert string to boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2lower(v: str):
    """Convert string to lowercase"""
    return v.lower()


def str2upper(v: str):
    """Convert string to uppercase"""
    return v.upper()


def init_logging(log_file, stdout=False):
    """Initialize logging configuration
    
    Args:
        log_file: Path to log file
        stdout: Whether to also log to stdout
        
    Returns:
        Logger instance
    """
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')

    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    if stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    logger.info('Making log output file: %s', log_file)
    if not os.path.exists(log_dir):
        logger.info('Creating log directory: %s', log_dir)

    return logger


def count_tag(data: list):
    """Count tags in the data
    
    Args:
        data: List of [sentence, quadruples] pairs
        
    Returns:
        Dictionary with tag counts
    """
    tag_all = []
    order = ["original", "narrow", "widen", "contraction", "spell"]
    
    for x, ys in data:
        for yi in ys:
            if type(yi) == tuple:
                yi = [yi]
            for yj in yi:
                tag = list(yj)[-1]
                tag_all.append(tag)
    
    count = Counter(tag_all)

    return {o: dict(count)[o] if o in count else 0 for o in order}


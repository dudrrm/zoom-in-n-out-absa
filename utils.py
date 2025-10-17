"""
Utility functions for ABSA evaluation
"""

import os
import sys
import json
import argparse
import logging


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
    """Initialize logging with file and optional stdout"""
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(module)s: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S'
    )

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


def load_json(path: str):
    """Load JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json(data, path: str, indent=2):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


"""Utility functions for model evaluation"""

import os
import json
import sys
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


def reformat_y(gt):
    """Reformat ground truth string"""
    gt = gt.replace("[A] it ", "[A] NULL ")
    gt = gt.replace("[NXT]", "####")
    return gt.strip()


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


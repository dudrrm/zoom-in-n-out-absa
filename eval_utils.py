"""
Evaluation utilities for ABSA
"""

import re
import numpy as np
from config import SENTTAG2SENTWORD, SENTTAG2OPINION, SENTWORD2OPINION


def extract_spans_para(seq, seq_type, delimeter="####", lower=True, order="ACSO"):
    """
    Extract quadruples from predicted/gold sequences
    
    Args:
        seq: Input sequence string
        seq_type: 'pred' or 'gold'
        delimeter: Delimiter between quadruples
        lower: Convert to lowercase
        order: Element order (e.g., "ACSO")
    
    Returns:
        List of quadruples
    """
    quads = []
    sents = [s.strip() for s in seq.split(delimeter)]
    
    for s in sents:
        try:
            tok_list = ["[C]", "[S]", "[A]", "[O]"]

            for tok in tok_list:
                if tok not in s:
                    s += " {} null".format(tok)
            
            index_ac = s.index("[C]")
            index_sp = s.index("[S]")
            index_at = s.index("[A]")
            index_ot = s.index("[O]")

            combined_list = [index_ac, index_sp, index_at, index_ot]
            arg_index_list = list(np.argsort(combined_list))

            result = []
            for i in range(len(combined_list)):
                start = combined_list[i] + 4
                sort_index = arg_index_list.index(i)
                if sort_index < 3:
                    next_ = arg_index_list[sort_index + 1]
                    re = s[start:combined_list[next_]]
                else:
                    re = s[start:]	
                
                if lower:
                    result.append(re.lower().strip())
                else:
                    result.append(re.strip())

            ac, sp, at, ot = result

            # if the aspect term is implicit
            if at.lower() == 'it':
                at = 'null'

        except ValueError:
            try:
                print(f'In {seq_type} seq, cannot decode: {s}')
            except UnicodeEncodeError:
                print(f'In {seq_type} seq, a string cannot be decoded')
            ac, at, sp, ot = '', '', '', ''
		
        qq = {"A": at, "C": ac, "S": sp, "O": ot}
        if order != "CASO":
            new_q = []
            for e in order.upper():
                new_q.append(qq[e])
            quads.append(tuple(new_q))
        else:
            quads.append((ac, at, sp, ot))	
    
    return quads


def compute_f1_scores(pred_pt, gold_pt, verbose=True):
    """
    Compute F1 scores with predicted and gold quadruples
    
    Args:
        pred_pt: List of predicted quadruples for each sample
        gold_pt: List of gold quadruples for each sample
        verbose: Print statistics
    
    Returns:
        precision, recall, f1
    """
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    if verbose:
        print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    return precision, recall, f1


def compute_scores(pred_seqs, gold_seqs, delimeter="####", verbose=False):
    """
    Compute model performance
    
    Args:
        pred_seqs: List of predicted sequences
        gold_seqs: List of gold sequences
        delimeter: Delimiter between quadruples
        verbose: Print statistics
    
    Returns:
        scores dict, all_labels, all_preds
    """
    assert len(pred_seqs) == len(gold_seqs), (len(pred_seqs), len(gold_seqs))
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        if isinstance(gold_seqs[i], list):
            gold_list = [tuple(q) for q in gold_seqs[i]]
        else:
            gold_list = extract_spans_para(gold_seqs[i], 'gold', delimeter, order="ACSO")
        
        pred_list = extract_spans_para(pred_seqs[i], 'pred', delimeter, order="ACSO")

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    precision, recall, f1 = compute_f1_scores(all_preds, all_labels, verbose=verbose)

    scores = {
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1': round(f1 * 100, 2)
    }

    return scores, all_labels, all_preds


def compute_score_multigt(pred_str: list, gold_list: list, delimeter: str = "####",
                          lower: bool = True, sentverb: bool = False, cspace: bool = True):
    """
    Compute scores with multiple ground truths (diverse expressions)
    
    Args:
        pred_str: List of predicted strings
        gold_list: List of gold labels (each can have multiple valid expressions)
        delimeter: Delimiter between quadruples
        lower: Lowercase comparison
        sentverb: Use sentiment verbs
        cspace: Use spaces in categories
    
    Returns:
        scores dict with precision, recall, f1
    """
    n_tp, n_gold, n_pred = 0, 0, 0
    
    for pred, gt in zip(pred_str, gold_list):
        pred_list = extract_spans_para(seq=pred, seq_type='pred', delimeter=delimeter, 
                                       lower=lower, order="ACSO")
        n_pred += len(pred_list)
        n_gold += len(gt)
    
        # Gather all GTs
        gt_all = []
        for gi in gt:
            for gii in gi:
                a, c, s, o = gii
                new_c = c.replace("style_options", "style&options").lower()
                new_s = s
                if s in SENTWORD2OPINION.keys() and sentverb:
                    new_s = SENTWORD2OPINION[s.lower()]
                elif s.lower() in SENTTAG2SENTWORD.keys() and (not sentverb):
                    new_s = SENTTAG2SENTWORD[s.lower()]

                gt_all.append(tuple([a.lower(), new_c, new_s.lower(), o.lower()]))
                
        # Verify answers
        for pi in pred_list:
            a, c, s, o = pi
            pi = (a, c.replace("style_options", "style&options").lower(), s, o)
            if pi in gt_all:
                n_tp += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    
    scores = {
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1': round(f1 * 100, 2)
    }
    return scores


"""Evaluation utilities for computing metrics"""

import re
import numpy as np
from const import senttag2sentword, senttag2opinion, sentword2opinion


e2idx = {"A": 0, "C": 1, "S": 2, "O": 3, "tag": 4}
abb2e = {"A": "Aspect", "O": "Opinion", "C": "Category", "S": "Sentiment"}


def extract_spans_para(seq, seq_type, delimeter="[SSEP]", lower=True, order="ACSO"):
    """Extract spans from prediction/gold string
    
    Args:
        seq: Input sequence string
        seq_type: Type of sequence ('pred' or 'gold')
        delimeter: Delimiter between quadruples
        lower: Whether to lowercase
        order: Element order
        
    Returns:
        List of extracted quadruples
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

            # If aspect term is implicit
            if at.lower() == 'it':
                at = 'null'

        except ValueError:
            try:
                print(f'In {seq_type} seq, cannot decode: {s}')
                pass
            except UnicodeEncodeError:
                print(f'In {seq_type} seq, a string cannot be decoded')
                pass
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
    """Compute F1 scores with prediction and gold quadruples
    
    Args:
        pred_pt: Predicted quadruples
        gold_pt: Gold quadruples
        verbose: Whether to print statistics
        
    Returns:
        (precision, recall, f1) tuple
    """
    # Number of true positive, gold standard, predictions
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


def compute_scores(pred_seqs, gold_seqs, delimeter="[SSEP]", verbose=False):
    """Compute model performance metrics
    
    Args:
        pred_seqs: List of predicted sequences
        gold_seqs: List of gold sequences
        delimeter: Delimiter between quadruples
        verbose: Whether to print detailed info
        
    Returns:
        (scores, all_labels, all_preds) tuple
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
    """Compute scores with multiple ground truths (expanded dataset)
    
    Args:
        pred_str: List of predicted strings
        gold_list: List of gold quadruple lists (with alternatives)
        delimeter: Delimiter between quadruples
        lower: Whether to lowercase
        sentverb: Whether to use sentiment verbalization
        cspace: Whether to use space in categories
        
    Returns:
        Dictionary with scores
    """
    n_tp, n_gold, n_pred = 0, 0, 0
    
    for pred, gt in zip(pred_str, gold_list):

        pred_list = extract_spans_para(seq=pred, seq_type='pred', delimeter=delimeter, 
                                       lower=lower, order="ACSO")
        n_pred += len(pred_list)
        n_gold += len(gt)
    
        # Gather all ground truths
        gt_all = []
        for gi in gt:
            for gii in gi:
                a, c, s, o = gii
                new_c = c.replace("style_options", "style&options").lower()
                new_s = s
                if s in sentword2opinion.keys() and sentverb:
                    new_s = sentword2opinion[s.lower()]
                elif s.lower() in senttag2sentword.keys() and (not sentverb):
                    new_s = senttag2sentword[s.lower()]

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


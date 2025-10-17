"""Data loading and preparation utilities"""

import os
import json
from const import e2idx, abb2e, senttag2sentword, senttag2opinion, sentword2opinion


def prepare_demo(fpath, trg_elements, delimeter="####", 
                 sent_verb: bool = False,
                 cspace: bool = True):
    """Prepare demonstration examples for few-shot prompting
    
    Args:
        fpath: Path to demo file (JSON format)
        trg_elements: Target elements to extract (e.g., "ACOS")
        delimeter: Delimiter between quadruples
        sent_verb: Whether to verbalize sentiment
        cspace: Whether to use space in category names
        
    Returns:
        List of demo examples with input and output
    """
    assert fpath.endswith("json"), f"Demo file must be JSON format: {fpath}"
    demos = load_data(fpath)

    new_demos = list()

    for x, ys in demos:
        
        new_y = ""
        for yi in ys:
            a, c, s, o = yi
            if not cspace:
                c = c.replace(" ", "#")
            c = c.replace("style_options", "style&options")
            
            # Handle sentiment verbalization
            if sent_verb:
                s = sentword2opinion.get(s.lower(), senttag2opinion.get(s.lower(), s))
            else:
                s = senttag2sentword.get(s.lower(), s.lower())
            
            e_dict = {"A": a, "C": c, "S": s, "O": o}
            for k in trg_elements:
                v = e_dict[k.upper()]
                if v == None: 
                    continue
                elif (v == "it" and k == "A"):
                    v = "null"
                new_y += f"[{k}] {v} "
            new_y += delimeter + " "
        new_y += "."
        new_y = new_y.replace(f"{delimeter} .", "")

        new_demos.append({
            "input": x.strip(),
            "output": new_y.strip()
        })
        
    return new_demos


def prepare_dataset_wo_args(fpath, target_element, task, 
                            a_null: bool = True, 
                            return_y_list: bool = False, 
                            lowercase: bool = True, 
                            delimeter: str = "####",
                            sent_verb: bool = False,
                            cspace: bool = True):
    """Prepare dataset for evaluation
    
    Args:
        fpath: Path to dataset file
        target_element: Target element order (e.g., "ACOS")
        task: Task type
        a_null: Whether to keep null aspects
        return_y_list: Whether to return list format
        lowercase: Whether to lowercase text
        delimeter: Delimiter between quadruples
        sent_verb: Whether to verbalize sentiment
        cspace: Whether to use space in category names
        
    Returns:
        List of [sentence, target] pairs
    """
    assert os.path.exists(fpath), f"File not found: {fpath}"
    
    # Load dataset
    test_xy = load_data(fpath, lowercase=lowercase)
    new_xy = get_para_targets(test_xy, target_element, task, 
                              return_y_list=return_y_list, a_null=a_null, 
                              delimeter=delimeter,
                              sent_verb=sent_verb,
                              cspace=cspace)

    return new_xy
    

def load_data(path: str, lowercase: bool = True):
    """Load data from file
    
    Args:
        path: Path to data file (.txt or .json)
        lowercase: Whether to lowercase text
        
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
            data.append([words, eval(tuples)])

    elif path.endswith("json"):
        with open(path, 'r') as f:
            data = json.load(f)

    else:
        raise NotImplementedError(f"Unsupported file format: {path}")

    return data


def get_para_targets(raw_xy, target_element, task, 
                     a_null: bool = False, 
                     return_y_list=False, 
                     delimeter="####", 
                     sent_verb=True,
                     cspace=False):
    """Obtain target strings in the specified element order
    
    Args:
        raw_xy: Raw data
        target_element: Target element order
        task: Task type
        a_null: Whether to keep null aspects
        return_y_list: Whether to return list format
        delimeter: Delimiter between quadruples
        sent_verb: Whether to verbalize sentiment
        cspace: Whether to use space in category names
        
    Returns:
        List of [sentence, formatted_target] pairs
    """
    # ASTE: parse aspect and opinion indices
    xy = []
    if task == 'aste':
        for x, y in raw_xy:
            assert len(y[0]) == 3
            parsed_label = []
            sent = x.split()
            for _tuple in y:
                parsed_tuple = parse_aste_tuple(_tuple, sent)
                parsed_label.append(parsed_tuple)
            xy.append([x, parsed_label])
    else:
        xy = raw_xy

    print(xy[:2])
    
    new_xy = []
    e_order = target_element
    for d in xy:
        if len(d) == 2:
            x, y = d
        elif len(d) == 3:
            x, y, _ = d
        else:
            raise NotImplementedError
            
        cur_targets = []
        for yi in y:
            new_y = []
            a, c, s, o = get_task_tuple(yi, task, a_null=a_null, sent_verb=sent_verb)
            if not cspace:
                c = c.replace(" ", "#")
            e_dict = {
                "A": a, 
                "C": c.replace("style_options", "style&options").replace("_", " "),
                "S": s, 
                "O": o
            }
            for k in e_order:
                v = e_dict[k]
                if v == None: 
                    continue
                if return_y_list:
                    new_y.append(v)
                else:
                    new_y.append(f"[{k}] {v}")

            if return_y_list: 
                cur_targets.append(new_y)
            else:
                cur_targets.append(" ".join(new_y))

        if return_y_list:
            new_xy.append([x, cur_targets])
        else:
            new_xy.append([x, f" {delimeter} ".join(cur_targets)])
            
    return new_xy


def parse_aste_tuple(_tuple, sent):
    """Parse ASTE tuple with indices to text spans
    
    Args:
        _tuple: Tuple with aspect/opinion indices and sentiment
        sent: Sentence split into words
        
    Returns:
        Parsed tuple with text spans
    """
    if isinstance(_tuple[0], str):
        res = _tuple
    elif isinstance(_tuple[0], list):
        # Parse aspect
        start_idx = _tuple[0][0]
        end_idx = _tuple[0][-1] if len(_tuple[0]) > 1 else start_idx
        at = ' '.join(sent[start_idx:end_idx + 1])

        # Parse opinion
        start_idx = _tuple[1][0]
        end_idx = _tuple[1][-1] if len(_tuple[1]) > 1 else start_idx
        ot = ' '.join(sent[start_idx:end_idx + 1])
        res = [at, ot, _tuple[2]]
    else:
        print(_tuple)
        raise NotImplementedError
    return res


def get_task_tuple(_tuple, task, a_null: bool = False, sent_verb: bool = False):
    """Get task-specific tuple format
    
    Args:
        _tuple: Input tuple
        task: Task type
        a_null: Whether to keep null aspects
        sent_verb: Whether to verbalize sentiment
        
    Returns:
        Tuple in (aspect, category, sentiment, opinion) format
    """
    if task == "aste":
        at, ot, sp = _tuple
        ac = None
    elif task == "tasd":
        at, ac, sp = _tuple
        ot = None
    elif task in ["asqp", "acos"]:
        at, ac, sp, ot = _tuple
    else:
        raise NotImplementedError(f"Task {task} not supported")

    # Handle sentiment
    if sp and sent_verb:
        sp = sentword2opinion.get(sp.lower(), senttag2opinion.get(sp.lower(), sp))
    elif sp:
        sp = senttag2sentword.get(sp.lower(), sp.lower())
    
    # Handle aspect
    if at and (at.lower() == 'null') and a_null:
        at = 'null'
    elif at and (at.lower() == 'null'):  # For implicit aspect term
        at = 'it'

    return at, ac, sp, ot


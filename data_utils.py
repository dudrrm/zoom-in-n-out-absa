"""
Data loading and processing utilities for ABSA
"""

import os
import json
from config import SENTTAG2SENTWORD, SENTTAG2OPINION, SENTWORD2OPINION


def load_data(path: str, lowercase: bool = True):
    """Load data from txt or json file"""
    data = list()

    if path.endswith("txt"):
        with open(path, 'r', encoding='utf-8') as f:
            d_str = [l.strip() for l in f.readlines() if len(l) > 0]
        for line in d_str:
            if lowercase:
                line = line.lower()
            words, tuples = line.split("####")
            data.append([words, eval(tuples)])

    elif path.endswith("json"):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        raise NotImplementedError(f"File format not supported: {path}")

    return data


def prepare_demo(fpath, trg_elements, delimeter="####", 
                 sent_verb: bool = False, cspace: bool = True):
    """
    Prepare demonstration examples for few-shot learning
    
    Args:
        fpath: Path to demo file
        trg_elements: Target element order (e.g., "ACOS")
        delimeter: Delimiter between quadruples
        sent_verb: Whether to verbalize sentiment (e.g., positive -> great)
        cspace: Whether to use spaces in category names
    """
    assert fpath.endswith("json"), "Demo file must be JSON format"
    demos = load_data(fpath)

    new_demos = list()

    for x, ys in demos:
        new_y = ""
        for yi in ys:
            a, c, s, o = yi
            if not cspace:
                c = c.replace(" ", "#")
            c = c.replace("style_options", "style&options")
            
            if sent_verb:
                s = SENTWORD2OPINION.get(s.lower(), SENTTAG2OPINION.get(s.lower(), s))
            else:
                s = SENTTAG2SENTWORD.get(s.lower(), s.lower())
            
            e_dict = {"A": a, "C": c, "S": s, "O": o}
            for k in trg_elements:
                v = e_dict[k.upper()]
                if v is None:
                    continue
                elif v == "it" and k == "A":
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
    """
    Prepare test dataset
    
    Args:
        fpath: Path to test file
        target_element: Target element order
        task: Task name (acos, asqp, aste, tasd)
        a_null: Whether to use 'null' for implicit aspects
        return_y_list: Return labels as list instead of string
        lowercase: Convert to lowercase
        delimeter: Delimiter between quadruples
        sent_verb: Verbalize sentiment
        cspace: Use spaces in categories
    """
    assert os.path.exists(fpath), f"File not found: {fpath}"
    
    # Load dataset
    test_xy = load_data(fpath, lowercase=lowercase)
    new_xy = get_para_targets(
        test_xy, target_element, task, 
        return_y_list=return_y_list, a_null=a_null, 
        delimeter=delimeter, sent_verb=sent_verb, cspace=cspace
    )

    return new_xy


def get_para_targets(raw_xy, target_element, task, 
                     a_null: bool = False, 
                     return_y_list=False, 
                     delimeter="####", 
                     sent_verb=True,
                     cspace=False):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    # ASTE: parse at & ot
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
                if v is None:
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
    """Parse ASTE tuple format"""
    if isinstance(_tuple[0], str):
        res = _tuple
    elif isinstance(_tuple[0], list):
        # parse at
        start_idx = _tuple[0][0]
        end_idx = _tuple[0][-1] if len(_tuple[0]) > 1 else start_idx
        at = ' '.join(sent[start_idx:end_idx + 1])

        # parse ot
        start_idx = _tuple[1][0]
        end_idx = _tuple[1][-1] if len(_tuple[1]) > 1 else start_idx
        ot = ' '.join(sent[start_idx:end_idx + 1])
        res = [at, ot, _tuple[2]]
    else:
        print(_tuple)
        raise NotImplementedError
    return res


def get_task_tuple(_tuple, task, a_null: bool = False, sent_verb: bool = False):
    """Extract task-specific tuple elements"""
    if task == "aste":
        at, ot, sp = _tuple
        ac = None
    elif task == "tasd":
        at, ac, sp = _tuple
        ot = None
    elif task in ["asqp", "acos"]:
        at, ac, sp, ot = _tuple
    else:
        raise NotImplementedError

    if sp and sent_verb:
        sp = SENTWORD2OPINION.get(sp.lower(), SENTTAG2OPINION.get(sp.lower(), sp))
    elif sp:
        sp = SENTTAG2SENTWORD.get(sp.lower(), sp.lower())
    
    if at and (at.lower() == 'null') and a_null:
        at = 'null'
    elif at and (at.lower() == 'null'):  # for implicit aspect term
        at = 'it'

    return at, ac, sp, ot


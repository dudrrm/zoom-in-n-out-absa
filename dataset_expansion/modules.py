"""Core modules for dataset expansion: generation and judging"""

import os
import json
import copy
from collections import Counter
import logging

# Import LangChain components
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from const import e2idx, abb2e
from prompts import sys_prompts, inout_formats, demos

logger = logging.getLogger()


##########################################
# Round 1: Generate alternative terms
##########################################

def demo_given_cond(demo_quad: list, cond_e: list, trg_e: str, 
                    format_input: str, format_output: str, format_cond: str = "",
                    step=["widen", "narrow", "contraction", "spell"]):
    """Format demo examples based on conditions
    
    Args:
        demo_quad: List of demo examples
        cond_e: Condition elements (e.g., ['S', 'O', 'C'])
        trg_e: Target element ('A' or 'O')
        format_input: Input format string
        format_output: Output format string
        format_cond: Condition format string
        step: Current step
        
    Returns:
        List of formatted demo examples
    """
    demo = []
    trg_name = abb2e[trg_e.upper()]

    for x, quads, new_gt in demo_quad:
        if isinstance(quads, list):
            a, c, s, o = copy.deepcopy(quads)
            quads = {"A": a, "C": c, "S": s, "O": o}

        gt = quads[trg_e]
        conditions = ''.join([format_cond.format(COND=abb2e[c_e], CONDTERM=quads[c_e]) for c_e in cond_e]).strip()
        input_text = format_input.format(X=x, CONDITIONS=conditions, TRG=trg_name, TRGTERM=gt).replace('\"\"', '\"')

        output = f"{new_gt}".replace("'", '"')

        demo.append({"input": input_text, "output": output})

    return demo


def generate(llms, inputs, trg_e, cond_e, demo_data, sys_prompt_format, 
             step, args=None):
    """Generate alternative expressions
    
    Args:
        llms: LLM model wrapper
        inputs: [sentence, quadruple]
        trg_e: Target element ('A' or 'O')
        cond_e: Condition elements
        demo_data: Demo examples
        sys_prompt_format: System prompt template
        step: Current step ('narrow', 'widen', etc.)
        args: Arguments
        
    Returns:
        List of generated expressions
    """
    x, quads = inputs

    if isinstance(quads, tuple):
        quads = list(quads)
    if isinstance(quads, list):
        a, c, s, o = copy.deepcopy(quads)
        quads = {"A": a, "C": c, "S": s, "O": o}
        
    quads["C"] = quads["C"].replace("#", " ")
    gt = quads[trg_e]        
    
    format_input = inout_formats["input"][step]
    format_output = inout_formats["output"][step]
    format_cond = inout_formats["condition"]

    outs = []

    for _ in range(args.sample):

        cond_names = ', '.join([abb2e[c_e] for c_e in cond_e])
        trg_e_name = abb2e[trg_e]
        
        if step in ["widen", "narrow"]:
            system_prompt = sys_prompt_format.format(CONDS=cond_names, TRG=trg_e_name)
        else:
            system_prompt = sys_prompt_format
        
        demo_list = demo_given_cond(demo_data, cond_e=cond_e, trg_e=trg_e,
                                    format_input=format_input, format_output=format_output, 
                                    format_cond=format_cond, step=step)

        example_prompt = ChatPromptTemplate.from_messages(
            [("user", "{input}"), ("assistant", "{output}")]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples = demo_list,
            example_prompt = example_prompt
        )
        
        conditions = ''.join([format_cond.format(COND=abb2e[c_e], CONDTERM=quads[c_e].lower()) for c_e in cond_e]).strip()

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            few_shot_prompt,
            ("user", format_input)
        ])

        output_parser = StrOutputParser()
        chain = prompt | llms.model | output_parser
            
        with get_openai_callback() as cb:
            out = chain.invoke({"X": x, "CONDITIONS": conditions, "TRG": trg_e_name, "TRGTERM": gt})

            try:
                cur_o = eval(out)
            except:
                cur_o = [] 

            # Add original GT if not in output
            if gt not in cur_o:
                cur_o.append(gt)

            outs.extend(cur_o)
            llms.prompt_tokens += cb.prompt_tokens
            llms.completion_tokens += cb.completion_tokens

        if (step in ["contraction", "spell"]) or (args.sample == 1):
            logger.info(f"\nStep: {step}\n<<Input>>: {x}\nOut: {out}\nConditions: {quads}\nGT: \"{gt}\"\nPreds: {outs}\n")
            return outs
            
    out_count = Counter(outs)
    result = [o_term for o_term, count in out_count.items() if count >= args.threshold]
    
    logger.info(f"\nStep: {step}\n<<Input>>: {x}\nOut: {out}\nConditions: {quads}\nGT: \"{gt}\"\nPreds (counter): {out_count}\nResult: {result}\n")

    return result


def generate_alternative(args, llms, trg_e, cond_e, datas, step=["narrow", "widen"]):
    """Generate alternative expressions for all data
    
    Args:
        args: Arguments
        llms: LLM model wrapper
        trg_e: Target element
        cond_e: Condition elements
        datas: List of data
        step: Current step
        
    Returns:
        (result, cost) tuple
    """
    trg_data = datas
        
    logger.info(f"[Current Step: {step}]\n")
    
    # Load demo examples
    if args.task in demos and args.dataset in demos[args.task]:
        demo_dict = demos[args.task][args.dataset].get(step, {}).get(trg_e, [])
    else:
        logger.warning(f"No demo found for task={args.task}, dataset={args.dataset}. Using empty demo.")
        demo_dict = []
    
    # Load prompt template
    sys_prompt_format = sys_prompts[step][trg_e]
    # Set target index
    trg_idx = e2idx[trg_e]
    
    print(f"[info] # of demos = {len(demo_dict)}")
    print(f"[info] Current system prompt format: \n{sys_prompt_format}\n")
    
    if args.now_debug:
        logger.warning("You are debugging now.")

    n_new_add = 0
    result = []

    for idx, data in enumerate(trg_data):
        x, quads = data
        new_quads = []

        for qi in quads:
            new_qi = copy.deepcopy(qi)
            new_qi_wo_tag = [tuple(q[:4]) for q in new_qi]
            org_q = [qi[0][j] for j in range(4)]

            for qj in qi:
                gt = qj[trg_idx]

                if gt.lower() == "null":
                    logger.info(f"Instance of test-idx-{idx} has implicit target term. Pass this example.")
                    continue

                pred = generate(llms, 
                                step=step,
                                inputs=[x, qj[:4]], 
                                trg_e=trg_e, cond_e=cond_e, 
                                demo_data=demo_dict, 
                                sys_prompt_format=sys_prompt_format, 
                                args=args)
                for p in pred:
                    new_q = copy.deepcopy(org_q)
                    new_q[trg_idx] = p

                    if tuple(new_q) not in new_qi_wo_tag:
                        new_qi.append(tuple(new_q + [step]))
                        new_qi_wo_tag.append(tuple(new_q))
                        n_new_add += 1

            new_quads.append(new_qi)
            logger.info(f"[{idx + 1}/{len(trg_data)}] Usage so far: {llms.get_usage()}")

        result.append([x, new_quads])
    
    logger.info(f"Total # of new terms added: {n_new_add}")
    cur_cost = llms.get_usage()["cost"]

    return result, cur_cost


##########################################
# Round 2: LLM judge
##########################################

def demo_format(demo_quad: list, format_input: str, trg_name: str):
    """Format demo examples for judging
    
    Args:
        demo_quad: List of demo examples
        format_input: Input format string
        trg_name: Target element name
        
    Returns:
        List of formatted demo examples
    """
    demo = []

    for x, quads, new, cot in demo_quad:
        a, c, s, o = copy.deepcopy(quads)
        gt = f"[A] {a} [C] {c} [S] {s} [O] {o}"
        input_text = format_input.format(X=x, GT=gt, NEW=new, TRGNAME=trg_name).replace('\"\"', '\"')
        demo.append({"input": input_text, "output": cot})

    return demo


def judge(llms, trg_e: str, inputs: list, demo_data: list, system_prompt: str):
    """Judge whether an alternative expression is valid
    
    Args:
        llms: LLM model wrapper
        trg_e: Target element
        inputs: [sentence, quadruple, new_expression]
        demo_data: Demo examples
        system_prompt: System prompt
        
    Returns:
        Judgment output (string)
    """
    format_input = inout_formats["input"]["judge"]

    trg_name = abb2e[trg_e.upper()]
    x, quads, new = inputs
    a, c, s, o = copy.deepcopy(quads)
    quads = {"A": a, "C": c, "S": s, "O": o}

    gt = f"[A] {a} [C] {c} [S] {s} [O] {o}"
    demo_list = demo_format(demo_data, format_input, trg_name)

    example_prompt = ChatPromptTemplate.from_messages(
        [("user", "{input}"), 
        ("assistant", "{output}")]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples = demo_list,
        example_prompt = example_prompt
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        ("user", format_input)
    ])

    output_parser = StrOutputParser()
    chain = prompt | llms.model | output_parser

    with get_openai_callback() as cb:
        out = chain.invoke({"X": x, "GT": gt, "NEW": new, "TRGNAME": trg_name})

        llms.prompt_tokens += cb.prompt_tokens
        llms.completion_tokens += cb.completion_tokens
        input_str = format_input.format(X=x, GT=gt, NEW=new, TRGNAME=trg_name).replace('\"\"', '\"')

    logger.info(f"\n<<Input>>:\n {input_str}\n<<Model output>>:\n{out}\n")

    return out


def generate_judge(args, llms, datas, trg_e):
    """Judge all generated alternatives
    
    Args:
        args: Arguments
        llms: LLM model wrapper
        datas: List of data with alternatives
        trg_e: Target element
        
    Returns:
        (result, cost) tuple
    """
    # Load demo and prompt
    if args.task in demos and args.dataset in demos[args.task]:
        demo_dict = demos[args.task][args.dataset].get("judge", {}).get(trg_e, [])
    else:
        logger.warning(f"No demo found for judging. Using empty demo.")
        demo_dict = []
        
    dtype = "rest" if "rest" in args.dataset else "laptop"
    sys_prompt_format = sys_prompts["judge"][f"{dtype}-{trg_e}"]

    trg_idx = e2idx[trg_e]

    n_judge = 0
    n_remain = 0
    n_filtered = 0

    result = []
    invalid = []

    for i, data in enumerate(datas):
        x, quads = data
        new_quads = []
        for qi in quads:
            
            if len(qi) == 1: 
                new_quads.append(qi)
                continue

            original_gt = qi[0]
            new_qi = [original_gt]
            
            for j in range(1, len(qi)):
                qj = qi[j]
                new = qj[trg_idx]

                if new.lower() == "null":
                    logger.info(f"Instance of test-idx-{i} has implicit target term. Pass this example.")
                    continue

                pred = judge(llms,
                            trg_e=trg_e,
                            inputs=[x, original_gt[:4], new],
                            demo_data=demo_dict,
                            system_prompt=sys_prompt_format)
                
                n_judge += 1
                if "invalid" in pred.lower():
                    n_filtered += 1
                    invalid.append([x, original_gt, qj, pred])
                else:
                    n_remain += 1
                    new_qi.append(qj)
                    
            new_quads.append(new_qi)
            logger.info(f"[{i+1}/{len(datas)}] Usage so far: {llms.get_usage()}")
        
        result.append([x, new_quads])

    logger.info(f"Total # of filtered terms: {n_filtered}")

    cur_cost = llms.get_usage()["cost"]

    return result, cur_cost
    

###############################################
# Step 6: Merge aspect and opinion predictions
###############################################

def remove_tag(new_quad):
    """Remove tags from quadruples"""
    result = []
    for qi in new_quad:
        new_qi = []
        for qj in qi:
            wotag = tuple(list(qj)[:4])
            new_qi.append(wotag)  
        result.append(new_qi)
    return result


def merge_AO_wo_overlap(results: dict) -> tuple:
    """Merge aspect and opinion with tag, avoiding overlaps
    
    Args:
        results: Dictionary with 'A' and 'O' results
        
    Returns:
        (merged_result, merged_result_wo_tag) tuple
    """
    merged_result = []
    merged_result_wo_tag = []

    n_filtered = 0
    n_quad = 0
    for i, (a, o) in enumerate(zip(results["A"], results["O"])):
        x, a_quads = a
        _, o_quads = o
        new_quads = []
        
        for a_qi, o_qi in zip(a_quads, o_quads):
            aspects = [(a_qj[0], a_qj[-1]) for a_qj in a_qi]
            opinions = [(o_qj[3], o_qj[-1]) for o_qj in o_qi]
            cate = a_qi[0][1]
            sent = a_qi[0][2]

            new_qi = []
            for a, atag in aspects:
                for o, otag in opinions:
                    
                    dependent = False
                    tag = (atag, otag)
                    if (a in o): dependent = True
                    elif (o in a): dependent = True

                    if dependent:
                        if tag == ("original", "original"): pass
                        else:
                            n_filtered += 1
                            continue
                    
                    new_qi.append((a, cate, sent, o, tag))
                    n_quad += 1
                    
            new_quads.append(new_qi)

        merged_result.append([x, new_quads])
        merged_result_wo_tag.append([x, remove_tag(new_quads)])

    logger.info(f"[Statistics] # original quads: {n_quad+n_filtered} | # filtered quads: {n_filtered} | # fin quads: {n_quad}")

    return merged_result, merged_result_wo_tag


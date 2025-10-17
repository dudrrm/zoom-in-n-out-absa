"""Prediction methods for ABSA tasks"""

import re
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompts import e2form, system_prompt
from const import cate_list
from models import LLMs


async def apred(llms: LLMs, x, demo, args):
    """Asynchronous prediction for ABSA task
    
    Args:
        llms: LLM model wrapper
        x: Input sentence
        demo: Demo examples
        args: Arguments
        
    Returns:
        Prediction string
    """
    # 1) System prompt
    cates = cate_list.get(args.dataset, [])
    vc = " " if args.cspace else "#"
    cur_cate = [c.replace(" ", vc) for c in cates]

    cur_form = ""
    for e in args.target_element:
        cur_form += e2form[e] + " "

    if args.sent_verb:
        sents = ['great', 'ok', 'bad']
    else:
        sents = ['positive', 'neutral', 'negative']
    
    # Select system prompt based on task and dataset
    prompt_key = f"detail-{args.task}-{args.dataset}"
    if prompt_key not in system_prompt:
        prompt_key = "naive-item"
    
    sys_prompt = system_prompt[prompt_key].format(
        FORMAT=cur_form.strip(), 
        CATE=cur_cate, 
        C=vc, 
        SENTCATE=sents
    )

    # 2) Demo
    example_prompt = ChatPromptTemplate.from_messages(
        [("user", "{input}"),
         ("assistant", "{output}")]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples = demo,
        example_prompt = example_prompt
    )

    # 3) Finalize prompt
    if args.constrained_decoding:
        # Create regex for constrained decoding
        regex = _get_regex(x, target_element=args.target_element, 
                          categories=cur_cate, delimeter=args.delimeter)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            few_shot_prompt,
            ("user", "{input}")
        ])
        
        output_parser = StrOutputParser()
        chain = prompt | llms.model.bind(extra_body={"guided_regex": regex}) | output_parser
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            few_shot_prompt,
            ("user", "{input}")
        ])
        
        output_parser = StrOutputParser()
        chain = prompt | llms.model | output_parser

    # Run
    with get_openai_callback() as cb:
        
        try:
            out = await chain.ainvoke({"input": x})
        except Exception as e:
            print(f"Error during prediction: {e}")
            out = ""
        
        llms.prompt_tokens += cb.prompt_tokens
        llms.completion_tokens += cb.completion_tokens
        
        # Update cost based on model
        if "gpt" in llms.model_name:
            # Cost calculation for OpenAI models (approximate)
            if "gpt-4" in llms.model_name:
                prompt_cost = cb.prompt_tokens * 0.03 / 1000
                completion_cost = cb.completion_tokens * 0.06 / 1000
            else:  # gpt-3.5-turbo
                prompt_cost = cb.prompt_tokens * 0.0015 / 1000
                completion_cost = cb.completion_tokens * 0.002 / 1000
            llms.total_cost += (prompt_cost + completion_cost)

        if args.debug:
            import pdb
            pdb.set_trace()

    return out


def _get_regex(input_text: str, target_element: str, categories: list, delimeter: str):
    """Generate regex pattern for constrained decoding
    
    Args:
        input_text: Input sentence
        target_element: Target element order (e.g., "ACOS")
        categories: List of category names
        delimeter: Delimiter between quadruples
        
    Returns:
        Regex pattern string
    """
    tokens = "| ".join(set(re.escape(token) for token in input_text.split()))
    categories_pattern = "|".join(category.strip() for category in categories)
    regex = ""
    
    for e in target_element:
        if e == "A":
            regex += rf" \[A\](?:(?: {tokens})+| null)"
        elif e == "O":
            regex += rf" \[O\](?:(?: {tokens})+| null)"
        elif e == "S":
            regex += r" \[S\] (?:positive|neutral|negative)"
        elif e == "C":
            regex += rf" \[C\] (?:{categories_pattern})"
    
    delimeter_pattern = r" " + re.escape(delimeter.strip())
    regex = rf"{regex}(?:{delimeter_pattern}{regex})*"
    
    return " ?" + regex[1:]


"""
Prediction methods for ABSA using LangChain
"""

import re
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompt import E2FORM, SYSTEM_PROMPT
from config import CATE_LIST
from models import LLMs


def _get_regex(input_text: str, target_element: str, categories: list[str], delimeter: str):
    """
    Generate regex pattern for constrained decoding
    
    Args:
        input_text: Input sentence
        target_element: Target element order (e.g., "ACOS")
        categories: List of valid categories
        delimeter: Delimiter between quadruples
    
    Returns:
        Regex pattern string
    """
    tokens = "| ".join(set(re.escape(token) for token in input_text.split()))
    categories = "|".join(category.strip() for category in categories)
    regex = ""
    for e in target_element:
        match e:
            case "A":
                regex += rf" \[A\](?:(?: {tokens})+| null)"
            case "O":
                regex += rf" \[O\](?:(?: {tokens})+| null)"
            case "S":
                regex += r" \[S\] (?:positive|neutral|negative)"
            case "C":
                regex += rf" \[C\] (?:{categories})"
    delimeter = r" " + delimeter.strip()
    regex = rf"{regex}(?:{delimeter}{regex})*"
    return " ?" + regex[1:]


async def apred(llms: LLMs, x, demo, args):
    """
    Async prediction method (main method for evaluation)
    
    Args:
        llms: LLMs object
        x: Input text
        demo: Demonstration examples
        args: Arguments
    
    Returns:
        Predicted output string
    """
    # 1) System prompt
    cates = CATE_LIST[args.dataset]
    vc = " " if args.cspace else "#"
    cur_cate = [c.replace(" ", vc) for c in cates]

    cur_form = ""
    for e in args.target_element:
        cur_form += E2FORM[e] + " "

    if args.sent_verb:
        sents = ['great', 'ok', 'bad']
    else:
        sents = ['positive', 'neutral', 'negative']
    
    sys_prompt = SYSTEM_PROMPT[f"detail-{args.task}-{args.dataset}"].format(
        FORMAT=cur_form.strip(), CATE=cur_cate, C=vc, SENTCATE=sents
    )

    # 2) Demo
    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{input}"),
        ("assistant", "{output}")
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=demo,
        example_prompt=example_prompt
    )

    # 3) Finalize prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        few_shot_prompt,
        ("user", "{input}")
    ])

    if args.constrained_decoding:
        # 4) Create Regex for constrained decoding
        regex = _get_regex(x, target_element=args.target_element, 
                          categories=cur_cate, delimeter=args.delimeter)
        output_parser = StrOutputParser()
        chain = prompt | llms.model.bind(extra_body={"guided_regex": regex}) | output_parser
    else:
        output_parser = StrOutputParser()
        chain = prompt | llms.model | output_parser

    # Run prediction
    with get_openai_callback() as cb:
        try:
            out = await chain.ainvoke({"input": x})
        except Exception as e:
            print(f"Error during prediction: {e}")
            out = ""
        
        llms.prompt_tokens += cb.prompt_tokens
        llms.completion_tokens += cb.completion_tokens

    return out


def pred(llms: LLMs, x, demo, args):
    """
    Synchronous prediction method
    
    Args:
        llms: LLMs object
        x: Input text
        demo: Demonstration examples
        args: Arguments
    
    Returns:
        Predicted output string
    """
    # 1) System prompt
    cates = CATE_LIST[args.dataset]
    vc = " " if args.cspace else "#"
    cur_cate = [c.replace(" ", vc) for c in cates]

    cur_form = ""
    for e in args.target_element:
        cur_form += E2FORM[e] + " "
    
    if args.sent_verb:
        sents = ['great', 'ok', 'bad']
    else:
        sents = ['positive', 'neutral', 'negative']

    sys_prompt = SYSTEM_PROMPT[f"detail-{args.task}-{args.dataset}"].format(
        FORMAT=cur_form.strip(), CATE=cur_cate, C=vc, SENTCATE=sents
    )

    # 2) Demo
    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{input}"), 
        ("assistant", "{output}")
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=demo,
        example_prompt=example_prompt
    )

    # 3) Finalize prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        few_shot_prompt,
        ("user", "{input}")
    ])

    # Chain
    output_parser = StrOutputParser()
    chain = prompt | llms.model | output_parser
    
    # Run
    with get_openai_callback() as cb:
        try:
            out = chain.invoke({"input": x})
        except Exception as e:
            print(f"Error during prediction: {e}")
            out = ""

        llms.prompt_tokens += cb.prompt_tokens
        llms.completion_tokens += cb.completion_tokens

    return out


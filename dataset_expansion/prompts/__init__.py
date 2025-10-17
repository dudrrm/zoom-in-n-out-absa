"""Import all prompts and formats"""

from .formats import gen_cond, gen_input, gen_out, spell_out, judge_intput
from .prompt import prompt_narrow_span, prompt_widen_span, prompt_judge

# Demo examples should be loaded from dataset-specific files
# You can add them as needed (e.g., acos_rest.py, asqp_rest16.py, etc.)
# For now, we provide a placeholder structure

# Example structure for demos:
# demos = {
#     "acos": {
#         "rest16": {...},
#         "laptop16-supcate": {...}
#     },
#     "asqp": {
#         "rest16": {...},
#         "rest15": {...}
#     },
# }

# Placeholder: Users should add their own demo examples
demos = {
    "acos": {},
    "asqp": {},
    "aste": {},
    "tasd": {},
}

sys_prompts = {
    "widen": prompt_widen_span,
    "narrow": prompt_narrow_span,
    "judge": prompt_judge,
}

inout_formats = {
    "input": {
        "widen": gen_input,
        "narrow": gen_input,
        "contraction": gen_input,
        "spell": gen_input,
        "judge": judge_intput,
    },
    "output": {
        "widen": gen_out,
        "narrow": gen_out,
        "contraction": spell_out,
        "spell": spell_out,
    },
    "condition": gen_cond,
}

__all__ = [
    'demos',
    'sys_prompts',
    'inout_formats',
    'gen_cond',
    'gen_input',
    'gen_out',
    'spell_out',
    'judge_intput',
    'prompt_narrow_span',
    'prompt_widen_span',
    'prompt_judge',
]


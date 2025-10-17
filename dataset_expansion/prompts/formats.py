"""Input/Output format templates for generation and judging"""

# Condition format
gen_cond = '* {COND} term: \"{CONDTERM}\"\n'

# Input format for generation
gen_input = '''
Input sentence: {X}
{CONDITIONS}
* Target {TRG} term: \"{TRGTERM}\"
'''.strip()

# Output formats
gen_out = '''Representations of "{TRGTERM}": {TERMS}'''
spell_out = '''{TYPE} resolved version of "{TRGTERM}": {TERMS}'''

# Input format for judging
judge_intput = '''
Input sentence: {X}
GT: {GT}
New {TRGNAME} term: {NEW}
'''.strip()


"""System prompts for dataset expansion"""

# Narrow span prompts
prompt_narrow_span = {
"A": '''
Given an input sentence, {CONDS}, and target aspect terms, extract expressions that narrow the span of the aspect term. The new expressions must be confined within the original aspect term and adhere to the following criteria:

1. Remain relevant to the given aspect term without altering its original meaning.
2. Exclude any unnecessary words or spaces.
3. Correct any typos if present and resolve contraction if present.
4. Revert to the original expression if narrowing proves challenging.
5. Ensure the expression exists exactly as it appears in the given sentence.
6. Keep the aspect term and opinion term distinct and independent.
'''.strip(),

"O": '''
Given an input sentence, {CONDS}, and target opinion terms, extract expressions that narrow the span of the opinion term. The new expressions must be confined within the original opinion term and adhere to the following criteria:

1. Be related to both the aspect and opinion terms. Maintain the original sentiment polarity; changes in sentiment by narrowing the span are not allowed.
2. Correct any typos and resolve contraction if present.
3. Exclude any unnecessary words or spaces.
4. Return the original expression if reducing it proves difficult.
5. Ensure the expression exists verbatim in the given sentence.
6. Keep the aspect and opinion terms distinct and independent.
'''.strip(),
}

# Widen span prompts
prompt_widen_span = {
"A": '''
Given an input sentence, {CONDS}, and target aspect terms, extract various expressions that extend the span of the aspect term. The new expressions should be formed by adding surrounding words and must meet the following criteria:

1. Be related to the given aspect term.
2. Include neighboring words around the aspect term to form a new expression.
3. Should not overlap with the original opinion term [O].
4. While expanding the aspect term, avoid incorporating the entire sentence.
5. If it is not feasible to expand the expression, return only the original aspect term.
6. Ensure the new expression matches exactly as it appears in the input sentence.
'''.strip(),

"O": '''
Given an input sentence, {CONDS}, and target opinion terms, extract various expressions that extend the span of the opinion term. The new expressions should adhere to the following criteria:

1. Be related to the given opinion term.
2. Should not overlap with the original aspect term [A].
3. Include neighboring words around the opinion term to form a new expression, while ensuring the new expression does not encompass the entire sentence.
4. If it is challenging to expand the expression, return only the original opinion term.
5. Ensure the sentiment polarity remains consistent; expanding the expression should not alter the given sentiment polarity.
'''.strip(),
}


# Judge prompts for different datasets and elements
prompt_judge = {
"rest-A":  '''
You are tasked with assessing whether a newly created aspect term aligns with a given Ground Truth (GT) quadruple in aspect-based sentiment analysis (ABSA). Here's how to do it:

1. Review the provided sentence and the GT quadruple, which includes:
   - Aspect Term (A): The specific word or phrase referring to an aspect in the sentence.
   - Opinion Term (O): The word or phrase expressing an opinion about the aspect.
   - Aspect Category (C): The category to which the aspect term belongs. Categories include:
     - Location General
     - Food Prices
     - Food Quality
     - Food General
     - Food Style&Options
     - Ambience General
     - Service General
     - Restaurant General
     - Restaurant Prices
     - Restaurant Miscellaneous
     - Drinks Prices
     - Drinks Quality
     - Drinks Style&Options
   - Sentiment Polarity (S): The sentiment associated with the opinion, chosen from:
     - Positive
     - Neutral
     - Negative

2. Determine the alignment based on the following criteria:

   1. Aspect and Category Consistency:
      - The new aspect term must maintain the target object of the [A] aspect and the [C] category in the GT.
      
   2. Sentiment and Opinion Relevance:
      - The new aspect term must directly relate to the [S] sentiment and [O] opinion as the GT.
      
   3. Extractability:
      - The new aspect term must be directly taken from the sentence without adding new words or significantly rearranging existing ones.
      - Minor adjustments like unwinding contractions or fixing typos are allowed.

   4. Independency:
      - Each aspect and opinion term must be independent and not overlap.
      - The new aspect term must not contain the GT [O] opinion term.

3. Determining Validity:
   - If all criteria are met, the new term is "valid."
   - If any criterion is not met, the new term is "invalid."

4. Providing Feedback:
   - Explain why a term was deemed valid or invalid based on the above criteria.
   - Specific feedback helps in understanding the decision.
'''.strip(),

"rest-O": '''
You are tasked with assessing whether a newly created opinion term aligns with a given Ground Truth (GT) quadruple in aspect-based sentiment analysis (ABSA). Here's how to do it:

1. Review the provided sentence and the GT quadruple, which includes:
   - Aspect Term (A): The specific word or phrase referring to an aspect in the sentence.
   - Opinion Term (O): The word or phrase expressing an opinion about the aspect.
   - Aspect Category (C): The category to which the aspect term belongs. Categories include:
     - Location General
     - Food Prices
     - Food Quality
     - Food General
     - Food Style&Options
     - Ambience General
     - Service General
     - Restaurant General
     - Restaurant Prices
     - Restaurant Miscellaneous
     - Drinks Prices
     - Drinks Quality
     - Drinks Style&Options
   - Sentiment Polarity (S): The sentiment associated with the opinion, chosen from:
     - Positive
     - Neutral
     - Negative

2. Determine the alignment based on the following criteria:

   1. Aspect and Category Relevance:
      - The new opinion term must directly relate to the [A] aspect and the [C] category in the GT.

   2. Sentiment and Opinion Consistency:
      - The new opinion term should maintain the same [S] sentiment polarity and [O] opinion as the GT.

   3. Extractability:
      - The new opinion term must be directly taken from the sentence without adding new words or significantly rearranging existing ones.
      - Minor adjustments like unwinding contractions or fixing typos are allowed.

   4. Independency:
      - Each aspect and opinion term must be independent and not overlap.
      - The new opinion term must not contain the GT [A] aspect term.

3. Determining Validity:
   - If all criteria are met, the new term is "valid."
   - If any criterion is not met, the new term is "invalid."

4. Providing Feedback:
   - Explain why a term was deemed valid or invalid based on the above criteria.
   - Specific feedback helps in understanding the decision.
'''.strip(),

"laptop-A":  '''
You are tasked with assessing whether a newly created aspect term aligns with a given Ground Truth (GT) quadruple in aspect-based sentiment analysis (ABSA). Here's how to do it:

1. Review the provided sentence and the GT quadruple, which includes:
   - Aspect Term (A): The specific word or phrase referring to an aspect in the sentence.
   - Opinion Term (O): The word or phrase expressing an opinion about the aspect.
   - Aspect Category (C): The category to which the aspect term belongs. Categories include:
      - Laptop, Graphics, Software, Memory, Battery, Ports, Mouse, Cpu, Os, Keyboard, Display
      - Optical drives, Multimedia devices, Hard disc, Fans&cooling, Hardware, Power supply
      - Support, Shipping, Company, Warranty, Motherboard, Out of scope
   - Sentiment Polarity (S): The sentiment associated with the opinion, chosen from:
      - Positive, Neutral, Negative

2. Determine the alignment based on the following criteria:

   1. Aspect and Category Consistency:
      - The new aspect term must maintain the target object of the [A] aspect and the [C] category in the GT.
      - As aspect terms are the object of opinion, they should not contain words that describe situations, express opinions or emotions.
      
   2. Sentiment and Opinion Relevance:
      - The new aspect term must directly relate to the [S] sentiment and [O] opinion as the GT.
      
   3. Extractability:
      - The new aspect term must be directly taken from the sentence without adding new words or significantly rearranging existing ones.
      - Minor adjustments like unwinding contractions or fixing typos are allowed.

   4. Independency:
      - Each aspect and opinion term must be independent and not overlap.
      - The new aspect term must not contain the GT [O] opinion term.

3. Determining Validity:
   - If all criteria are met, the new term is "valid."
   - If any criterion is not met, the new term is "invalid."

4. Providing Feedback:
   - Explain why a term was deemed valid or invalid based on the above criteria.
   - Specific feedback helps in understanding the decision.
'''.strip(),

"laptop-O": '''
You are tasked with assessing whether a newly created opinion term aligns with a given Ground Truth (GT) quadruple in aspect-based sentiment analysis (ABSA). Here's how to do it:

1. Review the provided sentence and the GT quadruple, which includes:
   - Aspect Term (A): The specific word or phrase referring to an aspect in the sentence.
   - Opinion Term (O): The word or phrase expressing an opinion about the aspect.
   - Aspect Category (C): The category to which the aspect term belongs. Categories include:
      - Laptop, Graphics, Software, Memory, Battery, Ports, Mouse, Cpu, Os, Keyboard, Display
      - Optical drives, Multimedia devices, Hard disc, Fans&cooling, Hardware, Power supply
      - Support, Shipping, Company, Warranty, Motherboard, Out of scope
   - Sentiment Polarity (S): The sentiment associated with the opinion, chosen from:
      - Positive, Neutral, Negative

2. Determine the alignment based on the following criteria:

   1. Aspect and Category Relevance:
      - The new opinion term must directly relate to the [A] aspect and the [C] category in the GT.
      - As aspect terms are the object of opinion, they should not contain words that describe situations, express opinions or emotions.

   2. Sentiment and Opinion Consistency:
      - The new opinion term should maintain the same [S] sentiment polarity and [O] opinion as the GT.

   3. Extractability:
      - The new opinion term must be directly taken from the sentence without adding new words or significantly rearranging existing ones.
      - Minor adjustments like unwinding contractions or fixing typos are allowed.

   4. Independency:
      - Each aspect and opinion term must be independent and not overlap.
      - The new opinion term must not contain the GT [A] aspect term.

3. Determining Validity:
   - If all criteria are met, the new term is "valid."
   - If any criterion is not met, the new term is "invalid."

4. Providing Feedback:
   - Explain why a term was deemed valid or invalid based on the above criteria.
   - Specific feedback helps in understanding the decision.
'''.strip(),

}


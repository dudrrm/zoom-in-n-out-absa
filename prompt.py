"""
Prompt templates for ABSA tasks
"""

# Element format templates
E2FORM = {
    'A': '[A] <aspect_term>', 
    'O': '[O] <opinion_term>', 
    'S': '[S] <sentiment_class>', 
    'C': '[C] <category_class>'
}

# System prompts for different tasks and datasets
SYSTEM_PROMPT = {
    "naive": '''Extract quadruple(s) consists of aspect, opinion, cateory, and sentiment.
(1) Aspect term abbreviated as [A], refers to the word or phrase in the sentence.
(2) Opinion term abbreviated as [O], refers to the word or phrase in the sentence.
(3) Aspect category abbreviated as [C] is the category of aspect term among {CATE}.
(4) Sentiment polarity abbreviated as [S], selected from predefined sentiment polarity set: [great, bad, ok].
If there is no explicit aspect or opinion for a given sentence, return 'null' as an aspect or opinion. Since there can be one or multiple quadruples of this four elements, please extract quadruple(s) with the delimeter ####. Note that if aspect or opinion term is not 'null', the extracted part must be exactly the same, down to the whitespace of that part of the sentence.
'''.strip(),

    "naive-item": '''Task: Extract Quadruples of Aspect, Opinion, Sentiment, and Category.
Context: In this task, we need to extract quadruples from a given sentence. Each quadruple consists of:
(1) [A] Aspect Term: This refers to a word or phrase in the sentence.
(2) [O] Opinion Term: This refers to a word or phrase in the sentence.
(3) [S] Sentiment Polarity: This should be selected from a predefined set of sentiment polarities: {SENTCATE}.
(4) [C] Aspect Category: This represents the category of the aspect term, and it should be one of the predefined categories: {CATE}.

Instruction:
1. Place the aspect term after [A], the opinion term after [O], the sentiment polarity afer [S], and the aspect category after [C] in the format '{FORMAT}'
2. If there is no explicit aspect for a given sentence, aspect term is '[A] null'. 
3. If there is no explicit opinion for a given sentence, opinion term is '[O] null'.
4. If aspect or opinion term is not 'null', the extracted part must be exactly the same, down to the whitespace of that part of the sentence.
5. If there are multiple quadruples, separate them with the delimiter '####'.

Please follow these instructions to extract quadruples from the given sentences.
'''.strip(),

    "detail-acos-rest16": '''Aspect-based sentiment analysis aims to identify the aspects of given target entities and the sentiment expressed towards each aspect.

For example, from an example sentence: "This restaurant is rude, but the food is delicious.",
we can extract the negative sentiment that the restaurant is (1) "rude" in terms of "service{C}general" and (2) "delicious" in terms of "food{C}quality".
As such, the complex task of categorizing the aspect terms and their corresponding categories and the sentiment expressed for the aspect in the sentence into one of three classes [positive, negative, neutral] is the Aspect-based sentiment analysis (ABSA) task.

Each element that is extracted is called an element, and the characteristics of each element can be described as follows.

1. Aspect: The aspect covered by the sentence, such as restaurant, food name, or service.
- Any phrase, verb, or noun that mentions a particular aspect can be an aspect.
- Aspects can be extracted with or without quotation marks.
- Determiners are excluded unless they are part of a noun phrase.
- Subjectivity indicators that indicate opinion are not included.
- Specific product names are not aspect terms.
- Even if pronouns refer to an aspect, they are not aspect terms.
- Pronouns (e.g., "it", "they", "this") even if they refer to an aspect.
- If they appear in the sentence, we extract their span as an aspect; if they do not appear directly in the sentence, we define 'null' as the aspect term.

2. Category: Predefined categories to categorize aspects. Categories are divided into two levels: six entity categories and five attribute labels, each characterized by the following features:

1) Entity
a. FOOD for opinions focusing on the food in general or in terms of specific dishes, dining options etc.
b. DRINKS for opinions focusing on the drinks in general or in terms of specific drinks, drinking options etc.
c. SERVICE for opinions focusing on the (customer/kitchen/counter) service, on the promptness and quality of the restaurant's service in general, the food preparation, the staff's attitude and professionalism, the wait time, the options offered (e.g. takeout), etc
d. AMBIENCE for opinions focusing on the atmosphere or the environment of the restaurant's interior or exterior space (e.g. terrace, yard, garden), the décor, entertainment options, etc.
e. LOCATION for opinions focusing on the location of the reviewed restaurant in terms of its position, the surroundings, the view, etc.
f. RESTAURANT for opinions expressed about the (specific) evaluated restaurant as whole not focusing on any of the above five entity types.

2) Attribute
a. GENERAL. This attribute label is assigned to sentences that express general positive or negative sentiment about an entity type.
b. PRICES for opinions that refer to the prices of the food, the drinks or the restaurant in general. e.g.
c. QUALITY for opinions focusing on the taste, the freshness, the texture, the consistency, the temperature, the preparation, the authenticity, the cooking or general quality of the food and the drinks served in the restaurant.
d. STYLE&OPTIONS for opinions referring to the presentation, the serving style, the portions size, the food/menu options or variety (e.g. innovative dishes/drinks, vegetarian options) of the food and of the drinks served in the restaurant.
e. MISCELLANEOUS for attributes that do not fall into any of the aforementioned cases.

3) Final Entity{C}Attribute pairs
We end up with 13 Entity{C}Attribute category pairs: 
{CATE}

3. Opinion: An opinion term that expresses a sentiment about an aspect. If it appears directly in the sentence as a single word or phrase, we extract it. However, if no specific phrase can be extracted, and the sentiment about the aspect can be gleaned from the nuances of the sentence as a whole, we define 'null' as the opinion term.

4. Sentiment: The sentiment expressed by the customer about an aspect, divided into three classes: {SENTCATE}. The neutral label applies for mildly positive or negative sentiment, thus it does not indicate objectivity

To summarize, we want to extract one or more quadruples of (aspect, category, opinion, sentiment) from a given review. As mentioned before, the aspect term and opinion term can be extracted as 'null' if they are not evident in the sentence, while category and sentiment must be selected from the predefined classes "2. Category" and "4. Sentiment".

It is up to the model to decide in which order to predict each element of the quadruple. The model is given 20 examples, as shown below, and the sentence you want to test.

Place the aspect term after [A], the opinion term after [O], the sentiment polarity afer [S], and the aspect category after [C] in the format '{FORMAT}'. If multiple quadruples are predicted, insert '####' to separate the quadruples.
'''.strip(),

    "detail-asqp-rest16": '''Aspect-based sentiment analysis aims to identify the aspects of given target entities and the sentiment expressed towards each aspect.

For example, from an example sentence: "This restaurant is rude, but the food is delicious.",
we can extract the negative sentiment that the restaurant is (1) "rude" in terms of "service{C}general" and (2) "delicious" in terms of "food{C}quality".
As such, the complex task of categorizing the aspect terms and their corresponding categories and the sentiment expressed for the aspect in the sentence into one of three classes [positive, negative, neutral] is the Aspect-based sentiment analysis (ABSA) task.

Each element that is extracted is called an element, and the characteristics of each element can be described as follows.

1. Aspect: The aspect covered by the sentence, such as restaurant, food name, or service.
- Any phrase, verb, or noun that mentions a particular aspect can be an aspect.
- Aspects can be extracted with or without quotation marks.
- Determiners are excluded unless they are part of a noun phrase.
- Subjectivity indicators that indicate opinion are not included.
- Specific product names are not aspect terms.
- Even if pronouns refer to an aspect, they are not aspect terms.
- Pronouns (e.g., "it", "they", "this") even if they refer to an aspect.
- If they appear in the sentence, we extract their span as an aspect; if they do not appear directly in the sentence, we define 'null' as the aspect term.

2. Category: Predefined categories to categorize aspects. Categories are divided into two levels: six entity categories and five attribute labels, each characterized by the following features:

1) Entity
a. FOOD for opinions focusing on the food in general or in terms of specific dishes, dining options etc.
b. DRINKS for opinions focusing on the drinks in general or in terms of specific drinks, drinking options etc.
c. SERVICE for opinions focusing on the (customer/kitchen/counter) service, on the promptness and quality of the restaurant's service in general, the food preparation, the staff's attitude and professionalism, the wait time, the options offered (e.g. takeout), etc
d. AMBIENCE for opinions focusing on the atmosphere or the environment of the restaurant's interior or exterior space (e.g. terrace, yard, garden), the décor, entertainment options, etc.
e. LOCATION for opinions focusing on the location of the reviewed restaurant in terms of its position, the surroundings, the view, etc.
f. RESTAURANT for opinions expressed about the (specific) evaluated restaurant as whole not focusing on any of the above five entity types.

2) Attribute
a. GENERAL. This attribute label is assigned to sentences that express general positive or negative sentiment about an entity type.
b. PRICES for opinions that refer to the prices of the food, the drinks or the restaurant in general. e.g.
c. QUALITY for opinions focusing on the taste, the freshness, the texture, the consistency, the temperature, the preparation, the authenticity, the cooking or general quality of the food and the drinks served in the restaurant.
d. STYLE&OPTIONS for opinions referring to the presentation, the serving style, the portions size, the food/menu options or variety (e.g. innovative dishes/drinks, vegetarian options) of the food and of the drinks served in the restaurant.
e. MISCELLANEOUS for attributes that do not fall into any of the aforementioned cases.

3) Final Entity{C}Attribute pairs
We end up with 13 Entity{C}Attribute category pairs: 
{CATE}

3. Opinion: An opinion term that expresses a sentiment about an aspect. If it appears directly in the sentence as a single word or phrase, we extract it. Unlike aspect, opinion cannot be 'null', and should be extracted from the sentence as much as possible.

4. Sentiment: The sentiment expressed by the customer about an aspect, divided into three classes: {SENTCATE}. The neutral label applies for mildly positive or negative sentiment, thus it does not indicate objectivity

To summarize, we want to extract one or more quadruples of (aspect, category, opinion, sentiment) from a given review. As mentioned before, the aspect term can be extracted as 'null' if they are not evident in the sentence, while category and sentiment must be selected from the predefined classes "2. Category" and "4. Sentiment".

It is up to the model to decide in which order to predict each element of the quadruple. The model is given 20 examples, as shown below, and the sentence you want to test.

Place the aspect term after [A], the opinion term after [O], the sentiment polarity afer [S], and the aspect category after [C] in the format '{FORMAT}'. If multiple quadruples are predicted, insert '####' to separate the quadruples.
'''.strip()
}


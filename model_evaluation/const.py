"""Constants for model evaluation"""

import json

# Element indices and abbreviations
e2idx = {"A": 0, "C": 1, "S": 2, "O": 3, "tag": 4}
abb2e = {"A": "Aspect", "O": "Opinion", "C": "Category", "S": "Sentiment"}

# Sentiment mappings
senttag2sentword = {'pos': 'positive', 'neg': 'negative', 'neu': 'neutral'}
senttag2opinion = {'pos': 'great', 'neg': 'bad', 'neu': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}
opinion2sent = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}

all_short2fullname = {
    "A": "aspect term", 
    "O": "opinion term", 
    "C": "aspect category", 
    "S": "sentiment polarity"
}

# All possible element orders
orders = [
    'AOSC', 'OCSA', 'OSAC', 'OSCA', 'OACS', 'AOCS', 
    'COAS', 'SAOC', 'OASC', 'SOAC', 'SOCA', 'ASOC', 
    'CAOS', 'SCAO', 'OCAS', 'COSA', 'CASO', 'CSAO', 
    'ACOS', 'ACSO', 'SCOA', 'CSOA', 'SACO', 'ASCO'
]

# Category lists for different datasets
rest_aspect_cate_list = [
    'location general', 
    'food prices', 'food quality', 'food general', 'food style&options',
    'ambience general', 'service general', 
    'restaurant general', 'restaurant prices', 'restaurant miscellaneous', 
    'drinks prices', 'drinks quality', 'drinks style&options'
]

laptop_aspect_cate_list = [
    'keyboard operation_performance', 'os operation_performance',
    'out_of_scope operation_performance', 'ports general',
    'optical_drives general', 'laptop operation_performance',
    'optical_drives operation_performance', 'optical_drives usability',
    'multimedia_devices general', 'keyboard general', 'os miscellaneous',
    'software operation_performance', 'display operation_performance',
    'shipping quality', 'hard_disc quality', 'motherboard general',
    'graphics general', 'multimedia_devices connectivity', 'display general',
    'memory operation_performance', 'os design_features',
    'out_of_scope usability', 'software design_features',
    'graphics design_features', 'ports connectivity',
    'support design_features', 'display quality', 'software price',
    'shipping general', 'graphics operation_performance',
    'hard_disc miscellaneous', 'display design_features',
    'cpu operation_performance', 'mouse general', 'keyboard portability',
    'hardware price', 'support quality', 'hardware quality',
    'motherboard operation_performance', 'multimedia_devices quality',
    'battery design_features', 'mouse usability', 'os price',
    'shipping operation_performance', 'laptop quality', 'laptop portability',
    'fans&cooling general', 'battery general', 'os usability',
    'hardware usability', 'optical_drives design_features',
    'fans&cooling operation_performance', 'memory general', 'company general',
    'power_supply general', 'hardware general', 'mouse design_features',
    'software general', 'keyboard quality', 'power_supply quality',
    'software quality', 'multimedia_devices usability',
    'power_supply connectivity', 'multimedia_devices price',
    'multimedia_devices operation_performance', 'ports design_features',
    'hardware operation_performance', 'shipping price',
    'hardware design_features', 'memory usability', 'cpu quality',
    'ports quality', 'ports portability', 'motherboard quality',
    'display price', 'os quality', 'graphics usability', 'cpu design_features',
    'hard_disc general', 'hard_disc operation_performance', 'battery quality',
    'laptop usability', 'company design_features',
    'company operation_performance', 'support general', 'fans&cooling quality',
    'memory design_features', 'ports usability', 'hard_disc design_features',
    'power_supply design_features', 'keyboard miscellaneous',
    'laptop miscellaneous', 'keyboard usability', 'cpu price',
    'laptop design_features', 'keyboard price', 'warranty quality',
    'display usability', 'support price', 'cpu general',
    'out_of_scope design_features', 'out_of_scope general',
    'software usability', 'laptop general', 'warranty general',
    'company price', 'ports operation_performance',
    'power_supply operation_performance', 'keyboard design_features',
    'support operation_performance', 'hard_disc usability', 'os general',
    'company quality', 'memory quality', 'software portability',
    'fans&cooling design_features', 'multimedia_devices design_features',
    'laptop connectivity', 'battery operation_performance', 'hard_disc price',
    'laptop price'
]

laptop_aspect_super_cate_list = [
    'support', 'shipping', 'memory', 'battery', 'ports', 'mouse', 
    'company', 'warranty', 'motherboard', 'laptop', 'graphics', 'software', 
    'out_of_scope', 'cpu', 'os', 'keyboard', 'display', 'optical_drives', 
    'multimedia_devices', 'hard_disc', 'fans&cooling', 'hardware', 'power_supply'
]

# Category mapping for different datasets
cate_list = {
    "rest14": rest_aspect_cate_list,
    "rest15": rest_aspect_cate_list,
    "rest16": rest_aspect_cate_list,
    "laptop14": laptop_aspect_cate_list,
    "laptop14-supcate": laptop_aspect_super_cate_list,
    "laptop16": laptop_aspect_cate_list,
    "laptop16-supcate": laptop_aspect_super_cate_list,
}


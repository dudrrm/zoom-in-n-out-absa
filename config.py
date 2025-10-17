"""
Configuration and constants for ABSA evaluation
Based on the NAACL 2025 paper implementation
"""

import json

# Element ordering permutations (24 possible orders for ACOS)
ORDERS = ['AOSC', 'OCSA', 'OSAC', 'OSCA', 'OACS', 'AOCS', 'COAS', 'SAOC', 'OASC', 'SOAC', 
          'SOCA', 'ASOC', 'CAOS', 'SCAO', 'OCAS', 'COSA', 'CASO', 'CSAO', 'ACOS', 'ACSO', 
          'SCOA', 'CSOA', 'SACO', 'ASCO']

# Sentiment mappings
SENTTAG2SENTWORD = {'pos': 'positive', 'neg': 'negative', 'neu': 'neutral'}
SENTTAG2OPINION = {'pos': 'great', 'neg': 'bad', 'neu': 'ok'}
SENTWORD2OPINION = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}
OPINION2SENT = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}

# Element descriptions
ALL_SHORT2FULLNAME = {
    "A": "aspect term", 
    "O": "opinion term", 
    "C": "aspect category", 
    "S": "sentiment polarity"
}

# Category lists for different datasets
REST_ASPECT_CATE_LIST = [
    'location general', 
    'food prices', 'food quality', 'food general', 'food style&options',
    'ambience general', 'service general', 
    'restaurant general', 'restaurant prices', 'restaurant miscellaneous', 
    'drinks prices', 'drinks quality', 'drinks style&options'
]

LAPTOP_ASPECT_CATE_LIST = [
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

LAPTOP_ASPECT_SUPER_CATE_LIST = [
    'support', 'shipping', 'memory', 'battery', 'ports', 'mouse', 'company', 
    'warranty', 'motherboard', 'laptop', 'graphics', 'software', 'out_of_scope', 
    'cpu', 'os', 'keyboard', 'display', 'optical_drives', 'multimedia_devices', 
    'hard_disc', 'fans&cooling', 'hardware', 'power_supply'
]

CATE_LIST = {
    "rest14": REST_ASPECT_CATE_LIST,
    "rest15": REST_ASPECT_CATE_LIST,
    "rest16": REST_ASPECT_CATE_LIST,
    "laptop14": LAPTOP_ASPECT_CATE_LIST,
    "laptop14-supcate": LAPTOP_ASPECT_SUPER_CATE_LIST,
    "laptop16": LAPTOP_ASPECT_CATE_LIST,
    "laptop16-supcate": LAPTOP_ASPECT_SUPER_CATE_LIST,
}

# Force words for constrained decoding
FORCE_WORDS = {
    'aste': {
        'rest15': list(SENTTAG2OPINION.values()),
        'rest16': list(SENTTAG2OPINION.values()),
        'rest14': list(SENTTAG2OPINION.values()),
        'laptop14': list(SENTTAG2OPINION.values())
    },
    'tasd': {
        "rest15": list(SENTWORD2OPINION.values()),
        "rest16": list(SENTWORD2OPINION.values())
    },
    'acos': {
        "rest16": REST_ASPECT_CATE_LIST + list(SENTWORD2OPINION.values()),
        "rest16-explicit": REST_ASPECT_CATE_LIST + list(SENTWORD2OPINION.values()),
        "laptop16": LAPTOP_ASPECT_CATE_LIST + list(SENTWORD2OPINION.values()),
        "laptop16-explicit": LAPTOP_ASPECT_CATE_LIST + list(SENTWORD2OPINION.values()),
        "laptop16-supcate": LAPTOP_ASPECT_SUPER_CATE_LIST + list(SENTWORD2OPINION.values()),
        "laptop16-supcate-explicit": LAPTOP_ASPECT_SUPER_CATE_LIST + list(SENTWORD2OPINION.values()),
    },
    'asqp': {
        "rest15": list(SENTWORD2OPINION.values()),
        "rest16": list(SENTWORD2OPINION.values()),
    }
}


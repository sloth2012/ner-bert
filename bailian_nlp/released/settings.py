# coding: utf8
import os

root_dir, _ = os.path.split(os.path.realpath(__file__))

for i in range(1):
    root_dir = os.path.dirname(root_dir)

MODEL_DIR = os.path.join(root_dir, 'datadrive/models')


USE_ERNIE = True
model_dir_name = 'chinese_L-12_H-768_A-12' if not USE_ERNIE else 'ernie'
model_name = 'bert' if not USE_ERNIE else 'ernie'

CHINESE_MODEL_DIR = os.path.join(MODEL_DIR, model_dir_name)

# 先主要处理中文的
DEFAULT_POS_MODEL_CONFIG_FILE = os.path.join(CHINESE_MODEL_DIR, 'pos.json')
DEFAULT_POS_MODEL_FILE = os.path.join(CHINESE_MODEL_DIR, 'pos.bin')

CHINESE_BERT_MODEL_FILE = os.path.join(CHINESE_MODEL_DIR, f'{model_name}_model.bin')
CHINESE_BERT_MODEL_CONFIG_FILE = os.path.join(CHINESE_MODEL_DIR, f'{model_name}_config.json')
CHINESE_BERT_VOCAB_FILE = os.path.join(CHINESE_MODEL_DIR, 'vocab.txt')

POS_DATA_DIR = os.path.join(root_dir, 'datadrive/bailian')

DEFAULT_POS_TRAIN_FILE = os.path.join(POS_DATA_DIR, 'pos/train.csv')
DEFAULT_POS_VALID_FILE = os.path.join(POS_DATA_DIR, 'pos/valid.csv')

# 默认自定义词典
DEFAULT_USER_DICT = os.path.join(root_dir, 'datadrive/dict/user_dict.txt')

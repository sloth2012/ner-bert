# coding: utf8
import os

root_dir, _ = os.path.split(os.path.realpath(__file__))
root_dir = os.path.dirname(root_dir)

MODEL_DIR = os.path.join(root_dir, 'datadrive/models')
CHINESE_MODEL_DIR = os.path.join(MODEL_DIR, 'chinese_L-12_H-768_A-12')

# 先主要处理中文的
DEFAULT_POS_MODEL_CONFIG_FILE = os.path.join(CHINESE_MODEL_DIR, 'pos.json')
DEFAULT_POS_MODEL_FILE = os.path.join(CHINESE_MODEL_DIR, 'pos.bin')

CHINESE_BERT_MODEL_FILE = os.path.join(CHINESE_MODEL_DIR, 'bert_model.bin')
CHINESE_BERT_MODEL_CONFIG_FILE = os.path.join(CHINESE_MODEL_DIR, 'bert_config.json')
CHINESE_BERT_VOCAB_FILE = os.path.join(CHINESE_MODEL_DIR, 'vocab.txt')

POS_DATA_DIR = os.path.join(root_dir, 'datadrive/bailian')

DEFAULT_POS_TRAIN_FILE = os.path.join(POS_DATA_DIR, 'pos/train.csv')
DEFAULT_POS_VALID_FILE = os.path.join(POS_DATA_DIR, 'pos/valid.csv')



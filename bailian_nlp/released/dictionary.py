#!/usr/bin/env python
# -*-coding:utf-8-*-

from . import trie
from ..web.utils.logger import getLogger
from ..modules.settings import UNKNOWN_TEXT_LABEL

# 未知标签
_UNKNOWN_LABEL = UNKNOWN_TEXT_LABEL
DELIMITER = '△' * 3


class Dictionary():
    '''
        自定义词典，逗号分隔
    '''

    def __init__(self):
        self.trie = trie.Trie()
        self.weights = {}
        self.labels = {}
        self.sizes = 0
        self.logger = getLogger(__name__)

    def delete_dict(self):
        self.trie = trie.Trie()
        self.weights = {}
        self.labels = {}
        self.sizes = 0

    def add_dict(self, path):
        words = []

        counter = 0
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                linelist = line.split(DELIMITER)

                word = linelist[0].strip()
                self.trie.add_keyword(word)

                weight = 1.0
                label = _UNKNOWN_LABEL  # 表示未知

                if len(linelist) == 2:
                    try:
                        weight = float(linelist[1])
                    except:
                        label = linelist[1]
                elif len(linelist) == 3:
                    try:
                        weight = float(linelist[2])
                        label = linelist[1]
                    except ValueError:
                        raise ValueError(f'词典每行格式必须满足：word{DELIMITER}label{DELIMITER}weight')

                self.weights[word] = weight
                self.labels[word] = label
                words.append(word)
                counter += 1

        self.logger.info(f'本次加载词条数：{counter}')
        self.sizes = len(self.weights)
        self.logger.info(f'当前总词条数: {self.sizes}')

    def parse_words(self, text):
        matchs = self.trie.parse_text(text)
        return matchs

    def get_weight(self, word):
        return self.weights.get(word, 0.1)

    def get_label(self, word):
        return self.labels.get(word, _UNKNOWN_LABEL)

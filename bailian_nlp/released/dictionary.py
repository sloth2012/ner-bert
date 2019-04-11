#!/usr/bin/env python
# -*-coding:utf-8-*-

from . import trie

# 未知标签
_UNKNOWN_LABEL = 'xx'


class Dictionary():
    '''
        自定义词典，逗号分隔
    '''

    def __init__(self):
        self.trie = trie.Trie()
        self.weights = {}
        self.labels = {}
        self.sizes = 0

    def delete_dict(self):
        self.trie = trie.Trie()
        self.weights = {}
        self.labels = {}
        self.sizes = 0

    def add_dict(self, path):
        words = []

        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                linelist = line.split(',')

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
                        raise ValueError('词典每行格式必须满足：word,label,weight')

                self.weights[word] = weight
                self.labels[word] = label
                words.append(word)
        self.sizes += len(self.weights)

    def parse_words(self, text):
        matchs = self.trie.parse_text(text)
        return matchs

    def get_weight(self, word):
        return self.weights.get(word, 0.1)

    def get_label(self, word):
        return self.labels.get(word, _UNKNOWN_LABEL)

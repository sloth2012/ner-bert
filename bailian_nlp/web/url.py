# coding: utf8

from .handler.pos import PosTaggerHandler

url = [
    # 词性标注
    (r'/pos', PosTaggerHandler)
]
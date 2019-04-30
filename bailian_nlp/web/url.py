# coding: utf8

from .handler.pos import PosTaggerHandler
from .handler.ping import PingHandler

url = [
    # 词性标注
    (r'/pos', PosTaggerHandler),
    (r'/_ping', PingHandler)
]
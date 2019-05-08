# coding: utf8

from .handler.pos import PosTaggerHandler
from .handler.pos_batch import BatchPosTaggerHandler
from .handler.ping import PingHandler

url = [
    # 词性标注
    (r'/pos', PosTaggerHandler),
    (r'/pos_batch', BatchPosTaggerHandler),
    (r'/_ping', PingHandler)
]
# coding: utf8

import logging
from tornado import log


class LogFormatter(log.LogFormatter):
    def __init__(self):
        super(LogFormatter, self).__init__(
            fmt='%(color)s[%(asctime)s %(filename)s:%(funcName)s:%(lineno)d %(levelname)s]%(end_color)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


default_streamHandler = logging.StreamHandler()
default_streamHandler.setFormatter(LogFormatter())

logging.root.handlers.clear()
logging.root.addHandler(default_streamHandler)


def getLogger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger


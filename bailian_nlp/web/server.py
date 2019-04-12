# coding: utf8

import logging

import tornado.httpserver
import tornado.ioloop
import urllib3
from tornado.options import options, define

define("port", default=50010, help="run on th given port", type=int)


def main():
    from bailian_nlp.web.utils.logger import getLogger, LogFormatter
    from bailian_nlp.web.application import application
    logger = getLogger('server')

    tornado.options.parse_command_line()

    default_streamHandler = logging.StreamHandler()
    default_streamHandler.setFormatter(LogFormatter())

    logging.root.handlers.clear()
    logging.root.addHandler(default_streamHandler)

    [i.setFormatter(LogFormatter()) for i in logging.getLogger().handlers]
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(options.port)

    logger.info('Development server is running at http://127.0.0.1:%s/' % options.port)
    logger.info('Quit the server with Control-C')

    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    import os
    import sys

    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    sys.path.insert(0, rootPath)

    urllib3.disable_warnings()

    main()

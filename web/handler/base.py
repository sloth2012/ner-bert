# coding: utf8

from tornado import web, websocket
from ..utils.logger import getLogger


class ApiBaseHandler(web.RequestHandler):
    name = 'Base'

    def __init__(self, application, request, **kwargs):
        super().__init__(application, request, **kwargs)
        self.logger = getLogger(self.name)


class WebsocketApiBaseHandler(websocket.WebSocketHandler):
    name = 'WebsocketBase'
    clients = set()

    def __init__(self, application, request, **kwargs):
        super().__init__(application, request, **kwargs)
        self.logger = getLogger(self.name)

    def check_origin(self, origin):
        return True

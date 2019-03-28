# coding: utf8

from .url import url
import tornado.web
import os

root_dir, _ = os.path.split(os.path.realpath(__file__))

setting = dict(
    template_path=os.path.join(root_dir, "resource/template"),
    static_path=os.path.join(root_dir, "resource/static"),
)
application = tornado.web.Application(
    handlers=url,
    **setting
)

# coding: utf8
from .base import ApiBaseHandler
from ..utils.common import timer


class PosTaggerHandler(ApiBaseHandler):
    name = 'PosTagger'

    @timer
    async def get(self, *args, **kwargs):
        text = self.get_argument('text', None)

        if text is None:
            self.write({
                'status': 'error',
                'msg': 'parameter text is null!'
            })

        else:
            try:
                from blnlp.models import pos_tagger

                # result = pos_tagger.cut(text)

                from aip import AipNlp

                client = AipNlp('11310322','YZS97m5TCvhNkVhaB1IiDcI7','b8GOkRyIY2SGO9ytewCI35wjaGPHLu6K')
                result = client.lexerCustom(text)

                # import jieba.posseg as psg
                # result = [(w.word, w.flag) for w in psg.cut(text)]

                self.write({
                    'status': 'success',
                    'result': result
                })

            except Exception as e:
                self.logger.error(e, exc_info=True)
                self.write({
                    'status': 'error',
                    'msg': e.__str__()
                })

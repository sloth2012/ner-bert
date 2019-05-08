# coding: utf8
from .base import ApiBaseHandler
from ..utils.common import timer


class PosTaggerHandler(ApiBaseHandler):
    name = 'PosTagger'

    async def post(self, *args, **kwargs):
        return await self.get(*args, **kwargs)

    @timer
    async def get(self, *args, **kwargs):
        text_arr = self.get_arguments('text', strip=False)

        if len(text_arr) == 0:
            self.write({
                'status': 'error',
                'msg': 'parameter text is null!'
            })

        else:
            try:
                from .. import global_var
                result = global_var.pos_tagger.lexerCustom(text_arr, ignore=True)

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

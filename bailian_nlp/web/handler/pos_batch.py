# coding: utf8
from .base import ApiBaseHandler
from ..utils.common import timer


class BatchPosTaggerHandler(ApiBaseHandler):
    name = 'BatchPosTagger'

    async def post(self, *args, **kwargs):
        return await self.get(*args, **kwargs)

    @timer
    async def get(self, *args, **kwargs):
        text = self.get_argument('text', None, strip=False)

        if text is None:
            self.write({
                'status': 'error',
                'msg': 'parameter text is null!'
            })

        else:
            try:
                import demjson
                text_arr = demjson.decode(text)

                if not isinstance(text_arr, list):
                    raise Exception(
                        'parameter text should be the json string that can decode as array!'
                    )

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

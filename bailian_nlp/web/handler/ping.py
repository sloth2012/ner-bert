# coding: utf8
from .base import ApiBaseHandler


class PingHandler(ApiBaseHandler):
    name = '_ping'

    async def post(self, *args, **kwargs):
        return await self.get(*args, **kwargs)

    async def get(self, *args, **kwargs):
        try:
            self.set_status(200)
            self.write({
                "status": "alive"
            })

        except Exception as e:
            self.logger.error(e, exc_info=True)
            raise

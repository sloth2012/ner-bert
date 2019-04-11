# coding: utf8

from .logger import getLogger

logger = getLogger(__name__)


def get_out_ip():
    from urllib.request import urlopen
    from json import load
    try:
        ip = load(urlopen('http://httpbin.org/ip'))['origin']
    except:
        ip = load(urlopen('https://api.ipify.org/?format=json'))['ip']

    return ip


def check_service(url, timeout=5):
    if not url:
        return False

    from urllib import request

    prefix = 'http'
    if not url.startswith(prefix):
        url = f'{prefix}://{url}'
    protocol, s1 = request.splittype(url)
    host, s2 = request.splithost(s1)
    host, port = request.splitport(host)

    import socket
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sk.settimeout(timeout)
    try:
        sk.connect((host, int(port)))
        logger.debug(f'{host}:{port} is available')
        is_valid = True
    except Exception:
        logger.info(f'{host}:{port} is not available')
        is_valid = False
    finally:
        sk.close()

    return is_valid


# 错误追踪
def try_except(func=None, error_callback=None, raise_error=False):
    import functools
    import asyncio

    def decorator(func):
        if not asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f'{func.__name__}:{str(e)}', exc_info=True)
                    if error_callback is not None:
                        error_callback(*args, **kwargs)
                    if raise_error:
                        raise e

        else:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f'{func.__name__}:{str(e)}', exc_info=True)
                    if error_callback is not None:
                        error_callback(*args, **kwargs)
                    if raise_error:
                        raise e

        return wrapper

    if func is not None:
        return decorator(func)
    else:
        return decorator


def get_mac_address():
    import uuid
    node = uuid.getnode()
    mac = uuid.UUID(int=node).hex[-12:]
    return mac


# 判断是否为中文
def is_chinese_char(char):
    if not char or len(char) != 1:
        raise ValueError('parameter char must be with length 1')
    return u'\u4e00' <= char <= u'\u9fff'


# 中文姓名规范化，主要识别是否为中文名，并剔除空格
def chinese_name_format(name):
    for ch in name:
        if ch == ' ' or is_chinese_char(ch):
            continue
        else:
            return name

    return name.replace(' ', '')


def re_compile(items, end=False):
    import re
    if items is None or len(items) == 0:
        return None
    return re.compile(join_items(items, end), re.I)


def join_items(items, end=False):
    rule = r'|'.join([f'({_item.strip()})' for _item in items if len(_item) > 0])
    if end:
        rule = f'({rule})$'
    return rule


# 装饰器，用来打印执行时间
def timer(func=None):
    from datetime import datetime
    import functools
    import asyncio

    def decorator(func):
        if not asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                st = datetime.now()
                res = func(*args, **kwargs)
                ed = datetime.now()

                cost = (ed - st).total_seconds()
                logger.debug(f'{func.__name__} cost {cost}s')
                return res
        else:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                st = datetime.now()
                res = await func(*args, **kwargs)
                ed = datetime.now()

                cost = (ed - st).total_seconds()
                logger.debug(f'{func.__name__} cost {cost}s')
                return res

        return wrapper

    if func is not None:
        return decorator(func)
    else:
        return decorator


# 区别与str的strip，严格去除有序字符串，而不是字符
def strip(text, strip_string):
    return rstrip(lstrip(text, strip_string), strip_string)


def lstrip(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def rstrip(text, suffix):
    if text.endswith(suffix):
        return text[:-len(suffix)]
    return text


if __name__ == '__main__':
    url = "http://52.82.46.226:50005"
    # url = 'http://150.109.50.113:50005'
    # url = 'http://192.144.192.205:50005'

    print(check_service(url))

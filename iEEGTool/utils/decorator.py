# -*- coding: UTF-8 -*-
'''
@Project ：EpiLocker 
@File    ：_decorator.py
@Author  ：Barry
@Date    ：2022/1/20 8:03 
'''
import sys
from decorator import decorator
import traceback

@decorator
def safe_event(fun, *args, **kwargs):
    """Protect against PyQt5 exiting on event-handling errors."""
    try:
        return fun(*args, **kwargs)
    except Exception:
        traceback.print_exc(file=sys.stderr)
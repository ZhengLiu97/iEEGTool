# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：numeric.py
@Author  ：Barry
@Date    ：2022/2/18 21:04 
"""

def isfloat(num):
    if isinstance(num, str):
        if '.' in num:
            int_part = num.split('.')[0]
            decimal_part = num.split('.')[1]
            if int_part.isdigit() and decimal_part.isdigit():
                return True
            else:
                return False
        else:
            return True
# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：utils.py
@Author  ：Barry
@Date    ：2022/3/26 20:06 
"""
def check_hemi(hemi):
    if hemi == 'Both':
        return ['lh', 'rh']
    elif hemi == 'Left':
        return 'lh'
    else:
        return 'rh'
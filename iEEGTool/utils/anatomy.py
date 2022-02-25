# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：anatomy.py
@Author  ：Barry
@Date    ：2022/2/24 23:29 
"""
import pandas as pd
from collections import OrderedDict

class Anatomy(object):

    def __init__(self, contacts: list, rois: dict) -> object:
        self.contacts = contacts
        self.rois = OrderedDict(rois)

        self.roi_df = pd.DataFrame()

        for index, roi_name in enumerate(rois):
            self.roi_df[roi_name] = self.rois[roi_name]

    def get_roi_df(self):
        return self.roi_df

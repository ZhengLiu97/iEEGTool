# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：anatomy.py
@Author  ：Barry
@Date    ：2022/2/24 23:29 
"""
import pandas as pd
from utils.process import get_chan_group

class Electrodes(object):

    def __init__(self):
        self.seg_name = []
        self.init_order = ['Channel', 'Group', 'x', 'y', 'z', 'issues']
        self.order = self.init_order
        self.electrodes_df = pd.DataFrame()

    def set_ch_names(self, ch_names):
        self.electrodes_df['Channel'] = ch_names
        ch_df = get_chan_group(chans=ch_names, return_df=True)
        self.electrodes_df['Group'] = ch_df['Group'].to_list()

    def set_ch_xyz(self, xyz):
        self.electrodes_df['x'] = xyz[0]
        self.electrodes_df['y'] = xyz[1]
        self.electrodes_df['z'] = xyz[2]

    def set_anatomy(self, seg_name, roi):
        self.seg_name.append(seg_name)
        self.seg_name.sort()
        self.order = self.init_order + self.seg_name
        self.electrodes_df[seg_name] = roi
        self.electrodes_df = self.electrodes_df[self.order]

    def rm_anatomy(self, seg_name):
        if seg_name in self.electrodes_df.columns:
            self.electrodes_df.drop(columns=seg_name)
            self.seg_name.remove(seg_name)
            self.order = self.init_order + self.seg_name

    def rm_chs(self, chs):
        if len(chs):
            elec_df = self.electrodes_df.copy()
            index = elec_df[elec_df['Channel'].isin(chs)].index
            if len(index):
                self.electrodes_df = elec_df.drop(index)

    def get_info(self):
        return self.electrodes_df


class ROI(object):
    def __init__(self):
        pass
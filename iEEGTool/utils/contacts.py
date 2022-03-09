# -*- coding: UTF-8 -*-
"""
@Project ：EpiLocker 
@File    ：_calculate_contact_pos.py
@Author  ：Barry
@Date    ：2022/1/20 1:52 
"""
import numpy as np
import re

from collections import OrderedDict

from utils.process import get_chan_group

def calc_ch_pos(tip, tail, ch_num, dist=3.5, extra_interval=None):
    """Calculate channels' position in the same shaft

    Parameters
    ----------
    tip: list | np.array shape (3, 1)
        the first contact's coordinates, should be in Surface RAS
    tail: list | np.array shape (3, 1)
        the tail's coordinates
    ch_num: int
        the number of contacts in this shaft, should be in Surface RAS
    dist: int | float
        the distance between the center of mass of contacts
    extra_interval: int | float
        not uniformly-spaced at the middle should be 10
        specify for 16 contacts

    Returns
    -------
    ch_pos: np.array shape (n_contacts, 3)
        The coordinates of the channels in this shaft

    """
    assert len(tip) == 3
    assert len(tail) == 3
    assert isinstance(ch_num, int)
    assert isinstance(dist, int) or isinstance(dist, float)
    if not isinstance(tip, np.ndarray):
        tip = np.asarray(tip)
    if not isinstance(tail, np.ndarray):
        tail = np.asarray(tail)
    xyz_diff = tail - tip
    line_len = np.sqrt(xyz_diff[0] ** 2 + xyz_diff[1] ** 2 + xyz_diff[2] ** 2)
    ch_pos = np.zeros((ch_num, 3))
    ch_pos[0, :] = tip
    xyz_unit = dist * xyz_diff / line_len
    for i in range(1, ch_num):
        ch_pos[i, :] = tip + xyz_unit * i
    if extra_interval is not None:
        assert ch_num == 16

        extra_dist = extra_interval - dist
        assert extra_dist > 0

        xyz_extra = extra_dist * xyz_diff / line_len
        ch_pos[ch_num//2:, :] += xyz_extra

    return ch_pos

def is_wm(roi_name):
    """Judge if roi is in white matter

    Parameters
    ----------
    roi_name : list
            ROI name

    Returns
    -------
    bool

    """
    wm_key = ['white', 'wm']
    roi_name = roi_name.lower()
    for key in wm_key:
        if key in roi_name:
            return True
    return False

def is_unknown(roi_name):
    """Judge if roi is in unknown

    Parameters
    ----------
    roi_name : list
            ROI name

    Returns
    -------
    bool

    """
    unknown_key = 'unknown'
    roi_name = roi_name.lower()
    if unknown_key in roi_name:
        return True
    else:
        return False

def is_gm(roi_name):
    """Judge if roi is in gray matter

    Parameters
    ----------
    roi_name : list
            ROI name
    Returns
    -------
    bool

    """
    if is_wm(roi_name):
        return False
    if is_unknown(roi_name):
        return False
    return True

def reorder_chs(chs):
    try:
        ch_group = OrderedDict(get_chan_group(chs))
        ch_names = []
        items = list(ch_group.values())
        for item in items:
            ch_names += item
        return ch_names
    except:
        print('This is not iEEG')
        return None

def reorder_chs_df(df):
    ch_names = df['Channel'].to_list()
    try:
        ch_group = OrderedDict(get_chan_group(ch_names))
        ch_names = []
        items = list(ch_group.values())
        for item in items:
            ch_names += item
        df['Channel'] = df['Channel'].astype('category').cat.set_categories(ch_names)
        return df.sort_values(by=['Channel'], ascending=True)
    except:
        print('This is not iEEG')
        return None


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    tip = (1, 1, 1)
    tail = (18, 18, 18)
    ch_pos = calc_ch_pos(tip, tail, 16, extra_interval=10)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(ch_pos[:, 0], ch_pos[:, 1], ch_pos[:, 2])
    plt.show(block=True)



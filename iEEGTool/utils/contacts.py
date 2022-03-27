# -*- coding: UTF-8 -*-
"""
@Project ：EpiLocker 
@File    ：_calculate_contact_pos.py
@Author  ：Barry
@Date    ：2022/1/20 1:52 
"""
import re
import numpy as np

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

def calc_bipolar_chs_pos(ch_pos, middle=False):
    """Calculate the bipolar channels' position based on the original coordinates
    Parameters
    ----------
    ch_pos : dict
        the coordinates of each contact

    middle : bool

    Returns
    -------
    bipolar_ch_pos : dict
        the coordinates of the bipolar contacts

    """
    from utils.process import get_bipolar_pair

    ch_names = list(ch_pos.keys())
    bipolar_pairs = get_bipolar_pair(ch_names)
    anode = []
    cathode = []
    for group in bipolar_pairs:
        anode += bipolar_pairs[group][0]
        cathode += bipolar_pairs[group][1]
    bipolar_ch_pos = OrderedDict()
    for index, _ in enumerate(anode):
        anode_name = anode[index]
        cathode_name = cathode[index]
        ch_name = f'{anode_name}-{cathode_name}'
        if middle:
            bipolar_ch_pos[ch_name] = np.round((ch_pos[anode_name] + ch_pos[cathode_name]) / 2, 3)
        else:
            bipolar_ch_pos[ch_name] = ch_pos[anode_name]

    return bipolar_ch_pos

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

def is_lh(roi_name):
    """Judge if roi is in left hemi

    Parameters
    ----------
    roi_name : list
            ROI name
    Returns
    -------
    bool

    """
    hemi_key = ['left', 'lh']
    roi_name = roi_name.lower()
    for key in hemi_key:
        if key in roi_name:
            return True
    return False

def reorder_chs(chs):
    """Reorder iEEG channels' name
    Parameters
    ----------
    chs : list of str
        The name of channels
    Returns
    -------
    sorted_chs : list of str
        The right sorted name of channels
    Notes
    -----
    The code is from
    https://stackoverflow.com/questions/71410219/reorder-a-list-with-elements-in-the-format-like-letternumber

    Because of the file exporting software of iEEG, the channels' names are always in the
    wrong order. The expected order is like
    ['A1', 'A2', 'A3', 'A11', 'A12', 'B1', 'B12', 'EC1', 'EC21']
    The code here considered both different length of channels' group and number using RegEx
    The idea of this code is
    1. separate the group and num
    2. save num to the corresponding group
    3. sort the group
    4. sort the num in each group
    5. merge the group+num

    """
    unsorted_chs_dict = {}
    post_node = {}
    for ch in chs:
        # Get the letter part and the number part
        # The letter is the group of this contact
        # and the number is the serial number of this contact
        quote = "'" in ch
        if quote:
            # In this case, the contact's name is like A'1 or A'1-A'2
            match = re.match(r"([A-Z]+)(')([0-9]+)", ch, re.I)
            ch_group = match.groups()[0] + match.groups()[1]
            # have to int the num so it would be sorted by the rule of int
            # not the rule of str
            ch_num = int(match.groups()[2])
        else:
            # In this case, the contact's name is like A1 or A1-A2
            match = re.match(r"([A-Z]+)([0-9]+)", ch, re.I)
            ch_group = match.groups()[0]
            ch_num = int(match.groups()[1])
        if ch_group in unsorted_chs_dict:
            unsorted_chs_dict[ch_group].append(ch_num)
        else:
            unsorted_chs_dict[ch_group] = [ch_num]
        ch_len = len(ch_group) + len(str(ch_num))
        if ch_len < len(ch):
            # if bipolar, restore its post channel's name with -
            post_node[f'{ch_group}{ch_num}'] = ch[ch_len:]
    # sort the channels' group, aka the keys of dict
    ch_group_sorted_dict = OrderedDict(sorted(unsorted_chs_dict.items()))

    sorted_chs = []
    for group in ch_group_sorted_dict:
        for ch_num in sorted(ch_group_sorted_dict[group]):
            ch_name = f'{group}{ch_num}'
            if ch_name in post_node:
                ch_name += post_node[ch_name]
            sorted_chs.append(ch_name)
    # print(sorted_chs)

    return sorted_chs

def reorder_chs_df(df):
    ch_names = df['Channel'].to_list()
    try:
        ch_names = reorder_chs(ch_names)
        # print(ch_names)
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



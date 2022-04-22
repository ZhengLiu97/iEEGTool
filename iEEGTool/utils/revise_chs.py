# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：revise_chs.py
@Author  ：Barry
@Date    ：2022/4/22 10:29 
"""
import numpy as np

def get_chan_group(chans, exclude=None):
    """Group iEEG channel
    Parameters
    ----------
    chans: list
        channels' name
    exclude: list
        channels need to be excluded
    Returns
    -------
    chan_group: dict  group: channels
        channels belong to each group
    """
    import re

    if isinstance(exclude, list):
        [chans.pop(chans.index(ch)) for ch in exclude]

    group_chs = dict()
    for ch in chans:
        match = r"([a-zA-Z]+')" if "'" in ch else r"([a-zA-Z]+)"
        group = re.match(match, ch, re.I).group()
        if group not in group_chs:
            group_chs[group] = [ch]
        else:
            group_chs[group].append(ch)
    return group_chs


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
        not uniformly-spaced at the middle should be 12
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
    xyz_diff = tail.astype(np.float64) - tip.astype(np.float64)
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

    return np.round(ch_pos, 3)

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
            match = re.match(r"([a-zA-Z]+)(')([0-9]+)", ch, re.I)
            ch_group = match.groups()[0] + match.groups()[1]
            # have to int the num so it would be sorted by the rule of int
            # not the rule of str
            ch_num = int(match.groups()[2])
        else:
            # In this case, the contact's name is like A1 or A1-A2
            match = re.match(r"([a-zA-Z]+)([0-9]+)", ch, re.I)
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

    return sorted_chs


if __name__ == '__main__':
    import argparse

    import os.path as op
    import pandas as pd

    help_text = f"Adjust special contacts' location"
    parser = argparse.ArgumentParser(description=help_text)

    parser.add_argument('-i', '--input_path', dest='fpath',
                        help='coordinates file path')
    parser.add_argument('-g', '--group', dest='group', type=list, default=[],
                        help='group of contacts to revise')
    parser.add_argument('-n', '--ch_num', dest='ch_num', type=int, default=16,
                        help='number of contacts')
    parser.add_argument('-d', '--dist', dest='dist', type=float, default=3.5,
                        help='distance of contacts')
    parser.add_argument('-e', '--extra_interval', dest='extra_interval', type=float, default=12,
                        help='extra interval of contacts')
    parser.add_argument('-s', '--save_path', dest='save_path', type=str, default='',
                        help='revised coordinates save path')
    args = parser.parse_args()
    fpath = args.fpath
    group = args.group
    ch_num = args.ch_num
    dist = args.dist
    extra_interval = args.extra_interval
    save_path = args.save_path
    if not op.exists(fpath):
        raise FileNotFoundError(f'No such file or directory: {fpath}')
    if not op.isfile(save_path):
        raise ValueError(f'Wrong save path format \n'
                         f'should be like *.txt')
    coords = pd.read_table(fpath)
    print(f'Loading coordinates from {fpath}')
    print(f'Revising group {group} using distance {dist} with extra interval {extra_interval}')

    ch_names = coords['Channel'].to_list()
    xyz = coords[['x', 'y', 'z']].to_numpy()
    ch_pos = dict(zip(ch_names, xyz))
    group_chs = get_chan_group(ch_names)
    if len(group):
        for g in group:
            if g not in group_chs:
                print(f'No group {g} in this file')
            else:
                g_chs = group_chs[g]
                tip_name = g_chs[0]
                tail_name = g_chs[-1]
                tip = ch_pos[tip_name]
                tail = ch_pos[tail_name]
                g_ch_pos = calc_ch_pos(tip, tail, ch_num, dist, extra_interval)
                for index, ch in enumerate(g_chs):
                    ch_index = ch_names.index(ch)
                    coords.loc[ch_index, ['x', 'y', 'z']] = g_ch_pos[index]

                # from matplotlib import pyplot as plt
                #
                # fig = plt.figure()
                # ax = fig.add_subplot(projection='3d')
                # ax.scatter(g_ch_pos[:, 0], g_ch_pos[:, 1], g_ch_pos[:, 2])
                # plt.show(block=True)
        coords.to_csv(save_path, sep='\t', index=None)
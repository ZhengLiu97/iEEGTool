# -*- coding: UTF-8 -*-
"""
@Project ：EpiLock 
@File    ：process.py
@Author  ：Barry
@Date    ：2021/11/15 4:07 
"""

import os
import time
import numpy as np
import traceback
import mne

from mne.transforms import apply_trans, invert_transform
from mne._freesurfer import _read_mri_info


def get_chan_group(chans, exclude=['E'], return_df=False):
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
    import pandas as pd
    try:
        chans.pop(chans.index('E'))
    except:
        pass

    chan_df = pd.DataFrame(columns=['Channel', 'Group'])
    chan_df['Channel'] = chans
    for chan in chans:
        num_start = re.search(f'\d', chan).start()
        chan_df.loc[chan_df['Channel'] == chan, ['Group']] = chan[:num_start]
    group = list(set(chan_df['Group']))
    group.sort()
    chan_group = {gname: [] for gname in group}
    for chan in chans:
        [chan_group[gname].append(chan) for gname in group if
         chan_df.loc[chan_df['Channel'] == chan]['Group'].item() == gname]
    # sort the channels in its group for reference, just in case
    for group in chan_group:
        if '-' not in chans[0]:
            if group.endswith('\''):
                chan_group[group].sort(key=lambda ch: (ch[0:len(group)],
                                                       int(ch[len(group):])))
            else:
                chan_group[group].sort(key=lambda ch: (ch[0:len(group)],
                                                       int(ch[len(group):])))
        else:
            if group.endswith('\''):
                chan_group[group].sort(key=lambda ch: (ch[0:len(group)],
                                                       int(ch[len(group):ch.index('-')])))
            else:
                chan_group[group].sort(key=lambda ch: (ch[0:len(group)],
                                                       int(ch[len(group):ch.index('-')])))
    if return_df:
        return chan_df
    else:
        return chan_group


def clean_chans(ieeg):
    import re

    ieeg.rename_channels({chan: chan[4:]
                          for chan in ieeg.ch_names
                          if 'POL' in chan})
    ieeg.rename_channels({chan: chan[:-4]
                          for chan in ieeg.ch_names
                          if 'Ref' in chan})
    ieeg.rename_channels({chan: chan[4:]
                               for chan in ieeg.ch_names
                               if 'EEG' in chan})
    print('Finish renaming channels')
    print('Start Dropping channels')
    drop_chans = []
    for chan in ieeg.ch_names:
        chan_num = re.findall("\d+", chan)
        if len(chan_num):
            chan_num = int(chan_num[0])
        if ('DC' in chan) or ('EKG' in chan) or ('BP' in chan) \
                or ('E' == chan) or chan[0].isdigit() or (chan_num > 16):
            drop_chans.append(chan)
    print(f"Dropping channels {drop_chans}")
    ieeg.drop_channels(drop_chans)

    return ieeg

def set_bipolar(ieeg):
    """Reference SEEG data using Bipolar Reference
    ieeg: instance of BaseRaw or BaseEpochs
          SEEG data

    return: instance of raw
            data and raw data of the first contact in each shafts
    """
    group_chan = get_chan_group(ieeg)
    ieeg.load_data()
    if 'EKG' in list(group_chan.keys()):
        group_chan.pop('EKG')
    group_ieeg = {
        group: ieeg.copy().pick_channels(group_chan[group]).reorder_channels(group_chan[group])
        for group in group_chan}
    group_ieeg_bipolar = {group: group_ieeg[group]._data[:-1, :] - group_ieeg[group]._data[1:, :]
                          for group in group_chan}

    for group in group_ieeg:
        group_ieeg[group].drop_channels(group_chan[group][-1])._data = \
            group_ieeg_bipolar[group]

    first_group = list(group_chan.keys())[0]
    bipolar_ieeg = group_ieeg[first_group]
    group_ieeg.pop(first_group)
    for name in group_ieeg:
        if not (name == 'DC'):
            bipolar_ieeg.add_channels([group_ieeg[name]])
    del ieeg
    del group_ieeg

    bipolar_chan = bipolar_ieeg.ch_names
    bipolar_group = get_chan_group(chans=bipolar_chan)

    for key in bipolar_group:
        b_group = bipolar_group[key]
        for index in range(len(b_group)):
            chan = b_group[index]
            if index == len(b_group) - 1:
                bipolar_name = chan + '-' + chan[0:len(key)] + str(int(chan[len(key):]) + 1)
                bipolar_ieeg.rename_channels({chan: bipolar_name})
            else:
                latter_chan = b_group[index + 1]
                bipolar_name = chan + '-' + latter_chan
                bipolar_ieeg.rename_channels({chan: bipolar_name})
            print('Successfully convert {:} to {:}'.format(chan, bipolar_name))

    return bipolar_ieeg

def get_bipolar_pair(ch_names):
    group = get_chan_group(chans=ch_names)
    group_pair = {name: [group[name][:-1], group[name][1:]] for name in group}
    return group_pair

def mne_bipolar(raw):
    from mne import set_bipolar_reference

    ch_names = raw.ch_names
    bipolar_pairs = get_bipolar_pair(ch_names)
    anode = []
    cathode = []
    for group in bipolar_pairs:
        anode += bipolar_pairs[group][0]
        cathode += bipolar_pairs[group][1]
    raw_bipolar = set_bipolar_reference(raw, anode=anode, cathode=cathode)
    return raw_bipolar

def get_montage(ch_pos, subject, subjects_dir):
    """Get montage given Surface RAS
    Parameters
    ----------
    ch_pos : dict
        Surface RAS (mri)
    subject
    subjects_dir

    Returns
    -------

    """
    subj_trans = mne.coreg.estimate_head_mri_t(subject, subjects_dir)
    mri_to_head_trans = mne.transforms.invert_transform(subj_trans)
    print('Start transforming mri to head')
    print(mri_to_head_trans)

    montage_mri = mne.channels.make_dig_montage(ch_pos, coord_frame='mri')
    montage = montage_mri.copy()
    montage.add_estimated_fiducials(subject, subjects_dir)
    montage.apply_trans(mri_to_head_trans)
    return montage_mri, montage

def set_montage(ieeg, ch_pos, subject, subjects_dir):
    print(f'Load files from {subjects_dir}/{subject}')
    _, montage = get_montage(ch_pos, subject, subjects_dir)
    ieeg.set_montage(montage, on_missing='ignore')
    return ieeg

def get_gm_chans(roi_df):
    ch_names = roi_df['Channel'].to_list()
    rois = roi_df['ROI'].to_list()
    ch_rois = dict(zip(ch_names, rois))
    gm_chs = []
    for ch in ch_rois:
        ch_roi = ch_rois[ch]
        if ('White' not in ch_roi) and ('Unknown' not in ch_roi) and \
                ('unknown' not in ch_roi) and ('white' not in ch_roi):
            gm_chs.append(ch)
    return gm_chs

def ras_to_tkras(ras, subject, subjects_dir):
    if not isinstance(ras, np.ndarray):
        ras = np.asarray(ras)
    t1_path = os.path.join(subjects_dir, subject, 'mri', 'T1.mgz')
    _, _, tkras_to_ras, _, _ = _read_mri_info(t1_path)
    ras_to_tkras = invert_transform(tkras_to_ras)
    print(ras_to_tkras)
    return apply_trans(ras_to_tkras, ras)


def tkras_to_ras(tkras, subject, subjects_dir):
    if not isinstance(tkras, np.ndarray):
        tkras = np.asarray(tkras)
    t1_path = os.path.join(subjects_dir, subject, 'mri', 'T1.mgz')
    _, _, tkras_to_ras, _, _ = _read_mri_info(t1_path)
    print(tkras_to_ras)
    return apply_trans(tkras_to_ras, tkras)
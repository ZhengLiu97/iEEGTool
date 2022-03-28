# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：reference.py
@Author  ：Barry
@Date    ：2022/3/28 14:47 
"""
import numpy as np

from mne.io.constants import FIFF
from mne.io.meas_info import create_info
from mne.io import BaseRaw, RawArray
from mne import BaseEpochs, Evoked, EpochsArray, EvokedArray


def set_laplacian_reference(ieeg, cen_nodes, adj_nodes, copy=True):
    """
    Re-reference selected seeg channels using a laplace referencing scheme.
    A Laplace reference takes the difference between one central channel with
    the average of adjacent channels. The exception, however, is for the top
    and bottom electrodes in the seeg-shaft, where only one adjacent electrode
    is subtracted (actually, the average of two same adjacent channels).

    Multiple central channels and adjacent channels can be specified.

    Parameters
    ----------
    ieeg : instance of Raw | Epochs | Evoked
        Data containing the unreferenced channels.
    cen_nodes : str | list of str
        The name(s) of the channel(s) to use as central channel(s) in the
        laplace reference.
    adj_nodes : list of strs
        The name(s) of the channel(s) to use as adjacent channels in the
        laplace reference.
    copy : bool
        Whether to operate on a copy of the data (True) or modify it in-place
        (False). Defaults to True.

    Returns
    -------
    ref_ieeg : instance of Raw | Epochs | Evoked
        Data with the specified channels re-referenced.

    For example,
    -----------
    central_ch=['A1', 'A2', 'A3']
    adj_nodes = [['A1', 'A1'], ['A1', 'A3'], ['A2', 'A2']]

    See Also
    --------
    set_eeg_reference : Convenience function for creating an EEG reference.

    Reference
    ---------
    Li, G., Jiang, S., Paraskevopoulou, S.E., Wang, M., Xu, Y., Wu, Z., ...
    & Schalk,G.(2018).Optimal referencing for stereo-electroencephalographic
    (SEEG) recordings. NeuroImage, 183, 327-335.

    """
    if copy:
        ieeg = ieeg.copy()

    if not isinstance(cen_nodes, list):
        cen_nodes = [cen_nodes]

    orig_ch_names = ieeg.ch_names

    drop_channels = set(orig_ch_names) - set(cen_nodes)
    ieeg.drop_channels(drop_channels)
    ieeg.reorder_channels(cen_nodes)

    if not isinstance(adj_nodes, list):
        raise ValueError('Adjacent channels are a list of at least two'
                         'adjacent channels! ')

    if len(cen_nodes) != len(adj_nodes):
        raise ValueError('Number of central channels (got %d) must equal '
                         'the number of adjacent groups of channels (got %d).'
                         % (len(cen_nodes), len(adj_nodes)))

    ch_name = cen_nodes
    ch_info = [{} for _ in cen_nodes]

    # Do laplacian reference by multiplying the data(channels x time) with
    # a matrix (n_central_channels x channels).
    multiplier = np.zeros((len(cen_nodes), len(ieeg.ch_names)))
    for idx, (a, c) in enumerate(zip(cen_nodes, adj_nodes)):
        multiplier[idx, ieeg.ch_names.index(a)] = 1
        adjacent_ch_n = len(c)
        for i in range(adjacent_ch_n):
            multiplier[idx, ieeg.ch_names.index(c[i])] += -1 / adjacent_ch_n

    ref_info = create_info(ch_names=ch_name, sfreq=ieeg.info['sfreq'],
                           ch_types=ieeg.get_channel_types(picks=cen_nodes))

    # Update "chs" in Reference-Info.
    for ch_idx, (an, info) in enumerate(zip(cen_nodes, ch_info)):
        an_idx = ieeg.ch_names.index(an)
        # Copy everything from anode (except ch_name).
        an_chs = {k: v for k, v in ieeg.info['chs'][an_idx].items()
                  if k != 'ch_name'}
        ref_info['chs'][ch_idx].update(an_chs)
        # Set coil-type to bipolar.
        ref_info['chs'][ch_idx]['coil_type'] = FIFF.FIFFV_COIL_EEG_CSD
        # Update with info from ch_info-parameter.
        ref_info['chs'][ch_idx].update(info)

    # Set other info-keys from original instance.
    pick_info = {k: v for k, v in ieeg.info.items() if k not in
                 ['chs', 'ch_names', 'bads', 'nchan', 'sfreq']}
    with ref_info._unlock():
        ref_info.update(pick_info)

    # Rereferencing of data.
    ref_data = multiplier @ ieeg._data

    if isinstance(ieeg, BaseRaw):
        ref_ieeg = RawArray(ref_data, ref_info, first_samp=ieeg.first_samp,
                            copy=None)
    elif isinstance(ieeg, BaseEpochs):
        ref_ieeg = EpochsArray(ref_data, ref_info, events=ieeg.events,
                               tmin=ieeg.tmin, event_id=ieeg.event_id,
                               metadata=ieeg.metadata)
    else:
        ref_ieeg = EvokedArray(ref_data, ref_info, tmin=ieeg.tmin,
                               comment=ieeg.comment, nave=ieeg.nave,
                               kind='average')

    new_channels = ', '.join([name for name in ch_name])
    print(f'Added the following laplacian channels:\n{new_channels}')

    return ref_ieeg
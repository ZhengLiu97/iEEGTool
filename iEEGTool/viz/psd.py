# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：psd.py
@Author  ：Barry
@Date    ：2022/3/10 19:22 
"""
import numpy as np
import mplcursors
from matplotlib import pyplot as plt
from matplotlib import patheffects
from functools import partial
from mne.viz.utils import _convert_psds


def plot_psd(info, psds, freqs, dB, average, method):
    fig, ax = plt.subplots(figsize=(6, 6))
    estimate = 'power' if dB else 'amplitude'
    ylabels = _convert_psds(psds, dB, estimate, 1000, 'mV', info.ch_names)
    if average:
        psds_mean = psds.mean(0).mean(0)
        psds_std = psds.mean(0).std(0)
        ax.plot(freqs, psds_mean, color='k')
        ax.fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std,
                        color='k', alpha=.33)
    else:
        lines = []
        psds_mean = psds.mean(0)
        for idx, psd in enumerate(psds_mean):
            line = ax.plot(freqs, psd, color='k', label=info.ch_names[idx])[0]
            lines.append(line)
        highlight_kwargs = dict(color="r")
        # annotation_kwargs = dict(color='k', arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.2"))
        cursor = mplcursors.cursor(lines, highlight=True, highlight_kwargs=highlight_kwargs, multiple=True)
        cursor.connect(
            "add", lambda sel: sel.annotation.set(
                text=sel.artist.get_label(),)
        )

    ax.set(title=f'{method} PSD ({estimate.capitalize()})', xlabel='Frequency (Hz)',
           ylabel=ylabels)
    plt.grid(color='k', linestyle=':', linewidth=0.5)
    fig.tight_layout()
    plt.show(block=True)
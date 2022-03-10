# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：psd.py
@Author  ：Barry
@Date    ：2022/3/10 19:22 
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patheffects
from functools import partial
from mne.viz.utils import _convert_psds


def butterfly_on_button_press(event, params):
    """Only draw once for picking."""
    print(params)
    if params['need_draw']:
        event.canvas.draw()
    else:
        text = params['texts'][0]
        text.set_alpha(0.)
        text.set_path_effects([])
        event.canvas.draw()
    params['need_draw'] = False

def plot_psd(info, psds, freqs, dB, average, method):
    fig, ax = plt.subplots(figsize=(12, 6))
    estimate = 'power' if dB else 'amplitude'
    ylabels = _convert_psds(psds, dB, estimate, 1000, 'mV', info.ch_names)
    if average:
        psds_mean = psds.mean(0).mean(0)
        psds_std = psds.mean(0).std(0)
        ax.plot(freqs, psds_mean, color='k')
        ax.fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std,
                        color='k', alpha=.33)
    else:
        texts = list()
        idxs = [list(range(len(info.ch_names)))]
        lines = list()

        psds_mean = psds.mean(0)
        for psd in psds_mean:
            lines.append(ax.plot(freqs, psd, color='k')[0])

        texts.append(ax.text(0, 0, '', zorder=3,
                             verticalalignment='baseline',
                             horizontalalignment='left',
                             fontweight='bold', alpha=0,
                             clip_on=True))

        path_effects = [patheffects.withStroke(linewidth=2, foreground="w",
                                               alpha=0.75)]
        params = dict(axes=[ax], texts=texts, lines=lines,
                      ch_names=info['ch_names'], idxs=idxs, need_draw=False,
                      path_effects=path_effects)
        plt.ion()
        fig.canvas.mpl_connect('button_press_event',
                               partial(butterfly_on_button_press,
                                       params=params))

    ax.set(title=f'{method} PSD ({estimate.capitalize()})', xlabel='Frequency (Hz)',
           ylabel=ylabels)
    plt.grid(color='k', linestyle=':', linewidth=0.5)
    fig.tight_layout()
    plt.show()
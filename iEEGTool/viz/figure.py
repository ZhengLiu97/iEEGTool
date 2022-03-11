# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：figure.py
@Author  ：Barry
@Date    ：2022/3/11 14:14 
"""
import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import seaborn as sns
import pandas as pd

from functools import partial
from matplotlib import pyplot as plt

NUM = 0

def key_press_event(params, event):
    global NUM

    fig = params['fig']
    fig_params = params['fig_params']
    matrix = params['matrix']
    ch_names = params['ch_names']
    title = params['title']
    title_item = params['title_item']

    key = event.key
    print(f'Pressed key is {key}')

    if key in ['up', 'right']:
        NUM += 1
    elif key in ['down', 'right']:
        NUM = NUM - 1 if NUM > 0 else matrix.shape[0]
    if NUM > matrix.shape[0] - 1:
        NUM = 0
    # print(NUM)

    plot_matrix = matrix[NUM, :, :]
    plot_matrix_df = pd.DataFrame(dict(zip(ch_names, plot_matrix)), index=ch_names).T

    fig.clf()
    # recreate an ax, do not use the former one
    ax = sns.heatmap(plot_matrix_df, center=0., cmap='Blues', linewidth=1., square=True, **fig_params)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=-0, ha="right", rotation_mode="anchor")

    if isinstance(title_item, list):
        ax.set_title(f'{title} ({title_item[NUM]})')
    if isinstance(title_item, str):
        ax.set_title(f'{title} ({title_item})')
    fig.tight_layout()
    fig.canvas.draw()


def create_heatmap(matrix, ch_names, mask, title='', title_item=None, unit=''):
    global NUM

    if len(matrix.shape) == 2:
        matrix = matrix.reshape(1, matrix.shape[0], matrix.shape[1])
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    first_matrix = matrix[0, :, :]

    first_matrix_df = pd.DataFrame(dict(zip(ch_names, first_matrix)), index=ch_names).T

    sns.heatmap(first_matrix_df, center=0., mask=mask, cmap='Blues',
                linewidth=1., square=True, ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=-0, ha="right", rotation_mode="anchor")

    if isinstance(title_item, list):
        if len(unit):
            title_item = [f'{item}{unit}' for item in title_item]
        ax.set_title(f'{title} ({title_item[NUM]})')
    if isinstance(title_item, str):
        if len(unit):
            title_item = f'{title_item}{unit}'
        ax.set_title(f'{title} ({title_item})')

    fig_params = dict(mask=mask)
    params = dict(matrix=matrix, ch_names=ch_names, fig=fig, title=title,
                  title_item=title_item, fig_params=fig_params)
    plt.ion()
    fig.canvas.mpl_connect('key_press_event', partial(key_press_event, params))
    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    matrix = np.random.random((5, 10, 10))
    ch_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']
    freqs = [1, 2, 3, 4, 5]
    mask = np.zeros_like(matrix[0])
    mask[np.triu_indices_from(mask)] = True
    create_heatmap(matrix, ch_names, mask, title='Connectivity', title_item='10', unit='Hz')

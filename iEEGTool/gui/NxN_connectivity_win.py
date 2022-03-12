# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：NxN_connectivity_win.py
@Author  ：Barry
@Date    ：2022/3/10 15:04 
"""
import mne
import numpy as np

from mne.io import BaseRaw
from mne.viz import circular_layout
from matplotlib import pyplot as plt
from mne_connectivity.viz import plot_connectivity_circle
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QButtonGroup, QMessageBox

from gui.NxN_connectivity_ui import Ui_MainWindow
from gui.list_win import ItemSelectionWin
from utils.process import make_epoch
from utils.config import color
from utils.log_config import create_logger
from utils.thread import ComputeSpectralConnectivity
from viz.figure import create_heatmap

logger = create_logger(filename='iEEGTool.log')


class NxNSpectraConWin(QMainWindow, Ui_MainWindow):

    def __init__(self, ieeg, method):
        super().__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('Spectral Connectivity')

        if isinstance(ieeg, BaseRaw):
            ieeg = make_epoch(ieeg)
        self.ieeg = ieeg

        fmin = str(int(ieeg.info['highpass']))
        fmax = str(int(ieeg.info['lowpass']))
        fmin = '0.1' if fmin == '0' else fmin
        self._freq_band_le.setText(fmin + ' ' + fmax)

        self.compute_chans = ieeg.ch_names

        self.method = method
        self.compute_params = dict()
        self.average_freq = True

        self.con = None
        self.freqs = None

        self._slot_connection()

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _slot_connection(self):
        self._select_chan_btn.clicked.connect(self._select_chan)
        self._compute_btn.clicked.connect(self._compute_con)
        self._plot_heatmap_btn.clicked.connect(self._plot_heatmap)
        self._plot_circular_btn.clicked.connect(self._plot_circular)
        self._plot_3d_btn.clicked.connect(self._plot_3d)

    def _select_chan(self):
        self._select_chan_win = ItemSelectionWin(self.ieeg.ch_names)
        self._select_chan_win.SELECTION_SIGNAL.connect(self._get_compute_chans)
        self._select_chan_win.show()

    def _get_compute_chans(self, chans):
        self.compute_chans = chans
        if len(chans) == 1:
            self._fig_chans = chans
        logger.info(f"Selected channels are {chans}")

    def get_compute_params(self):
        freq_band = self._freq_band_le.text().split(' ')
        if len(freq_band) != 2:
            QMessageBox.warning(self, 'Frequency', 'Wrong frequency input!')
            return
        fmin = float(freq_band[0])
        fmax = float(freq_band[1])
        mt_bandwidth = float(self._bandwidth_le.text())
        block_size = int(self._n_jobs_le.text())
        self.compute_params = dict(fmin=fmin, fmax=fmax, method=self.method,
                                   mode='multitaper', mt_bandwidth=mt_bandwidth,
                                   sfreq=self.ieeg.info['sfreq'], mt_adaptive=True,
                                   mt_low_bias=True, block_size=block_size,
                                   faverage=False)
        print(self.compute_params)

    def _compute_con(self):
        logger.info(f'Start Compute Speatral Connectivity using {self.method}')
        self.get_compute_params()
        ieeg = self.ieeg.copy()
        if len(self.compute_chans):
            ieeg.pick_channels(self.compute_chans)
        self._compute_spectral_con_thread = ComputeSpectralConnectivity(ieeg, self.compute_params)
        self._compute_spectral_con_thread.COMPUTE_SIGNAL.connect(self._get_con)
        self._compute_spectral_con_thread.start()

    def _get_con(self, con):
        self.con = con
        self.freqs = con.freqs
        QMessageBox.information(self, 'Connectivity', 'Finish computing connectivity!')

    def _plot_heatmap(self):
        con = self.con.get_data('dense')
        con = con.transpose((2, 0, 1))
        freqs = [f'{round(freq, 2)}' for freq in self.freqs]
        if self._average_freq_cbx.currentText() == 'True':
            fmin = self.compute_params['fmin']
            fmax = self.compute_params['fmax']
            con = con.mean(axis=0)
            mask = np.zeros_like(con)
            mask[np.triu_indices_from(mask)] = True
            create_heatmap(con, ch_names=self.compute_chans, mask=mask,
                           title=f'Spectral Connectivity ({self.method})',
                           title_item=f'{fmin}-{fmax}', unit='Hz')

        else:
            mask = np.zeros_like(con[0])
            mask[np.triu_indices_from(mask)] = True
            create_heatmap(con, ch_names=self.compute_chans, mask=mask,
                           title=f'Spectral Connectivity ({self.method})',
                           title_item=freqs, unit='Hz')

    def _plot_circular(self):
        con = self.con.get_data('dense').mean(2)
        node_colors = []
        ch_names = self.compute_chans
        for index in range(len(ch_names)):
            node_colors.append(color[index % 20])
        node_angles = circular_layout(ch_names, ch_names, start_pos=90, group_sep=5,
                                      group_boundaries=[0, len(ch_names) // 2])
        # fig = plt.figure(figsize=(8, 8), facecolor='black')
        threshold = float(self._threshold_le.text())
        n_lines = len(con[con > threshold])
        if n_lines == 0:
            QMessageBox.warning(self, 'Connectivity', 'Threshold is too big. No Lines plotted！')
            return
        plot_connectivity_circle(con, ch_names, n_lines=n_lines, node_angles=node_angles,
                                 node_colors=node_colors,
                                 title=f'Spectral Connectivity {self.method}')

    def _plot_3d(self):
        pass


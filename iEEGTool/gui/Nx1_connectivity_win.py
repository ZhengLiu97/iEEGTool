# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：1xN_connectivity_win.py
@Author  ：Barry
@Date    ：2022/3/12 16:59 
"""
import mne
import numpy as np
import pandas as pd

from mne.io import BaseRaw
from mne.viz import circular_layout
from matplotlib import pyplot as plt
from mne_connectivity.viz import plot_connectivity_circle
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QButtonGroup, QMessageBox, \
                            QAbstractItemView

from gui.Nx1_connectivity_ui import Ui_MainWindow
from gui.list_win import ItemSelectionWin
from utils.process import make_epoch
from utils.log_config import create_logger
from utils.thread import ComputeSpectralConnectivity
from viz.figure import create_nx1_lineplot

logger = create_logger(filename='iEEGTool.log')


class Nx1SpectraConWin(QMainWindow, Ui_MainWindow):

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

        self.chanA = []
        self.chanB = []
        self.chan_pairs = []

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
        self._select_chanA_btn.clicked.connect(self._select_chanA)
        self._select_chanB_btn.clicked.connect(self._select_chanB)
        self._compute_btn.clicked.connect(self._compute_con)
        self._lineplot_btn.clicked.connect(self._plot_line)

    def _select_chanA(self):
        self._select_chanA_win = ItemSelectionWin(self.ieeg.ch_names)
        self._select_chanA_win._list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self._select_chanA_win.SELECTION_SIGNAL.connect(self._get_chanA)
        self._select_chanA_win.show()

    def _select_chanB(self):
        self._select_chanB_win = ItemSelectionWin(self.ieeg.ch_names)
        self._select_chanB_win.SELECTION_SIGNAL.connect(self._get_chanB)
        self._select_chanB_win.show()

    def _get_chanA(self, chanA):
        self.chanA = chanA
        logger.info(f"Selected channel A is {chanA}")

    def _get_chanB(self, chanB):
        self.chanB = chanB
        logger.info(f"Selected channel B is {chanB}")

    def get_compute_params(self):
        freq_band = self._freq_band_le.text().split(' ')
        if len(freq_band) != 2:
            QMessageBox.warning(self, 'Frequency', 'Wrong frequency input!')
            return
        fmin = float(freq_band[0])
        fmax = float(freq_band[1])
        mt_bandwidth = float(self._bandwidth_le.text())
        block_size = int(self._n_jobs_le.text())

        if self.chanA[0] in self.chanB:
            self.chanB.remove(self.chanA[0])

        indiceA = [self.ieeg.ch_names.index(ch) for ch in self.chanA] * len(self.chanB)
        indiceB = [self.ieeg.ch_names.index(ch) for ch in self.chanB]

        indices = (indiceA, indiceB)

        # for index in range(len(self.chanB)):
        #     ch_pairs = f'{self.chanA[0]} {self.chanB[index]}'
        #     self.chan_pairs.append(ch_pairs)
        # print(f'Channel pairs are {self.chan_pairs}')

        self.compute_params = dict(fmin=fmin, fmax=fmax, method=self.method,
                                   indices=indices, mode='multitaper',
                                   mt_bandwidth=mt_bandwidth, sfreq=self.ieeg.info['sfreq'],
                                   mt_adaptive=True, mt_low_bias=True, block_size=block_size,
                                   faverage=False)
        print(self.compute_params)

    def _compute_con(self):
        logger.info(f'Start Compute Spectral Connectivity using {self.method}')
        self.get_compute_params()
        if len(self.chanB) < 0 or len(self.chanA) == 0:
            return

        self._select_chan_pair_cbx.addItem('All pairs')
        self._select_chan_pair_cbx.addItems(self.chanB)

        ieeg = self.ieeg.copy()
        self._compute_spectral_con_thread = ComputeSpectralConnectivity(ieeg, self.compute_params)
        self._compute_spectral_con_thread.COMPUTE_SIGNAL.connect(self._get_con)
        self._compute_spectral_con_thread.start()

    def _get_con(self, con):
        self.con = con
        self.freqs = con.freqs
        QMessageBox.information(self, 'Connectivity', 'Finish computing connectivity!')

    def _plot_line(self):
        if self.con is not None:
            con = self.con.get_data()
            ch_all = self.chanA + self.chanB
            chA_data = self.ieeg.copy().pick_channels(self.chanA).get_data()[0, :, :].reshape(1, -1)
            chB_data = self.ieeg.copy().pick_channels(self.chanB).get_data()[0, :, :].reshape(len(self.chanB), -1)
            ieeg_data = np.concatenate((chA_data, chB_data), axis=0)

            ieeg_df = pd.DataFrame(dict(zip(ch_all, ieeg_data)), index=self.ieeg.times)

            con_df = pd.DataFrame(dict(zip(self.chanB, con)), index=self.con.freqs)

            pair = self._select_chan_pair_cbx.currentText()
            step = int(self._freq_step_le.text())
            if pair == 'All pairs':
                source = self.chanA[0]
                create_nx1_lineplot(None, con_df[::step], source)
            else:
                source = self.chanA[0]
                target = pair
                create_nx1_lineplot(ieeg_df[[source, target]][::step] * 1e3,
                                    con_df[[target]][::step], source)
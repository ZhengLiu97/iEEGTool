# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：psd_multitaper_win.py
@Author  ：Barry
@Date    ：2022/3/10 0:39 
"""
import mne
import numpy as np

from mne.io import BaseRaw
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QButtonGroup, QMessageBox

from gui.psd_multitaper_ui import Ui_MainWindow
from gui.list_win import ItemSelectionWin
from utils.process import make_epoch
from utils.log_config import create_logger
from utils.thread import ComputePSD
from viz.psd import plot_psd

logger = create_logger(filename='iEEGTool.log')


class MultitaperPSDWin(QMainWindow, Ui_MainWindow):

    def __init__(self, ieeg):
        super().__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('Power Spectral Density')

        if isinstance(ieeg, BaseRaw):
            ieeg = make_epoch(ieeg)
        self.ieeg = ieeg

        fmin = str(int(ieeg.info['highpass']))
        fmax = str(int(ieeg.info['lowpass']))
        fmin = '0.1' if fmin == '0' else fmin
        self._freq_band_le.setText(fmin + ' ' + fmax)

        self._compute_chans = ieeg.ch_names
        self._fig_chans = None

        self.compute_params = {}
        self.fig_params = {}

        self.psd = None
        self.freqs = None

        self._slot_connection()

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _slot_connection(self):
        self._select_chan_btn.clicked.connect(self._select_chan)
        self._compute_btn.clicked.connect(self._compute_psd)
        self._plot_btn.clicked.connect(self._plot_psd)

    def _select_chan(self):
        self._select_chan_win = ItemSelectionWin(self.ieeg.ch_names)
        self._select_chan_win.SELECTION_SIGNAL.connect(self._get_compute_chans)
        self._select_chan_win.show()

    def _get_compute_chans(self, chans):
        self._compute_chans = chans
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
        bandwidth = float(self._bandwidth_le.text())
        n_jobs = int(self._n_jobs_le.text())

        self.compute_params['fmin'] = fmin
        self.compute_params['fmax'] = fmax
        self.compute_params['bandwidth'] = bandwidth
        self.compute_params['n_jobs'] = n_jobs
        self.compute_params['adaptive'] = True
        self.compute_params['low_bias'] = True

    def get_fig_params(self):
        average = self._average_chs_cbx.currentText()
        self.fig_params['average'] = True if average == 'True' else False
        log = self._log_trans_cbx.currentText()
        self.fig_params['log'] = True if log == 'True' else False

    def _compute_psd(self):
        logger.info('Start computing PSD using Multitaper')
        self.get_compute_params()
        ieeg = self.ieeg.copy()
        if len(self._compute_chans):
            ieeg.pick_channels(self._compute_chans)
        self._compute_psd_multitaper_thread = ComputePSD(ieeg, compute_method='multitaper',
                                                         params=self.compute_params)
        self._compute_psd_multitaper_thread.PSD_SIGNAL.connect(self._get_psd)
        self._compute_psd_multitaper_thread.start()

    def _get_psd(self, result):
        QMessageBox.information(self, 'PSD', 'Finish computing PSD!')
        self.psd = result['psd']
        self.freqs = result['freqs']

    def _plot_psd(self):
        if self.psd is not None:
            self.get_fig_params()
            plot_psd(self.ieeg.info, self.psd.copy(), self.freqs,
                     self.fig_params['log'], self.fig_params['average'], method='Multitaper')
# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：csd_fourier_win.py
@Author  ：Barry
@Date    ：2022/3/11 18:51 
"""
import mne
import numpy as np

from mne.io import BaseRaw
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QButtonGroup, QMessageBox

from gui.csd_fourier_ui import Ui_MainWindow
from gui.list_win import ItemSelectionWin
from utils.process import make_epoch
from utils.log_config import create_logger
from utils.thread import ComputePSD
from viz.figure import create_heatmap

logger = create_logger(filename='iEEGTool.log')


class FourierCSDWin(QMainWindow, Ui_MainWindow):

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
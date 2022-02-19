# -*- coding: UTF-8 -*-
'''
@Project ：iEEGTool 
@File    ：fir_filter_win.py
@Author  ：Barry
@Date    ：2022/2/19 3:24 
'''
import numpy as np

from PyQt5.QtWidgets import QMainWindow, QMessageBox, QDesktopWidget
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIntValidator

from gui.fir_filter_ui import Ui_MainWindow
from utils.thread import FIRFilter

class FIRFilterWin(QMainWindow, Ui_MainWindow):
    IEEG_SIGNAL = pyqtSignal(object)

    def __init__(self, ieeg):
        super(FIRFilterWin, self).__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('FIR Filter')

        self.ieeg = ieeg

        int_validator = QIntValidator()
        self._lfreq_le.setValidator(int_validator)
        self._hfreq_le.setValidator(int_validator)
        self._notch_freq_le.setValidator(int_validator)
        self._notch_level_le.setValidator(int_validator)
        self._njobs_le.setValidator(int_validator)

        self._ok_btn.clicked.connect(self._run_filter)
        self._cancel_btn.clicked.connect(self.close)

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _run_filter(self):
        params = dict()

        lfreq = self._lfreq_le.text()
        hfreq = self._hfreq_le.text()
        notch_freq = self._notch_freq_le.text()
        notch_level = self._notch_level_le.text()
        n_jobs = self._njobs_le.text()
        phase = self._phase_cbx.currentText()
        window = self._win_type_cbx.currentText()

        lfreq = int(lfreq) if len(lfreq) else None
        hfreq = int(hfreq) if len(hfreq) else None
        notch_freq = int(notch_freq) if len(notch_freq) else None
        notch_level = int(notch_level) if len(notch_level) else None
        n_jobs = int(n_jobs) if len(n_jobs) else 1

        if notch_freq is not None and notch_level is None:
            notch_level = 1
        if notch_freq is not None:
            notch_freqs = np.arange(notch_freq, (notch_freq * notch_level) + 1, notch_freq)
        else:
            notch_freqs = None

        params['n_jobs'] = n_jobs
        params['phase'] = phase
        params['fir_window'] = window

        self.filter_thread = FIRFilter(self.ieeg, lfreq, hfreq, notch_freqs, params)
        self.filter_thread.IEEG_SIGNAL.connect(self._get_filtered_ieeg)
        self.filter_thread.start()

    def _get_filtered_ieeg(self, ieeg):
        self.close()
        self.IEEG_SIGNAL.emit(ieeg)
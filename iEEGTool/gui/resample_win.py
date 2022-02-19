# -*- coding: UTF-8 -*-
'''
@Project ：iEEGTool 
@File    ：resample_win.py
@Author  ：Barry
@Date    ：2022/2/18 20:29 
'''
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget
from PyQt5.QtCore import pyqtSignal

from gui.resample_ui import Ui_MainWindow
from utils.log_config import create_logger
from utils.thread import ResampleiEEG

logger = create_logger(filename='iEEGTool.log')


class ResampleWin(QMainWindow, Ui_MainWindow):
    RESAMPLE_SIGNAL = pyqtSignal(object)

    def __init__(self, ieeg):
        super(ResampleWin, self).__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('Resample iEEG')

        self.ieeg = ieeg

        self._start_btn.clicked.connect(self._resample_ieeg)
        self._start_btn.clicked.connect(self.close)

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _resample_ieeg(self):
        if self._resampling_rate_le.text().isdigit():
            resampling_rate = float(self._resampling_rate_le.text())
            logger.info(f"Start resampling iEEG to {resampling_rate}Hz")
            self._resample_thread = ResampleiEEG(self.ieeg, resampling_rate)
            self._resample_thread.RESAMPLE_SIGNAL.connect(self._return_ieeg)
            self._resample_thread.start()

    def _return_ieeg(self, ieeg):
        self.RESAMPLE_SIGNAL.emit(ieeg)






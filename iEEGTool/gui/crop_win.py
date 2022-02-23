# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool
@File    ：crop_win.py
@Author  ：Barry
@Date    ：2022/2/18 21:00
"""
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QDesktopWidget
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QDoubleValidator

from gui.crop_ui import Ui_MainWindow
from utils.log_config import create_logger
from utils.numeric import isfloat

logger = create_logger(filename='iEEGTool.log')


class CropWin(QMainWindow, Ui_MainWindow):
    CROP_SIGNAL = pyqtSignal(float, float)

    def __init__(self, tmin, tmax):
        super(CropWin, self).__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('Crop iEEG')

        self.tmax = tmax

        self._tmin_le.setText(str(tmin))
        self._tmax_le.setText(str(tmax))

        validator = QDoubleValidator()
        self._tmin_le.setValidator(validator)
        self._tmax_le.setValidator(validator)

        self._start_btn.clicked.connect(self._return_params)

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _return_params(self):
        tmin = self._tmin_le.text()
        tmax = self._tmax_le.text()
        if isfloat(tmin) and isfloat(tmax):
            tmin, tmax = float(tmin), float(tmax)
            if tmin < tmax:
                tmax = self.tmax if tmax > self.tmax else tmax
                if tmin > self.tmax:
                    QMessageBox.warning(self, 'Input', f'tmin must be less than or '
                                                       f'equal to the max time {self.tmax}!')
                else:
                    self.CROP_SIGNAL.emit(tmin, tmax)
                    self.close()
            else:
                QMessageBox.warning(self, 'Input', 'tmin should be smaller than tmax!')
        else:
            QMessageBox.warning(self, 'Input', 'time value can only be numeric!')
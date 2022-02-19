# -*- coding: UTF-8 -*-
'''
@Project ：iEEGTool 
@File    ：compute_ei_win.py
@Author  ：Barry
@Date    ：2022/2/20 1:53 
'''
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QDesktopWidget
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QIcon, QFont, QPixmap

from gui.compute_ei_ui import Ui_MainWindow
from utils.log_config import create_logger

logger = create_logger(filename='iEEGTool.log')


class EIWin(QMainWindow, Ui_MainWindow):

    def __init__(self, ieeg):
        super(EIWin, self).__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('Epileptogenicity Index')

        self.ieeg = ieeg

        int_validator = QIntValidator()
        self._lfreq_low_le.setValidator(int_validator)
        self._lfreq_high_le.setValidator(int_validator)
        self._hfreq_low_le.setValidator(int_validator)
        self._hfreq_high_le.setValidator(int_validator)
        self._decay_le.setValidator(int_validator)
        self._duration_le.setValidator(int_validator)

        float_validator = QDoubleValidator()
        self._win_le.setValidator(float_validator)
        self._step_le.setValidator(float_validator)
        self._bias_le.setValidator(float_validator)
        self._threshold_le.setValidator(float_validator)
        self._ez_threshold_le.setValidator(float_validator)

        self._compute_btn.clicked.connect(self._compute_ei)
        self._display_table_btn.clicked.connect(self._display_table)

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _set_icon(self):
        import_icon = QIcon()
        import_icon.addPixmap(QPixmap("icon/mri.svg"), QIcon.Normal, QIcon.Off)
        self._import_ei_action.setIcon(import_icon)

        save_icon = QIcon()
        save_icon.addPixmap(QPixmap("icon/save.svg"), QIcon.Normal, QIcon.Off)
        self._save_excel_action.setIcon(save_icon)

    @staticmethod
    def _compute_ei():
        logger.info("Start compute EI")

    @staticmethod
    def _display_table():
        logger.info("Display EI Table!")


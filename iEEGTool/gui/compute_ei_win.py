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
from utils.thread import ComputeEI

logger = create_logger(filename='iEEGTool.log')


class EIWin(QMainWindow, Ui_MainWindow):

    def __init__(self, ieeg):
        super(EIWin, self).__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('Epileptogenicity Index')
        self._set_icon()

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

        # self._viz_ieeg_action.triggered.connect(self._viz_ieeg)

        self._compute_btn.clicked.connect(self._compute_ei)
        self._display_table_btn.clicked.connect(self._display_table)

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _set_icon(self):
        import_icon = QIcon()
        import_icon.addPixmap(QPixmap("icon/folder.svg"), QIcon.Normal, QIcon.Off)
        self._import_ei_action.setIcon(import_icon)

        save_icon = QIcon()
        save_icon.addPixmap(QPixmap("icon/save.svg"), QIcon.Normal, QIcon.Off)
        self._save_excel_action.setIcon(save_icon)

        wave_icon = QIcon()
        wave_icon.addPixmap(QPixmap("icon/square-wave.svg"), QIcon.Normal, QIcon.Off)
        self._viz_ieeg_action.setIcon(wave_icon)

        bar_icon = QIcon()
        bar_icon.addPixmap(QPixmap("icon/bar-chart.svg"), QIcon.Normal, QIcon.Off)
        self._bar_chart_action.setIcon(bar_icon)

        brain_icon = QIcon()
        brain_icon.addPixmap(QPixmap("icon/brain.svg"), QIcon.Normal, QIcon.Off)
        self._3d_vis_action.setIcon(brain_icon)

        help_icon = QIcon()
        help_icon.addPixmap(QPixmap("icon/help.svg"), QIcon.Normal, QIcon.Off)
        self._help_action.setIcon(help_icon)

    def _compute_ei(self):
        window = float(self._win_le.text())
        step = float(self._step_le.text())
        low = [float(self._lfreq_low_le.text()), float(self._lfreq_high_le.text())]
        high = [float(self._hfreq_low_le.text()), float(self._hfreq_high_le.text())]
        bias = float(self._bias_le.text())
        threshold = float(self._threshold_le.text())
        tau = float(self._decay_le.text())
        H = float(self._duration_le.text())

        params = {'window': window, 'step': step, 'low': low, 'high': high, 'bias': bias,
                  'threshold': threshold, 'tau': tau, 'H': H}

        logger.info("Start computing EI")
        logger.info(f"Epileptogenicity Index params are {params}")

        self._compute_ei_thread = ComputeEI(self.ieeg, params)
        self._compute_ei_thread.EI_SIGNAL.connect(self._get_ei)
        self._compute_ei_thread.start()

    def _get_ei(self, result):
        logger.info("Finish computing EI")
        ei = result[0]
        U_n = result[1]


    @staticmethod
    def _display_table():
        logger.info("Display EI Table!")


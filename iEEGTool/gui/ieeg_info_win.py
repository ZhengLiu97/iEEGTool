# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：ieeg_info_win.py
@Author  ：Barry
@Date    ：2022/2/28 21:39 
"""
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget

from gui.ieeg_info_ui import Ui_MainWindow


class iEEGInfoWin(QMainWindow, Ui_MainWindow):

    def __init__(self, info):
        super(iEEGInfoWin, self).__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('iEEG Information')

        self._epoch_num_lb.setText(str(info['epoch_num']))
        self._ch_num_lb.setText(str(info['ch_num']))
        self._ch_group_lb.setText(str(info['ch_group']))
        self._time_lb.setText(str(info['time']))
        self._sfreq_lb.setText(str(info['sfreq']))
        self._fmin_lb.setText(str(info['fmin']))
        self._fmax_lb.setText(str(info['fmax']))
        self._data_size_lb.setText(str(info['data_size']))

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
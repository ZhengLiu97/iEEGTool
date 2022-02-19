# -*- coding: UTF-8 -*-
'''
@Project ：iEEGTool 
@File    ：info_win.py
@Author  ：Barry
@Date    ：2022/2/18 22:03 
'''
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QDesktopWidget
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIntValidator

from gui.info_ui import Ui_MainWindow
from utils.decorator import safe_event


class InfoWin(QMainWindow, Ui_MainWindow):
    INFO_PARAM = pyqtSignal(dict)

    def __init__(self, info):
        super(InfoWin, self).__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('Subject Information')

        subject_name = info['subject_name']
        age = info['age']
        gender = info['gender']
        self._name_le.setText(subject_name)
        self._age_le.setText(age)
        self._gender_le.setText(gender)

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    @safe_event
    def closeEvent(self, event):
        info = dict()
        info['subject_name'] = self._name_le.text()
        info['age'] = self._age_le.text()
        info['gender'] = self._gender_le.text()
        self.INFO_PARAM.emit(info)


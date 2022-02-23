# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：list_win.py
@Author  ：Barry
@Date    ：2022/2/21 12:50 
"""
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget
from PyQt5.QtCore import pyqtSignal
from gui.list_ui import Ui_MainWindow


class ItemSelectionWin(QMainWindow, Ui_MainWindow):
    SELECTION_SIGNAL = pyqtSignal(list)

    def __init__(self, items):
        super(ItemSelectionWin, self).__init__()
        self.setupUi(self)
        self._center_win()

        self._list_widget.addItems(items)

        self._ok_btn.clicked.connect(self._return_items)

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _return_items(self):
        items_selected = []
        selected_item = self._list_widget.selectedItems()
        items_selected.append([item.text() for item in list(selected_item)])
        items_selected = items_selected[0]
        self.SELECTION_SIGNAL.emit(items_selected)
        self.close()
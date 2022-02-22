# -*- coding: UTF-8 -*-
'''
@Project ：iEEGTool 
@File    ：table_win.py
@Author  ：Barry
@Date    ：2022/2/21 13:13 
'''
from PyQt5.QtWidgets import QMainWindow, QApplication, QDesktopWidget, \
                            QAbstractItemView, QHeaderView, QTableWidgetItem
from PyQt5.QtCore import pyqtSignal, Qt
from gui.table_ui import Ui_MainWindow


class TableWin(QMainWindow, Ui_MainWindow):

    def __init__(self, table):
        super(TableWin, self).__init__()
        self.setupUi(self)
        self._center_win()

        self.table = table

        self._init_tables()

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _init_tables(self):
        table = self.table
        columns = list(table.columns)
        index_len = table.index.stop
        self._table_widget.setColumnCount(len(columns))
        self._table_widget.setRowCount(index_len)
        self._table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        # self._table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table_widget.setHorizontalHeaderLabels(columns)
        table_npy = table.to_numpy()
        for index, info in enumerate(table_npy):
            info = list(info)
            for col, item in enumerate(info):
                if '[' in str(item):
                    item = str(item)[1:-1]
                    item = item.replace(',', '    ')
                item = QTableWidgetItem(str(item))
                self._table_widget.setItem(index, col, item)
        for i in range(table_npy.shape[0]):
            for j in range(table_npy.shape[1]):
                self._table_widget.item(i, j).setTextAlignment(Qt.AlignCenter)

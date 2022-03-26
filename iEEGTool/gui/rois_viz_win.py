# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：rois_viz_win.py
@Author  ：Barry
@Date    ：2022/3/27 0:34 
"""
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QTableWidgetItem, QHeaderView, \
                            QAbstractItemView, QToolTip
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QCursor

from gui.rois_viz_ui import Ui_MainWindow
from viz.surface import check_hemi
from utils.process import get_chan_group
from utils.contacts import is_lh


class ROIsWin(QMainWindow, Ui_MainWindow):
    CLOSE_SIGNAL = pyqtSignal(bool)

    def __init__(self, subject, subjects_dir, ch_info):
        super().__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('ROIs Visualization')

        self.subject = subject
        self.subjects_dir = subjects_dir
        self.ch_info = ch_info

        stats = list(self.ch_info.groupby(by='ROI'))
        rois = [stat[0] for stat in stats]
        chs = [', '.join(list(stat[1]['Channel'])) for stat in stats]

        rois_chs = dict(zip(rois, chs))

        self.roi_info = pd.DataFrame(rois_chs.items(), columns=['ROI', 'Channel'])

        self._info_table.setMouseTracking(True)
        self._info_table.cellEntered.connect(self._show_tooltip)

        self._init_brain(subject, subjects_dir)
        self.add_items2table(self.roi_info)

        # self._slot_connection()

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _init_brain(self, subject, subjects_dir):
        self._plotter.add_brain(subject, subjects_dir, ['lh', 'rh'], 'pial', 0.1)

    def add_items2table(self, ch_info):
        columns = list(ch_info.columns)
        index_len = len(ch_info.index)
        self._info_table.setColumnCount(len(columns))
        self._info_table.setRowCount(index_len)
        self._info_table.setHorizontalHeaderLabels(columns)
        table_npy = ch_info.to_numpy()
        for index, info in enumerate(table_npy):
            info = list(info)
            for col, item in enumerate(info):
                item = QTableWidgetItem(str(item))
                if col == 0:
                    item.setCheckState(Qt.Unchecked)
                item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                self._info_table.setItem(index, col, item)
        for i in range(table_npy.shape[0]):
            for j in range(table_npy.shape[1]):
                self._info_table.item(i, j).setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self._info_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._info_table.setColumnWidth(0, 220)
        self._info_table.horizontalHeader().setStretchLastSection(True)

    def _show_tooltip(self, i, j):
        item = self._info_table.item(i, j).text()
        if len(item) > 20:
            QToolTip.showText(QCursor.pos(), item)

    def closeEvent(self, a0: QCloseEvent) -> None:
        self.CLOSE_SIGNAL.emit(True)
# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：electrodes_viz_win.py
@Author  ：Barry
@Date    ：2022/3/26 18:37 
"""
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QTableWidgetItem, QHeaderView, \
                            QAbstractItemView, QToolTip
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QCursor

from gui.electrodes_viz_ui import Ui_MainWindow
from viz.surface import check_hemi
from utils.process import get_chan_group


class ElectrodesWin(QMainWindow, Ui_MainWindow):
    CLOSE_SIGNAL = pyqtSignal(bool)

    def __init__(self, subject, subjects_dir, ch_info, parcellation):
        super().__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('Electrodes Visualization')

        self.subject = subject
        self.subjects_dir = subjects_dir
        self.ch_info = ch_info
        self.parcellation = parcellation

        ch_names = ch_info['Channel'].to_list()
        self.ch_group = get_chan_group(ch_names)
        self._group_cbx.addItems(self.ch_group.keys())

        self._info_table.setMouseTracking(True)
        self._info_table.cellEntered.connect(self._show_tooltip)

        self._init_brain(subject, subjects_dir)
        self._init_ch_info(ch_info, parcellation)
        self._init_electrodes(ch_info)

        self._slot_connection()

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _init_brain(self, subject, subjects_dir):
        self._plotter.add_brain(subject, subjects_dir, ['lh', 'rh'], 'pial', 0.1)

    def _init_ch_info(self, ch_info, parcellation):
        if parcellation is not None:
            ch_info = ch_info[['Channel', parcellation]]
        else:
            ch_info = ch_info[['Channel', 'x', 'y', 'z']]

        first_gp = list(self.ch_group.keys())[0]
        init_info = ch_info[ch_info['Channel'].isin(self.ch_group[first_gp])]
        self.add_items2table(init_info)

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
                self._info_table.item(i, j).setTextAlignment(Qt.AlignCenter)

        self._info_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._info_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._info_table.horizontalHeader().setStretchLastSection(True)

    def _init_electrodes(self, ch_info):
        ch_names = ch_info['Channel'].to_list()
        ch_coords = ch_info[['x', 'y', 'z']].to_numpy()
        self._plotter.add_chs(ch_names, ch_coords)

    def _slot_connection(self):
        self._brain_gp.clicked.connect(self._enable_brain_viz)
        self._transparency_slider.valueChanged.connect(self._set_brain_transparency)
        self._hemi_cbx.currentTextChanged.connect(self._set_brain_hemi)
        self._group_cbx.currentTextChanged.connect(self._change_info)

    def _enable_brain_viz(self):
        viz = self._brain_gp.isChecked()
        hemi = check_hemi(self._hemi_cbx.currentText())
        self._plotter.enable_brain_viz(viz, hemi)

    def _set_brain_transparency(self, transparency):
        transparency = float(transparency) / 100
        self._plotter.set_brain_opacity(transparency)

    def _set_brain_hemi(self):
        hemi = check_hemi(self._hemi_cbx.currentText())
        self._plotter.set_brain_hemi(hemi)

    def _change_info(self):
        group = self._group_cbx.currentText()
        ch_names = self.ch_group[group]
        print(ch_names)
        if self.parcellation is not None:
            ch_info = self.ch_info[['Channel', self.parcellation]]
        else:
            ch_info = self.ch_info[['Channel', 'x', 'y', 'z']]

        info = ch_info[ch_info['Channel'].isin(ch_names)]
        self.add_items2table(info)

    def _show_tooltip(self, i, j):
        item = self._info_table.item(i, j).text()
        if len(item) > 39:
            QToolTip.showText(QCursor.pos(), item)

    def closeEvent(self, a0: QCloseEvent) -> None:
        self.CLOSE_SIGNAL.emit(True)
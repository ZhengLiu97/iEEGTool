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

        self.ch_anatomy = None
        self.chs_wm = []
        self.chs_gm = []
        self.chs_unknown = []

        self.subject = subject
        self.subjects_dir = subjects_dir
        self.ch_info = ch_info
        self.parcellation = parcellation

        self.ch_names = ch_info['Channel'].to_list()
        self.ch_group = get_chan_group(self.ch_names)
        self._group_cbx.addItems(self.ch_group.keys())
        self.group_viz = {group: True for group in self.ch_group}
        self.electrode_viz = {ch_name: True for ch_name in self.ch_names}

        if parcellation is not None:
            self.ch_info_tb = self.ch_info[['Channel', self.parcellation]]
            rois = ch_info[parcellation].to_list()
            self.rois_num = {roi: 0 for roi in rois} # to find if the roi need removing
            issues = ch_info['issue'].to_numpy()
            self.ch_anatomy = dict(zip(self.ch_names, rois))

            ch_names = np.array(self.ch_names)
            self.chs_wm = list(ch_names[issues == 'White'])
            self.chs_gm = list(ch_names[issues == 'Gray'])
            self.chs_unknown = list(ch_names[issues == 'Unknown'])
        else:
            self.ch_info_tb = self.ch_info[['Channel', 'x', 'y', 'z']]

        self._info_table.setMouseTracking(True)
        self._info_table.cellEntered.connect(self._show_tooltip)

        self._init_brain(subject, subjects_dir)
        self._init_ch_info()
        self._init_electrodes(ch_info)

        self._slot_connection()

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _init_brain(self, subject, subjects_dir):
        self._plotter.add_brain(subject, subjects_dir, ['lh', 'rh'], 'pial', 0.1)

    def _init_ch_info(self):
        first_gp = list(self.ch_group.keys())[0]
        init_info = self.ch_info_tb[self.ch_info_tb['Channel'].isin(self.ch_group[first_gp])]
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
        self._electrodes_gp.clicked.connect(self._enable_chs)
        self._group_cbx.currentTextChanged.connect(self._change_info)
        self._select_cbx.currentTextChanged.connect(self._enable_chs_viz)
        self._display_cbx.stateChanged.connect(self._enable_chs_group_viz)

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

    def _enable_chs(self):
        viz = self._electrodes_gp.isChecked()
        self.group_viz = {group: viz for group in self.ch_group}
        self.electrode_viz = {ch_name: viz for ch_name in self.ch_names}
        self._plotter.enable_chs_viz(self.ch_names, viz)
        self._plotter.enable_group_label_viz(self.ch_group, viz)

    def _enable_chs_group_viz(self):
        nviz_chs = []
        viz = self._display_cbx.isChecked()
        group = self._group_cbx.currentText()
        condition = self._select_cbx.currentText()
        if viz != self.group_viz[group]:
            ch_names = self.ch_group[group]
            if viz:
                # if this group is not viz
                # then disable all of them
                # if this group is gonna viz
                # then find which chs need enabling
                # according to the condition
                if condition == 'Gray matter':
                    [nviz_chs.append(ch_name) for ch_name in ch_names
                                    if ch_name not in self.chs_gm]
                elif condition == 'White matter':
                    [nviz_chs.append(ch_name) for ch_name in ch_names
                                    if ch_name not in self.chs_wm]
                print(nviz_chs)
            self._plotter.enable_chs_viz(ch_names, viz)
            if len(nviz_chs):
                nviz = not viz
                self._plotter.enable_chs_viz(nviz_chs, nviz)
            self._plotter.enable_group_label_viz(group, viz)
            self.group_viz[group] = viz
            for ch_name in ch_names:
                self.electrode_viz[ch_name] = viz
            if len(nviz_chs):
                for ch_name in nviz_chs:
                    self.electrode_viz[ch_name] = not viz

    def _enable_chs_viz(self):
        nviz_chs = []
        viz_chs = []
        condition = self._select_cbx.currentText()
        if condition == 'All':
            for group in self.group_viz:
                if self.group_viz[group]:
                    ch_names = self.ch_group[group]
                    viz_chs += ch_names
                    for ch_name in ch_names:
                        self.electrode_viz[ch_name] = True
        elif condition == 'Gray matter':
            for group in self.group_viz:
                if self.group_viz[group]:
                    ch_names = self.ch_group[group]
                    for ch_name in ch_names:
                        if ch_name in self.chs_gm:
                            self.electrode_viz[ch_name] = True
                            viz_chs.append(ch_name)
                        else:
                            self.electrode_viz[ch_name] = False
                            nviz_chs.append(ch_name)
        else:
            for group in self.group_viz:
                if self.group_viz[group]:
                    ch_names = self.ch_group[group]
                    for ch_name in ch_names:
                        if ch_name in self.chs_wm:
                            self.electrode_viz[ch_name] = True
                            viz_chs.append(ch_name)
                        else:
                            self.electrode_viz[ch_name] = False
                            nviz_chs.append(ch_name)
        if len(nviz_chs):
            self._plotter.enable_chs_viz(nviz_chs, False)
        if len(viz_chs):
            self._plotter.enable_chs_viz(viz_chs, True)

    def _change_info(self):
        group = self._group_cbx.currentText()
        ch_names = self.ch_group[group]

        info = self.ch_info_tb[self.ch_info_tb['Channel'].isin(ch_names)]
        self.add_items2table(info)
        viz = self._display_cbx.isChecked()
        if viz != self.group_viz[group]:
            self._display_cbx.setChecked(self.group_viz[group])

    def _show_tooltip(self, i, j):
        item = self._info_table.item(i, j).text()
        if len(item) > 39:
            QToolTip.showText(QCursor.pos(), item)

    def closeEvent(self, a0: QCloseEvent) -> None:
        self.CLOSE_SIGNAL.emit(True)
# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：ei_viz_win.py
@Author  ：Barry
@Date    ：2022/4/1 19:39 
"""
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QTableWidgetItem, QHeaderView, \
                            QAbstractItemView, QToolTip, QColorDialog, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QCursor

from gui.ei_viz_ui import Ui_MainWindow
from viz.surface import check_hemi
from utils.decorator import safe_event
from utils.config import view_dict


class EIWin(QMainWindow, Ui_MainWindow):
    CLOSE_SIGNAL = pyqtSignal(bool)

    def __init__(self, subject, subjects_dir, ch_info, ei_info, seg_name, parcellation):
        super().__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('Epileptogenicity Index')

        self.subject = subject
        self.subjects_dir = subjects_dir
        self.ch_info = ch_info
        self.ei_info = ei_info
        self.seg_name = seg_name
        self.parcellation = parcellation

        self.ch_names = ch_info['Channel'].to_list()

        coords = ch_info[['x', 'y', 'z']].to_numpy()
        self.ch_pos = dict(zip(self.ch_names, coords))

        self.ch_info_tb = self.ch_info[['Channel', 'norm_EI', seg_name]]
        rois = ch_info[seg_name].to_list()
        self.ch_rois = dict(zip(self.ch_names, rois))

        self._init_rois(subject, subjects_dir, set(rois), parcellation)

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
        self._plotter.view_vector(view_dict['front'][0], view_dict['front'][1])

    def _init_rois(self, subject, subjects_dir, rois, aseg):
        self._plotter.add_rois(subject, subjects_dir, rois, aseg)

    def _init_ch_info(self):
        pass

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
        self._plotter.enable_ch_name_viz(ch_names, False)

    def _slot_connection(self):
        self._brain_gp.clicked.connect(self._enable_brain_viz)
        self._transparency_slider.valueChanged.connect(self._set_brain_transparency)
        self._hemi_cbx.currentTextChanged.connect(self._set_brain_hemi)

        self._ei_gp.clicked.connect(self._enable_ei_vz)
        self._electrodes_cbx.clicked.connect(self._enable_electrodes_viz)
        self._ez_cbx.clicked.connect(self._enable_ez_viz)

        self._background_color_action.triggered.connect(self._set_background_color)
        self._brain_color_action.triggered.connect(self._set_brain_color)

        self._screenshot_action.triggered.connect(self._screenshot)

        # cannot use lambda for don't know why
        # but if using lambda to simplify
        # we cannot open the window at the second time
        self._front_action.triggered.connect(self._set_front_view)
        self._back_action.triggered.connect(self._set_back_view)
        self._left_action.triggered.connect(self._set_left_view)
        self._right_action.triggered.connect(self._set_right_view)
        self._top_action.triggered.connect(self._set_top_view)
        self._bottom_action.triggered.connect(self._set_bottom_view)

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

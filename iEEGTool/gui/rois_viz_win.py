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
                            QAbstractItemView, QToolTip, QColorDialog
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QCursor

from gui.rois_viz_ui import Ui_MainWindow
from viz.surface import check_hemi, create_roi_surface
from utils.process import get_chan_group
from utils.contacts import is_lh
from utils.config import view_dict
from utils.decorator import safe_event


class ROIsWin(QMainWindow, Ui_MainWindow):
    CLOSE_SIGNAL = pyqtSignal(bool)

    def __init__(self, subject, subjects_dir, ch_info, parcellation):
        super().__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('ROIs Visualization')

        self.subject = subject
        self.subjects_dir = subjects_dir
        self.ch_info = ch_info
        self.parcellation = parcellation

        stats = list(self.ch_info.groupby(by='ROI'))
        rois = [stat[0] for stat in stats]
        chs = [list(stat[1]['Channel']) for stat in stats]
        chs_tb = [', '.join(list(stat[1]['Channel'])) for stat in stats]

        self.viz_chs = set()

        self.ch_names = ch_info['Channel'].to_list()
        coords = ch_info[['x', 'y', 'z']].to_numpy()
        self.ch_pos = dict(zip(self.ch_names, coords))

        self.rois_chs = dict(zip(rois, chs))
        self.rois_chs_tb = dict(zip(rois, chs_tb))

        self.roi_info = pd.DataFrame(self.rois_chs_tb.items(), columns=['ROI', 'Channel'])

        self._info_table.setMouseTracking(True)
        self._info_table.cellEntered.connect(self._show_tooltip)

        self.roi_viz = {roi: False for roi in rois}

        self._init_rois(subject, subjects_dir, rois)
        self._init_brain(subject, subjects_dir)
        self.add_items2table(self.roi_info)
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

    def _init_electrodes(self, ch_info):
        ch_names = ch_info['Channel'].to_list()
        group = get_chan_group(ch_names).keys()
        ch_coords = ch_info[['x', 'y', 'z']].to_numpy()
        self._plotter.add_chs(ch_names, ch_coords)
        self._plotter.enable_chs_viz(ch_names, False)
        self._plotter.enable_group_label_viz(group, False)

    def _init_rois(self, subject, subjects_dir, rois):
        self._plotter.add_rois(subject, subjects_dir, rois, self.parcellation)
        for roi in rois:
            self._plotter.enable_rois_viz(roi, False)

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

    def _slot_connection(self):
        self._brain_gp.clicked.connect(self._enable_brain_viz)
        self._transparency_slider.valueChanged.connect(self._set_brain_transparency)
        self._hemi_cbx.currentTextChanged.connect(self._set_brain_hemi)
        self._chs_cbx.clicked.connect(self._enable_chs_viz)
        self._chs_name_cbx.stateChanged.connect(self._enable_chs_name_viz)
        self._info_table.cellClicked.connect(self._enable_roi_viz)

        # cannot use lambda for don't know why
        # but if using lambda to simplify
        # we cannot open the window the second time
        self._front_action.triggered.connect(self._set_front_view)
        self._back_action.triggered.connect(self._set_back_view)
        self._left_action.triggered.connect(self._set_left_view)
        self._right_action.triggered.connect(self._set_right_view)
        self._top_action.triggered.connect(self._set_top_view)
        self._bottom_action.triggered.connect(self._set_bottom_view)

        self._background_color_action.triggered.connect(self._set_background_color)
        self._brain_color_action.triggered.connect(self._set_brain_color)

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

    def _enable_roi_viz(self, i, j):
        if j == 0:
            roi = self._info_table.item(i, j).text()
            viz = bool(self._info_table.item(i, 0).checkState())
            self.roi_viz[roi] = viz
            self._plotter.enable_rois_viz(roi, viz)
            if not viz:
                ch_names = self.rois_chs[roi]
                self.viz_chs = self.viz_chs - set(ch_names)
                self._plotter.enable_chs_viz(ch_names, False)

            self._enable_chs_viz()

    def _enable_chs_viz(self):
        viz = self._chs_cbx.isChecked()
        viz_rois = [roi for roi in self.roi_viz if self.roi_viz[roi]]
        ch_names = []
        if len(viz_rois):
            for roi in viz_rois:
                ch_names += self.rois_chs[roi]
        if len(ch_names):
            self.viz_chs = self.viz_chs.union(set(ch_names))
            self._plotter.enable_chs_viz(ch_names, viz)

        if not viz:
            self._chs_name_cbx.setChecked(False)
        self._chs_name_cbx.setEnabled(viz)

    def _enable_chs_name_viz(self):
        viz = self._chs_name_cbx.isChecked()
        # print(self.viz_chs)
        if len(self.viz_chs):
            for ch_name in self.viz_chs:
                coords = self.ch_pos[ch_name]
                self._plotter.enable_ch_name_viz(ch_name, coords, viz)

    def _set_background_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            # 第四位为透明度 color必须在0-1之间
            color = color.getRgbF()[:-1]
            print(f"change brain color to {color}")
            self._plotter.set_background_color(color)

    def _set_brain_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            # 第四位为透明度 color必须在0-1之间
            color = color.getRgbF()[:-1]
            print(f"change brain color to {color}")
            self._plotter.set_brain_color(color)

    def _set_front_view(self):
        view = view_dict['front']
        self._plotter.view_vector(view[0], view[1])

    def _set_back_view(self):
        view = view_dict['back']
        self._plotter.view_vector(view[0], view[1])

    def _set_left_view(self):
        view = view_dict['left']
        self._plotter.view_vector(view[0], view[1])

    def _set_right_view(self):
        view = view_dict['right']
        self._plotter.view_vector(view[0], view[1])

    def _set_top_view(self):
        view = view_dict['top']
        self._plotter.view_vector(view[0], view[1])

    def _set_bottom_view(self):
        view = view_dict['bottom']
        self._plotter.view_vector(view[0], view[1])

    def _show_tooltip(self, i, j):
        item = self._info_table.item(i, j).text()
        if len(item) > 20:
            QToolTip.showText(QCursor.pos(), item)

    @safe_event
    def closeEvent(self, a0: QCloseEvent) -> None:
        self.CLOSE_SIGNAL.emit(True)
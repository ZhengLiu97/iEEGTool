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
                            QAbstractItemView, QToolTip, QColorDialog
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QCursor

from gui.electrodes_viz_ui import Ui_MainWindow
from viz.surface import check_hemi
from utils.process import get_chan_group
from utils.contacts import is_lh, is_gm
from utils.decorator import safe_event
from utils.config import view_dict


class ElectrodesWin(QMainWindow, Ui_MainWindow):
    CLOSE_SIGNAL = pyqtSignal(bool)

    def __init__(self, subject, subjects_dir, ch_info, seg_name, parcellation):
        super().__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('Electrodes Visualization')

        self.ch_rois = None
        self.chs_wm = []
        self.chs_gm = []
        self.chs_unknown = []
        self.group_hemi = {}

        self.subject = subject
        self.subjects_dir = subjects_dir
        self.ch_info = ch_info
        self.seg_name = seg_name
        self.parcellation = parcellation

        self.roi_viz_signal = True

        self.ch_names = ch_info['Channel'].to_list()
        self.ch_group = get_chan_group(self.ch_names)
        self._group_cbx.addItems(self.ch_group.keys())
        self.group_viz = {group: True for group in self.ch_group}
        self.group_roi_viz = {group: False for group in self.ch_group}
        self.electrode_viz = {ch_name: True for ch_name in self.ch_names}

        coords = ch_info[['x', 'y', 'z']].to_numpy()
        self.ch_pos = dict(zip(self.ch_names, coords))

        if seg_name is not None:
            self.ch_info_tb = self.ch_info[['Channel', seg_name]]
            rois = ch_info[seg_name].to_list()
            issues = ch_info['issue'].to_numpy()
            self.ch_rois = dict(zip(self.ch_names, rois))

            self._init_rois(subject, subjects_dir, set(rois), parcellation)
            self.rois_num = {roi: 0 for roi in set(rois)}

            for group in self.ch_group:
                ch_name = self.ch_group[group][0]
                roi = self.ch_rois[ch_name]
                self.group_hemi[group] = 'lh' if is_lh(roi) else 'rh'
            print(self.group_hemi)
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
        self._plotter.view_vector(view_dict['front'][0], view_dict['front'][1])

    def _init_rois(self, subject, subjects_dir, rois, aseg):
        self._plotter.add_rois(subject, subjects_dir, rois, aseg)
        for roi in rois:
            self._plotter.enable_rois_viz(roi, False)

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
        self._elec_hemi_cbx.currentTextChanged.connect(self._set_chs_viz)
        self._roi_cbx.clicked.connect(self._enable_rois_viz)
        self._group_cbx.currentTextChanged.connect(self._change_info)
        self._select_cbx.currentTextChanged.connect(self._set_chs_viz)
        self._display_cbx.stateChanged.connect(self._enable_chs_group_viz)
        self._info_table.cellClicked.connect(self._enable_chs_name_viz)

        self._background_color_action.triggered.connect(self._set_background_color)
        self._brain_color_action.triggered.connect(self._set_brain_color)

        # cannot use lambda for don't know why
        # but if using lambda to simplify
        # we cannot open the window the second time
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

    def _enable_chs(self):
        viz = self._electrodes_gp.isChecked()
        self.group_viz = {group: viz for group in self.ch_group}
        self.electrode_viz = {ch_name: viz for ch_name in self.ch_names}
        self._plotter.enable_chs_viz(self.ch_names, viz)
        self._plotter.enable_group_label_viz(self.ch_group, viz)

    def _set_chs_viz(self):
        nviz_chs = set()
        hemi = check_hemi(self._elec_hemi_cbx.currentText())

        matter = self._select_cbx.currentText()

        viz_group = {group for group in self.group_viz if self.group_viz[group]}

        # find all the group that shouldn't viz
        nviz_group = {group for group in self.group_viz if not self.group_viz[group]}
        if len(viz_group):
            for group in viz_group:
                if self.group_hemi[group] not in hemi:
                    nviz_group.add(group)
        if len(nviz_group):
            for group in nviz_group:
                nviz_chs = nviz_chs.union(set(self.ch_group[group]))
        viz_chs = set(self.ch_names) - nviz_chs
        viz_group = set(self.ch_group.keys()) - nviz_group

        # now we know all the group shouldn't viz and their chs
        # then we find all the viz_chs not in the right matter
        if matter == 'Gray matter':
            for ch in viz_chs:
                if ch not in self.chs_gm:
                    nviz_chs.add(ch)
        elif matter == 'White matter':
            for ch in viz_chs:
                if ch not in self.chs_wm:
                    nviz_chs.add(ch)
        # do it again
        viz_chs = set(self.ch_names) - nviz_chs

        # now we know all the chs should viz and nviz
        # we update their state of viz
        if len(viz_chs):
            self._plotter.enable_chs_viz(viz_chs, True)
            for ch in viz_chs:
                self.electrode_viz[ch] = True
        if len(nviz_chs):
            self._plotter.enable_chs_viz(nviz_chs, False)
            for ch in nviz_chs:
                self.electrode_viz[ch] = False
        self._plotter.enable_group_label_viz(viz_group, True)
        self._plotter.enable_group_label_viz(nviz_group, False)

    def _enable_chs_group_viz(self):
        viz = self._display_cbx.isChecked()
        group = self._group_cbx.currentText()
        self.group_viz[group] = viz
        self._set_chs_viz()

    def _enable_chs_name_viz(self, i, j):
        if j == 0:
            ch_name = self._info_table.item(i, j).text()
            viz = bool(self._info_table.item(i, j).checkState())
            coords = self.ch_pos[ch_name]
            self._plotter.enable_ch_name_viz(ch_name, coords, viz)

    def _enable_rois_viz(self):
        # if self.roi_viz_signal:
        viz = self._roi_cbx.isChecked()
        group = self._group_cbx.currentText()
        ch_names = self.ch_group[group]
        rois = self.ch_info[self.ch_info['Channel'].isin(ch_names)][self.seg_name].to_list()
        rois = set(rois)
        bad_rois = []
        for roi in rois:
            if not is_gm(roi):
                bad_rois.append(roi)
        rois = rois - set(bad_rois)
        self.group_roi_viz[group] = viz
        for roi in rois:
            if viz:
                self.rois_num[roi] += 1
                print(roi, self.rois_num[roi])
                self._plotter.enable_rois_viz(roi, viz)
            else:
                self.rois_num[roi] -= 1
                print(roi, self.rois_num[roi])
                if self.rois_num[roi] == 0:
                    self._plotter.enable_rois_viz(roi, viz)

    def _change_info(self):
        group = self._group_cbx.currentText()
        ch_names = self.ch_group[group]

        roi_viz = self._roi_cbx.isChecked()
        if roi_viz != self.group_roi_viz[group]:
            self._roi_cbx.setChecked(self.group_roi_viz[group])

        info = self.ch_info_tb[self.ch_info_tb['Channel'].isin(ch_names)]
        self.add_items2table(info)
        viz = self._display_cbx.isChecked()
        if viz != self.group_viz[group]:
            self._display_cbx.setChecked(self.group_viz[group])

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
        if len(item) > 39:
            QToolTip.showText(QCursor.pos(), item)

    @safe_event
    def closeEvent(self, a0: QCloseEvent) -> None:
        self.CLOSE_SIGNAL.emit(True)
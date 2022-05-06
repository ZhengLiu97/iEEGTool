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
from viz.surface import check_hemi, create_chs_sphere
from utils.decorator import safe_event
from utils.config import view_dict, contact_kwargs, text_kwargs


class ElectrodesWin(QMainWindow, Ui_MainWindow):
    CLOSE_SIGNAL = pyqtSignal(bool)

    def __init__(self, subject, subjects_dir, threshold, ez_chs,
                 ch_info, ei_info, mri_path):
        super().__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('Epileptogenicity Index')

        self.subject = subject
        self.subjects_dir = subjects_dir
        self.ez_chs = ez_chs
        self.ch_info = ch_info  # anatomy
        self.ei_info = ei_info  # EI value
        self.mri_path = mri_path

        self.threshold = threshold

        self.ch_names = ch_info['Channel'].to_list()

        coords = ch_info[['x', 'y', 'z']].to_numpy()
        self.ch_pos = dict(zip(self.ch_names, coords))

        self.ei_info_tb = self.ei_info[['Channel', 'norm_EI', 'ROI']].\
                                       sort_values(by='norm_EI', ascending=False)

        self.ez = set(ch_info[ch_info['Channel'].isin(ez_chs)]['ROI'].to_list())
        self._init_rois(mri_path)

        self._info_table.setMouseTracking(True)
        self._info_table.cellEntered.connect(self._show_tooltip)

        self.actors = {}

        self._init_brain(subject, subjects_dir)
        self.add_items2table(self.ei_info_tb)

        self._init_electrodes()

        self._slot_connection()

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _init_brain(self, subject, subjects_dir):
        self._plotter.add_brain(subject, subjects_dir, ['lh', 'rh'], 'pial', 0.1)
        self._plotter.view_vector(view_dict['front'][0], view_dict['front'][1])

    def _init_rois(self, mri_path):
        self._plotter.add_rois(self.ez, mri_path)
        self._plotter.add_rois_text(self.ez)
        for roi in self.ez:
            self._plotter.enable_rois_viz(roi, False)
        text_actors = self._plotter.text_actors
        [text_actors[actor].SetVisibility(False) for actor in text_actors]

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
                item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                self._info_table.setItem(index, col, item)
        for i in range(table_npy.shape[0]):
            for j in range(table_npy.shape[1]):
                self._info_table.item(i, j).setTextAlignment(Qt.AlignCenter)

        self._info_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._info_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._info_table.horizontalHeader().setStretchLastSection(True)

    def _init_electrodes(self):
        all_chs = self.ei_info['Channel'].to_list()

        ez_chs = self.ez_chs
        ez_df = self.ch_info.copy()
        ez_df = ez_df[ez_df['Channel'].isin(ez_chs)]
        ez_pos = ez_df[['x', 'y', 'z']].to_numpy()
        ez_spheres = create_chs_sphere(ez_pos, radius=1.5)
        ch_spheres = dict(zip(ez_chs, ez_spheres))
        for ch_name in ch_spheres:
            self.actors[ch_name] = self._plotter.add_mesh(ch_spheres[ch_name], name=ch_name,
                                                          color='r', **contact_kwargs)
            self.actors[f'{ch_name} name'] = self._plotter.add_point_labels(self.ch_pos[ch_name] + 1,
                                                                   [ch_name], name=f'{ch_name} name',
                                                                   **text_kwargs)

        nez_chs = set(all_chs) - set(ez_chs)
        nez_df = self.ch_info.copy()
        nez_pos = nez_df[nez_df['Channel'].isin(nez_chs)][['x', 'y', 'z']].to_numpy()
        nez_spheres = create_chs_sphere(nez_pos, radius=0.8)
        nch_spheres = dict(zip(nez_chs, nez_spheres))
        for ch_name in nch_spheres:
            self.actors[ch_name] = self._plotter.add_mesh(nch_spheres[ch_name], name=ch_name,
                                                          color='b', **contact_kwargs)
        [self.actors[actor].SetVisibility(False) for actor in self.actors]

    def _slot_connection(self):
        self._brain_gp.clicked.connect(self._enable_brain_viz)
        self._transparency_slider.valueChanged.connect(self._set_brain_transparency)
        self._hemi_cbx.currentTextChanged.connect(self._set_brain_hemi)

        self._ei_gp.clicked.connect(self._enable_ei_vz)
        self._electrodes_cbx.stateChanged.connect(self._enable_electrodes_viz)
        self._ez_cbx.stateChanged.connect(self._enable_ez_viz)

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

    def _enable_ei_vz(self):
        viz = self._ei_gp.isChecked()
        self._electrodes_cbx.setEnabled(viz)
        self._ez_cbx.setEnabled(viz)

    def _enable_electrodes_viz(self):
        viz = self._electrodes_cbx.isChecked()
        [self.actors[actor].SetVisibility(viz) for actor in self.actors]

    def _enable_ez_viz(self):
        viz = self._ez_cbx.isChecked()
        rois = self.ez
        [self._plotter.enable_rois_viz(roi, viz) for roi in rois]
        text_actors = self._plotter.text_actors
        [text_actors[actor].SetVisibility(viz) for actor in text_actors]

    def _screenshot(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Screenshot', filter="Screenshot (*..jpeg)")
        if len(fname):
            self._plotter.screenshot(fname)

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

    def closeEvent(self, a0: QCloseEvent) -> None:
        self.CLOSE_SIGNAL.emit(True)

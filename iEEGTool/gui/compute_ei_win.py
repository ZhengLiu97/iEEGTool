# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：compute_ei_win.py
@Author  ：Barry
@Date    ：2022/2/20 1:53 
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from collections import OrderedDict
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QDesktopWidget, QFileDialog
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QIcon, QFont, QPixmap

from gui.compute_ei_ui import Ui_MainWindow
from gui.ei_viz_win import ElectrodesWin
from gui.list_win import ItemSelectionWin
from gui.table_win import TableWin
from utils.log_config import create_logger
from utils.thread import ComputeEI
from utils.process import get_chan_group
from utils.config import color
from utils.contacts import reorder_chs_df
from utils.decorator import safe_event

logger = create_logger(filename='iEEGTool.log')


class EIWin(QMainWindow, Ui_MainWindow):
    ANATOMY_SIGNAL = pyqtSignal(str)

    def __init__(self, ieeg, subject, subjects_dir, anatomy=None, seg_name=None, parcellation=None):
        super().__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('Epileptogenicity Index')
        self._set_icon()

        self.ieeg = ieeg
        self.subject = subject
        self.subjects_dir = subjects_dir
        self.anatomy = anatomy
        self.seg_name = seg_name
        self.parcellation = parcellation
        self.chans = ieeg.ch_names
        self.ei = None
        self.ei_anatomy = pd.DataFrame()

        self._ei_table_win = None
        self._3d_viz_win = None

        int_validator = QIntValidator()
        self._lfreq_low_le.setValidator(int_validator)
        self._lfreq_high_le.setValidator(int_validator)
        self._hfreq_low_le.setValidator(int_validator)
        self._hfreq_high_le.setValidator(int_validator)
        self._decay_le.setValidator(int_validator)
        self._duration_le.setValidator(int_validator)

        float_validator = QDoubleValidator()
        self._win_le.setValidator(float_validator)
        self._step_le.setValidator(float_validator)
        self._bias_le.setValidator(float_validator)
        self._threshold_le.setValidator(float_validator)
        self._ez_threshold_le.setValidator(float_validator)

        self._load_anatomy_action.triggered.connect(self._load_anatomy)
        self._viz_ieeg_action.triggered.connect(self._viz_ieeg)
        self._bar_chart_action.triggered.connect(self._plot_ei_barchart)
        self._3d_vis_action.triggered.connect(self._ez_viz)

        self._select_chan_btn.clicked.connect(self._select_chans)
        self._compute_btn.clicked.connect(self._compute_ei)
        self._display_table_btn.clicked.connect(self._display_table)
        self._save_excel_action.triggered.connect(self._save_excel)

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _set_icon(self):
        anatomy_icon = QIcon()
        anatomy_icon.addPixmap(QPixmap("icon/coordinate.svg"), QIcon.Normal, QIcon.Off)
        self._load_anatomy_action.setIcon(anatomy_icon)

        import_icon = QIcon()
        import_icon.addPixmap(QPixmap("icon/folder.svg"), QIcon.Normal, QIcon.Off)
        self._import_ei_action.setIcon(import_icon)

        save_icon = QIcon()
        save_icon.addPixmap(QPixmap("icon/save.svg"), QIcon.Normal, QIcon.Off)
        self._save_excel_action.setIcon(save_icon)

        wave_icon = QIcon()
        wave_icon.addPixmap(QPixmap("icon/square-wave.svg"), QIcon.Normal, QIcon.Off)
        self._viz_ieeg_action.setIcon(wave_icon)

        bar_icon = QIcon()
        bar_icon.addPixmap(QPixmap("icon/bar-chart.svg"), QIcon.Normal, QIcon.Off)
        self._bar_chart_action.setIcon(bar_icon)

        brain_icon = QIcon()
        brain_icon.addPixmap(QPixmap("icon/brain.svg"), QIcon.Normal, QIcon.Off)
        self._3d_vis_action.setIcon(brain_icon)

        help_icon = QIcon()
        help_icon.addPixmap(QPixmap("icon/help.svg"), QIcon.Normal, QIcon.Off)
        self._help_action.setIcon(help_icon)

    def _load_anatomy(self):
        self.ANATOMY_SIGNAL.emit('EI')

    def set_anatomy(self, subject, subjects_dir, anatomy):
        self.subject = subject
        self.subjects_dir = subjects_dir
        self.ei_anatomy = pd.DataFrame()
        if self.ei is not None:
            ch_names = self.ei['Channel'].to_list()
        else:
            ch_names = self.chans

        anatomy = anatomy[anatomy['Channel'].isin(ch_names)]

        ana_ch = anatomy['Channel'].to_list()
        extra_chans = list(set(ch_names).difference(set(ana_ch)))
        print('Extra channels are: ', extra_chans)
        # reorder the anatomy df using ei df
        # anatomy['Channel'] = anatomy['Channel'].astype('category').cat.set_categories(ch_names)
        # anatomy = anatomy.sort_values(by=['Channel'], ascending=True)

        chs = anatomy['Channel'].to_list()
        rois = anatomy[self.seg_name].to_list()
        x = anatomy['x'].to_list()
        y = anatomy['y'].to_list()
        z = anatomy['z'].to_list()
        chs_rois_dict = dict(zip(chs, rois))
        chs_x = dict(zip(chs, x))
        chs_y = dict(zip(chs, y))
        chs_z = dict(zip(chs, z))

        rois_ordered = []
        for ch in ch_names:
            if ch in chs_rois_dict:
                rois_ordered.append(chs_rois_dict[ch])
            else:
                rois_ordered.append(None)

        x_ordered = []
        for ch in ch_names:
            if ch in chs_x:
                x_ordered.append(chs_x[ch])
            else:
                x_ordered.append(np.nan)

        y_ordered = []
        for ch in ch_names:
            if ch in chs_y:
                y_ordered.append(chs_y[ch])
            else:
                y_ordered.append(np.nan)

        z_ordered = []
        for ch in ch_names:
            if ch in chs_z:
                z_ordered.append(chs_z[ch])
            else:
                z_ordered.append(np.nan)

        # print(ch_names)
        self.ei_anatomy['Channel'] = ch_names
        self.ei_anatomy['x'] = x_ordered
        self.ei_anatomy['y'] = y_ordered
        self.ei_anatomy['z'] = z_ordered
        self.ei_anatomy[self.seg_name] = rois_ordered
        self.ei[self.seg_name] = rois_ordered

        print(self.ei_anatomy)

    def _select_chans(self):
        self._select_chans_win = ItemSelectionWin(self.ieeg.ch_names)
        self._select_chans_win.SELECTION_SIGNAL.connect(self._get_selected_chans)
        self._select_chans_win.show()

    def _get_selected_chans(self, chans):
        self.chans = chans
        logger.info(f"Selected channels are {chans}")

    def _compute_ei(self):
        window = float(self._win_le.text())
        step = float(self._step_le.text())
        low = [float(self._lfreq_low_le.text()), float(self._lfreq_high_le.text())]
        high = [float(self._hfreq_low_le.text()), float(self._hfreq_high_le.text())]
        bias = float(self._bias_le.text())
        threshold = float(self._threshold_le.text())
        tau = float(self._decay_le.text())
        H = float(self._duration_le.text())

        params = {'window': window, 'step': step, 'low': low, 'high': high, 'bias': bias,
                  'threshold': threshold, 'tau': tau, 'H': H}

        logger.info("Start computing EI")
        logger.info(f"Epileptogenicity Index params are {params}")

        if len(self.chans) != len(self.ieeg.ch_names):
            ieeg = self.ieeg.copy().pick_channels(self.chans)
        else:
            ieeg = self.ieeg
        self._compute_ei_thread = ComputeEI(ieeg, params)
        self._compute_ei_thread.EI_SIGNAL.connect(self._get_ei)
        self._compute_ei_thread.start()

    def _get_ei(self, result):
        logger.info("Finish computing EI")
        self.ei = result[0]
        self.U_n = result[1]

        df = reorder_chs_df(self.ei)
        if df is not None:
            self.ei = df

        if self.anatomy is not None:
            self.set_anatomy(self.subject, self.subjects_dir, self.anatomy.copy())

        onset = self.ei.detection_time.min()

        sz_ch_num = 0
        for i in range(len(self.ei)):
            if not np.isnan(self.ei.iloc[i].detection_time):
                sz_ch_num += 1
        if sz_ch_num == 0:
            QMessageBox.information(self, 'EI calculation', f'No Seizure detected!')
        else:
            sz_ch = self.chans[np.argmin(self.ei.detection_time)]
            QMessageBox.information(self, 'EI calculation', f'Finish calculating EI\n'
                                                            f'Onset at {onset} second in {sz_ch} \n'
                                                            f'{sz_ch_num} Channels detected')

    def _plot_ei_barchart(self):
        if self.ei is not None:
            ch_names = self.chans
            if len(self.ei):
                ch_color = {}
                min_ei = float(self._ez_threshold_le.text())
                if min_ei > 0.99:
                    min_ei = 1
                    self._ez_threshold_le.setText(str(0.99))
                try:
                    group_chans = get_chan_group(chans=ch_names)

                    for idx, gp in enumerate(group_chans):
                        for ch in group_chans[gp]:
                            ch_color[ch] = color[idx]
                except:
                    for ch in ch_names:
                        ch_color[ch] = 'g'

                # fig, ax = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
                # sns.barplot(data=self.ei, x='Channel', y='norm_ER', ci=None, palette=ch_color, ax=ax[0])
                # plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                # ax[0].set_title('Energy Ratio')
                #
                # sns.barplot(data=self.ei, x='Channel', y='norm_EI', ci=None, palette=ch_color, ax=ax[1])
                # ax[1].hlines(min_ei, xmin=-.5, xmax=len(ch_names), colors='r', linestyles='--')
                # ax[1].hlines(1., xmin=-.5, xmax=len(ch_names), colors='r', linestyles='--')
                # ax[1].set_xlim(-.5, len(ch_names))
                # ax[1].set_ylim(0, 1.2)
                # plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                #
                # for index, p in enumerate(ax[1].patches):
                #     _x = p.get_x() + p.get_width() / 2
                #     _y = p.get_y() + p.get_height() + 0.01
                #     ei = round(self.ei.iloc[index].norm_EI, 3)
                #     if ei > 0.3:
                #         ax[1].text(_x, _y, ei, ha="center")
                #
                # ax[1].set_title('Epileptogenicity Index')

                fig, ax = plt.subplots(1, 1, figsize=(20, 4), sharex=True)
                sns.barplot(data=self.ei, x='Channel', y='norm_EI', ci=None, palette=ch_color, ax=ax)
                ax.hlines(min_ei, xmin=-.5, xmax=len(ch_names), colors='r', linestyles='--')
                ax.hlines(1., xmin=-.5, xmax=len(ch_names), colors='r', linestyles='--')
                ax.set_xlim(-.5, len(ch_names))
                ax.set_ylim(0, 1.2)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

                for index, p in enumerate(ax.patches):
                    _x = p.get_x() + p.get_width() / 2
                    _y = p.get_y() + p.get_height() + 0.01
                    ei = round(self.ei.iloc[index].norm_EI, 3)
                    if ei > 0.3:
                        ax.text(_x, _y, ei, ha="center")

                ax.set_title('Epileptogenicity Index')

                fig.tight_layout()
                plt.show()

    def _viz_ieeg(self):
        if self.ei is not None:
            logger.info('Plot EI in iEEG')
            ieeg = self.ieeg.copy()
            ch_names = ieeg.ch_names
            annot = ieeg.annotations
            ei_df = self.ei
            for i in range(len(ch_names)):
                if not np.isnan(ei_df.iloc[i].detection_idx):
                    onset = ei_df.iloc[i].detection_time
                    ch = [[ei_df.iloc[i].Channel]]
                    des = f"{ch[0][0]} EI {ei_df.iloc[i].norm_EI}"
                    annot.append(onset, 0, des, ch)
            ieeg.set_annotations(annot)
            ieeg.plot(scalings='auto', color='k')

    def _display_table(self):
        logger.info("Display EI Table!")
        if self.ei is not None:
            columns = ['Channel', 'detection_time', 'alarm_time', 'ER', 'norm_EI']
            if self.seg_name is not None:
                columns.append(self.seg_name)
            ei = self.ei[columns]
            ei = ei.sort_values(by='norm_EI', ascending=False)
            self._ei_table_win = TableWin(ei)
            self._ei_table_win.show()

    def _ez_viz(self):
        threshold = float(self._ez_threshold_le.text())
        if len(self.ei_anatomy):
            ei_info = self.ei.copy()

            ez_chs = ei_info[ei_info['norm_EI'] >= threshold]['Channel'].to_list()

            ch_info = self.ei_anatomy.copy()

            ez = set(ch_info[ch_info['Channel'].isin(ez_chs)][self.seg_name].to_list())
            if list(ez)[0] is not None:
                self._3d_viz_win = ElectrodesWin(self.subject, self.subjects_dir, threshold, ez_chs,
                                                 ch_info, ei_info, self.seg_name, self.parcellation)
                self._3d_viz_win.CLOSE_SIGNAL.connect(self._close_3d_win)
                self._3d_viz_win.show()
            else:
                QMessageBox.warning(self, 'Anatomy', 'Wrong Montage!')

    def _close_3d_win(self):
        self._3d_viz_win = None

    def _save_excel(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Export', filter="EI (*.xlsx)")
        if len(fname):
            if 'xlsx' not in fname:
                fname += '.xlsx'
            self.ei.to_excel(fname, index=None)
            QMessageBox.information(self, 'Export', 'Finish exporting EI!')

    @safe_event
    def closeEvent(self, event):
        if self._ei_table_win is not None:
            self._ei_table_win.close()
        if self._3d_viz_win is not None:
            self._3d_viz_win.close()
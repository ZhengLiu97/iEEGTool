# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：compute_hfo_win.py
@Author  ：Barry
@Date    ：2022/2/23 20:05 
"""
import mne
import numpy as np
import seaborn as sns
import pandas as pd

from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QMessageBox, QFileDialog
from PyQt5.QtGui import QIcon, QPixmap
from matplotlib import pyplot as plt
from mne_hfo import RMSDetector, events_to_annotations, compute_chs_hfo_rates

from gui.compute_hfo_ui import Ui_MainWindow
from gui.list_win import ItemSelectionWin
from gui.table_win import TableWin
from utils.log_config import create_logger
from utils.thread import ComputeHFOsRate
from utils.process import get_chan_group
from utils.config import color

logger = create_logger(filename='iEEGTool.log')


class RMSHFOWin(QMainWindow, Ui_MainWindow):

    def __init__(self, ieeg):
        super(RMSHFOWin, self).__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('High-Frequency Oscillations')
        self._set_icon()

        self.ieeg = ieeg
        self.chans = ieeg.ch_names
        self.params = None
        self.detector = None
        self.hfo_rate_df = None

        self._slot_connection()

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _set_icon(self):
        save_icon = QIcon()
        save_icon.addPixmap(QPixmap("icon/save.svg"), QIcon.Normal, QIcon.Off)
        self._save_hfo_rates_action.setIcon(save_icon)

        wave_icon = QIcon()
        wave_icon.addPixmap(QPixmap("icon/square-wave.svg"), QIcon.Normal, QIcon.Off)
        self._viz_ieeg_action.setIcon(wave_icon)

        bar_icon = QIcon()
        bar_icon.addPixmap(QPixmap("icon/bar-chart.svg"), QIcon.Normal, QIcon.Off)
        self._plot_barchart_action.setIcon(bar_icon)

        help_icon = QIcon()
        help_icon.addPixmap(QPixmap("icon/help.svg"), QIcon.Normal, QIcon.Off)
        self._help_action.setIcon(help_icon)

    def _slot_connection(self):
        self._save_hfo_rates_action.triggered.connect(self._save_hfo)
        self._viz_ieeg_action.triggered.connect(self._viz_ieeg)
        self._plot_barchart_action.triggered.connect(self._plot_hfo_rate)

        self._select_chan_btn.clicked.connect(self._select_chan)
        self._compute_btn.clicked.connect(self._compute_hfo)
        self._display_table_btn.clicked.connect(self._display_hfo_table)

    def _select_chan(self):
        self._select_chan_win = ItemSelectionWin(self.ieeg.ch_names)
        self._select_chan_win.SELECTION_SIGNAL.connect(self._get_chans)
        self._select_chan_win.show()

    def _get_chans(self, chans):
        self.chans = chans
        logger.info(f"Selected channels are {self.chans}")

    def get_compute_params(self):
        filter_band = self._freq_band_le.text().split(' ')
        if len(filter_band) != 2:
            print(filter_band)
            QMessageBox.warning(self, 'Frequency', 'Wrong frequency input!')
            return
        filter_band = (float(filter_band[0]), float(filter_band[1]))
        threshold = float(self._threshold_le.text())
        win_size = int(self._win_size_le.text())
        overlap = float(self._overlap_le.text())
        sfreq = self.ieeg.info['sfreq']
        self.params = {
            'filter_band': filter_band, 'threshold': threshold,
            'win_size': win_size, 'overlap': overlap, 'sfreq': sfreq}
        print(self.params)

    def _compute_hfo(self):
        self.get_compute_params()
        detector = RMSDetector(**self.params)
        if len(self.chans) != len(self.ieeg.ch_names):
            ieeg = self.ieeg.copy().pick_channels(self.chans)
        else:
            ieeg = self.ieeg
        self._compute_hfo_thread = ComputeHFOsRate(ieeg, detector)
        self._compute_hfo_thread.HFO_SIGNAL.connect(self._get_hfo_rate)
        self._compute_hfo_thread.start()

    def _get_hfo_rate(self, detector):
        self.detector = detector
        self.hfo_df = self.detector.df_
        if len(self.hfo_df):
            self.hfo_rate_df = pd.DataFrame()
            self.hfo_rate_dict = compute_chs_hfo_rates(annot_df=self.hfo_df, rate='s')
            self.hfo_rate_df['Channel'] = list(self.hfo_rate_dict.keys())
            self.hfo_rate_df['HFO_Rate (sec)'] = list(self.hfo_rate_dict.values())
            QMessageBox.information(self, 'HFO', f'HFOs Detected!\n'
                                                 f'{len(self.hfo_rate_df)} Channels have HFOs')
        else:
            self.hfo_rate_df = None
            QMessageBox.information(self, 'HFO', 'No HFOs Detected!')

    def _plot_hfo_rate(self):
        if self.hfo_rate_df is not None:
            logger.info('Display HFOs Rate')
            group_chans = get_chan_group(chans=self.chans)
            ch_color = {}
            for idx, gp in enumerate(group_chans):
                for ch in group_chans[gp]:
                    ch_color[ch] = color[idx]

            fig, ax = plt.subplots(figsize=(20, 3))
            sns.barplot(data=self.hfo_rate_df, x='Channel', y='HFO_Rate (sec)',
                        palette=ch_color, ax=ax)
            ax.set_title('High Frequency Oscillations Rate')
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            fig.tight_layout()
            plt.show()

    def _display_hfo_table(self):
        if self.hfo_rate_df is not None:
            self._hfo_table_win = TableWin(self.hfo_rate_df)
            self._hfo_table_win.show()

    def _viz_ieeg(self):
        onset = self.hfo_df['onset']
        duration = self.hfo_df['duration']
        ch_names = self.hfo_df['channels']
        ch_names = [[ch_name] for ch_name in ch_names]
        des = [f"HFOs {ch_names[i][0]}" for i in range(len(ch_names))]
        annot = mne.Annotations(onset=onset, duration=duration, description=des, ch_names=ch_names)
        ieeg = self.ieeg.copy()
        ieeg.set_annotations(annot)
        ieeg.plot(scalings='auto', n_channels=20, color='k')

    def _save_hfo(self):
        if (self.hfo_df is not None) and (self.hfo_rate_df is not None):
            fname, _ = QFileDialog.getSaveFileName(self, 'Export HFO to Excel', 'data',
                                                     filter="HFO Rates (*.xlsx)")
            if len(fname):
                self.hfo_df.to_excel(fname, index=None)
                hfo_rate_fname = fname[:fname.rfind('.')] + '_hfo_rate.xlsx'
                self.hfo_rate_df.to_excel(hfo_rate_fname, index=None)
                QMessageBox.information(self, 'Save HFO Rates', 'Finish saving HFO Rates')

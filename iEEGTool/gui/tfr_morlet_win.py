# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：tfr_morlet_win.py
@Author  ：Barry
@Date    ：2022/2/23 11:55 
"""
import mne
import numpy as np

from mne.io import BaseRaw
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QButtonGroup, QMessageBox
from PyQt5.QtCore import pyqtSignal

from gui.tfr_morlet_ui import Ui_MainWindow
from gui.list_win import ItemSelectionWin
from utils.process import make_epoch
from utils.log_config import create_logger
from utils.thread import ComputeTFR

logger = create_logger(filename='iEEGTool.log')


class TFRMorletWin(QMainWindow, Ui_MainWindow):

    def __init__(self, ieeg):
        super(TFRMorletWin, self).__init__()
        self.setupUi(self)
        self._center_win()
        self.setWindowTitle('Time-Frequency Response')
        if isinstance(ieeg, BaseRaw):
            self.ieeg = make_epoch(ieeg)
        else:
            self.ieeg = ieeg
        self._compute_chans = ieeg.ch_names
        self._fig_chans = None
        self._compute_params = dict()
        self._fig_params = dict()
        self._tfr = None
        self.mode = 'zscore'

        fmin = str(int(ieeg.info['highpass']))
        fmax = str(int(ieeg.info['lowpass']))
        fmin = '0.1' if fmin == '0' else fmin
        self._freq_band_le.setText(fmin + ' ' + fmax)

        tmin = str(0)
        tmax = str(round(self.ieeg.tmax, 3))
        self._fig_time_le.setText(tmin + ' ' + tmax)

        self._radio_group = QButtonGroup(self)
        self._radio_group.addButton(self._zscore_btn, 0)
        self._radio_group.addButton(self._zlogratio_btn, 1)
        self._radio_group.addButton(self._mean_btn, 2)
        self._radio_group.addButton(self._ratio_btn, 3)
        self._radio_group.addButton(self._logratio_btn, 4)
        self._radio_group.addButton(self._percent_btn, 5)

        self._slot_connection()

    def _center_win(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _slot_connection(self):
        self._select_chan_btn.clicked.connect(self._select_chan)
        self._compute_btn.clicked.connect(self._compute_tfr)
        self._fig_select_chan_btn.clicked.connect(self._select_fig_chans)
        self._plot_btn.clicked.connect(self._plot_tfr)
        self._radio_group.buttonClicked.connect(self._get_mode)

    def _get_mode(self):
        mode_id = self._radio_group.checkedId()
        mode = ['zscore', 'zlogratio', 'mean', 'ratio', 'logratio', 'percent']
        self.mode = mode[mode_id]
        logger.info(f"Set mode as {self.mode}")

    def _select_chan(self):
        self._select_chan_win = ItemSelectionWin(self.ieeg.ch_names)
        self._select_chan_win.SELECTION_SIGNAL.connect(self._get_compute_chans)
        self._select_chan_win.show()

    def _get_compute_chans(self, chans):
        self._compute_chans = chans
        if len(chans) == 1:
            self._fig_chans = chans
        logger.info(f"Selected channels are {chans}")

    def _select_fig_chans(self):
        self._select_fig_chan_win = ItemSelectionWin(self._compute_chans)
        self._select_fig_chan_win.SELECTION_SIGNAL.connect(self._get_fig_chans)
        self._select_fig_chan_win.show()

    def _get_fig_chans(self, chans):
        self._fig_chans = chans
        logger.info(f"Selected channels are {chans}")

    def get_compute_params(self):
        freq_band = self._freq_band_le.text().split(' ')
        if len(freq_band) != 2:
            QMessageBox.warning(self, 'Frequency', 'Wrong frequency input!')
            return
        fmin = float(freq_band[0])
        fmax = float(freq_band[1])
        step = float(self._freq_step_le.text())
        denom_ncycles = float(self._denom_ncycles_le.text())
        n_jobs = int(self._njobs_le.text())
        log_freq = self._log_freq_cbx.currentText()
        log_freq = True if log_freq == 'True' else False
        if log_freq:
            num = (fmax - fmin) // step
            freqs = np.logspace(*np.log10([fmin, fmax]), num=int(num))
        else:
            freqs = np.arange(fmin, fmax, step)
        n_cycles = freqs / denom_ncycles
        self._compute_params['freqs'] = freqs
        self._compute_params['n_cycles'] = n_cycles
        self._compute_params['n_jobs'] = n_jobs
        self._compute_params['average'] = True
        self._compute_params['use_fft'] = True
        self._compute_params['decim'] = 4 if log_freq else 1

    def _compute_tfr(self):
        ieeg = self.ieeg.copy()
        if len(self._compute_chans):
            ieeg.pick_channels(self._compute_chans)
        self.get_compute_params()
        self._fig_freq_band_le.setText(self._freq_band_le.text())
        self._compute_tfr_morlet_thread = ComputeTFR(ieeg, compute_method='morlet',
                                                     params=self._compute_params)
        self._compute_tfr_morlet_thread.COMPUTE_SIGNAL.connect(self._get_tfr)
        self._compute_tfr_morlet_thread.start()

    def _get_tfr(self, tfr):
        self._tfr = tfr
        logger.info('Calculating TFR using Morlet finished!')
        QMessageBox.information(self, 'TFR', 'Calculating TFR using Morlet finished!')

    def get_fig_params(self):
        time = self._fig_time_le.text().split(' ')
        if len(time) != 2:
            print(time)
            QMessageBox.warning(self, 'Time', 'Wrong time input!')
            return
        tmin = float(time[0])
        tmax = float(time[1])

        freq = self._fig_freq_band_le.text().split(' ')
        if len(freq) != 2:
            print(freq)
            QMessageBox.warning(self, 'Frequency', 'Wrong frequency input!')
            return
        fmin = float(freq[0])
        fmax = float(freq[1])

        baseline = self._fig_baseline_le.text().split(' ')
        if len(baseline) != 2:
            print(baseline)
            QMessageBox.warning(self, 'Baseline', 'Wrong baseline input!')
            return
        baseline = (float(baseline[0]), float(baseline[1]))

        log_trans = self._fig_log_cbx.currentText()
        log_trans = True if log_trans == 'True' else False

        self._fig_params['baseline'] = baseline
        self._fig_params['tmin'] = tmin
        self._fig_params['tmax'] = tmax
        self._fig_params['fmin'] = fmin
        self._fig_params['fmax'] = fmax
        self._fig_params['vmin'] = self._tfr.data.min()
        self._fig_params['cmap'] = 'jet'
        self._fig_params['dB'] = log_trans
        self._fig_params['mode'] = self.mode
        logger.info(f'Plot params are {self._fig_params}')

    def _plot_tfr(self):
        if self._tfr is not None:
            logger.info("Plotting TFR")
            self.get_fig_params()
            if self._fig_chans is not None:
                index = [self._compute_chans.index(chan) for chan in self._fig_chans]
                if len(index):
                    print(index)
                    self._tfr.plot(index, title='auto', **self._fig_params)
            else:
                QMessageBox.warning(self, 'Channel', 'Please select channels!')
